package tools

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"grading-gateway/internal/database"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/models"
	"grading-gateway/pb"

	"gorm.io/gorm"
)

// ProcessExamSubmission 是负责处理单份学生试卷所有图片的高并发方法
func ProcessExamSubmission(examIDStr string, studentID string, imagePaths []string) {
	log.Printf("\n======================================================\n")
	log.Printf("[试卷调度] 启动异步批改, 试卷ID: %s, 学生: %s, 共 %d 张图片\n", examIDStr, studentID, len(imagePaths))
	startTime := time.Now()

	examID, _ := strconv.ParseUint(examIDStr, 10, 32)

	// 1. 获取试卷及其所有题目
	var exam models.Exam
	if err := database.DB.Preload("Questions").First(&exam, examID).Error; err != nil {
		log.Printf("[错误] 找不到试卷ID %s\n", examIDStr)
		return
	}

	if len(exam.Questions) == 0 {
		log.Printf("[错误] 试卷没有任何题目，无法批改\n")
		return
	}

	// 构建题目列表字符串，给大模型分类图片时使用
	var qListBuilder strings.Builder
	for _, q := range exam.Questions {
		qListBuilder.WriteString(fmt.Sprintf("题号 %d: %s\n", q.QuestionNumber, q.QuestionText))
	}
	questionListStr := qListBuilder.String()

	// 2. 清理历史记录，并创建新的 StudentExam
	var studentExam models.StudentExam
	database.DB.Where("exam_id = ? AND student_id = ?", examID, studentID).First(&studentExam)
	if studentExam.ID != 0 {
		log.Printf("发现该生存在旧记录，执行覆盖删除...")
		database.DB.Delete(&studentExam)
	}

	studentExam = models.StudentExam{
		ExamID:    uint(examID),
		StudentID: studentID,
	}
	if err := database.DB.Create(&studentExam).Error; err != nil {
		log.Printf("[错误] 无法创建学生试卷记录: %v", err)
		return
	}

	// 3. Map阶段：并发对所有图片进行 OCR 提取和题号识别
	var wgImages sync.WaitGroup
	var mu sync.Mutex
	var fullOCRText strings.Builder

	type imageResult struct {
		Index      int
		ImagePath  string
		OCRText    string
		QuestionID *uint
	}
	imgResults := make([]imageResult, len(imagePaths))

	for i, path := range imagePaths {
		wgImages.Add(1)
		go func(idx int, p string) {
			defer wgImages.Done()

			// 读取文件
			content, err := os.ReadFile(p)
			if err != nil {
				log.Printf("读取图片失败: %s, %v", p, err)
				return
			}
			filename := filepath.Base(p)

			// 3.1 请求 Python 进行 OCR
			extCtx, cancel1 := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel1()
			resOCR, err := grpcclient.Client.ExtractText(extCtx, &pb.ExtractRequest{
				Filename:    filename,
				FileContent: content,
			})

			ocrText := ""
			if err == nil && resOCR != nil {
				ocrText = resOCR.TextContent
			}

			// 3.2 如果文本长度可观，呼叫大模型判定该图属于哪道题 (可选步骤，目前已不需存入DB，但可用于日志追溯)
			var qID *uint
			if len(ocrText) > 5 {
				idCtx, cancel2 := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel2()
				idRes, idErr := grpcclient.Client.IdentifyQuestionNumber(idCtx, &pb.IdentifyQuestionRequest{
					OcrText:         ocrText,
					QuestionListStr: questionListStr,
				})

				if idErr == nil && idRes != nil && idRes.QuestionNumber > 0 {
					for _, q := range exam.Questions {
						if q.QuestionNumber == int(idRes.QuestionNumber) {
							qid := q.ID
							qID = &qid
							break
						}
					}
				}
			}

			// 保存当前图片的独立结果
			mu.Lock()
			imgResults[idx] = imageResult{
				Index:      idx,
				ImagePath:  filepath.ToSlash(p), // 注意：如果是windows路径转换为 URL path
				OCRText:    ocrText,
				QuestionID: qID,
			}
			mu.Unlock()

		}(i, path)
	}

	wgImages.Wait()
	log.Printf("[试卷调度] 所有图片OCR及题号识别并发完成！")

	// 汇总所有 OCR 文本，并写入数据库图片记录
	for _, res := range imgResults {
		if res.ImagePath == "" {
			continue
		}
		fullOCRText.WriteString(fmt.Sprintf("\n[图片 %d 内容]:\n%s\n", res.Index+1, res.OCRText))

		// 转换路径格式使得前端可以直接通过 /uploads/ 访问
		relativePath := "/" + filepath.ToSlash(res.ImagePath)

		database.DB.Create(&models.StudentExamImage{
			StudentExamID:  studentExam.ID,
			ImagePath:      relativePath,
			ImageIndex:     res.Index + 1, // 保存图片页码
			ExamQuestionID: res.QuestionID,
		})
	}
	finalOCRText := fullOCRText.String()

	// 4. Reduce/Map阶段：带着全卷文本，并发对所有题目进行判分
	var wgQuestions sync.WaitGroup
	var totalScore float64
	var allFeedbacks []string

	for _, q := range exam.Questions {
		wgQuestions.Add(1)
		go func(question models.ExamQuestion) {
			defer wgQuestions.Done()

			qCtx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			// 呼叫大模型评分
			gradeRes, err := grpcclient.Client.GradeExamQuestion(qCtx, &pb.GradeExamQuestionRequest{
				QuestionText:    question.QuestionText,
				StandardAnswer:  question.StandardAnswer,
				Rubric:          question.Rubric,
				MaxScore:        float32(question.MaxScore),
				FullStudentText: finalOCRText,
			})

			score := float64(0.0)
			feedback := "【系统容错】大模型未返回评语或调用超时。"
			studentAnswer := "未能提取答案"

			if err == nil && gradeRes != nil {
				score = float64(gradeRes.Score)
				feedback = gradeRes.Feedback
				studentAnswer = gradeRes.StudentAnswerExtracted
			}

			// 保存单题答案
			ans := models.StudentExamAnswer{
				StudentExamID:  studentExam.ID,
				ExamQuestionID: question.ID,
				OCRText:        studentAnswer,
				Score:          score,
				Feedback:       feedback,
			}

			// 开启事务级别锁保证线程安全保存
			database.DB.Transaction(func(tx *gorm.DB) error {
				if err := tx.Create(&ans).Error; err != nil {
					return err
				}
				mu.Lock()
				totalScore += score
				allFeedbacks = append(allFeedbacks, fmt.Sprintf("题号 %d (满分 %.1f): 得分 %.1f, 评语: %s",
					question.QuestionNumber, question.MaxScore, score, feedback))
				mu.Unlock()
				return nil
			})

			log.Printf(" -> 题目 %d 批改完成 (%.1f/%.1f)", question.QuestionNumber, score, question.MaxScore)

		}(q)
	}

	wgQuestions.Wait()
	log.Printf("[试卷调度] 所有题目并发批改完成！开始生成总评...")

	// 5. Final Reduce阶段：生成整卷总评
	sumCtx, cancelSum := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancelSum()

	summaryRes, err := grpcclient.Client.SummarizeExam(sumCtx, &pb.SummarizeExamRequest{
		AllFeedback: allFeedbacks,
	})

	summaryText := "生成总结报告失败。"
	if err == nil && summaryRes != nil {
		summaryText = summaryRes.SummaryReport
	}

	// 6. 保存总计报告
	report := models.ExamReport{
		StudentExamID: studentExam.ID,
		TotalScore:    totalScore,
		Summary:       summaryText,
	}
	database.DB.Create(&report)

	log.Printf("\n[试卷调度] 学生 %s 答卷全流程解析及评分完成！总得分: %.1f, 总耗时: %v\n======================================================\n", studentID, totalScore, time.Since(startTime))
}
