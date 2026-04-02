package tools

import (
	"archive/zip"
	"context"
	"encoding/json"
	"fmt"
	"grading-gateway/internal/database"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/models"
	"grading-gateway/pb"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"golang.org/x/text/encoding/simplifiedchinese"
)

type GradingTask struct {
	StudentID    string
	StudentName  string
	AssignmentID string
	FilePaths    []string
}

// 提取压缩文件内容并分组
func extractAndGroupTasks(assignmentID string, zipPath string) ([]GradingTask, error) {
	reader, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	extractDir := filepath.Join(filepath.Dir(zipPath), fmt.Sprintf("extracted_%s", assignmentID))
	os.MkdirAll(extractDir, os.ModePerm)

	studentMap := make(map[string]*GradingTask)

	for _, file := range reader.File {
		if file.FileInfo().IsDir() {
			continue
		}

		filename := file.Name
		if !utf8.ValidString(filename) {
			decoded, err := simplifiedchinese.GB18030.NewDecoder().String(filename)
			if err == nil {
				filename = decoded
			}
		}

		filename = filepath.ToSlash(filename)
		if strings.Contains(filename, "__MACOSX") || strings.HasSuffix(filename, ".DS_Store") {
			continue
		}

		var studentID, studentName string
		parts := strings.Split(filename, "/")
		if len(parts) > 1 {
			studentID = parts[0]
		} else {
			baseName := filepath.Base(filename)
			studentID = strings.TrimSuffix(baseName, filepath.Ext(baseName))
		}

		if studentID == "" {
			continue
		}

		// 根据 '-' 分割学号和姓名
		if strings.Contains(studentID, "-") {
			splitResult := strings.SplitN(studentID, "-", 2)
			studentID = splitResult[0]
			studentName = splitResult[1]
		} else {
			studentName = "未知"
		}

		safeFilename := fmt.Sprintf("%s_%s", studentID, filepath.Base(filename))
		savePath := filepath.Join(extractDir, safeFilename)

		if err := extractSingleFile(file, savePath); err != nil {
			continue
		}

		if _, exists := studentMap[studentID]; !exists {
			studentMap[studentID] = &GradingTask{
				StudentID:    studentID,
				StudentName:  studentName,
				AssignmentID: assignmentID,
				FilePaths:    []string{},
			}
		}
		studentMap[studentID].FilePaths = append(studentMap[studentID].FilePaths, savePath)
	}

	var tasks []GradingTask
	for _, task := range studentMap {
		tasks = append(tasks, *task)
	}
	return tasks, nil
}

// 功能函数，提取单个压缩文件
func extractSingleFile(file *zip.File, destPath string) error {
	rc, err := file.Open()
	if err != nil {
		return err
	}
	defer rc.Close()
	destFile, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer destFile.Close()
	_, err = io.Copy(destFile, rc)
	return err
}

// processPipeline Go 核心调度中心
func ProcessPipeline(assignmentID string, zipPath string) {
	log.Printf("\n======================================================\n")
	log.Printf("[调度中心] 启动全新批改流线, 作业ID: %s\n", assignmentID)
	startTime := time.Now()

	var assignment models.Assignment
	if err := database.DB.First(&assignment, assignmentID).Error; err != nil {
		log.Printf("[错误] 找不到作业ID %s\n", assignmentID)
		return
	}

	tasks, err := extractAndGroupTasks(assignmentID, zipPath)
	if err != nil || len(tasks) == 0 {
		log.Printf("[错误] 解压失败或包内无文件\n")
		return
	}
	totalTasks := int32(len(tasks))
	log.Printf("[调度中心] 物理拆包完成，共分发 %d 个学生的独立作业堆。\n", totalTasks)

	// Map 1 阶段：并发调用 Go 原生的文件解析引擎
	var wgMap1 sync.WaitGroup
	var studentTextsMap sync.Map
	var map1Completed int32 = 0

	// 并发度20，并发提取文件
	extractConcurrencyLimit := make(chan struct{}, 20)

	// 预编译正则表达式，用于清理连续3个以上的莫名其妙空行
	newlineRegex := regexp.MustCompile(`\n{3,}`)

	for _, task := range tasks {
		wgMap1.Add(1)
		go func(t GradingTask) {
			defer wgMap1.Done()

			extractConcurrencyLimit <- struct{}{}
			defer func() { <-extractConcurrencyLimit }()

			var mergedStudentText strings.Builder
			for _, filePath := range t.FilePaths {
				fileBytes, err := os.ReadFile(filePath)
				if err != nil {
					continue
				}

				// 提取文件
				extractedText := ExtractContent(filepath.Base(filePath), fileBytes, 0)

				if extractedText != "" {
					if !strings.HasPrefix(extractedText, "---") {
						mergedStudentText.WriteString(fmt.Sprintf("--- 文件开始: %s ---\n\n%s\n\n--- 文件结束: %s ---\n\n", filepath.Base(filePath), extractedText, filepath.Base(filePath)))
					} else {
						// 针对嵌套压缩包返回的已经带有分隔符的文本，直接拼接
						mergedStudentText.WriteString(extractedText + "\n")
					}
				}
			}

			// 清理掉所有无法被 UTF-8 识别的乱码字节（如 GBK 遗留等），用  代替
			// 防止 gRPC 序列化崩溃 和 MySQL Error 1366
			finalText := strings.ToValidUTF8(mergedStudentText.String(), "")

			// 换行符清洗逻辑
			// 1. 统一处理 Windows 和老式 Mac 的换行符为 \n
			finalText = strings.ReplaceAll(finalText, "\r\n", "\n")
			finalText = strings.ReplaceAll(finalText, "\r", "\n")
			// 2. 将连续出现 3个及以上的 \n 强制压缩为 2个 \n
			finalText = newlineRegex.ReplaceAllString(finalText, "\n\n")

			if strings.TrimSpace(finalText) == "" {
				finalText = "【系统容错记录】文件破损、非文本格式或内容全空。"
			}
			studentTextsMap.Store(t.StudentID, finalText)

			currentProg := atomic.AddInt32(&map1Completed, 1)
			log.Printf("[Map 1 进度] 文件纯Go解析完成: %d/%d | 当前 Goroutine 数: %d", currentProg, totalTasks, runtime.NumGoroutine())
		}(task)
	}

	wgMap1.Wait()
	log.Printf("[调度中心] Map 1 阶段(Go 原生解析引擎) 全部跑完，耗时: %v！", time.Since(startTime))

	// Reduce 阶段：全局查重
	plagMap := make(map[string]string)
	studentTextsMap.Range(func(key, value interface{}) bool {
		plagMap[key.(string)] = value.(string)
		return true
	})

	log.Printf("[调度中心] 发起全班查重，数据包大小: %d 份...", len(plagMap))
	// 查重涉及批量大模型调用，耗时极长，不用超时处理
	// ctxPlag, cancelPlag := context.WithTimeout(context.Background(), 600*time.Second)
	ctxPlag := context.Background()
	plagRes, errPlag := grpcclient.Client.CheckPlagiarism(ctxPlag, &pb.PlagiarismRequest{StudentTexts: plagMap})
	// cancelPlag()

	var globalPlagiarismResult map[string]interface{}
	if errPlag != nil {
		log.Printf("[降级警告] 全局查重失败: %v\n", errPlag)
		globalPlagiarismResult = make(map[string]interface{})
	} else {
		json.Unmarshal([]byte(plagRes.PlagiarismResultsJson), &globalPlagiarismResult)
	}

	// Map 2 阶段：并发调用 AI 大模型评分
	numWorkers := 10 // Python 现在是推理机，分配 10 个网络协程连接
	jobs := make(chan GradingTask, len(tasks))
	var wgMap2 sync.WaitGroup
	var map2Completed int32 = 0

	for w := 1; w <= numWorkers; w++ {
		wgMap2.Add(1)
		go func(workerID int) {
			defer wgMap2.Done()

			for t := range jobs {
				textVal, _ := studentTextsMap.Load(t.StudentID)
				studentText := textVal.(string)

				var studentPlagJSON string = "[]"
				if reportList, ok := globalPlagiarismResult[t.StudentID]; ok {
					b, _ := json.Marshal(reportList)
					studentPlagJSON = string(b)
				}

				// A. AIGC
				// ctxAIGC, cancelAIGC := context.WithTimeout(context.Background(), 30*time.Second)
				ctxAIGC := context.Background()
				aigcRes, errAIGC := grpcclient.Client.DetectAIGC(ctxAIGC, &pb.AIGCRequest{TextContent: studentText})
				// cancelAIGC()

				aigcJSON := "{}"
				if errAIGC == nil {
					aigcJSON = aigcRes.AigcReportJson
				}

				// B. 评分
				// ctxGrade, cancelGrade := context.WithTimeout(context.Background(), 300*time.Second)
				ctxGrade := context.Background()
				gradeRes, errGrade := grpcclient.Client.GradeHomework(ctxGrade, &pb.GradeRequest{
					StudentId:             t.StudentID,
					Question:              assignment.Question,
					RubricJson:            assignment.Rubric,
					StudentText:           studentText,
					PlagiarismReportsJson: studentPlagJSON,
					AigcReportJson:        aigcJSON,
				})
				// cancelGrade()

				finalScore := float64(-1.0)
				finalFeedback := "【系统容错记录】AI 评分请求超时或失败"
				mergedContent := studentText
				matchJSON := "{}"

				if errGrade == nil && gradeRes != nil {
					finalScore = float64(gradeRes.TotalScore)
					finalFeedback = gradeRes.Feedback
					mergedContent = gradeRes.MergedContent
					matchJSON = gradeRes.CodeDocMatchReportJson
				}

				database.SaveAssignment(assignmentID, t.StudentID, t.StudentName, finalScore, finalFeedback, mergedContent, studentPlagJSON, aigcJSON, matchJSON)

				currentProg := atomic.AddInt32(&map2Completed, 1)
				log.Printf(" <- [Worker %d] 学生 [%s] 批改完毕！进度: %d/%d", workerID, t.StudentID, currentProg, totalTasks)
			}
		}(w)
	}

	for _, t := range tasks {
		jobs <- t
	}
	close(jobs)

	wgMap2.Wait()
	log.Printf("\n[调度中心] 作业 %s 批改任务全流程结束！\n总耗时: %v\n======================================================\n", assignmentID, time.Since(startTime))

	// 启动成绩池化处理
	go AddPoolingToPipeline(assignmentID)
}
