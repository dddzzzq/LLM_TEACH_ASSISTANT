package tools

import (
	"archive/zip"
	"context"
	"encoding/json"
	"fmt"
	"grading-gateway/internal/database"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/models"
	"grading-gateway/pb"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"golang.org/x/text/encoding/simplifiedchinese"

	"grading-gateway/internal/cache"
	"strconv"
)

type GradingTask struct {
	StudentID    string
	StudentName  string
	AssignmentID string
	FilePaths    []string
	RootPath     string
	Sources      []materials.ArchiveSource
	SkippedFiles []materials.SkippedFile
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
				RootPath:     extractDir,
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
func ProcessPipeline(assignmentID string, zipPath string) error {
	return ProcessPipelineForJob(assignmentID, zipPath, models.AsyncJob{})
}

func ProcessPipelineForJob(assignmentID string, zipPath string, job models.AsyncJob) error {
	log.Printf("\n======================================================\n")
	log.Printf("[调度中心] 启动全新批改流线, 作业ID: %s\n", assignmentID)
	startTime := time.Now()
	saveSubmission := database.SaveSubmission
	var intent struct {
		Regrade bool `json:"regrade"`
	}
	if json.Unmarshal([]byte(job.PayloadJSON), &intent) == nil && intent.Regrade && job.MaterialBatchID != "" {
		saveSubmission = database.SaveRegradedSubmission
	}

	// 1. 获取作业信息及评分标准 (走缓存)
	parsedID, _ := strconv.Atoi(assignmentID)
	assignmentPtr, err := cache.GetAssignmentWithCache(context.Background(), uint(parsedID))
	if err != nil || assignmentPtr == nil {
		log.Printf("[错误] 找不到作业ID %s (或缓存读取失败)\n", assignmentID)
		return fmt.Errorf("找不到作业ID %s", assignmentID)
	}
	assignment := *assignmentPtr
	if job.QuestionSnapshot != "" {
		assignment.Question, assignment.Rubric = job.QuestionSnapshot, job.RubricSnapshot
	}

	var tasks []GradingTask
	if job.MaterialBatchID != "" {
		var batch models.MaterialBatch
		if err := database.DB.First(&batch, "id = ? AND grading_job_id = ?", job.MaterialBatchID, job.ID).Error; err != nil {
			return fmt.Errorf("材料批次与批改任务不匹配")
		}
		if batch.ManifestPath != zipPath || batch.AssignmentID != uint(parsedID) {
			return fmt.Errorf("材料路径或作业不匹配")
		}
		manifest, loadErr := materials.LoadManifest(batch.ManifestPath, batch.ManifestSHA256)
		if loadErr != nil {
			return loadErr
		}
		for _, student := range manifest.Students {
			paths := []string{}
			for _, file := range student.Files {
				paths = append(paths, filepath.Join(filepath.Dir(batch.ManifestPath), filepath.FromSlash(file.Path)))
			}
			tasks = append(tasks, GradingTask{StudentID: student.ID, StudentName: student.Name, AssignmentID: assignmentID, FilePaths: paths, RootPath: filepath.Dir(batch.ManifestPath), Sources: manifest.StudentSources(student), SkippedFiles: student.SkippedFiles})
		}
	} else {
		tasks, err = extractAndGroupTasks(assignmentID, zipPath)
	}
	if err != nil || len(tasks) == 0 {
		log.Printf("[错误] 解压失败或包内无文件\n")
		return fmt.Errorf("解压失败或包内无学生文件")
	}
	totalTasks := int32(len(tasks))
	log.Printf("[调度中心] 物理拆包完成，共分发 %d 个学生的独立作业堆。\n", totalTasks)

	// Map 1 阶段：并发调用 Go 原生的文件解析引擎
	var wgMap1 sync.WaitGroup
	var studentTextsMap sync.Map
	var map1Completed int32 = 0

	// 并发度20，并发提取文件
	extractConcurrencyLimit := make(chan struct{}, 20)

	for _, task := range tasks {
		wgMap1.Add(1)
		go func(t GradingTask) {
			defer wgMap1.Done()

			extractConcurrencyLimit <- struct{}{}
			defer func() { <-extractConcurrencyLimit }()

			parsed := prepareStudentMaterials(context.Background(), t, cachedStudentOCR)
			studentTextsMap.Store(t.StudentID, parsed)

			currentProg := atomic.AddInt32(&map1Completed, 1)
			log.Printf("[Map 1 进度] 学生 %s 解析完成，读取状态: %s, 文件数: %d | 进度: %d/%d", t.StudentID, parsed.Reading.Status, len(parsed.Files), currentProg, totalTasks)
		}(task)
	}

	wgMap1.Wait()
	log.Printf("[调度中心] Map 1 阶段(Go 原生解析引擎) 全部跑完，耗时: %v！", time.Since(startTime))

	// Reduce 阶段：全局查重
	plagMap := make(map[string]string)
	studentTextsMap.Range(func(key, value interface{}) bool {
		parsed := value.(studentMaterials)
		if parsed.Reading.Status == "COMPLETE" {
			plagMap[key.(string)] = parsed.jsonText()
		}
		return true
	})

	log.Printf("[调度中心] 发起全班查重，数据包大小: %d 份...", len(plagMap))
	// 查重涉及批量大模型调用，耗时极长，不用超时处理
	ctxPlag, cancelPlag := context.WithTimeout(context.Background(), 60*time.Minute)
	// defer cancelPlag()
	// ctxPlag := context.Background()
	plagResults, errPlag := checkPlagiarismBatched(ctxPlag, grpcclient.Client, plagMap, 8<<20)
	cancelPlag()

	var globalPlagiarismResult map[string]interface{}
	if errPlag != nil {
		log.Printf("[降级警告] 全局查重失败: %v\n", errPlag)
		globalPlagiarismResult = make(map[string]interface{})
	} else {
		globalPlagiarismResult = plagResults
	}

	// Map 2 阶段：并发调用 AI 大模型评分
	numWorkers := 10 // Python 现在是推理机，分配 10 个网络协程连接
	jobs := make(chan GradingTask, len(tasks))
	var wgMap2 sync.WaitGroup
	var map2Completed int32 = 0
	var failedGrades int32

	for w := 1; w <= numWorkers; w++ {
		wgMap2.Add(1)
		go func(workerID int) {
			defer wgMap2.Done()

			for t := range jobs {
				textVal, _ := studentTextsMap.Load(t.StudentID)
				parsed := textVal.(studentMaterials)
				studentText := parsed.jsonText()
				if len(studentText) > 24<<20 {
					parsed.Reading.Status = "NEEDS_REVIEW"
					parsed.Reading.ReviewStage = "TRANSPORT"
					parsed.Reading.Warnings = append(parsed.Reading.Warnings, "材料传输超过24MiB预算，未截断后评分")
				}
				readingJSON, _ := json.Marshal(parsed.Reading)
				// Partial reading is diagnostic information, not a reason to
				// withhold a score for the student's other readable material.
				if len(studentText) > 24<<20 {
					feedback := "材料传输超过预算，待复核，未截断后评分。"
					atomic.AddInt32(&failedGrades, 1)
					if err := saveSubmission(models.Submission{AssignmentID: uint(parsedID), StudentID: t.StudentID, StudentName: t.StudentName, Score: -1, Feedback: feedback, MergeContent: parsed.analysisText(), MaterialBatchID: job.MaterialBatchID, GradingJobID: job.ID, ReadingStatus: "NEEDS_REVIEW", ReadingReport: string(readingJSON)}); err != nil {
						log.Printf("学生 [%s] 待复核记录保存失败: %v", t.StudentID, err)
					}
					currentProg := atomic.AddInt32(&map2Completed, 1)
					log.Printf(" <- [Worker %d] 学生 [%s] 待复核，继续其他学生！进度: %d/%d", workerID, t.StudentID, currentProg, totalTasks)
					continue
				}

				var studentPlagJSON string = "[]"
				if reportList, ok := globalPlagiarismResult[t.StudentID]; ok {
					b, _ := json.Marshal(reportList)
					studentPlagJSON = string(b)
				}

				// A. AIGC
				ctxAIGC, cancelAIGC := context.WithTimeout(context.Background(), 30*time.Second)
				// ctxAIGC := context.Background()

				aigcRes, errAIGC := grpcclient.Client.DetectAIGC(ctxAIGC, &pb.AIGCRequest{TextContent: studentText})
				cancelAIGC()

				aigcJSON := "{}"
				if errAIGC == nil {
					aigcJSON = aigcRes.AigcReportJson
				}

				// B. 评分
				ctxGrade, cancelGrade := context.WithTimeout(context.Background(), 60*time.Minute)
				// ctxGrade := context.Background()
				gradeRes, errGrade := grpcclient.Client.GradeHomework(ctxGrade, &pb.GradeRequest{
					StudentId:             t.StudentID,
					Question:              assignment.Question,
					RubricJson:            assignment.Rubric,
					StudentText:           studentText,
					PlagiarismReportsJson: studentPlagJSON,
					AigcReportJson:        aigcJSON,
				})
				cancelGrade()

				finalScore := float64(-1.0)
				finalFeedback := "【系统容错记录】AI 评分请求超时或失败"
				mergedContent := parsed.analysisText()
				matchJSON := "{}"
				readingStatus := "ERROR"
				scoreDetails := "[]"
				if errGrade == nil && gradeRes != nil {
					finalFeedback = gradeRes.Feedback
					var result struct {
						Schema  string          `json:"schema"`
						Text    string          `json:"text"`
						Reading readingReport   `json:"reading"`
						Details json.RawMessage `json:"score_details"`
					}
					if json.Unmarshal([]byte(gradeRes.MergedContent), &result) == nil && result.Schema == "grading-result.v1" {
						readingStatus = result.Reading.Status
						readingJSON, _ = json.Marshal(result.Reading)
						scoreDetails = string(result.Details)
						mergedContent = result.Text
					} else {
						finalFeedback = "AI 节点未返回完整读取回执，待复核"
					}
				} else if errGrade != nil {
					finalFeedback = "AI 评分通信失败：" + grpcFailureCode(errGrade) + "，请重试或复核"
				}

				if errGrade == nil && gradeRes != nil && (readingStatus == "COMPLETE" || readingStatus == "PARTIAL") && gradeRes.TotalScore >= 0 && !math.IsNaN(float64(gradeRes.TotalScore)) && !math.IsInf(float64(gradeRes.TotalScore), 0) {
					finalScore = float64(gradeRes.TotalScore)
					finalFeedback = gradeRes.Feedback
					matchJSON = gradeRes.CodeDocMatchReportJson
				} else {
					atomic.AddInt32(&failedGrades, 1)
				}

				if err := saveSubmission(models.Submission{AssignmentID: uint(parsedID), StudentID: t.StudentID, StudentName: t.StudentName, Score: finalScore, Feedback: finalFeedback, MergeContent: mergedContent, PlagiarismReport: studentPlagJSON, AIGCReport: aigcJSON, CodeDocMatchReport: matchJSON, MaterialBatchID: job.MaterialBatchID, GradingJobID: job.ID, ReadingStatus: readingStatus, ReadingReport: string(readingJSON), ScoreDetails: scoreDetails}); err != nil {
					log.Printf("学生 [%s] 批改结果保存失败: %v", t.StudentID, err)
					if finalScore >= 0 {
						atomic.AddInt32(&failedGrades, 1)
					}
				}

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
	if failedGrades > 0 {
		return fmt.Errorf("%d/%d 份学生作业需复核或评分失败，其他结果已保存", failedGrades, totalTasks)
	}
	log.Printf("\n[调度中心] 作业 %s 批改任务全流程结束！\n总耗时: %v\n======================================================\n", assignmentID, time.Since(startTime))

	// 启动成绩池化处理
	// A prepared batch keeps rubric-based scores stable and never pools historical batches.
	if job.MaterialBatchID == "" {
		go AddPoolingToPipeline(assignmentID)
	}
	return nil
}
