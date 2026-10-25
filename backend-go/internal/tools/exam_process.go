package tools

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"gorm.io/gorm"
	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/models"
	"grading-gateway/pb"
)

type examImageResult struct {
	Path, Text string
	QuestionID *uint
}

// parallelExamSteps keeps input ordering and returns errors instead of fabricating zero scores.
func parallelExamSteps(n int, work func(int) error) error {
	errors := make([]error, n)
	var wg sync.WaitGroup
	slots := make(chan struct{}, 4)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) { defer wg.Done(); slots <- struct{}{}; defer func() { <-slots }(); errors[i] = work(i) }(i)
	}
	wg.Wait()
	for _, err := range errors {
		if err != nil {
			return err
		}
	}
	return nil
}
func evaluateExam(exam models.Exam, paths []string, client pb.ComputeServiceClient) ([]examImageResult, []models.StudentExamAnswer, float64, string, error) {
	if client == nil || len(paths) == 0 || len(exam.Questions) == 0 {
		return nil, nil, 0, "", fmt.Errorf("试卷、图片或评分服务未就绪")
	}
	var questions strings.Builder
	for _, q := range exam.Questions {
		fmt.Fprintf(&questions, "题号 %d: %s\n", q.QuestionNumber, q.QuestionText)
	}
	images := make([]examImageResult, len(paths))
	err := parallelExamSteps(len(paths), func(i int) error {
		data, err := os.ReadFile(paths[i])
		if err != nil {
			return fmt.Errorf("第 %d 页图片读取失败", i+1)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		ocr, err := client.ExtractText(ctx, &pb.ExtractRequest{Filename: filepath.Base(paths[i]), FileContent: data})
		if err != nil || ocr == nil || strings.TrimSpace(ocr.TextContent) == "" {
			return fmt.Errorf("第 %d 页识别失败或没有可读文本，请检查完整清晰的答卷；未按零分处理", i+1)
		}
		images[i] = examImageResult{Path: filepath.ToSlash(paths[i]), Text: ocr.TextContent}
		idctx, idcancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer idcancel()
		identified, iderr := client.IdentifyQuestionNumber(idctx, &pb.IdentifyQuestionRequest{OcrText: ocr.TextContent, QuestionListStr: questions.String()})
		if iderr == nil && identified != nil {
			for _, q := range exam.Questions {
				if q.QuestionNumber == int(identified.QuestionNumber) {
					id := q.ID
					images[i].QuestionID = &id
					break
				}
			}
		}
		return nil
	})
	if err != nil {
		return nil, nil, 0, "", err
	}
	var full strings.Builder
	for i, img := range images {
		fmt.Fprintf(&full, "\n[图片 %d 内容]:\n%s\n", i+1, img.Text)
	}
	answers := make([]models.StudentExamAnswer, len(exam.Questions))
	feedbacks := make([]string, len(exam.Questions))
	err = parallelExamSteps(len(exam.Questions), func(i int) error {
		q := exam.Questions[i]
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()
		grade, err := client.GradeExamQuestion(ctx, &pb.GradeExamQuestionRequest{QuestionText: q.QuestionText, StandardAnswer: q.StandardAnswer, Rubric: q.Rubric, MaxScore: float32(q.MaxScore), FullStudentText: full.String()})
		if err != nil || grade == nil {
			return fmt.Errorf("第 %d 题评分失败，未保存为零分", q.QuestionNumber)
		}
		score := float64(grade.Score)
		if math.IsNaN(score) || math.IsInf(score, 0) || score < 0 || score > q.MaxScore {
			return fmt.Errorf("第 %d 题评分超出有效范围", q.QuestionNumber)
		}
		answers[i] = models.StudentExamAnswer{ExamQuestionID: q.ID, OCRText: grade.StudentAnswerExtracted, Score: score, Feedback: grade.Feedback}
		feedbacks[i] = fmt.Sprintf("题号 %d (满分 %.1f): 得分 %.1f, 评语: %s", q.QuestionNumber, q.MaxScore, score, grade.Feedback)
		return nil
	})
	if err != nil {
		return nil, nil, 0, "", err
	}
	total := 0.0
	for _, answer := range answers {
		total += answer.Score
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	summary, err := client.SummarizeExam(ctx, &pb.SummarizeExamRequest{AllFeedback: feedbacks})
	if err != nil || summary == nil || strings.TrimSpace(summary.SummaryReport) == "" {
		return nil, nil, 0, "", fmt.Errorf("整卷总评生成失败")
	}
	return images, answers, total, summary.SummaryReport, nil
}

// ProcessExamSubmission computes a complete result before atomically replacing prior results.
func ProcessExamSubmission(examIDStr, studentID string, imagePaths []string) error {
	id, err := strconv.ParseUint(examIDStr, 10, 32)
	if err != nil || id == 0 {
		return fmt.Errorf("试卷 ID 无效")
	}
	exam, err := cache.GetExamWithCache(context.Background(), uint(id))
	if err != nil || exam == nil {
		return fmt.Errorf("无法读取试卷")
	}
	images, answers, total, summary, err := evaluateExam(*exam, imagePaths, grpcclient.Client)
	if err != nil {
		return err
	}
	err = database.DB.Transaction(func(tx *gorm.DB) error {
		if err := tx.Where("exam_id = ? AND student_id = ?", id, studentID).Delete(&models.StudentExam{}).Error; err != nil {
			return err
		}
		record := models.StudentExam{ExamID: uint(id), StudentID: studentID}
		if err := tx.Create(&record).Error; err != nil {
			return err
		}
		for i, img := range images {
			path := img.Path
			if relative, e := filepath.Rel(".", path); e == nil && !strings.HasPrefix(relative, "..") {
				path = "/" + filepath.ToSlash(relative)
			}
			if err := tx.Create(&models.StudentExamImage{StudentExamID: record.ID, ImagePath: path, ImageIndex: i + 1, ExamQuestionID: img.QuestionID}).Error; err != nil {
				return err
			}
		}
		for i := range answers {
			answers[i].StudentExamID = record.ID
		}
		if err := tx.Create(&answers).Error; err != nil {
			return err
		}
		return tx.Create(&models.ExamReport{StudentExamID: record.ID, TotalScore: total, Summary: summary}).Error
	})
	if err == nil {
		cache.InvalidateExamCache(context.Background(), uint(id))
	}
	return err
}
