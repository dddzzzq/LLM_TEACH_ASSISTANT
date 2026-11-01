package database

import (
	"errors"
	"fmt"
	"log"
	"strconv"
	"strings"

	"grading-gateway/internal/models"

	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// DB 全局数据库示例
var DB *gorm.DB

func InitDB(dsn string) {
	var err error
	// 必须使用 = 而不是 := ，这样才能把连接赋给全局的 DB 变量
	DB, err = gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Fatalf("fail to connect to database: %v", err)
	}

	fmt.Println("成功连接到数据库！正在进行模型映射检查...")

	// AutoMigrate: 自动同步 Go 结构体到 MySQL 的表结构。不会删除已有数据，只会新增字段。
	err = DB.AutoMigrate(
		&models.User{},
		&models.ChatSession{},
		&models.ChatMessage{},
		&models.AgentRun{},
		&models.AgentEvent{},
		&models.ToolDefinition{},
		&models.Assignment{},
		&models.Submission{},
		&models.AsyncJob{},
		&models.RPAJob{},
		&models.RPAFile{},
		&models.MaterialBatch{},
		&models.PortalConnection{},
		&models.PortalOperation{},
		&models.PortalOperationEvent{},
		&models.PortalCatalogSnapshot{},
		&models.PortalCourse{},
		&models.PortalHomework{},
		&models.PortalHomeworkDetail{},
		&models.PortalCourseClass{},
		&models.PortalAssignmentBinding{},
		&models.PortalWorkflow{},
		&models.PortalWorkflowItem{},
	)
	if err != nil {
		log.Fatalf("自动迁移表结构失败: %v", err)
	}
	fmt.Println("表结构映射完成！")
}

func SaveAssignment(assignmentID, studentID, studentName string, score float64, feedback, mergedContent, plagJSON, aigcJSON, matchJSON string, provenance ...string) error {
	aID, _ := strconv.ParseUint(assignmentID, 10, 32)

	// 再次清洗确保所有存入 DB 的字符都是绝对纯净的 UTF-8
	safeMergedContent := strings.ToValidUTF8(mergedContent, "")
	safeFeedback := strings.ToValidUTF8(feedback, "")

	submission := models.Submission{
		StudentID:          studentID,
		StudentName:        studentName,
		AssignmentID:       uint(aID),
		Score:              score,
		Feedback:           safeFeedback,
		MergeContent:       safeMergedContent,
		PlagiarismReport:   plagJSON,
		AIGCReport:         aigcJSON,
		CodeDocMatchReport: matchJSON,
		IsHumanReviewed:    false,
	}
	if len(provenance) >= 2 {
		submission.MaterialBatchID, submission.GradingJobID = provenance[0], provenance[1]
	}
	return SaveSubmission(submission)
}

func SaveSubmission(submission models.Submission) error {
	submission.MergeContent = strings.ToValidUTF8(submission.MergeContent, "")
	submission.Feedback = strings.ToValidUTF8(submission.Feedback, "")
	if err := DB.Create(&submission).Error; err != nil {
		log.Printf("[DB错误] 无法保存学生 %s 记录: %v\n", submission.StudentID, err)
		return err
	}
	return nil
}

// Explicit regrading refreshes AI fields in this batch only. Human review fields
// are deliberately absent from Updates, including reviews saved while AI runs.
func SaveRegradedSubmission(submission models.Submission) error {
	if submission.MaterialBatchID == "" || submission.GradingJobID == "" {
		return fmt.Errorf("重新批改缺少材料批次或任务ID")
	}
	submission.MergeContent = strings.ToValidUTF8(submission.MergeContent, "")
	submission.Feedback = strings.ToValidUTF8(submission.Feedback, "")
	return DB.Transaction(func(tx *gorm.DB) error {
		var existing models.Submission
		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("assignment_id = ? AND material_batch_id = ? AND student_id = ?", submission.AssignmentID, submission.MaterialBatchID, submission.StudentID).Order("id DESC").First(&existing).Error
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return tx.Create(&submission).Error
		}
		if err != nil {
			return err
		}
		return tx.Model(&existing).Updates(map[string]any{
			"student_name": submission.StudentName, "score": submission.Score, "feedback": submission.Feedback,
			"merge_content": submission.MergeContent, "plagiarism_reports": submission.PlagiarismReport,
			"aigc_reports": submission.AIGCReport, "code_doc_match_report": submission.CodeDocMatchReport,
			"grading_job_id": submission.GradingJobID, "reading_status": submission.ReadingStatus,
			"reading_report": submission.ReadingReport, "score_details": submission.ScoreDetails,
		}).Error
	})
}
