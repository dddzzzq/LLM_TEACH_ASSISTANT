package database

import (
	"fmt"
	"log"
	"strconv"
	"strings"

	"grading-gateway/internal/models"

	"gorm.io/driver/mysql"
	"gorm.io/gorm"
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
		&models.Assignment{},
		&models.Submission{},
		&models.Exam{},
		&models.ExamQuestion{},
		&models.StudentExam{},
		&models.StudentExamImage{},
		&models.StudentExamAnswer{},
		&models.ExamReport{},
		&models.AsyncJob{},
	)
	if err != nil {
		log.Fatalf("自动迁移表结构失败: %v", err)
	}
	fmt.Println("表结构映射完成！")
}

func SaveAssignment(assignmentID, studentID, studentName string, score float64, feedback, mergedContent, plagJSON, aigcJSON, matchJSON string) {
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
	if err := DB.Create(&submission).Error; err != nil {
		log.Printf("[DB错误] 无法保存学生 %s 记录: %v\n", studentID, err)
	}
}
