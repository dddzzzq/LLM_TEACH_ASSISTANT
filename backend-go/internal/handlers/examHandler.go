package handlers

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
	"grading-gateway/internal/mq"
	"grading-gateway/internal/schemas"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"gorm.io/gorm/clause"
)

// 创建试卷
func CreateExam(c *gin.Context) {
	var req schemas.ExamCreate
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	exam := models.Exam{
		Name:          req.Name,
		QuestionCount: 0,
		TotalScore:    0,
	}
	if err := database.DB.Create(&exam).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建试卷失败"})
		return
	}
	c.JSON(http.StatusCreated, exam)
}

// 获取试卷列表
func GetExams(c *gin.Context) {
	var exams []models.Exam
	if err := database.DB.Order("id desc").Find(&exams).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取列表失败"})
		return
	}
	if exams == nil {
		exams = []models.Exam{}
	}
	c.JSON(http.StatusOK, exams)
}

// 获取单个试卷（含所有题目）
func GetExam(c *gin.Context) {
	id := c.Param("id")
	var exam models.Exam
	if err := database.DB.Preload("Questions").First(&exam, id).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "未找到试卷"})
		return
	}
	c.JSON(http.StatusOK, exam)
}

// 删除试卷
func DeleteExam(c *gin.Context) {
	id := c.Param("id")

	// 使用 Select(clause.Associations) 明确指示 GORM 触发级联删除
	if err := database.DB.Select(clause.Associations).Delete(&models.Exam{}, id).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除失败: " + err.Error()})
		return
	}

	c.Status(http.StatusNoContent)
}

// 添加试卷题目
func AddExamQuestion(c *gin.Context) {
	examID := c.Param("id")
	var req schemas.ExamQuestionCreate
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var exam models.Exam
	if err := database.DB.First(&exam, examID).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "未找到归属试卷"})
		return
	}

	question := models.ExamQuestion{
		ExamID:         exam.ID,
		QuestionNumber: req.QuestionNumber,
		QuestionText:   req.QuestionText,
		StandardAnswer: req.StandardAnswer,
		Rubric:         req.Rubric,
		MaxScore:       req.MaxScore,
	}

	// 开启事务：添加题目并更新试卷总分
	tx := database.DB.Begin()
	if err := tx.Create(&question).Error; err != nil {
		tx.Rollback()
		c.JSON(http.StatusInternalServerError, gin.H{"error": "添加题目失败"})
		return
	}

	exam.QuestionCount += 1
	exam.TotalScore += req.MaxScore
	if err := tx.Save(&exam).Error; err != nil {
		tx.Rollback()
		c.JSON(http.StatusInternalServerError, gin.H{"error": "更新试卷总分失败"})
		return
	}
	tx.Commit()

	c.JSON(http.StatusCreated, question)
}

// 提交学生试卷并开启后台并发处理
func UploadStudentExam(c *gin.Context) {
	examID := c.Param("id")
	studentID := c.PostForm("student_id")
	if studentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "student_id 不能为空"})
		return
	}

	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无法解析表单数据"})
		return
	}
	files := form.File["images"]
	if len(files) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请至少上传一张图片"})
		return
	}

	// 保存图片以在前端展示
	uploadDir := filepath.Join("uploads", "exams", examID, studentID)
	os.MkdirAll(uploadDir, os.ModePerm)

	var savedPaths []string
	for _, file := range files {
		// 生成随机名字，防止中文乱码和重复
		ext := filepath.Ext(file.Filename)
		newFileName := uuid.New().String() + ext
		savePath := filepath.Join(uploadDir, newFileName)

		if err := c.SaveUploadedFile(file, savePath); err == nil {
			savedPaths = append(savedPaths, savePath)
		}
	}

	// 生成 UUID 作为 jobID
	jobID := uuid.New().String()
	log.Printf("创建异步试卷任务: job=%s, exam=%s, student=%s", jobID, examID, studentID)

	ctx := context.Background()

	// 1. 在 MySQL 中创建 AsyncJob 记录，状态设为 PENDING
	err = mq.CreateAsyncJob(jobID, models.JobTypeExam, examID, studentID)
	if err != nil {
		log.Printf("ERROR: Failed to create async job in database: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建异步任务失败"})
		return
	}

	// 2. 在 Redis 中缓存初始状态
	if err := cache.SetJobStatus(ctx, jobID, string(models.JobStatusPending), "任务已创建，等待处理"); err != nil {
		log.Printf("WARNING: Failed to cache job status in Redis: %v", err)
		// 不返回错误，继续执行
	}

	// 3. 调用 mq.PublishExamTask 将任务推送到 Kafka
	if err := mq.PublishExamTask(jobID, examID, studentID, savedPaths); err != nil {
		log.Printf("ERROR: Failed to publish exam task to Kafka: %v", err)
		// 更新任务状态为 FAILED
		cache.SetJobStatus(ctx, jobID, string(models.JobStatusFailed), "发布到消息队列失败")
		mq.UpdateAsyncJobStatus(jobID, models.JobStatusFailed, "发布到消息队列失败")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "任务提交失败，请重试"})
		return
	}

	// 4. 修改 HTTP 响应，返回 http.StatusAccepted (202)，并在 JSON 中包含 job_id
	c.JSON(http.StatusAccepted, gin.H{
		"message": fmt.Sprintf("已收到学生 %s 的 %d 张试卷图片，任务已加入队列，系统将异步处理批改，请稍后通过 job_id 查询状态。", studentID, len(savedPaths)),
		"job_id":  jobID,
	})
}

// 获取一次考试的所有学生成绩概览
func GetExamResultsSummary(c *gin.Context) {
	examID := c.Param("id")

	// 获取用户角色和用户名
	role, roleErr := middleware.GetRoleFromContext(c)
	username, usernameErr := middleware.GetUsernameFromContext(c)

	var res []schemas.ExamResultSummary
	var err error

	// 学生只能查看自己的成绩
	if roleErr == nil && usernameErr == nil && role == "student" {
		err = database.DB.Raw(`
			SELECT 
				se.student_id, 
				er.total_score, 
				er.student_exam_id 
			FROM exam_reports er 
			JOIN student_exams se ON se.id = er.student_exam_id 
			WHERE se.exam_id = ? AND se.student_id = ?
		`, examID, username).Scan(&res).Error
	} else {
		// 教师/管理员获取所有人成绩
		err = database.DB.Raw(`
			SELECT 
				se.student_id, 
				er.total_score, 
				er.student_exam_id 
			FROM exam_reports er 
			JOIN student_exams se ON se.id = er.student_exam_id 
			WHERE se.exam_id = ?
		`, examID).Scan(&res).Error
	}

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取成绩单失败"})
		return
	}

	if res == nil {
		res = []schemas.ExamResultSummary{}
	}
	c.JSON(http.StatusOK, res)
}

// 获取单个学生的详细报告
func GetStudentDetailedReport(c *gin.Context) {
	studentExamID := c.Param("student_exam_id")
	var studentExam models.StudentExam

	if err := database.DB.Preload("Report").Preload("Answers.Question").Preload("Images").First(&studentExam, studentExamID).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "未找到该学生的答卷信息"})
		return
	}

	// 检查权限：学生只能查看自己的报告
	role, roleErr := middleware.GetRoleFromContext(c)
	username, usernameErr := middleware.GetUsernameFromContext(c)
	if roleErr == nil && usernameErr == nil && role == "student" {
		if studentExam.StudentID != username {
			c.JSON(http.StatusForbidden, gin.H{"error": "无权查看他人的试卷报告"})
			return
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"student_id": studentExam.StudentID,
		"exam_id":    studentExam.ExamID,
		"report":     studentExam.Report,
		"answers":    studentExam.Answers,
		"images":     studentExam.Images,
	})
}

// 删除单个学生的成绩
func DeleteStudentExamResult(c *gin.Context) {
	studentExamID := c.Param("student_exam_id")
	if err := database.DB.Delete(&models.StudentExam{}, studentExamID).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除记录失败"})
		return
	}
	c.Status(http.StatusNoContent)
}
