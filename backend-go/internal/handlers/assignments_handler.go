package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/grading"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
	"grading-gateway/internal/schemas"
	"grading-gateway/internal/tools"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"gorm.io/gorm"
)

// 创建作业handler
func CreateAssignment(c *gin.Context) {
	var req schemas.AssignmentCreate
	if err := c.ShouldBindJSON(&req); err != nil {
		// 处理失败
		log.Printf("创建作业请求格式存在问题： %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"detail": "创建作业数据请求格式有误", "error": err.Error()})
		return
	}

	// 请求成功，创建作业并入库
	// json化rubric
	rubricBytes, _ := json.Marshal(req.Rubric)
	var assignment = models.Assignment{
		OwnerID:    middleware.GetUserIDFromContextMust(c),
		CourseName: req.CourseName,
		ClassName:  req.ClassName,
		TaskName:   req.TaskName,
		Question:   req.Question,
		Rubric:     string(rubricBytes),
	}

	// 入库失败
	if err := database.DB.Create(&assignment).Error; err != nil {
		log.Printf("创建作业入库失败：%v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"message": "创建作业入库失败", "error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, schemas.AssignmentResponse{
		Summary:    &schemas.AssignmentSummary{Status: "EMPTY"},
		CourseName: assignment.CourseName,
		ClassName:  assignment.ClassName,
		ID:         assignment.ID,
		TaskName:   assignment.TaskName,
		Question:   assignment.Question,
		Rubric:     req.Rubric,
	})
}

// 获取所有作业handler
func GetAssignments(c *gin.Context) {
	// 从数据库中获取所有assignmnets
	var assignments []models.Assignment
	query := database.DB.Order("id desc")
	role, _ := middleware.GetRoleFromContext(c)
	owner, _ := middleware.GetUserIDFromContext(c)
	if role == "student" {
		username, _ := middleware.GetUsernameFromContext(c)
		query = query.Where("EXISTS (SELECT 1 FROM submissions WHERE submissions.assignment_id = assignments.id AND submissions.student_id = ?)", username)
	} else if role != "admin" {
		query = query.Where("owner_id = ?", owner)
	}
	if err := query.Find(&assignments).Error; err != nil {
		log.Printf("获取所有作业失败：%v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"message": "获取所有作业失败！", "error": err.Error()})
		return
	}

	// 获取成功后返回json
	ids := make([]uint, 0, len(assignments))
	for _, assignment := range assignments {
		ids = append(ids, assignment.ID)
	}
	summaries, err := assignmentSummaries(c.Request.Context(), database.DB, ids)
	if err != nil {
		// Progress is additive metadata; assignments remain accessible if a
		// statistics query is temporarily unavailable.
		log.Printf("读取作业进度失败：%v", err)
	}
	var res []schemas.AssignmentResponse
	for _, ass := range assignments {
		var rubricMap map[string]interface{}
		if ass.Rubric != "" {
			if err := json.Unmarshal([]byte(ass.Rubric), &rubricMap); err != nil {
				rubricMap = make(map[string]interface{})
			}
		} else {
			rubricMap = make(map[string]interface{})
		}

		var summary *schemas.AssignmentSummary
		if value, ok := summaries[ass.ID]; ok {
			summary = &value
		}
		res = append(res, schemas.AssignmentResponse{
			PortalHomeworkID: ass.PortalHomeworkID, RequirementsStatus: ass.RequirementsStatus,
			Summary:    summary,
			ID:         ass.ID,
			CourseName: ass.CourseName,
			ClassName:  ass.ClassName,
			TaskName:   ass.TaskName,
			Question:   ass.Question,
			Rubric:     rubricMap,
		})
	}

	if res == nil {
		res = []schemas.AssignmentResponse{}
	}
	c.JSON(http.StatusOK, res)
}

// 获取单个作业详情handler
func GetAssignment(c *gin.Context) {
	idStr := c.Param("id")
	assignmentID, err := strconv.ParseUint(idStr, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "无效的作业ID"})
		return
	}

	// 获取用户角色和用户名
	role, roleErr := middleware.GetRoleFromContext(c)
	username, usernameErr := middleware.GetUsernameFromContext(c)

	// 1. 【动静分离 - 静态部分】：从 Redis 缓存中获取作业基础信息（带防穿透、防击穿保护）
	// Portal requirements are completed in the background; read the current
	// assignment so an earlier empty cache cannot hide newly synchronized text.
	assignmentBase := &models.Assignment{}
	err = database.DB.WithContext(c.Request.Context()).First(assignmentBase, uint(assignmentID)).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			log.Printf("获取单个作业未找到：%v", err)
			c.JSON(http.StatusNotFound, gin.H{"message": "获取单个作业详情出错", "detail": "作业未找到"})
			return
		}
		log.Printf("获取单个作业出错：%v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"message": "系统繁忙，获取作业详情失败"})
		return
	}

	// 2. 【动静分离 - 动态部分】：根据角色权限，单独去 MySQL 查询属于该用户的 Submissions
	var submissions []models.Submission
	subQuery := database.DB.Where("assignment_id = ?", assignmentID)

	// 如果是学生，只查询自己的提交
	if roleErr == nil && usernameErr == nil && role == "student" {
		subQuery = subQuery.Where("student_id = ?", username).Order("score DESC")
	} else {
		// 教师/管理员获取所有提交
		subQuery = subQuery.Order("score DESC")
	}

	if err := subQuery.Find(&submissions).Error; err != nil {
		log.Printf("获取作业提交记录出错：%v", err)
	}

	// 3. 组装数据并返回 JSON
	var rubricMap map[string]interface{}
	if assignmentBase.Rubric != "" {
		json.Unmarshal([]byte(assignmentBase.Rubric), &rubricMap)
	} else {
		rubricMap = make(map[string]interface{})
	}

	var formattedSubmissions []map[string]interface{}
	for _, sub := range submissions {
		// 这里刻意不返回 merged_content，以减小单次作业详情响应体积
		var plag []map[string]interface{}
		if sub.PlagiarismReport != "" && sub.PlagiarismReport != "[]" {
			_ = json.Unmarshal([]byte(sub.PlagiarismReport), &plag)
		} else {
			plag = make([]map[string]interface{}, 0)
		}

		var aigc map[string]interface{}
		if sub.AIGCReport != "" && sub.AIGCReport != "{}" {
			_ = json.Unmarshal([]byte(sub.AIGCReport), &aigc)
		}

		var match map[string]interface{}
		if sub.CodeDocMatchReport != "" && sub.CodeDocMatchReport != "{}" {
			_ = json.Unmarshal([]byte(sub.CodeDocMatchReport), &match)
		}

		formattedSubmissions = append(formattedSubmissions, map[string]interface{}{
			"material_batch_id": sub.MaterialBatchID, "grading_job_id": sub.GradingJobID,
			"id":                    sub.ID,
			"student_id":            sub.StudentID,
			"student_name":          sub.StudentName,
			"score":                 sub.Score,
			"feedback":              sub.Feedback,
			"plagiarism_reports":    plag,
			"aigc_report":           aigc,
			"code_doc_match_report": match,
			"assignment_id":         sub.AssignmentID,
			"is_human_reviewed":     sub.IsHumanReviewed,
			"human_feedback":        sub.HumanFeedback,
			"human_score":           sub.HumanScore,
		})
	}

	if formattedSubmissions == nil {
		formattedSubmissions = make([]map[string]interface{}, 0)
	}

	c.JSON(http.StatusOK, gin.H{
		"portal_homework_id": assignmentBase.PortalHomeworkID, "requirements_status": assignmentBase.RequirementsStatus,
		"id":          assignmentBase.ID,
		"course_name": assignmentBase.CourseName,
		"class_name":  assignmentBase.ClassName,
		"task_name":   assignmentBase.TaskName,
		"question":    assignmentBase.Question,
		"rubric":      rubricMap,
		"submissions": formattedSubmissions,
	})
}

func GetAssignmentSubmissions(c *gin.Context) {
	assignmentID := c.Param("id")

	// 获取用户角色和用户名
	role, roleErr := middleware.GetRoleFromContext(c)
	username, usernameErr := middleware.GetUsernameFromContext(c)

	var submissions = make([]models.Submission, 0)
	query := database.DB.Where("assignment_id = ?", assignmentID)
	if batch := c.Query("batch_id"); batch != "" {
		query = query.Where("material_batch_id = ?", batch)
	}
	query = submissionClassScope(query, database.DB, assignmentID, c.Query("class_id"))

	// 如果是学生，只能查看自己的提交
	if roleErr == nil && usernameErr == nil && role == "student" {
		query = query.Where("student_id = ?", username)
	}

	if err := query.Order("score DESC").Find(&submissions).Error; err != nil {
		log.Printf("获取所有提交出错: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"message": "获取所有提交出错", "error": err.Error()})
		return
	}

	// 格式化输出
	var formattedSubmissions []map[string]interface{}
	for _, sub := range submissions {
		formattedSubmissions = append(formattedSubmissions, tools.FormatSubmission(sub))
	}

	if formattedSubmissions == nil {
		formattedSubmissions = make([]map[string]interface{}, 0)
	}

	c.JSON(http.StatusOK, formattedSubmissions)
}

// 删除作业handler
func DeleteAssignment(c *gin.Context) {
	id := c.Param("id")

	// 获取一致性服务
	consistencySvc := database.GetConsistencyService()
	ctx := context.Background()

	//
	assignmentKey := fmt.Sprintf("assignment:%s", id)

	// 使用一致性服务删除缓存和数据
	err := consistencySvc.DeleteThenInvalidate(ctx, "assignment", assignmentKey, func() error {
		// 这是实际的MySQL删除操作
		return database.DB.Delete(&models.Assignment{}, id).Error
	})

	if err != nil {
		log.Printf("删除作业失败： %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "删除作业失败"})
		return
	}

	// 清理作业相关的提交缓存
	consistencySvc.InvalidateByPattern(ctx, fmt.Sprintf("*assignment:%s*", id))
	consistencySvc.InvalidateByPattern(ctx, fmt.Sprintf("*submission:*:%s*", id))
	// 【新增】：主动清理我们刚刚引入的 DB 旁路缓存
	assignmentID, _ := strconv.ParseUint(id, 10, 32)
	cache.InvalidateAssignmentCache(ctx, uint(assignmentID))

	c.Status(http.StatusNoContent)
}

// 上传学生提交handler
func UploadAssignment(c *gin.Context) {
	actor, ok := gradingActor(c)
	if !ok {
		return
	}
	assignmentID := c.Param("id")
	id, parseErr := grading.ResourceID(assignmentID)
	var target models.Assignment
	if parseErr != nil || database.DB.First(&target, id).Error != nil {
		c.JSON(404, gin.H{"detail": "作业不存在"})
		return
	}
	file, err := c.FormFile("batch_file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "请上传本次作业 ZIP/RAR/7z"})
		return
	}
	uploadDir := "./uploads/assignments"
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		c.JSON(500, gin.H{"detail": "无法创建上传目录"})
		return
	}
	uploadDir = filepath.Join(uploadDir, uuid.NewString())
	if err := os.MkdirAll(uploadDir, 0700); err != nil {
		c.JSON(500, gin.H{"detail": "无法创建批次上传目录"})
		return
	}
	savePath := filepath.Join(uploadDir, filepath.Base(strings.ReplaceAll(file.Filename, `\`, "/")))
	if err := c.SaveUploadedFile(file, savePath); err != nil {
		c.JSON(500, gin.H{"detail": "文件保存失败"})
		return
	}
	path, _ := filepath.Abs(savePath)
	autoGrade := c.PostForm("auto_grade") == "true"
	if autoGrade && !checkGradingTarget(c, actor, "homework", assignmentID) {
		return
	}
	batch, err := materials.Default().Create(c.Request.Context(), materials.Actor{UserID: actor.UserID, Role: actor.Role}, materials.Request{ClassID: c.PostForm("class_id"), SourcePath: path, AssignmentID: id, AutoGrade: autoGrade}, c.GetHeader("Idempotency-Key"))
	if err != nil {
		c.JSON(400, gin.H{"detail": err.Error()})
		return
	}
	message := "作业材料已导入，正在整理；完成后请主动点击开始批改"
	if autoGrade {
		message = "作业材料已导入，整理完成后自动批改"
	}
	c.JSON(202, gin.H{"outcome": "accepted", "job_id": batch.ID, "job_type": "homework_materials", "batch_id": batch.ID, "message": message})
}

// 一键清空所有学生提交handler
func ClearAllSubmissions(c *gin.Context) {
	assignmentID := c.Param("id")

	if err := database.DB.Where("assignment_id = ?", assignmentID).Delete(&models.Submission{}).Error; err != nil {
		log.Printf("一键清空失败： %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "一键清空失败", "error": err.Error()})
		return
	}

	c.Status(http.StatusNoContent)
}
