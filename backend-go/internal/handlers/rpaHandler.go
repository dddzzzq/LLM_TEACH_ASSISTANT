package handlers

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"grading-gateway/internal/database"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
	"grading-gateway/internal/rpa"
)

func CreateRPAJob(c *gin.Context) {
	userID, _ := middleware.GetUserIDFromContext(c)
	var target rpa.Target
	if err := c.ShouldBindJSON(&target); err != nil {
		c.JSON(400, gin.H{"error": "请提供课程与作业名称"})
		return
	}
	job, err := rpa.CreateWithKey(c.Request.Context(), userID, target, c.GetHeader("Idempotency-Key"))
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusAccepted, rpa.Detail(job))
}

func ListRPAJobs(c *gin.Context) {
	userID, _ := middleware.GetUserIDFromContext(c)
	var jobs []models.RPAJob
	if err := database.DB.Where("user_id = ?", userID).Order("created_at DESC").Limit(50).Find(&jobs).Error; err != nil {
		c.JSON(500, gin.H{"error": "无法读取任务"})
		return
	}
	c.JSON(200, jobs)
}

func GetRPAJob(c *gin.Context) {
	userID, _ := middleware.GetUserIDFromContext(c)
	job, err := rpa.Owned(userID, c.Param("id"))
	if err != nil {
		c.JSON(404, gin.H{"error": "任务不存在或无权访问"})
		return
	}
	c.Header("Cache-Control", "no-store")
	c.JSON(200, rpa.Detail(job))
}

func ControlRPAJob(c *gin.Context) {
	userID, _ := middleware.GetUserIDFromContext(c)
	job, err := rpa.Owned(userID, c.Param("id"))
	if err != nil {
		c.JSON(404, gin.H{"error": "任务不存在或无权访问"})
		return
	}
	action := c.Param("action")
	if action == "map-file" {
		if job.Status == "CANCELLED" {
			c.JSON(409, gin.H{"error": "任务已取消"})
			return
		}
		var payload struct {
			FileID    string `json:"file_id"`
			ClassName string `json:"class_name"`
		}
		if c.ShouldBindJSON(&payload) != nil || strings.TrimSpace(payload.ClassName) == "" || len([]rune(payload.ClassName)) > 255 {
			c.JSON(400, gin.H{"error": "请提供有效的班级名称"})
			return
		}
		result := database.DB.Model(&models.RPAFile{}).Where("id = ? AND job_id = ? AND grading_status = ?", payload.FileID, job.ID, "WAITING_RUBRIC").Updates(map[string]any{"class_name": strings.TrimSpace(payload.ClassName), "assignment_id": 0, "message": "用户已核对班级，等待匹配作业"})
		if result.Error != nil || result.RowsAffected == 0 {
			c.JSON(409, gin.H{"error": "附件不存在或已开始批改"})
			return
		}
		c.JSON(200, gin.H{"ok": true})
		return
	}
	allowed := map[string]bool{"pause": true, "resume": true, "cancel": true, "input": true, "class-scope": true, "bind-exports": true, "restart": true, "confirm-target": true}
	if !allowed[action] {
		c.JSON(404, gin.H{"error": "未知操作"})
		return
	}
	if action == "restart" {
		err := rpa.Start(c.Request.Context(), job.ID, true)
		if err != nil {
			c.JSON(409, gin.H{"error": err.Error()})
			return
		}
		c.JSON(200, gin.H{"ok": true})
		return
	}
	if action == "cancel" && (job.Status == "QUEUED" || job.Status == "INTERRUPTED") {
		if err := rpa.CancelQueued(c.Request.Context(), job); err != nil {
			c.JSON(500, gin.H{"error": "取消失败"})
			return
		}
		c.JSON(200, gin.H{"status": "CANCELLED"})
		return
	}
	var payload any
	if action == "input" || action == "class-scope" || action == "bind-exports" || action == "confirm-target" {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 16*1024)
		if err := json.NewDecoder(c.Request.Body).Decode(&payload); err != nil {
			c.JSON(400, gin.H{"error": "操作参数无效"})
			return
		}
	}
	raw, err := rpa.WorkerRequest(c.Request.Context(), "POST", "/tasks/"+job.ID+"/"+action, payload)
	if err != nil {
		workerError(c, err)
		return
	}
	if action != "input" {
		_ = rpa.SaveState(job.ID, raw)
	}
	c.Header("Cache-Control", "no-store")
	c.Data(200, "application/json", raw)
}

func RPABrowserFrame(c *gin.Context) {
	userID, _ := middleware.GetUserIDFromContext(c)
	job, err := rpa.Owned(userID, c.Param("id"))
	if err != nil {
		c.JSON(404, gin.H{"error": "任务不存在或无权访问"})
		return
	}
	raw, err := rpa.WorkerRequest(c.Request.Context(), "GET", "/tasks/"+job.ID+"/frame", nil)
	if err != nil {
		workerError(c, err)
		return
	}
	c.Header("Cache-Control", "no-store")
	c.Data(200, "application/json", raw)
}

func workerError(c *gin.Context, err error) {
	var we *rpa.WorkerError
	if errors.As(err, &we) {
		c.JSON(we.Status, gin.H{"error": we.Message})
		return
	}
	c.JSON(503, gin.H{"error": err.Error()})
}
