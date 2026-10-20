package handlers

import (
	"errors"
	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/rpa"
)

func GetRPAAttention(c *gin.Context) {
	userID, _ := middleware.GetUserIDFromContext(c)
	notices, err := rpa.Attention(userID)
	if err != nil {
		c.JSON(500, gin.H{"error": "读取待处理事项失败"})
		return
	}
	c.Header("Cache-Control", "no-store")
	c.JSON(200, notices)
}

func GetRPARequirements(c *gin.Context) {
	userID, _ := middleware.GetUserIDFromContext(c)
	job, err := rpa.Owned(userID, c.Param("id"))
	if err != nil {
		c.JSON(404, gin.H{"error": "任务不存在或无权访问"})
		return
	}
	if job.Status == "CANCELLED" {
		c.JSON(409, gin.H{"error": "任务已取消"})
		return
	}
	data, err := rpa.FileRequirements(job.ID, c.Query("file_id"))
	if errors.Is(err, gorm.ErrRecordNotFound) {
		c.JSON(404, gin.H{"error": "附件或作业不存在"})
		return
	}
	if err != nil {
		c.JSON(409, gin.H{"error": "附件已开始批改或尚未匹配作业，请刷新任务"})
		return
	}
	c.Header("Cache-Control", "no-store")
	c.JSON(200, data)
}

func supplementRPARequirements(c *gin.Context, jobID string) {
	var payload struct {
		FileID   string         `json:"file_id" binding:"required"`
		Question string         `json:"question" binding:"required"`
		Rubric   map[string]any `json:"rubric" binding:"required"`
	}
	if c.ShouldBindJSON(&payload) != nil {
		c.JSON(400, gin.H{"error": "请填写题目要求和 JSON 对象形式的评分标准"})
		return
	}
	err := rpa.SupplementRequirements(jobID, payload.FileID, payload.Question, payload.Rubric)
	if errors.Is(err, gorm.ErrRecordNotFound) {
		c.JSON(404, gin.H{"error": "附件或作业不存在"})
		return
	}
	if errors.Is(err, rpa.ErrRequirementsChanged) || errors.Is(err, rpa.ErrInvalidRequirements) {
		c.JSON(409, gin.H{"error": err.Error()})
		return
	}
	if err != nil {
		c.JSON(500, gin.H{"error": "保存题目与评分标准失败，请稍后重试"})
		return
	}
	c.JSON(200, gin.H{"message": "题目与评分标准已保存，系统将自动继续批改"})
}
