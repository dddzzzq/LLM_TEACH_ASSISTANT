package handlers

import (
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
	"grading-gateway/internal/schemas"
	"grading-gateway/internal/tools"
	"net/http"

	"github.com/gin-gonic/gin"
)

// 获取学生提交handler
func GetSubmission(c *gin.Context) {
	id := c.Param("id")
	var sub models.Submission
	if err := database.DB.First(&sub, id).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"detail": "未找到该批改记录"})
		return
	}
	c.JSON(http.StatusOK, tools.FormatSubmission(sub))
}

// 删除单个提交handler
func DeleteSubmission(c *gin.Context) {
	id := c.Param("id")
	if err := database.DB.Delete(&models.Submission{}, id).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "删除失败", "error": err.Error()})
		return
	}
	c.Status(http.StatusNoContent)
}

// 更新学生评分或评语handler
func UpdateSubmission(c *gin.Context) {
	id := c.Param("id")
	var req schemas.SubmissionUpdate
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "请求数据格式有误", "error": err.Error()})
		return
	}
	var sub models.Submission
	if err := database.DB.First(&sub, id).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"detail": "未找到该批改记录"})
		return
	}
	if req.Score != nil {
		sub.Score = *req.Score
	}
	if req.Feedback != nil {
		sub.Feedback = *req.Feedback
	}
	if req.HumanScore != nil {
		sub.HumanScore = *req.HumanScore
	}
	if req.HumanFeedback != nil {
		sub.HumanFeedback = *req.HumanFeedback
	}
	if req.IsHumanReviewed != nil {
		sub.IsHumanReviewed = *req.IsHumanReviewed
	} else if req.HumanScore != nil || req.HumanFeedback != nil {
		sub.IsHumanReviewed = true
	}
	if err := database.DB.Save(&sub).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "数据保存失败", "error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, tools.FormatSubmission(sub))
}
