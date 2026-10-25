package handlers

import (
	"errors"
	"fmt"
	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
	"grading-gateway/internal/schemas"
	"grading-gateway/internal/tools"
	"log"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

// 获取学生提交handler（学生只能查看自己的提交）
func GetSubmission(c *gin.Context) {
	idStr := c.Param("id")
	submissionID, err := strconv.ParseUint(idStr, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "无效的提交ID"})
		return
	}

	// 1. 【缓存获取】：从 Redis 缓存中获取提交记录（带防穿透、防击穿保护）
	sub, err := cache.GetSubmissionWithCache(c.Request.Context(), uint(submissionID))
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"detail": "未找到该批改记录"})
			return
		}
		log.Printf("获取单个提交出错：%v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "系统繁忙，获取提交详情失败"})
		return
	}

	// 2. 【权限检查】：学生只能查看自己的提交
	role, roleErr := middleware.GetRoleFromContext(c)
	username, usernameErr := middleware.GetUsernameFromContext(c)
	if roleErr == nil && usernameErr == nil && role == "student" {
		if sub.StudentID != username {
			c.JSON(http.StatusForbidden, gin.H{"detail": "无权查看他人的提交"})
			return
		}
	}

	c.JSON(http.StatusOK, tools.FormatSubmission(*sub))
}

// 删除单个提交handler（只有教师/管理员可以删除）
func DeleteSubmission(c *gin.Context) {
	id := c.Param("id")

	// 先查询提交信息，获取assignment_id用于清理缓存
	var submission models.Submission
	if err := database.DB.First(&submission, id).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"detail": "未找到该批改记录"})
		return
	}

	// 获取一致性服务
	consistencySvc := database.GetConsistencyService()
	ctx := c.Request.Context()

	// 先删除Redis中的相关缓存，再删除MySQL
	submissionKey := fmt.Sprintf("submission:%s", id)

	// 使用一致性服务删除缓存和数据
	err := consistencySvc.DeleteThenInvalidate(ctx, "submission", submissionKey, func() error {
		// 这是实际的MySQL删除操作
		return database.DB.Delete(&models.Submission{}, id).Error
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "删除失败", "error": err.Error()})
		return
	}

	// 清理作业相关的提交缓存
	consistencySvc.InvalidateByPattern(ctx, fmt.Sprintf("*assignment:submissions:%d*", submission.AssignmentID))

	// 失效提交记录缓存
	submissionID, _ := strconv.ParseUint(id, 10, 32)
	cache.InvalidateSubmissionCache(ctx, uint(submissionID))

	c.Status(http.StatusNoContent)
}

// 更新学生评分或评语handler（教师/管理员可以更新，学生不能修改）
func UpdateSubmission(c *gin.Context) {
	id := c.Param("id")
	var req schemas.SubmissionUpdate
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "请求数据格式有误", "error": err.Error()})
		return
	}

	// 检查权限：学生不能修改任何提交
	role, roleErr := middleware.GetRoleFromContext(c)
	if roleErr == nil && role == "student" {
		c.JSON(http.StatusForbidden, gin.H{"detail": "学生无权修改提交"})
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

	// 失效提交记录缓存，因为内容已更新
	ctx := c.Request.Context()
	cache.InvalidateSubmissionCache(ctx, sub.ID)

	c.JSON(http.StatusOK, tools.FormatSubmission(sub))
}
