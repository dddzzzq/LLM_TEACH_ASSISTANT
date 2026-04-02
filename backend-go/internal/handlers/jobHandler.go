package handlers

import (
	"context"
	"log"
	"net/http"

	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"

	"github.com/gin-gonic/gin"
)

// GetJobStatus 获取异步任务状态
// @Summary 获取异步任务状态
// @Description 根据 job_id 获取异步任务状态，优先从 Redis 查询，如果 Redis 查不到则作为兜底查询 MySQL
// @Tags Jobs
// @Accept json
// @Produce json
// @Param job_id path string true "任务 ID"
// @Success 200 {object} map[string]interface{} "任务状态信息"
// @Failure 404 {object} map[string]interface{} "任务未找到"
// @Failure 500 {object} map[string]interface{} "服务器内部错误"
// @Router /jobs/{job_id} [get]
func GetJobStatus(c *gin.Context) {
	jobID := c.Param("job_id")
	if jobID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "job_id 不能为空",
		})
		return
	}

	ctx := context.Background()

	// 1. 优先从 Redis 中获取任务状态
	redisStatus, err := cache.GetJobStatus(ctx, jobID)
	if err != nil {
		// Redis 查询出错，记录日志但不返回错误，继续尝试 MySQL
		log.Printf("WARNING: Failed to get job status from Redis for %s: %v", jobID, err)
	}

	// 如果 Redis 中有数据，直接返回
	if redisStatus != nil && len(redisStatus) > 0 {
		status := redisStatus["status"]
		message := redisStatus["message"]
		updated := redisStatus["updated"]

		c.JSON(http.StatusOK, gin.H{
			"job_id":  jobID,
			"status":  status,
			"message": message,
			"updated": updated,
			"source":  "redis",
		})
		return
	}

	// 2. Redis 中查不到，作为兜底去 MySQL 的 AsyncJob 表中查询
	var asyncJob models.AsyncJob
	result := database.DB.Where("id = ?", jobID).First(&asyncJob)
	if result.Error != nil {
		// 记录找不到的错误
		log.Printf("ERROR: AsyncJob %s not found in MySQL: %v", jobID, result.Error)
		c.JSON(http.StatusNotFound, gin.H{
			"error":   "任务不存在",
			"job_id":  jobID,
			"message": "未找到指定的任务记录",
		})
		return
	}

	// 3. 返回 MySQL 中的数据
	c.JSON(http.StatusOK, gin.H{
		"job_id":       asyncJob.ID,
		"status":       string(asyncJob.Status),
		"message":      asyncJob.Message,
		"job_type":     string(asyncJob.JobType),
		"reference_id": asyncJob.ReferenceID,
		"student_id":   asyncJob.StudentID,
		"created_at":   asyncJob.CreatedAt,
		"updated_at":   asyncJob.UpdatedAt,
		"source":       "mysql",
	})
}
