package handlers

import (
	"context"
	"errors"
	"log"
	"net/http"

	"grading-gateway/internal/cache"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
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

	fromRedis, fromDB, err := cache.GetJobStatusWithFallback(ctx, jobID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			log.Printf("ERROR: AsyncJob %s not found in MySQL: %v", jobID, err)
			c.JSON(http.StatusNotFound, gin.H{
				"error":   "任务不存在",
				"job_id":  jobID,
				"message": "未找到指定的任务记录",
			})
			return
		}
		log.Printf("ERROR: GetJobStatusWithFallback %s: %v", jobID, err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "查询任务状态失败",
			"job_id":  jobID,
			"message": err.Error(),
		})
		return
	}

	if len(fromRedis) > 0 {
		c.JSON(http.StatusOK, gin.H{
			"job_id":  jobID,
			"status":  fromRedis["status"],
			"message": fromRedis["message"],
			"updated": fromRedis["updated"],
			"source":  "redis",
		})
		return
	}

	if fromDB != nil {
		c.JSON(http.StatusOK, gin.H{
			"job_id":       fromDB.ID,
			"status":       string(fromDB.Status),
			"message":      fromDB.Message,
			"job_type":     string(fromDB.JobType),
			"reference_id": fromDB.ReferenceID,
			"student_id":   fromDB.StudentID,
			"created_at":   fromDB.CreatedAt,
			"updated_at":   fromDB.UpdatedAt,
			"source":       "mysql",
		})
		return
	}

	c.JSON(http.StatusInternalServerError, gin.H{
		"error":  "unexpected empty job status",
		"job_id": jobID,
	})
}
