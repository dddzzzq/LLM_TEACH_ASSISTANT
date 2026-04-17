package cache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"time"

	"grading-gateway/internal/database"
	"grading-gateway/internal/models"

	"github.com/redis/go-redis/v9"
	"gorm.io/gorm"
)

const (
	jobStatusKeyPrefix = "job:status:"
	jobStatusTTL       = 24 * time.Hour
)

// JobStatus 表示任务状态信息
type JobStatus struct {
	Status  string `json:"status"`
	Message string `json:"message"`
	Updated string `json:"updated"` // RFC3339 时间戳
}

// SetJobStatus 将任务状态通过 Hash (HSET) 存入 Redis
func SetJobStatus(ctx context.Context, jobID, status, message string) error {
	key := jobStatusKeyPrefix + jobID

	statusData := JobStatus{
		Status:  status,
		Message: message,
		Updated: time.Now().UTC().Format(time.RFC3339),
	}

	// 将结构体转换为 JSON 字符串
	statusJSON, err := json.Marshal(statusData)
	if err != nil {
		return fmt.Errorf("failed to marshal job status: %w", err)
	}

	// 使用 HSET 存储哈希字段
	client := database.GetRedisClient()
	if client == nil {
		return fmt.Errorf("redis client not initialized")
	}

	err = client.HSet(ctx, key, "status", status, "message", message, "updated", statusData.Updated, "data", string(statusJSON)).Err()
	if err != nil {
		return fmt.Errorf("failed to set job status in redis: %w", err)
	}

	// 设置过期时间
	err = client.Expire(ctx, key, jobStatusTTL).Err()
	if err != nil {
		log.Printf("WARNING: Failed to set expiration for job status key %s: %v", key, err)
		// 不返回错误，因为主操作已成功
	}

	log.Printf("DEBUG: Job status set for %s: %s - %s", jobID, status, message)
	return nil
}

// GetJobStatus 获取任务的当前状态
func GetJobStatus(ctx context.Context, jobID string) (map[string]string, error) {
	key := jobStatusKeyPrefix + jobID
	client := database.GetRedisClient()
	if client == nil {
		return nil, fmt.Errorf("redis client not initialized")
	}

	// 获取所有哈希字段
	result, err := client.HGetAll(ctx, key).Result()
	if err != nil {
		if err == redis.Nil {
			return nil, nil // 键不存在，返回空
		}
		return nil, fmt.Errorf("failed to get job status from redis: %w", err)
	}

	if len(result) == 0 {
		return nil, nil
	}

	return result, nil
}

// GetJobStatusWithFallback 优先从 Redis Hash 读取；未命中或 Redis 报错时回退到 MySQL AsyncJob。
// 命中 MySQL 后会异步调用 SetJobStatus 回写 Redis（与 Cache Aside 一致）。
// 返回值：Redis 有数据时 fromRedis 非空、fromDB 为 nil；回退 DB 成功时 fromRedis 为空、fromDB 非空；不存在时为 ErrRecordNotFound。
func GetJobStatusWithFallback(ctx context.Context, jobID string) (fromRedis map[string]string, fromDB *models.AsyncJob, err error) {
	redisStatus, rerr := GetJobStatus(ctx, jobID)
	if rerr != nil {
		log.Printf("WARNING: Failed to get job status from Redis for %s: %v", jobID, rerr)
	} else if len(redisStatus) > 0 {
		return redisStatus, nil, nil
	}

	var asyncJob models.AsyncJob
	res := database.DB.Where("id = ?", jobID).First(&asyncJob)
	if res.Error != nil {
		if errors.Is(res.Error, gorm.ErrRecordNotFound) {
			return nil, nil, gorm.ErrRecordNotFound
		}
		return nil, nil, res.Error
	}

	go func() {
		updateCtx := context.Background()
		if err := SetJobStatus(updateCtx, jobID, string(asyncJob.Status), asyncJob.Message); err != nil {
			log.Printf("WARNING: Failed to update Redis cache for job %s from MySQL fallback: %v", jobID, err)
		} else {
			log.Printf("DEBUG: Updated Redis cache for job %s from MySQL fallback", jobID)
		}
	}()

	return nil, &asyncJob, nil
}

// DeleteJobStatus 删除任务状态缓存
func DeleteJobStatus(ctx context.Context, jobID string) error {
	key := jobStatusKeyPrefix + jobID
	client := database.GetRedisClient()
	if client == nil {
		return fmt.Errorf("redis client not initialized")
	}

	err := client.Del(ctx, key).Err()
	if err != nil {
		return fmt.Errorf("failed to delete job status from redis: %w", err)
	}
	return nil
}
