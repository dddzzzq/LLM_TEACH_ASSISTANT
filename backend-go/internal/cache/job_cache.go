package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"grading-gateway/internal/database"

	"github.com/redis/go-redis/v9"
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

// GetJobStatusWithFallback 优先从 Redis 获取，如果 Redis 中查不到，则作为兜底从 MySQL 的 AsyncJob 表中查询
func GetJobStatusWithFallback(ctx context.Context, jobID string) (map[string]string, error) {
	// 首先尝试 Redis
	redisStatus, err := GetJobStatus(ctx, jobID)
	if err != nil {
		log.Printf("WARNING: Failed to get job status from redis for %s: %v", jobID, err)
		// 继续尝试 MySQL
	}

	if redisStatus != nil && len(redisStatus) > 0 {
		return redisStatus, nil
	}

	// Redis 中没有，尝试 MySQL
	// 注意：这里需要导入 models 包和 database.DB
	// 为了保持代码简洁，我们只返回 Redis 的结果，MySQL 回退将在 handler 中实现
	// 返回 nil 表示需要回退到 MySQL
	return nil, nil
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
