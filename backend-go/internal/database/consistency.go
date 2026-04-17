package database

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/redis/go-redis/v9"
	"gorm.io/gorm"
)

// ConsistencyService 提供数据库与Redis一致性的服务
type ConsistencyService struct {
	db    *gorm.DB
	redis *redis.Client
}

// NewConsistencyService 创建一致性服务实例
func NewConsistencyService(db *gorm.DB, redis *redis.Client) *ConsistencyService {
	return &ConsistencyService{
		db:    db,
		redis: redis,
	}
}

// InvalidateCache 删除指定键的Redis缓存（Cache Aside模式）
// 业务规范：所有写入/更新数据的操作，必须先完成MySQL的更新，然后再调用此方法删除Redis中的key
func (cs *ConsistencyService) InvalidateCache(ctx context.Context, key string) error {
	// 删除Redis缓存
	err := cs.redis.Del(ctx, key).Err()
	if err != nil {
		// 记录日志但不返回错误，避免因为Redis问题影响主流程
		log.Printf("[Consistency Warning] Failed to invalidate cache (key: %s): %v", key, err)
		// 异步重试删除操作
		go cs.retryDeleteOperation(ctx, key)
	}

	log.Printf("[Consistency Debug] Successfully invalidated cache (key: %s)", key)
	return nil
}

// WriteJobWithCache 专用于AsyncJob的双写操作（已废弃，请使用cache.SetJobStatus和InvalidateCache）
// 注意：此方法仅用于兼容旧代码，新代码应直接调用cache.SetJobStatus并在DB更新后调用InvalidateCache
func (cs *ConsistencyService) WriteJobWithCache(ctx context.Context, jobID, status, message string) error {
	key := "job:status:" + jobID
	// 不再写入缓存，而是删除缓存，遵循Cache Aside模式
	// DB更新应在调用此方法前完成
	return cs.InvalidateCache(ctx, key)
}

// DeleteThenInvalidate 应该先删除数据库，再删除redis
func (cs *ConsistencyService) DeleteThenInvalidate(ctx context.Context, dataType string, key string, deleteFunc func() error) error {
	// 1. 先删除MySQL数据
	err := deleteFunc()
	if err != nil {
		return fmt.Errorf("failed to delete %s from MySQL: %w", dataType, err)
	}

	// 2. 再删除Redis缓存
	err = cs.redis.Del(ctx, key).Err()
	if err != nil {
		log.Printf("[Consistency Warning] Failed to delete %s cache from Redis (key: %s): %v", dataType, key, err)
		// 继续执行，不因为Redis失败而中断
	}

	log.Printf("[Consistency Debug] Successfully deleted %s data (key: %s)", dataType, key)
	return nil
}

// GetWithFallback 优先从 Redis 获取，失败则回退到 MySQL，并把结果写入 result。
// result 必须为指向目标类型的指针（与 json.Unmarshal 约定一致），例如 &models.Assignment{}。
// fallbackFunc 应返回可 JSON 序列化的值（通常为结构体指针或 map）。
func (cs *ConsistencyService) GetWithFallback(ctx context.Context, dataType, key string, result interface{}, fallbackFunc func() (interface{}, error)) error {
	if result == nil {
		return fmt.Errorf("GetWithFallback: result must be a non-nil pointer")
	}

	// 1. 首先尝试从Redis获取
	cachedJSON, err := cs.redis.Get(ctx, key).Result()
	if err == nil && cachedJSON != "" {
		// Redis命中
		err = json.Unmarshal([]byte(cachedJSON), result)
		if err == nil {
			log.Printf("[Consistency Debug] Cache hit for %s (key: %s)", dataType, key)
			return nil
		}
		log.Printf("[Consistency Warning] Failed to unmarshal cached %s data: %v", dataType, err)
	}

	// 2. Redis未命中或解析失败，回退到MySQL
	log.Printf("[Consistency Debug] Cache miss for %s (key: %s), falling back to MySQL", dataType, key)

	data, err := fallbackFunc()
	if err != nil {
		return fmt.Errorf("failed to get %s from MySQL: %w", dataType, err)
	}

	// 3. 将 MySQL 结果序列化并写入 result（与 Redis 命中路径使用相同的 JSON 语义）
	dataJSON, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("marshal %s from MySQL: %w", dataType, err)
	}
	if err := json.Unmarshal(dataJSON, result); err != nil {
		return fmt.Errorf("copy %s into result: %w", dataType, err)
	}

	// 4. 异步回写 Redis（复用同一 payload，避免二次 Marshal）
	go func(payload []byte) {
		// 使用 WithoutCancel 继承原有 Trace 但解除取消绑定，避免父级 Context 取消影响异步操作
		// 然后添加超时控制，防止 Goroutine 无限期阻塞
		asyncCtx := context.WithoutCancel(ctx)
		timeoutCtx, cancel := context.WithTimeout(asyncCtx, 5*time.Second)
		defer cancel()

		var ttl time.Duration
		switch dataType {
		case "job":
			ttl = 2 * time.Hour
		case "assignment":
			ttl = 1 * time.Hour
		case "submission":
			ttl = 1 * time.Hour
		default:
			ttl = 30 * time.Minute
		}

		err := cs.redis.Set(timeoutCtx, key, payload, ttl).Err()
		if err != nil {
			log.Printf("[Consistency Warning] Failed to cache %s data after fallback: %v", dataType, err)
		}
	}(append([]byte(nil), dataJSON...))

	return nil
}

// InvalidateByPattern 通过模式匹配删除Redis缓存
func (cs *ConsistencyService) InvalidateByPattern(ctx context.Context, pattern string) error {
	var cursor uint64
	var keys []string
	var err error

	// 扫描匹配模式的键
	for {
		keys, cursor, err = cs.redis.Scan(ctx, cursor, pattern, 100).Result()
		if err != nil {
			return fmt.Errorf("failed to scan Redis keys with pattern %s: %w", pattern, err)
		}

		if len(keys) > 0 {
			// 批量删除
			err = cs.redis.Del(ctx, keys...).Err()
			if err != nil {
				log.Printf("[Consistency Warning] Failed to delete keys with pattern %s: %v", pattern, err)
			} else {
				log.Printf("[Consistency Debug] Deleted %d keys with pattern %s", len(keys), pattern)
			}
		}

		if cursor == 0 {
			break
		}
	}

	return nil
}

// CheckAndRepairConsistency 检查并修复MySQL与Redis的不一致
func (cs *ConsistencyService) CheckAndRepairConsistency(ctx context.Context, checkFunc func() ([]InconsistencyItem, error)) error {
	inconsistencies, err := checkFunc()
	if err != nil {
		return fmt.Errorf("failed to check consistency: %w", err)
	}

	for _, item := range inconsistencies {
		log.Printf("[Consistency Repair] Found inconsistency for %s (key: %s): %s",
			item.DataType, item.Key, item.Description)

		switch item.Action {
		case "delete_cache":
			err := cs.redis.Del(ctx, item.Key).Err()
			if err != nil {
				log.Printf("[Consistency Repair] Failed to delete cache for %s: %v", item.Key, err)
			}
		case "update_cache":
			if item.Data != nil {
				dataJSON, err := json.Marshal(item.Data)
				if err == nil {
					err = cs.redis.Set(ctx, item.Key, dataJSON, item.TTL).Err()
					if err != nil {
						log.Printf("[Consistency Repair] Failed to update cache for %s: %v", item.Key, err)
					}
				}
			}
		}
	}

	return nil
}

// InconsistencyItem 表示不一致的数据项
type InconsistencyItem struct {
	DataType    string
	Key         string
	Description string
	Action      string // "delete_cache", "update_cache"
	Data        interface{}
	TTL         time.Duration
}

// retryCacheOperation 重试缓存操作（已废弃，仅用于兼容）
// 注意：此方法仅用于CheckAndRepairConsistency中的update_cache操作
func (cs *ConsistencyService) retryCacheOperation(ctx context.Context, dataType, key string, data []byte, ttl time.Duration) {
	maxRetries := 3
	for i := 0; i < maxRetries; i++ {
		// 使用select监听Context取消，防止野协程
		select {
		case <-time.After(time.Duration(i+1) * time.Second): // 指数退避
			err := cs.redis.Set(ctx, key, data, ttl).Err()
			if err == nil {
				log.Printf("[Consistency Retry] Successfully cached %s data after %d retries", dataType, i+1)
				return
			}
			log.Printf("[Consistency Retry] Attempt %d failed to cache %s data: %v", i+1, dataType, err)
		case <-ctx.Done():
			log.Printf("[Consistency Retry] Context cancelled, aborting retry for %s (key: %s)", dataType, key)
			return
		}
	}

	log.Printf("[Consistency Error] Failed to cache %s data after %d retries", dataType, maxRetries)
}

// retryDeleteOperation 重试删除缓存操作（用于InvalidateCache）
func (cs *ConsistencyService) retryDeleteOperation(ctx context.Context, key string) {
	maxRetries := 3
	for i := 0; i < maxRetries; i++ {
		select {
		case <-time.After(time.Duration(i+1) * time.Second): // 指数退避
			err := cs.redis.Del(ctx, key).Err()
			if err == nil {
				log.Printf("[Consistency Retry] Successfully deleted cache after %d retries (key: %s)", i+1, key)
				return
			}
			log.Printf("[Consistency Retry] Attempt %d failed to delete cache (key: %s): %v", i+1, key, err)
		case <-ctx.Done():
			log.Printf("[Consistency Retry] Context cancelled, aborting delete retry for key: %s", key)
			return
		}
	}

	log.Printf("[Consistency Error] Failed to delete cache after %d retries (key: %s)", maxRetries, key)
}

// GetConsistencyService 获取全局一致性服务实例
func GetConsistencyService() *ConsistencyService {
	return &ConsistencyService{
		db:    DB,
		redis: RedisClient,
	}
}
