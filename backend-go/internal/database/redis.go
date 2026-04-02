package database

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

// RedisClient 全局 Redis 客户端实例
var RedisClient *redis.Client

// RedisConfig Redis 配置
type RedisConfig struct {
	Addr     string
	Password string
	DB       int
}

// DefaultRedisConfig 从环境变量获取默认 Redis 配置
func DefaultRedisConfig() *RedisConfig {
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		addr = "localhost:6379"
	}

	password := os.Getenv("REDIS_PASSWORD")
	if password == "" {
		password = ""
	}

	db := 0
	if env := os.Getenv("REDIS_DB"); env != "" {
		if val, err := strconv.Atoi(env); err == nil && val >= 0 {
			db = val
		}
	}

	return &RedisConfig{
		Addr:     addr,
		Password: password,
		DB:       db,
	}
}

// InitRedis 初始化 Redis 连接
func InitRedis(config *RedisConfig) {
	if config == nil {
		config = DefaultRedisConfig()
	}

	RedisClient = redis.NewClient(&redis.Options{
		Addr:     config.Addr,
		Password: config.Password,
		DB:       config.DB,
	})

	// 测试连接
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := RedisClient.Ping(ctx).Result()
	if err != nil {
		log.Fatalf("Failed to connect to Redis: %v", err)
	}

	fmt.Println("成功连接到 Redis！")
}

// CloseRedis 关闭 Redis 连接
func CloseRedis() {
	if RedisClient != nil {
		err := RedisClient.Close()
		if err != nil {
			log.Printf("Error closing Redis connection: %v", err)
		}
	}
}

// GetRedisClient 获取 Redis 客户端
func GetRedisClient() *redis.Client {
	return RedisClient
}

// IsRedisAvailable 检查 Redis 是否可用
func IsRedisAvailable() bool {
	if RedisClient == nil {
		return false
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := RedisClient.Ping(ctx).Result()
	return err == nil
}

// RetryRedisOperation 重试 Redis 操作（用于处理临时故障）
func RetryRedisOperation(operation func() error, maxRetries int, baseDelay time.Duration) error {
	var lastErr error
	for i := 0; i < maxRetries; i++ {
		err := operation()
		if err == nil {
			return nil
		}

		lastErr = err
		// 检查是否是网络错误或连接错误
		if i < maxRetries-1 {
			delay := baseDelay * time.Duration(1<<uint(i)) // 指数退避
			log.Printf("Redis operation failed (attempt %d/%d), retrying in %v: %v", i+1, maxRetries, delay, err)
			time.Sleep(delay)
		}
	}
	return fmt.Errorf("Redis operation failed after %d attempts: %v", maxRetries, lastErr)
}

// RedisKey 工具函数：生成 Redis 键
func RedisKey(parts ...string) string {
	return "chat:memory:" + redisKeyJoin(parts...)
}

// redisKeyJoin 连接 Redis 键的各个部分
func redisKeyJoin(parts ...string) string {
	result := ""
	for i, part := range parts {
		if i > 0 {
			result += ":"
		}
		result += part
	}
	return result
}

// SetWithExpiry 设置键值并指定过期时间
func SetWithExpiry(ctx context.Context, key string, value interface{}, expiry time.Duration) error {
	return RetryRedisOperation(func() error {
		return RedisClient.Set(ctx, key, value, expiry).Err()
	}, 3, 100*time.Millisecond)
}

// GetString 获取字符串值
func GetString(ctx context.Context, key string) (string, error) {
	var result string
	err := RetryRedisOperation(func() error {
		var err error
		result, err = RedisClient.Get(ctx, key).Result()
		return err
	}, 3, 100*time.Millisecond)
	return result, err
}

// DeleteKey 删除键
func DeleteKey(ctx context.Context, key string) error {
	return RetryRedisOperation(func() error {
		return RedisClient.Del(ctx, key).Err()
	}, 3, 100*time.Millisecond)
}

// ListPush 向列表头部添加元素
func ListPush(ctx context.Context, key string, values ...interface{}) error {
	return RetryRedisOperation(func() error {
		return RedisClient.LPush(ctx, key, values...).Err()
	}, 3, 100*time.Millisecond)
}

// ListRange 获取列表范围内的元素
func ListRange(ctx context.Context, key string, start, stop int64) ([]string, error) {
	var result []string
	err := RetryRedisOperation(func() error {
		var err error
		result, err = RedisClient.LRange(ctx, key, start, stop).Result()
		return err
	}, 3, 100*time.Millisecond)
	return result, err
}

// ListTrim 修剪列表，只保留指定范围内的元素
func ListTrim(ctx context.Context, key string, start, stop int64) error {
	return RetryRedisOperation(func() error {
		return RedisClient.LTrim(ctx, key, start, stop).Err()
	}, 3, 100*time.Millisecond)
}

// ExpireKey 设置键的过期时间
func ExpireKey(ctx context.Context, key string, expiry time.Duration) error {
	return RetryRedisOperation(func() error {
		return RedisClient.Expire(ctx, key, expiry).Err()
	}, 3, 100*time.Millisecond)
}

// KeyExists 检查键是否存在
func KeyExists(ctx context.Context, key string) (bool, error) {
	var result int64
	err := RetryRedisOperation(func() error {
		var err error
		result, err = RedisClient.Exists(ctx, key).Result()
		return err
	}, 3, 100*time.Millisecond)
	return result > 0, err
}
