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
	"gorm.io/gorm" // 引入 gorm 用于判断 ErrRecordNotFound，为了防止缓存穿透设置空值
)

const (
	AssignmentCacheTTL = 24 * time.Hour
	ExamCacheTTL       = 24 * time.Hour
	SubmissionCacheTTL = 2 * time.Hour // 提交记录缓存时间，因为批改状态可能会变

	// 缓存穿透防护配置
	EmptyCacheValue = "EMPTY_RECORD"  // 标志数据库中不存在该记录
	EmptyCacheTTL   = 5 * time.Minute // 空值缓存时间较短，防止长时间占用内存或掩盖后续真实数据写入

	// 缓存击穿防护配置
	LockTTL        = 5 * time.Second       // 分布式锁超时时间，防止进程崩溃导致死锁
	LockRetrySleep = 50 * time.Millisecond // 拿不到锁时的自旋休眠时间
	MaxRetries     = 30                    // 最大自旋次数，防止无限死循环 (约等待 1.5 秒)
)

func GetAssignmentWithCache(ctx context.Context, assignmentID uint) (*models.Assignment, error) {
	client := database.GetRedisClient()
	if client == nil {
		return getAssignmentFromDB(assignmentID)
	}

	key := fmt.Sprintf("cache:assignment:%d", assignmentID)
	lockKey := fmt.Sprintf("lock:assignment:%d", assignmentID)

	// 使用 for 循环实现未拿到锁时的自旋等待
	for i := 0; i < MaxRetries; i++ {
		val, err := client.Get(ctx, key).Result()

		// 1. 缓存命中
		if err == nil {
			// 【防御缓存穿透】：如果是我们之前设置的空值标识，直接返回错误，不打 DB
			if val == EmptyCacheValue {
				log.Printf("INFO: Blocked penetrating request for Assignment %d", assignmentID)
				return nil, gorm.ErrRecordNotFound
			}

			var assignment models.Assignment
			if err := json.Unmarshal([]byte(val), &assignment); err != nil {
				log.Printf("WARNING: Failed to unmarshal assignment cache: %v", err)
				return getAssignmentFromDB(assignmentID)
			}
			return &assignment, nil
		}

		// 2. 缓存未命中 (redis.Nil)，可能发生【缓存击穿】
		if err == redis.Nil {
			// 【防御缓存击穿】：使用 SetArgs 获取 Redis 分布式锁
			err := client.SetArgs(ctx, lockKey, "1", redis.SetArgs{
				Mode: "NX",
				TTL:  LockTTL,
			}).Err()

			if err == nil {
				// 成功拿到锁，去数据库查询
				assignment, dbErr := getAssignmentFromDB(assignmentID)

				if dbErr != nil {
					// 判断是否为数据真的不存在
					if errors.Is(dbErr, gorm.ErrRecordNotFound) {
						// 【防御缓存穿透】：数据不存在，缓存一个极短时间的空标记
						client.Set(ctx, key, EmptyCacheValue, EmptyCacheTTL)
					}
					// 无论如何，释放锁
					client.Del(ctx, lockKey)
					return nil, dbErr
				}

				// 查询成功，回写正常数据的缓存
				jsonData, _ := json.Marshal(assignment)
				if err := client.Set(ctx, key, jsonData, AssignmentCacheTTL).Err(); err != nil {
					log.Printf("WARNING: Failed to write assignment %d to cache: %v", assignmentID, err)
					// 继续执行，返回数据给调用者
				} else {
					log.Printf("INFO: Assignment %d cached successfully", assignmentID)
				}

				// 释放锁
				client.Del(ctx, lockKey)
				log.Printf("INFO: Assignment %d loaded from DB and cached (Lock Acquired)", assignmentID)

				return assignment, nil
			}

			// 没拿到锁，说明别的线程正在查 DB，休眠一会进入下一次 for 循环重新读缓存
			time.Sleep(LockRetrySleep)
			continue
		}

		// 3. Redis 发生其他异常（如网络波动），降级走数据库
		log.Printf("WARNING: Failed to get assignment cache due to redis err: %v", err)
		return getAssignmentFromDB(assignmentID)
	}

	// 达到最大重试次数，系统繁忙保护
	return nil, errors.New("system busy: timeout waiting for cache lock")
}

func InvalidateAssignmentCache(ctx context.Context, assignmentID uint) error {
	client := database.GetRedisClient()
	if client != nil {
		key := fmt.Sprintf("cache:assignment:%d", assignmentID)
		return client.Del(ctx, key).Err()
	}
	return nil
}

func getAssignmentFromDB(id uint) (*models.Assignment, error) {
	var assignment models.Assignment
	if err := database.DB.First(&assignment, id).Error; err != nil {
		return nil, err
	}
	return &assignment, nil
}

// ============== 试卷 (Exam) 的缓存处理逻辑完全同理 ============== //

func GetExamWithCache(ctx context.Context, examID uint) (*models.Exam, error) {
	client := database.GetRedisClient()
	if client == nil {
		return getExamFromDB(examID)
	}

	key := fmt.Sprintf("cache:exam:%d", examID)
	lockKey := fmt.Sprintf("lock:exam:%d", examID)

	for i := 0; i < MaxRetries; i++ {
		val, err := client.Get(ctx, key).Result()

		if err == nil {
			// 【防御缓存穿透】
			if val == EmptyCacheValue {
				log.Printf("INFO: Blocked penetrating request for Exam %d", examID)
				return nil, gorm.ErrRecordNotFound
			}

			var exam models.Exam
			if err := json.Unmarshal([]byte(val), &exam); err != nil {
				log.Printf("WARNING: Failed to unmarshal exam cache: %v", err)
				return getExamFromDB(examID)
			}
			return &exam, nil
		}

		if err == redis.Nil {
			// 【防御缓存击穿】：使用 SetArgs 获取 Redis 分布式锁
			err := client.SetArgs(ctx, lockKey, "1", redis.SetArgs{
				Mode: "NX",
				TTL:  LockTTL,
			}).Err()

			if err == nil {
				// 成功拿到锁，去数据库查询
				exam, dbErr := getExamFromDB(examID)

				if dbErr != nil {
					// 【防御缓存穿透】：设置空值
					if errors.Is(dbErr, gorm.ErrRecordNotFound) {
						client.Set(ctx, key, EmptyCacheValue, EmptyCacheTTL)
					}
					client.Del(ctx, lockKey)
					return nil, dbErr
				}

				jsonData, _ := json.Marshal(exam)
				if err := client.Set(ctx, key, jsonData, ExamCacheTTL).Err(); err != nil {
					log.Printf("WARNING: Failed to write exam %d to cache: %v", examID, err)
					// 继续执行，返回数据给调用者
				} else {
					log.Printf("INFO: Exam %d cached successfully", examID)
				}
				client.Del(ctx, lockKey)

				log.Printf("INFO: Exam %d loaded from DB and cached (Lock Acquired)", examID)
				return exam, nil
			}

			time.Sleep(LockRetrySleep)
			continue
		}

		log.Printf("WARNING: Failed to get exam cache: %v", err)
		return getExamFromDB(examID)
	}

	return nil, errors.New("system busy: timeout waiting for cache lock")
}

func InvalidateExamCache(ctx context.Context, examID uint) error {
	client := database.GetRedisClient()
	if client != nil {
		key := fmt.Sprintf("cache:exam:%d", examID)
		return client.Del(ctx, key).Err()
	}
	return nil
}

func getExamFromDB(id uint) (*models.Exam, error) {
	var exam models.Exam
	if err := database.DB.Preload("Questions").First(&exam, id).Error; err != nil {
		return nil, err
	}
	return &exam, nil
}

// ============== 提交记录 (Submission) 的缓存处理逻辑 ============== //

func GetSubmissionWithCache(ctx context.Context, submissionID uint) (*models.Submission, error) {
	client := database.GetRedisClient()
	if client == nil {
		return getSubmissionFromDB(submissionID)
	}

	key := fmt.Sprintf("cache:submission:%d", submissionID)
	lockKey := fmt.Sprintf("lock:submission:%d", submissionID)

	for i := 0; i < MaxRetries; i++ {
		val, err := client.Get(ctx, key).Result()

		// 1. 缓存命中
		if err == nil {
			// 【防御缓存穿透】：如果是我们之前设置的空值标识，直接返回错误，不打 DB
			if val == EmptyCacheValue {
				log.Printf("INFO: Blocked penetrating request for Submission %d", submissionID)
				return nil, gorm.ErrRecordNotFound
			}

			var submission models.Submission
			if err := json.Unmarshal([]byte(val), &submission); err != nil {
				log.Printf("WARNING: Failed to unmarshal submission cache: %v", err)
				return getSubmissionFromDB(submissionID)
			}
			return &submission, nil
		}

		// 2. 缓存未命中 (redis.Nil)，可能发生【缓存击穿】
		if err == redis.Nil {
			// 【防御缓存击穿】：使用 SetArgs 获取 Redis 分布式锁
			err := client.SetArgs(ctx, lockKey, "1", redis.SetArgs{
				Mode: "NX",
				TTL:  LockTTL,
			}).Err()

			if err == nil {
				// 成功拿到锁，去数据库查询
				submission, dbErr := getSubmissionFromDB(submissionID)

				if dbErr != nil {
					// 判断是否为数据真的不存在
					if errors.Is(dbErr, gorm.ErrRecordNotFound) {
						// 【防御缓存穿透】：数据不存在，缓存一个极短时间的空标记
						client.Set(ctx, key, EmptyCacheValue, EmptyCacheTTL)
					}
					// 无论如何，释放锁
					client.Del(ctx, lockKey)
					return nil, dbErr
				}

				// 查询成功，回写正常数据的缓存
				jsonData, _ := json.Marshal(submission)
				if err := client.Set(ctx, key, jsonData, SubmissionCacheTTL).Err(); err != nil {
					log.Printf("WARNING: Failed to write submission %d to cache: %v", submissionID, err)
					// 继续执行，返回数据给调用者
				} else {
					log.Printf("INFO: Submission %d cached successfully", submissionID)
				}

				// 释放锁
				client.Del(ctx, lockKey)
				log.Printf("INFO: Submission %d loaded from DB and cached (Lock Acquired)", submissionID)

				return submission, nil
			}

			// 没拿到锁，说明别的线程正在查 DB，休眠一会进入下一次 for 循环重新读缓存
			time.Sleep(LockRetrySleep)
			continue
		}

		// 3. Redis 发生其他异常（如网络波动），降级走数据库
		log.Printf("WARNING: Failed to get submission cache due to redis err: %v", err)
		return getSubmissionFromDB(submissionID)
	}

	// 达到最大重试次数，系统繁忙保护
	return nil, errors.New("system busy: timeout waiting for cache lock")
}

func InvalidateSubmissionCache(ctx context.Context, submissionID uint) error {
	client := database.GetRedisClient()
	if client != nil {
		key := fmt.Sprintf("cache:submission:%d", submissionID)
		return client.Del(ctx, key).Err()
	}
	return nil
}

func getSubmissionFromDB(id uint) (*models.Submission, error) {
	var submission models.Submission
	if err := database.DB.First(&submission, id).Error; err != nil {
		return nil, err
	}
	return &submission, nil
}
