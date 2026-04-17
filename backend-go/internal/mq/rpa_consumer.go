package mq

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strconv"
	"strings"
	"time"

	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/models"
	"grading-gateway/pb"

	"github.com/IBM/sarama"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

// RPAConsumer 实现 sarama.ConsumerGroupHandler，专门处理RPA抓取任务
type RPAConsumer struct{}

// Setup 在消费者组分配分区时调用
func (RPAConsumer) Setup(session sarama.ConsumerGroupSession) error {
	log.Printf("[RPA Consumer] ✅ 消费者组会话初始化完成")
	log.Printf("[RPA Consumer]    - Member ID: %s", session.MemberID())
	log.Printf("[RPA Consumer]    - Generation ID: %d", session.GenerationID())
	return nil
}

// Cleanup 在消费者组释放分区时调用
func (RPAConsumer) Cleanup(session sarama.ConsumerGroupSession) error {
	log.Printf("[RPA Consumer] 🔄 消费者组会话清理完成")
	log.Printf("[RPA Consumer]    - Member ID: %s", session.MemberID())
	return nil
}

// ConsumeClaim 处理分配给此消费者的消息
func (RPAConsumer) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[RPA Consumer] 🎯 开始消费分区消息")
	log.Printf("[RPA Consumer]    - Partition: %d", claim.Partition())
	log.Printf("[RPA Consumer]    - Initial Offset: %d", claim.InitialOffset())
	log.Printf("[RPA Consumer]    - Topic: %s", claim.Topic())
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	for msg := range claim.Messages() {
		ctx := context.Background()
		log.Printf("────────────────────────────────────────────────────────────────────────────")
		log.Printf("[RPA Consumer] 📨 收到新消息")
		log.Printf("[RPA Consumer]    - Topic: %s", msg.Topic)
		log.Printf("[RPA Consumer]    - Partition: %d", msg.Partition)
		log.Printf("[RPA Consumer]    - Offset: %d", msg.Offset)
		log.Printf("[RPA Consumer]    - Timestamp: %s", msg.Timestamp.Format("2006-01-02 15:04:05"))
		log.Printf("────────────────────────────────────────────────────────────────────────────")

		// 处理RPA抓取任务
		if err := handleRPAFetchTask(ctx, msg.Value); err != nil {
			log.Printf("[RPA Consumer] ❌ 处理RPA抓取任务失败: %v", err)
			log.Printf("[RPA Consumer] ⚠️  消息不会标记为完成，将触发Kafka重试机制")
			// 注意：不标记消息为完成，让 Kafka 重试
			continue
		}

		// 处理成功，提交 offset
		session.MarkMessage(msg, "")
		session.Commit()
		log.Printf("[RPA Consumer] ✅ 消息处理成功，已提交Offset")
		log.Printf("[RPA Consumer]    - Committed Offset: %d", msg.Offset)
	}

	return nil
}

// handleRPAFetchTask 处理RPA抓取任务
func handleRPAFetchTask(ctx context.Context, message []byte) error {
	var task RPAFetchMessage
	if err := json.Unmarshal(message, &task); err != nil {
		log.Printf("[RPA Consumer] ❌ 消息反序列化失败: %v", err)
		return fmt.Errorf("failed to unmarshal RPA fetch task: %w", err)
	}

	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[RPA Consumer] 🚀 开始处理RPA抓取任务")
	log.Printf("[RPA Consumer] 📋 任务信息:")
	log.Printf("[RPA Consumer]    - Job ID: %s", task.JobID)
	log.Printf("[RPA Consumer]    - 用户名: %s", task.Username)
	log.Printf("[RPA Consumer]    - 课程名称: %s", task.CourseName)
	log.Printf("[RPA Consumer]    - 作业名称: %s", task.AssignmentName)
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// ============================
	// 环节 A：状态流转与下载
	// ============================
	log.Printf("[RPA Consumer] 📦 环节 A: 状态流转与下载")
	log.Printf("[RPA Consumer] ─────────────────────────────────────────────────────────────────────")

	// 1. 更新 AsyncJob 表中该 jobID 的状态为 "PROCESSING"
	log.Printf("[RPA Consumer] 📋 步骤 A1/7: 更新MySQL任务状态为PROCESSING...")
	if err := updateRPAJobStatus(task.JobID, models.JobStatusProcessing, "开始RPA抓取作业"); err != nil {
		log.Printf("[RPA Consumer] ⚠️  更新MySQL任务状态失败: %v", err)
	} else {
		log.Printf("[RPA Consumer] ✅ MySQL任务状态已更新为 PROCESSING")
	}

	// 同时更新 Redis 缓存状态
	log.Printf("[RPA Consumer] 📋 步骤 A2/7: 更新Redis缓存状态为PROCESSING...")
	if err := cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusProcessing), "开始RPA抓取作业"); err != nil {
		log.Printf("[RPA Consumer] ⚠️  更新Redis缓存状态失败: %v", err)
	} else {
		log.Printf("[RPA Consumer] ✅ Redis缓存状态已更新为 PROCESSING")
	}

	// 2. 检查 gRPC 客户端是否可用
	log.Printf("[RPA Consumer] 📋 步骤 A3/7: 检查gRPC客户端连接状态...")
	if grpcclient.Client == nil {
		errMsg := "gRPC客户端未初始化，无法连接到Python RPA服务"
		log.Printf("[RPA Consumer] ❌ %s", errMsg)
		updateRPAJobStatus(task.JobID, models.JobStatusFailed, errMsg)
		cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusFailed), errMsg)
		log.Printf("[RPA Consumer] ✅ 已更新任务状态为 FAILED")
		return nil // 返回nil跳过重试
	}
	log.Printf("[RPA Consumer] ✅ gRPC客户端连接正常")

	// 3. 构建 gRPC 请求
	log.Printf("[RPA Consumer] 📋 步骤 A4/7: 构建gRPC请求参数...")
	fetchRequest := &pb.FetchRequest{
		Username:       task.Username,
		Password:       task.Password,
		CourseName:     task.CourseName,
		AssignmentName: task.AssignmentName,
	}
	log.Printf("[RPA Consumer] ✅ gRPC请求参数构建完成")
	log.Printf("[RPA Consumer]    - Username: %s", fetchRequest.Username)
	log.Printf("[RPA Consumer]    - CourseName: %s", fetchRequest.CourseName)
	log.Printf("[RPA Consumer]    - AssignmentName: %s", fetchRequest.AssignmentName)

	// 4. 设置超时上下文（5分钟）
	log.Printf("[RPA Consumer] 📋 步骤 A5/7: 设置gRPC调用超时时间(20分钟)...")
	grpcCtx, cancel := context.WithTimeout(ctx, 20*time.Minute)
	defer cancel()
	log.Printf("[RPA Consumer] ✅ 超时上下文已设置")

	// 5. 调用 Python 侧的 gRPC 服务
	log.Printf("[RPA Consumer] 📋 步骤 A6/7: 调用Python gRPC服务执行RPA抓取...")
	log.Printf("[RPA Consumer] 🔄 正在调用 FetchPortalHomework 接口...")
	log.Printf("[RPA Consumer] ⏳ 这可能需要几分钟时间，请耐心等待...")

	startTime := time.Now()
	response, err := grpcclient.Client.FetchPortalHomework(grpcCtx, fetchRequest)
	elapsed := time.Since(startTime)

	if err != nil {
		errMsg := fmt.Sprintf("gRPC调用失败: %v", err)
		log.Printf("[RPA Consumer] ❌ %s", errMsg)
		log.Printf("[RPA Consumer] ⏱️  gRPC调用耗时: %v", elapsed)
		updateRPAJobStatus(task.JobID, models.JobStatusFailed, errMsg)
		cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusFailed), errMsg)
		log.Printf("[RPA Consumer] ✅ 已更新任务状态为 FAILED")
		return nil // 返回nil跳过重试
	}

	log.Printf("[RPA Consumer] ✅ gRPC调用成功")
	log.Printf("[RPA Consumer] ⏱️  gRPC调用耗时: %v", elapsed)

	// 6. 检查响应状态
	log.Printf("[RPA Consumer] 📋 步骤 A7/7: 检查RPA执行结果...")
	if !response.Success {
		errMsg := fmt.Sprintf("RPA执行失败: %s", response.Message)
		log.Printf("[RPA Consumer] ❌ %s", errMsg)
		updateRPAJobStatus(task.JobID, models.JobStatusFailed, errMsg)
		cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusFailed), errMsg)
		log.Printf("[RPA Consumer] ✅ 已更新任务状态为 FAILED")
		return nil // 返回nil跳过重试
	}

	// 7. 检查是否下载到文件
	if len(response.FilePaths) == 0 {
		errMsg := "未下载到任何文件，请检查课程名称和作业名称是否正确"
		log.Printf("[RPA Consumer] ⚠️  %s", errMsg)
		updateRPAJobStatus(task.JobID, models.JobStatusSuccess, errMsg)
		cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusSuccess), errMsg)
		log.Printf("[RPA Consumer] ✅ 任务完成(无文件下载)")
		return nil // 任务完成，但没有文件
	}

	log.Printf("[RPA Consumer] ✅ RPA执行成功")
	log.Printf("[RPA Consumer] 📥 共下载 %d 个文件:", len(response.FilePaths))
	for i, filePath := range response.FilePaths {
		log.Printf("[RPA Consumer]    %d. %s", i+1, filePath)
	}

	// ============================
	// 环节 C：投递给批改流水线
	// ============================
	log.Printf("────────────────────────────────────────────────────────────────────────────")
	log.Printf("[RPA Consumer] 🚀 环节 C: 投递给批改流水线")
	log.Printf("[RPA Consumer] ─────────────────────────────────────────────────────────────────────")

	var gradingJobIDs []string
	var successCount int
	var failCount int

	for i, filePath := range response.FilePaths {
		log.Printf("[RPA Consumer] 📋 处理文件 %d/%d:", i+1, len(response.FilePaths))
		log.Printf("[RPA Consumer]    - 文件路径: %s", filePath)

		// ============================
		// 环节 B：为每个文件独立匹配 Assignment
		// ============================
		// 从文件路径中提取班级信息
		// 文件路径格式: ./uploads/rpa/xxx/班级2CS20510502-第一次作业(附件).zip
		className := extractClassNameFromFilePath(filePath)
		log.Printf("[RPA Consumer] 📝 从文件路径提取班级信息: %s", className)

		assignmentID, err := matchOrCreateAssignment(ctx, task.CourseName, className, task.AssignmentName)
		if err != nil {
			log.Printf("[RPA Consumer] ⚠️  匹配/创建Assignment失败: %v", err)
			log.Printf("[RPA Consumer] 📝 将继续投递任务，AssignmentID将设为0")
			// 即使匹配失败，也继续投递到批改队列，assignmentID可以为0
		} else {
			log.Printf("[RPA Consumer] ✅ Assignment匹配成功")
			log.Printf("[RPA Consumer]    - 班级: %s, Assignment ID: %d", className, assignmentID)
		}

		// 生成新的 gradingJobID
		gradingJobID := uuid.New().String()
		log.Printf("[RPA Consumer]    - 生成批改任务ID: %s", gradingJobID)

		// 往 AsyncJob 表插入一条新记录
		log.Printf("[RPA Consumer]    - 创建批改任务记录到MySQL...")
		gradingAsyncJob := models.AsyncJob{
			ID:          gradingJobID,
			JobType:     "grade_homework",
			ReferenceID: task.JobID, // 关联RPA任务ID
			StudentID:   task.Username,
			Status:      models.JobStatusPending,
			Message:     "RPA抓取完成，等待批改",
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		}

		// 1. 先写入MySQL
		if err := database.DB.Create(&gradingAsyncJob).Error; err != nil {
			log.Printf("[RPA Consumer] ❌ 创建批改AsyncJob失败: %v", err)
			failCount++
			continue
		}
		log.Printf("[RPA Consumer] ✅ 批改任务MySQL记录已创建")

		// 2. 然后写入Redis缓存
		if err := cache.SetJobStatus(ctx, gradingJobID, string(models.JobStatusPending), "RPA抓取完成，等待批改"); err != nil {
			log.Printf("[RPA Consumer] ⚠️  创建批改AsyncJob Redis缓存失败: %v", err)
			log.Printf("[RPA Consumer] ⚠️  继续执行，因为MySQL已成功，但缓存不一致")
			// 不中断流程，因为MySQL写入成功
		} else {
			log.Printf("[RPA Consumer] ✅ 批改任务Redis缓存已创建")
		}

		// 构造 GradingTaskMessage，包含AssignmentID
		log.Printf("[RPA Consumer]    - 构造批改任务消息...")
		gradingMessage := HomeworkTaskMessage{
			JobID:        gradingJobID,
			AssignmentID: assignmentID,
			ZipPath:      filePath,
		}
		log.Printf("[RPA Consumer] ✅ 批改任务消息构造完成")
		log.Printf("[RPA Consumer]    - JobID: %s", gradingMessage.JobID)
		log.Printf("[RPA Consumer]    - AssignmentID: %d", gradingMessage.AssignmentID)
		log.Printf("[RPA Consumer]    - ZipPath: %s", gradingMessage.ZipPath)

		// 投递到 topic_grading_homework 队列
		log.Printf("[RPA Consumer]    - 投递到Kafka topic_grading_homework...")
		if err := publishMessage(TopicGradingHomework, gradingMessage); err != nil {
			log.Printf("[RPA Consumer] ❌ 投递批改任务到Kafka失败: %v", err)
			// 更新状态为FAILED
			database.DB.Model(&gradingAsyncJob).Updates(models.AsyncJob{
				Status:    models.JobStatusFailed,
				Message:   fmt.Sprintf("投递批改任务失败: %v", err),
				UpdatedAt: time.Now(),
			})
			log.Printf("[RPA Consumer] ✅ 已更新批改任务状态为 FAILED")
			failCount++
			continue
		}

		gradingJobIDs = append(gradingJobIDs, gradingJobID)
		successCount++
		log.Printf("[RPA Consumer] ✅ 批改任务已成功投递到Kafka")
		log.Printf("[RPA Consumer]    - 批改任务ID: %s", gradingJobID)
	}

	// 更新RPA任务状态为SUCCESS
	log.Printf("────────────────────────────────────────────────────────────────────────────")
	log.Printf("[RPA Consumer] 📊 任务处理统计:")
	log.Printf("[RPA Consumer]    - 总文件数: %d", len(response.FilePaths))
	log.Printf("[RPA Consumer]    - 成功投递: %d", successCount)
	log.Printf("[RPA Consumer]    - 投递失败: %d", failCount)

	successMsg := fmt.Sprintf("RPA抓取完成，共下载 %d 个文件，成功投递 %d 个批改任务，失败 %d 个",
		len(response.FilePaths), successCount, failCount)

	log.Printf("[RPA Consumer] 📋 更新RPA任务状态为SUCCESS...")
	if err := updateRPAJobStatus(task.JobID, models.JobStatusSuccess, successMsg); err != nil {
		log.Printf("[RPA Consumer] ⚠️  更新RPA任务状态失败: %v", err)
	} else {
		log.Printf("[RPA Consumer] ✅ MySQL任务状态已更新为 SUCCESS")
	}

	if err := cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusSuccess), successMsg); err != nil {
		log.Printf("[RPA Consumer] ⚠️  更新Redis缓存状态失败: %v", err)
	} else {
		log.Printf("[RPA Consumer] ✅ Redis缓存状态已更新为 SUCCESS")
	}

	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[RPA Consumer] 🎉 RPA抓取任务处理完成")
	log.Printf("[RPA Consumer] 🆔 RPA任务ID: %s", task.JobID)
	log.Printf("[RPA Consumer] 📋 批改任务ID列表: %v", gradingJobIDs)
	log.Printf("[RPA Consumer] 📤 已投递 %d 个批改任务到 topic_grading_homework", successCount)
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	return nil
}

// matchOrCreateAssignment 通过Redis Cache-Aside匹配或创建Assignment
func matchOrCreateAssignment(ctx context.Context, courseName, className, assignmentName string) (uint, error) {
	log.Printf("[RPA Consumer] 🔍 开始匹配/创建Assignment...")
	log.Printf("[RPA Consumer]    - 课程名称: %s", courseName)
	log.Printf("[RPA Consumer]    - 班级名称: %s", className)
	log.Printf("[RPA Consumer]    - 作业名称: %s", assignmentName)

	// 设计 Redis Key：ai_grade:assignment:{course_name}:{class_name}:{assignment_name}
	redisKey := fmt.Sprintf("ai_grade:assignment:%s:%s:%s", courseName, className, assignmentName)
	log.Printf("[RPA Consumer]    - Redis Key: %s", redisKey)

	// 1. 查缓存：尝试从 Redis 获取该 Key
	log.Printf("[RPA Consumer] 📋 步骤 B1/3: 查询Redis缓存...")
	redisClient := database.GetRedisClient()
	cachedAssignmentID := uint64(0) // 记录缓存的Assignment ID
	if redisClient != nil {
		val, err := redisClient.Get(ctx, redisKey).Result()
		if err == nil {
			// 命中缓存，反序列化拿到 AssignmentID
			assignmentID, parseErr := strconv.ParseUint(val, 10, 32)
			if parseErr == nil {
				cachedAssignmentID = assignmentID // 记录缓存的ID
				log.Printf("[RPA Consumer] ✅ Redis缓存命中")
				log.Printf("[RPA Consumer]    - Assignment ID: %d", assignmentID)
			} else {
				log.Printf("[RPA Consumer] ⚠️  解析缓存AssignmentID失败: %v", parseErr)
			}
		} else if err != redis.Nil {
			log.Printf("[RPA Consumer] ⚠️  Redis查询错误: %v", err)
		} else {
			log.Printf("[RPA Consumer] ℹ️  Redis缓存未命中")
		}
	} else {
		log.Printf("[RPA Consumer] ⚠️  Redis客户端未初始化，跳过缓存查询")
	}

	// 2. 查DB/降级/新建：若未命中（redis.Nil）或 Redis 宕机，直接查 MySQL
	log.Printf("[RPA Consumer] 📋 步骤 B2/3: 查询MySQL数据库...")
	var assignment models.Assignment
	result := database.DB.Where("course_name = ? AND class_name = ? AND task_name = ?", courseName, className, assignmentName).First(&assignment)

	if result.Error == nil {
		// 找到记录
		log.Printf("[RPA Consumer] ✅ MySQL查询成功")
		log.Printf("[RPA Consumer]    - Assignment ID: %d", assignment.ID)
		log.Printf("[RPA Consumer]    - Task Name: %s", assignment.TaskName)

		// 如果缓存的ID与数据库中的ID不一致，更新缓存
		if cachedAssignmentID > 0 && cachedAssignmentID != uint64(assignment.ID) {
			log.Printf("[RPA Consumer] ⚠️  检测到缓存数据不一致，更新缓存...")
			log.Printf("[RPA Consumer]    - 缓存ID: %d, 数据库ID: %d", cachedAssignmentID, assignment.ID)
			if redisClient != nil {
				redisClient.Del(ctx, redisKey) // 先删除旧缓存
			}
		}

		// 写缓存
		log.Printf("[RPA Consumer] 📝 写入Redis缓存...")
		if redisClient != nil {
			setErr := redisClient.Set(ctx, redisKey, strconv.Itoa(int(assignment.ID)), 7*24*time.Hour).Err()
			if setErr != nil {
				log.Printf("[RPA Consumer] ⚠️  写入Redis缓存失败: %v", setErr)
			} else {
				log.Printf("[RPA Consumer] ✅ Redis缓存写入成功")
			}
		}

		return assignment.ID, nil
	}

	// 如果数据库查询失败，检查是否有缓存失效问题（缓存的ID在数据库中不存在）
	if cachedAssignmentID > 0 {
		log.Printf("[RPA Consumer] ⚠️  检测到缓存失效，清除旧缓存...")
		if redisClient != nil {
			redisClient.Del(ctx, redisKey)
			log.Printf("[RPA Consumer] ✅ 已清除失效缓存")
		}
	}

	// 3. 若 MySQL 中也没有记录，则自动 Create 一条 Assignment，状态设为 PENDING_RUBRIC
	log.Printf("[RPA Consumer] ℹ️  MySQL中未找到Assignment记录")
	log.Printf("[RPA Consumer] 📋 步骤 B3/3: 自动创建新的Assignment...")

	newAssignment := models.Assignment{
		CourseName: courseName,
		ClassName:  className,
		TaskName:   assignmentName,
		Question:   assignmentName + "（自动创建）",
		Rubric:     "", // 空的评分标准，等待教师补充
	}

	if err := database.DB.Create(&newAssignment).Error; err != nil {
		log.Printf("[RPA Consumer] ❌ 创建Assignment失败: %v", err)
		return 0, fmt.Errorf("failed to create assignment: %w", err)
	}

	log.Printf("[RPA Consumer] ✅ 新Assignment已创建")
	log.Printf("[RPA Consumer]    - Assignment ID: %d", newAssignment.ID)
	log.Printf("[RPA Consumer]    - Task Name: %s", newAssignment.TaskName)

	// 写缓存
	log.Printf("[RPA Consumer] 📝 写入Redis缓存...")
	if redisClient != nil {
		setErr := redisClient.Set(ctx, redisKey, strconv.Itoa(int(newAssignment.ID)), 7*24*time.Hour).Err()
		if setErr != nil {
			log.Printf("[RPA Consumer] ⚠️  写入Redis缓存失败: %v", setErr)
		} else {
			log.Printf("[RPA Consumer] ✅ Redis缓存写入成功")
		}
	}

	return newAssignment.ID, nil
}

// updateRPAJobStatus 更新RPA任务状态
func updateRPAJobStatus(jobID string, status models.AsyncJobStatus, message string) error {
	var job models.AsyncJob
	result := database.DB.Where("id = ?", jobID).First(&job)
	if result.Error != nil {
		log.Printf("[RPA Consumer] ⚠️  AsyncJob %s 未找到: %v", jobID, result.Error)
		return result.Error
	}

	job.Status = status
	job.Message = message
	job.UpdatedAt = time.Now()

	return database.DB.Save(&job).Error
}

// StartRPAConsumer 启动 RPA 消费者组
func StartRPAConsumer() error {
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[RPA Consumer] 🚀 启动RPA消费者组")

	// 为RPA消费者创建独立的消费者组，避免与批改消费者竞争
	rpaConsumerConfig := sarama.NewConfig()
	rpaConsumerConfig.Consumer.Return.Errors = true
	rpaConsumerConfig.Consumer.Offsets.Initial = sarama.OffsetOldest

	rpaConsumerGroup, err := sarama.NewConsumerGroup([]string{"localhost:9092"}, "rpa-consumer-group", rpaConsumerConfig)
	if err != nil {
		log.Printf("[RPA Consumer] ❌ 创建RPA消费者组失败: %v", err)
		return fmt.Errorf("failed to create RPA consumer group: %w", err)
	}

	ctx := context.Background()
	consumer := RPAConsumer{}
	topics := []string{TopicRPAFetch}

	log.Printf("[RPA Consumer] 📋 消费者配置:")
	log.Printf("[RPA Consumer]    - Topic: %s", TopicRPAFetch)
	log.Printf("[RPA Consumer]    - Consumer Group: rpa-consumer-group")

	// 启动消费循环
	go func() {
		for {
			log.Printf("[RPA Consumer] 🔄 开始消费循环...")
			err := rpaConsumerGroup.Consume(ctx, topics, consumer)
			if err != nil {
				log.Printf("[RPA Consumer] ❌ 消费者组错误: %v", err)
				log.Printf("[RPA Consumer] ⏳ 等待5秒后重试...")
				// 等待一段时间后重试
				time.Sleep(5 * time.Second)
				continue
			}

			// 如果上下文被取消，退出循环
			if ctx.Err() != nil {
				log.Printf("[RPA Consumer] 🛑 消费者上下文已取消: %v", ctx.Err())
				return
			}
		}
	}()

	log.Printf("[RPA Consumer] ✅ RPA消费者已启动")
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	return nil
}

// extractClassNameFromFilePath 从文件路径中提取班级名称
// 文件路径格式: ./uploads/rpa/xxx/班级2CS20510502-第一次作业(附件).zip
// 需要提取班级: 2CS20510502
func extractClassNameFromFilePath(filePath string) string {
	// 使用正则表达式提取班级信息
	// 匹配模式: "班级" 后面跟着字母数字组合，直到遇到 "-"
	re := regexp.MustCompile(`班级([A-Za-z0-9]+)-`)
	matches := re.FindStringSubmatch(filePath)

	if len(matches) >= 2 {
		return matches[1]
	}

	// 如果正则匹配失败，尝试手动解析
	// 找到 "班级" 的位置
	classIndex := strings.Index(filePath, "班级")
	if classIndex == -1 {
		return "未知班级"
	}

	// 从 "班级" 后面开始提取
	start := classIndex + len("班级")
	if start >= len(filePath) {
		return "未知班级"
	}

	// 找到 "-" 的位置
	endIndex := strings.Index(filePath[start:], "-")
	if endIndex == -1 {
		return "未知班级"
	}

	return filePath[start : start+endIndex]
}
