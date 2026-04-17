package mq

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
	"grading-gateway/internal/tools"

	"github.com/IBM/sarama"
)

// JobItem 表示一个待处理的任务项，包含原始消息和会话信息
// Worker将从jobChan中读取JobItem，处理完成后提交Offset
type JobItem struct {
	Message *sarama.ConsumerMessage
	Session sarama.ConsumerGroupSession
	Claim   sarama.ConsumerGroupClaim
}

// GradingConsumer 实现 sarama.ConsumerGroupHandler
// 引入Worker Pool模式提升吞吐量，支持并发处理8个任务
type GradingConsumer struct {
	jobChan        chan JobItem   // 任务通道，缓冲大小20
	wg             sync.WaitGroup // 等待组，用于优雅停机
	workerCount    int            // 工作协程数量
	stopChan       chan struct{}  // 停止信号通道
	workersStopped bool           // 工作协程是否已停止
}

// NewGradingConsumer 创建并初始化一个新的GradingConsumer实例
// 启动Worker Pool，包含8个Worker协程，任务通道缓冲大小为20
func NewGradingConsumer() *GradingConsumer {
	consumer := &GradingConsumer{
		jobChan:        make(chan JobItem, 20), // 缓冲通道，防止生产者过快
		workerCount:    8,                      // 8个Worker，略小于Python端的10个max_workers
		stopChan:       make(chan struct{}),
		workersStopped: false,
	}

	// 启动Worker协程
	for i := 0; i < consumer.workerCount; i++ {
		consumer.wg.Add(1)
		go consumer.worker(i + 1)
	}

	log.Printf("[Grading Consumer] 🚀 Worker Pool已启动: %d个Worker, 通道缓冲20", consumer.workerCount)
	return consumer
}

// worker 是工作协程，负责从jobChan中读取任务并处理
// 关键：Offset提交必须在任务处理完成后进行，避免消息丢失
func (c *GradingConsumer) worker(id int) {
	defer c.wg.Done()

	log.Printf("[Grading Consumer] 👷 Worker %d 已启动", id)

	for {
		select {
		case job, ok := <-c.jobChan:
			if !ok {
				// 通道关闭，退出Worker
				log.Printf("[Grading Consumer] 👷 Worker %d 停止工作，任务通道已关闭", id)
				return
			}

			// 处理任务
			log.Printf("────────────────────────────────────────────────────────────────────────────")
			log.Printf("[Grading Consumer] 👷 Worker %d 开始处理任务", id)
			log.Printf("[Grading Consumer]    - Topic: %s", job.Message.Topic)
			log.Printf("[Grading Consumer]    - Partition: %d", job.Message.Partition)
			log.Printf("[Grading Consumer]    - Offset: %d", job.Message.Offset)

			// 根据主题调用相应的处理函数
			ctx := context.Background()
			var err error

			switch job.Message.Topic {
			case TopicGradingHomework:
				err = handleHomeworkTask(ctx, job.Message.Value)
			case TopicGradingExam:
				err = handleExamTask(ctx, job.Message.Value)
			default:
				log.Printf("[Grading Consumer] ⚠️  Worker %d 收到未知主题 %s，忽略消息", id, job.Message.Topic)
				continue
			}

			if err != nil {
				log.Printf("[Grading Consumer] ❌ Worker %d 处理任务失败: %v", id, err)
				log.Printf("[Grading Consumer] ⚠️  消息不会标记为完成，将触发Kafka重试机制")
				// 注意：不标记消息为完成，让 Kafka 重试
				continue
			}

			// 任务处理成功，提交Offset（关键：在Worker中提交，确保处理完成后再提交）
			job.Session.MarkMessage(job.Message, "")
			job.Session.Commit()
			log.Printf("[Grading Consumer] ✅ Worker %d 任务处理成功，已提交Offset", id)
			log.Printf("[Grading Consumer]    - Committed Offset: %d", job.Message.Offset)

		case <-c.stopChan:
			// 收到停止信号，退出Worker
			log.Printf("[Grading Consumer] 👷 Worker %d 收到停止信号，准备退出", id)
			return
		}
	}
}

// Setup 在消费者组分配分区时调用
func (c *GradingConsumer) Setup(session sarama.ConsumerGroupSession) error {
	log.Printf("[Grading Consumer] ✅ 消费者组会话初始化完成")
	log.Printf("[Grading Consumer]    - Member ID: %s", session.MemberID())
	log.Printf("[Grading Consumer]    - Generation ID: %d", session.GenerationID())
	return nil
}

// Cleanup 在消费者组释放分区时调用
func (c *GradingConsumer) Cleanup(session sarama.ConsumerGroupSession) error {
	log.Printf("[Grading Consumer] 🔄 消费者组会话清理完成")
	log.Printf("[Grading Consumer]    - Member ID: %s", session.MemberID())

	// 优雅停机：关闭任务通道，等待所有Worker完成
	if !c.workersStopped {
		log.Printf("[Grading Consumer] 🛑 开始优雅停机流程...")
		c.Stop()
	}

	return nil
}

// ConsumeClaim 处理分配给此消费者的消息
// 使用Worker Pool模式：将消息打包为JobItem发送到jobChan，由Worker处理
func (c *GradingConsumer) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[Grading Consumer] 🎯 开始消费分区消息")
	log.Printf("[Grading Consumer]    - Partition: %d", claim.Partition())
	log.Printf("[Grading Consumer]    - Initial Offset: %d", claim.InitialOffset())
	log.Printf("[Grading Consumer]    - Topic: %s", claim.Topic())
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	for msg := range claim.Messages() {
		log.Printf("────────────────────────────────────────────────────────────────────────────")
		log.Printf("[Grading Consumer] 📨 收到新消息")
		log.Printf("[Grading Consumer]    - Topic: %s", msg.Topic)
		log.Printf("[Grading Consumer]    - Partition: %d", msg.Partition)
		log.Printf("[Grading Consumer]    - Offset: %d", msg.Offset)
		log.Printf("[Grading Consumer]    - Timestamp: %s", msg.Timestamp.Format("2006-01-02 15:04:05"))
		log.Printf("────────────────────────────────────────────────────────────────────────────")

		// 创建任务项
		job := JobItem{
			Message: msg,
			Session: session,
			Claim:   claim,
		}

		// 将任务发送到jobChan（Worker Pool模式）
		// 注意：这里不会立即提交Offset，Offset提交由Worker在任务完成后进行
		log.Printf("[Grading Consumer] 📤 将任务投递到Worker Pool，等待Worker处理...")
		log.Printf("[Grading Consumer]    - 当前通道长度: %d/%d", len(c.jobChan), cap(c.jobChan))

		select {
		case c.jobChan <- job:
			log.Printf("[Grading Consumer] ✅ 任务已成功投递到Worker Pool")
			// 任务已进入处理队列，Offset将在Worker处理完成后提交
		case <-c.stopChan:
			// 收到停止信号，退出消费循环
			log.Printf("[Grading Consumer] 🛑 收到停止信号，停止消费新消息")
			return nil
		}
	}

	return nil
}

// Stop 优雅停止Worker Pool
// 1. 关闭jobChan阻止新任务投递
// 2. 发送停止信号给所有Worker
// 3. 等待所有Worker完成当前任务
func (c *GradingConsumer) Stop() {
	if c.workersStopped {
		return
	}

	log.Printf("[Grading Consumer] 🛑 开始优雅停止Worker Pool...")

	// 1. 关闭任务通道，防止新任务投递
	close(c.jobChan)
	c.workersStopped = true

	// 2. 发送停止信号给所有Worker
	close(c.stopChan)

	// 3. 等待所有Worker完成
	log.Printf("[Grading Consumer] ⏳ 等待Worker完成当前任务...")
	c.wg.Wait()

	log.Printf("[Grading Consumer] ✅ Worker Pool已完全停止")
}

// handleHomeworkTask 处理作业批改任务
func handleHomeworkTask(ctx context.Context, message []byte) error {
	var task HomeworkTaskMessage
	if err := json.Unmarshal(message, &task); err != nil {
		log.Printf("[Grading Consumer] ❌ 消息反序列化失败: %v", err)
		return fmt.Errorf("failed to unmarshal homework task: %w", err)
	}

	// 加锁保证幂等性
	redisClient := database.GetRedisClient()
	if redisClient != nil {
		lockKey := "lock:job:" + task.JobID
		acquired, err := redisClient.SetNX(ctx, lockKey, 1, 30*time.Minute).Result()
		if err != nil {
			log.Printf("ERROR: Failed to acquire redis lock for homework task %s: %v", task.JobID, err)
			return err
		}
		if !acquired {
			log.Printf("INFO: Task %s is already being processed or completed. Skipping.", task.JobID)
			return nil
		}
	}

	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[Grading Consumer] 🚀 开始处理作业批改任务")
	log.Printf("[Grading Consumer] �� 任务信息:")
	log.Printf("[Grading Consumer]    - Job ID: %s", task.JobID)
	log.Printf("[Grading Consumer]    - Assignment ID: %d", task.AssignmentID)
	log.Printf("[Grading Consumer]    - Zip Path: %s", task.ZipPath)
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 1. 更新 MySQL AsyncJob 状态为 PROCESSING（先数据库）
	log.Printf("[Grading Consumer] 📋 步骤 1/4: 更新MySQL任务状态为PROCESSING...")
	if err := updateAsyncJobStatus(task.JobID, models.JobStatusProcessing, "开始处理作业批改"); err != nil {
		log.Printf("[Grading Consumer] ⚠️  更新MySQL状态失败: %v", err)
	} else {
		log.Printf("[Grading Consumer] ✅ MySQL任务状态已更新为 PROCESSING")
	}

	// 2. 更新 Redis 状态为 PROCESSING（后缓存）
	log.Printf("[Grading Consumer] 📋 步骤 2/4: 更新Redis缓存状态为PROCESSING...")
	if err := cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusProcessing), "开始处理作业批改"); err != nil {
		log.Printf("[Grading Consumer] ⚠️  更新Redis状态失败: %v", err)
	} else {
		log.Printf("[Grading Consumer] ✅ Redis缓存状态已更新为 PROCESSING")
	}

	// 3. 调用 tools.ProcessPipeline 函数
	log.Printf("[Grading Consumer] 📋 步骤 3/4: 调用批改流水线处理作业...")
	log.Printf("[Grading Consumer] 🔄 正在调用 ProcessPipeline 接口...")

	startTime := time.Now()
	defer func() {
		elapsed := time.Since(startTime)
		log.Printf("[Grading Consumer] ⏱️  批改任务总耗时: %v", elapsed)
	}()

	// 捕获 panic
	defer func() {
		if r := recover(); r != nil {
			errMsg := fmt.Sprintf("Panic in ProcessPipeline: %v", r)
			log.Printf("[Grading Consumer] ❌ PANIC: %s", errMsg)
			cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusFailed), errMsg)
			updateAsyncJobStatus(task.JobID, models.JobStatusFailed, errMsg)
		}
	}()

	tools.ProcessPipeline(fmt.Sprintf("%d", task.AssignmentID), task.ZipPath)

	log.Printf("[Grading Consumer] ✅ 批改流水线执行完成")

	// 4. 处理完成，更新状态为 SUCCESS
	log.Printf("[Grading Consumer] 📋 步骤 4/4: 更新任务状态为SUCCESS...")
	updateAsyncJobStatus(task.JobID, models.JobStatusSuccess, "作业批改完成")
	cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusSuccess), "作业批改完成")

	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[Grading Consumer] 🎉 作业批改任务处理完成")
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	return nil
}
func handleExamTask(ctx context.Context, message []byte) error {
	var task ExamTaskMessage
	if err := json.Unmarshal(message, &task); err != nil {
		log.Printf("[Grading Consumer] ❌ 消息反序列化失败: %v", err)
		return fmt.Errorf("failed to unmarshal exam task: %w", err)
	}

	// 加锁保证幂等性
	redisClient := database.GetRedisClient()
	if redisClient != nil {
		lockKey := "lock:job:" + task.JobID
		acquired, err := redisClient.SetNX(ctx, lockKey, 1, 30*time.Minute).Result()
		if err != nil {
			log.Printf("ERROR: Failed to acquire redis lock for exam task %s: %v", task.JobID, err)
			return err
		}
		if !acquired {
			log.Printf("INFO: Task %s is already being processed or completed. Skipping.", task.JobID)
			return nil
		}
	}

	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[Grading Consumer] 🚀 开始处理试卷批改任务")
	log.Printf("[Grading Consumer] 📋 任务信息:")
	log.Printf("[Grading Consumer]    - Job ID: %s", task.JobID)
	log.Printf("[Grading Consumer]    - Exam ID: %s", task.ExamID)
	log.Printf("[Grading Consumer]    - Student ID: %s", task.StudentID)
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 1. 更新 MySQL AsyncJob 状态为 PROCESSING
	updateAsyncJobStatus(task.JobID, models.JobStatusProcessing, "开始处理试卷批改")
	// 2. 更新 Redis 状态为 PROCESSING
	cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusProcessing), "开始处理试卷批改")

	// 3. 调用 tools.ProcessExamSubmission 函数
	startTime := time.Now()
	defer func() {
		log.Printf("[Grading Consumer] ⏱️  试卷批改任务总耗时: %v", time.Since(startTime))
	}()

	// 捕获 panic
	defer func() {
		if r := recover(); r != nil {
			errMsg := fmt.Sprintf("Panic in ProcessExamSubmission: %v", r)
			log.Printf("[Grading Consumer] ❌ PANIC: %s", errMsg)
			cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusFailed), errMsg)
			updateAsyncJobStatus(task.JobID, models.JobStatusFailed, errMsg)
		}
	}()

	tools.ProcessExamSubmission(task.ExamID, task.StudentID, task.ImagePaths)

	// 4. 处理完成，更新状态为 SUCCESS
	updateAsyncJobStatus(task.JobID, models.JobStatusSuccess, "试卷批改完成")
	cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusSuccess), "试卷批改完成")

	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[Grading Consumer] 🎉 试卷批改任务处理完成")
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	return nil
}
func updateAsyncJobStatus(jobID string, status models.AsyncJobStatus, message string) error {
	log.Printf("[Grading Consumer] 📝 更新AsyncJob状态...")
	log.Printf("[Grading Consumer]    - Job ID: %s", jobID)
	log.Printf("[Grading Consumer]    - Status: %s", status)
	log.Printf("[Grading Consumer]    - Message: %s", message)

	// 查找现有的 AsyncJob
	var job models.AsyncJob
	result := database.DB.Where("id = ?", jobID).First(&job)
	if result.Error != nil {
		// 如果找不到记录，可能是任务创建时出错了，我们尝试创建一条记录
		// 但这种情况不应该发生，因为任务创建时会先插入记录
		log.Printf("[Grading Consumer] ⚠️  AsyncJob %s 未找到，尝试创建新记录: %v", jobID, result.Error)
		job = models.AsyncJob{
			ID:        jobID,
			Status:    status,
			Message:   message,
			UpdatedAt: time.Now(),
		}
		// 由于缺少 JobType 和 ReferenceID 等信息，我们只能尽力保存
		if err := database.DB.Create(&job).Error; err != nil {
			log.Printf("[Grading Consumer] ❌ 创建AsyncJob记录失败: %v", err)
			return err
		}
		log.Printf("[Grading Consumer] ✅ 新AsyncJob记录已创建")
		return nil
	}

	// 更新状态和消息
	job.Status = status
	job.Message = message
	job.UpdatedAt = time.Now()

	if err := database.DB.Save(&job).Error; err != nil {
		log.Printf("[Grading Consumer] ❌ 更新AsyncJob记录失败: %v", err)
		return err
	}

	log.Printf("[Grading Consumer] ✅ AsyncJob状态已更新")
	return nil
}

// UpdateAsyncJobStatus 更新 MySQL 中的 AsyncJob 记录（供外部使用）
func UpdateAsyncJobStatus(jobID string, status models.AsyncJobStatus, message string) error {
	return updateAsyncJobStatus(jobID, status, message)
}

// StartKafkaConsumer 启动 Kafka 消费者组
func StartKafkaConsumer(topics []string) error {
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[Grading Consumer] 🚀 启动批改任务消费者组")

	consumerGroup := GetConsumerGroup()
	if consumerGroup == nil {
		log.Printf("[Grading Consumer] ❌ Kafka消费者组未初始化")
		return fmt.Errorf("Kafka consumer group not initialized")
	}

	ctx := context.Background()
	consumer := NewGradingConsumer() // 使用Worker Pool模式的消费者

	log.Printf("[Grading Consumer] 📋 消费者配置:")
	log.Printf("[Grading Consumer]    - Topics: %v", topics)
	log.Printf("[Grading Consumer]    - Consumer Group: grading-consumer-group")
	log.Printf("[Grading Consumer]    - Worker Count: %d", consumer.workerCount)
	log.Printf("[Grading Consumer]    - Job Channel Buffer: %d", cap(consumer.jobChan))

	// 启动消费循环
	go func() {
		for {
			log.Printf("[Grading Consumer] 🔄 开始消费循环...")
			err := consumerGroup.Consume(ctx, topics, consumer)
			if err != nil {
				log.Printf("[Grading Consumer] ❌ 消费者组错误: %v", err)
				log.Printf("[Grading Consumer] ⏳ 等待5秒后重试...")
				// 等待一段时间后重试
				time.Sleep(5 * time.Second)
				continue
			}

			// 如果上下文被取消，退出循环
			if ctx.Err() != nil {
				log.Printf("[Grading Consumer] 🛑 消费者上下文已取消: %v", ctx.Err())
				return
			}
		}
	}()

	log.Printf("[Grading Consumer] ✅ 批改任务消费者已启动")
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	return nil
}

// CreateAsyncJob 创建异步任务记录（供生产者使用）
func CreateAsyncJob(jobID string, jobType models.AsyncJobType, referenceID string, studentID string) error {
	log.Printf("[Grading Consumer] 📝 创建AsyncJob记录...")
	log.Printf("[Grading Consumer]    - Job ID: %s", jobID)
	log.Printf("[Grading Consumer]    - Job Type: %s", jobType)
	log.Printf("[Grading Consumer]    - Reference ID: %s", referenceID)
	log.Printf("[Grading Consumer]    - Student ID: %s", studentID)

	job := models.AsyncJob{
		ID:          jobID,
		JobType:     jobType,
		ReferenceID: referenceID,
		StudentID:   studentID,
		Status:      models.JobStatusPending,
		Message:     "任务已创建，等待处理",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	// 1. 先写入MySQL
	if err := database.DB.Create(&job).Error; err != nil {
		log.Printf("[Grading Consumer] ❌ 创建AsyncJob记录失败: %v", err)
		return fmt.Errorf("failed to create async job: %w", err)
	}
	log.Printf("[Grading Consumer] ✅ AsyncJob MySQL记录已创建")

	// 2. 然后写入Redis缓存
	ctx := context.Background()
	if err := cache.SetJobStatus(ctx, jobID, string(models.JobStatusPending), "任务已创建，等待处理"); err != nil {
		log.Printf("[Grading Consumer] ⚠️  创建AsyncJob Redis缓存失败: %v", err)
		log.Printf("[Grading Consumer] ⚠️  继续执行，因为MySQL已成功，但缓存不一致")
		// 不返回错误，因为MySQL写入成功，业务可以继续
	} else {
		log.Printf("[Grading Consumer] ✅ AsyncJob Redis缓存已创建")
	}

	log.Printf("[Grading Consumer] 🎉 AsyncJob双写完成")
	return nil
}
