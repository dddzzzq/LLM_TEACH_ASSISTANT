package mq

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
	"grading-gateway/internal/tools"

	"github.com/IBM/sarama"
)

// GradingConsumer 实现 sarama.ConsumerGroupHandler
type GradingConsumer struct{}

// Setup 在消费者组分配分区时调用
func (GradingConsumer) Setup(session sarama.ConsumerGroupSession) error {
	log.Printf("Consumer group session setup: member_id=%s, generation_id=%d", session.MemberID(), session.GenerationID())
	return nil
}

// Cleanup 在消费者组释放分区时调用
func (GradingConsumer) Cleanup(session sarama.ConsumerGroupSession) error {
	log.Printf("Consumer group session cleanup: member_id=%s", session.MemberID())
	return nil
}

// ConsumeClaim 处理分配给此消费者的消息
func (GradingConsumer) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
	log.Printf("Starting to consume from partition %d, initial offset %d", claim.Partition(), claim.InitialOffset())

	for msg := range claim.Messages() {
		ctx := context.Background()
		log.Printf("Received message: topic=%s, partition=%d, offset=%d", msg.Topic, msg.Partition, msg.Offset)

		// 根据主题分发处理
		switch msg.Topic {
		case TopicGradingHomework:
			if err := handleHomeworkTask(ctx, msg.Value); err != nil {
				log.Printf("ERROR handling homework task: %v", err)
				// 注意：不标记消息为完成，让 Kafka 重试
				continue
			}
		case TopicGradingExam:
			if err := handleExamTask(ctx, msg.Value); err != nil {
				log.Printf("ERROR handling exam task: %v", err)
				continue
			}
		default:
			log.Printf("WARNING: Unknown topic %s, ignoring", msg.Topic)
		}

		// 处理成功，提交 offset
		session.MarkMessage(msg, "")
		session.Commit()
	}

	return nil
}

// handleHomeworkTask 处理作业批改任务
func handleHomeworkTask(ctx context.Context, message []byte) error {
	var task HomeworkTaskMessage
	if err := json.Unmarshal(message, &task); err != nil {
		return fmt.Errorf("failed to unmarshal homework task: %w", err)
	}

	log.Printf("Processing homework task: job=%s, assignment=%s", task.JobID, task.AssignmentID)

	// 1. 更新 Redis 状态为 PROCESSING
	if err := cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusProcessing), "开始处理作业批改"); err != nil {
		log.Printf("WARNING: Failed to update Redis status for job %s: %v", task.JobID, err)
	}

	// 2. 更新 MySQL AsyncJob 状态为 PROCESSING
	if err := updateAsyncJobStatus(task.JobID, models.JobStatusProcessing, "开始处理作业批改"); err != nil {
		log.Printf("WARNING: Failed to update MySQL async job status for job %s: %v", task.JobID, err)
	}

	// 3. 调用现有的 tools.ProcessPipeline 函数
	// 注意：这是一个同步调用，会阻塞当前消费者协程
	// 由于我们设置了并发度1，这可以保护后端 Python 节点不过载
	startTime := time.Now()
	defer func() {
		log.Printf("Homework task %s completed in %v", task.JobID, time.Since(startTime))
	}()

	// 捕获 panic，防止消费者崩溃
	defer func() {
		if r := recover(); r != nil {
			errMsg := fmt.Sprintf("Panic in ProcessPipeline: %v", r)
			log.Printf("PANIC: %s", errMsg)
			// 更新状态为 FAILED
			cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusFailed), errMsg)
			updateAsyncJobStatus(task.JobID, models.JobStatusFailed, errMsg)
		}
	}()

	tools.ProcessPipeline(task.AssignmentID, task.ZipPath)

	// 4. 处理完成，更新状态为 SUCCESS
	if err := cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusSuccess), "作业批改完成"); err != nil {
		log.Printf("WARNING: Failed to update Redis status for job %s: %v", task.JobID, err)
	}
	if err := updateAsyncJobStatus(task.JobID, models.JobStatusSuccess, "作业批改完成"); err != nil {
		log.Printf("WARNING: Failed to update MySQL async job status for job %s: %v", task.JobID, err)
	}

	log.Printf("Homework task %s processed successfully", task.JobID)
	return nil
}

// handleExamTask 处理试卷批改任务
func handleExamTask(ctx context.Context, message []byte) error {
	var task ExamTaskMessage
	if err := json.Unmarshal(message, &task); err != nil {
		return fmt.Errorf("failed to unmarshal exam task: %w", err)
	}

	log.Printf("Processing exam task: job=%s, exam=%s, student=%s", task.JobID, task.ExamID, task.StudentID)

	// 1. 更新 Redis 状态为 PROCESSING
	if err := cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusProcessing), "开始处理试卷批改"); err != nil {
		log.Printf("WARNING: Failed to update Redis status for job %s: %v", task.JobID, err)
	}

	// 2. 更新 MySQL AsyncJob 状态为 PROCESSING
	if err := updateAsyncJobStatus(task.JobID, models.JobStatusProcessing, "开始处理试卷批改"); err != nil {
		log.Printf("WARNING: Failed to update MySQL async job status for job %s: %v", task.JobID, err)
	}

	// 3. 调用现有的 tools.ProcessExamSubmission 函数
	startTime := time.Now()
	defer func() {
		log.Printf("Exam task %s completed in %v", task.JobID, time.Since(startTime))
	}()

	// 捕获 panic
	defer func() {
		if r := recover(); r != nil {
			errMsg := fmt.Sprintf("Panic in ProcessExamSubmission: %v", r)
			log.Printf("PANIC: %s", errMsg)
			// 更新状态为 FAILED
			cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusFailed), errMsg)
			updateAsyncJobStatus(task.JobID, models.JobStatusFailed, errMsg)
		}
	}()

	tools.ProcessExamSubmission(task.ExamID, task.StudentID, task.ImagePaths)

	// 4. 处理完成，更新状态为 SUCCESS
	if err := cache.SetJobStatus(ctx, task.JobID, string(models.JobStatusSuccess), "试卷批改完成"); err != nil {
		log.Printf("WARNING: Failed to update Redis status for job %s: %v", task.JobID, err)
	}
	if err := updateAsyncJobStatus(task.JobID, models.JobStatusSuccess, "试卷批改完成"); err != nil {
		log.Printf("WARNING: Failed to update MySQL async job status for job %s: %v", task.JobID, err)
	}

	log.Printf("Exam task %s processed successfully", task.JobID)
	return nil
}

// updateAsyncJobStatus 更新 MySQL 中的 AsyncJob 记录（内部使用）
func updateAsyncJobStatus(jobID string, status models.AsyncJobStatus, message string) error {
	// 查找现有的 AsyncJob
	var job models.AsyncJob
	result := database.DB.Where("id = ?", jobID).First(&job)
	if result.Error != nil {
		// 如果找不到记录，可能是任务创建时出错了，我们尝试创建一条记录
		// 但这种情况不应该发生，因为任务创建时会先插入记录
		log.Printf("WARNING: AsyncJob %s not found in database, creating new record", jobID)
		job = models.AsyncJob{
			ID:        jobID,
			Status:    status,
			Message:   message,
			UpdatedAt: time.Now(),
		}
		// 由于缺少 JobType 和 ReferenceID 等信息，我们只能尽力保存
		return database.DB.Create(&job).Error
	}

	// 更新状态和消息
	job.Status = status
	job.Message = message
	job.UpdatedAt = time.Now()

	return database.DB.Save(&job).Error
}

// UpdateAsyncJobStatus 更新 MySQL 中的 AsyncJob 记录（供外部使用）
func UpdateAsyncJobStatus(jobID string, status models.AsyncJobStatus, message string) error {
	return updateAsyncJobStatus(jobID, status, message)
}

// StartKafkaConsumer 启动 Kafka 消费者组
func StartKafkaConsumer(topics []string) error {
	consumerGroup := GetConsumerGroup()
	if consumerGroup == nil {
		return fmt.Errorf("Kafka consumer group not initialized")
	}

	ctx := context.Background()
	consumer := GradingConsumer{}

	// 启动消费循环
	go func() {
		for {
			log.Printf("Starting consumer group for topics: %v", topics)
			err := consumerGroup.Consume(ctx, topics, consumer)
			if err != nil {
				log.Printf("ERROR: Consumer group error: %v", err)
				// 等待一段时间后重试
				time.Sleep(5 * time.Second)
				continue
			}

			// 如果上下文被取消，退出循环
			if ctx.Err() != nil {
				log.Printf("Consumer context cancelled: %v", ctx.Err())
				return
			}
		}
	}()

	log.Printf("Kafka consumer started for topics: %v", topics)
	return nil
}

// CreateAsyncJob 创建异步任务记录（供生产者使用）
func CreateAsyncJob(jobID string, jobType models.AsyncJobType, referenceID string, studentID string) error {
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

	if err := database.DB.Create(&job).Error; err != nil {
		return fmt.Errorf("failed to create async job: %w", err)
	}

	log.Printf("AsyncJob created: id=%s, type=%s, reference=%s", jobID, jobType, referenceID)
	return nil
}
