package mq

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/IBM/sarama"
)

const (
	TopicRPAFetch        = "topic_rpa_fetch"
	TopicGradingHomework = "topic_grading_homework"
)

// publishMessage 通用消息发布函数
func publishMessage(topic string, message interface{}) error {
	mu.RLock()
	currentProducer := producer
	mu.RUnlock()
	if currentProducer == nil {
		return fmt.Errorf("Kafka producer unavailable")
	}

	messageJSON, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	msg := &sarama.ProducerMessage{
		Topic: topic,
		Value: sarama.ByteEncoder(messageJSON),
	}

	partition, offset, err := currentProducer.SendMessage(msg)
	if err != nil {
		return fmt.Errorf("failed to send message to Kafka topic %s: %w", topic, err)
	}

	log.Printf("Message sent to topic %s partition %d offset %d (Payload Size: %d bytes)", topic, partition, offset, len(messageJSON))
	return nil
}

// PublishHomeworkTask 将作业批改任务推送到 Topic: topic_grading_homework
func PublishHomeworkTask(jobID string, assignmentID uint, zipPath string) error {
	message := HomeworkTaskMessage{
		JobID:        jobID,
		AssignmentID: assignmentID,
		ZipPath:      zipPath,
	}

	if err := publishMessage(TopicGradingHomework, message); err != nil {
		log.Printf("ERROR: Failed to publish homework task for job %s: %v", jobID, err)
		return err
	}

	log.Printf("Homework task published successfully: job=%s, assignment=%d", jobID, assignmentID)
	return nil
}

// PublishRPAFetchTask 将RPA抓取任务推送到 Topic: topic_rpa_fetch
func PublishRPAFetchTask(message RPAFetchMessage) error {
	if err := publishMessage(TopicRPAFetch, message); err != nil {
		log.Printf("ERROR: Failed to publish RPA fetch task for job %s: %v", message.JobID, err)
		return err
	}

	log.Printf("RPA fetch task published successfully: job=%s", message.JobID)
	return nil
}

func IsKafkaAvailable() bool {
	producer := GetProducer()
	_, _, err := producer.SendMessage(&sarama.ProducerMessage{
		Topic: "health-check",
		Value: sarama.ByteEncoder([]byte("ping")),
	})
	// 我们不关心消息是否发送成功，只关心生产者是否能够正常工作
	return err == nil
}
