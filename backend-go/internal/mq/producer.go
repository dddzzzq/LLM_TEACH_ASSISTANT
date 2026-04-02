package mq

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/IBM/sarama"
)

const (
	TopicGradingHomework = "topic_grading_homework"
	TopicGradingExam     = "topic_grading_exam"
)

// HomeworkTaskMessage 作业批改任务消息结构
type HomeworkTaskMessage struct {
	JobID        string `json:"job_id"`
	AssignmentID string `json:"assignment_id"`
	ZipPath      string `json:"zip_path"`
}

// ExamTaskMessage 试卷批改任务消息结构
type ExamTaskMessage struct {
	JobID      string   `json:"job_id"`
	ExamID     string   `json:"exam_id"`
	StudentID  string   `json:"student_id"`
	ImagePaths []string `json:"image_paths"`
}

// publishMessage 通用消息发布函数
func publishMessage(topic string, message interface{}) error {
	producer := GetProducer()

	messageJSON, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	msg := &sarama.ProducerMessage{
		Topic: topic,
		Value: sarama.ByteEncoder(messageJSON),
	}

	partition, offset, err := producer.SendMessage(msg)
	if err != nil {
		return fmt.Errorf("failed to send message to Kafka topic %s: %w", topic, err)
	}

	log.Printf("Message sent to topic %s partition %d offset %d: %s", topic, partition, offset, string(messageJSON))
	return nil
}

// PublishHomeworkTask 将作业批改任务推送到 Topic: topic_grading_homework
func PublishHomeworkTask(jobID, assignmentID, zipPath string) error {
	message := HomeworkTaskMessage{
		JobID:        jobID,
		AssignmentID: assignmentID,
		ZipPath:      zipPath,
	}

	if err := publishMessage(TopicGradingHomework, message); err != nil {
		log.Printf("ERROR: Failed to publish homework task for job %s: %v", jobID, err)
		return err
	}

	log.Printf("Homework task published successfully: job=%s, assignment=%s", jobID, assignmentID)
	return nil
}

// PublishExamTask 将试卷批改任务推送到 Topic: topic_grading_exam
func PublishExamTask(jobID, examID, studentID string, imagePaths []string) error {
	message := ExamTaskMessage{
		JobID:      jobID,
		ExamID:     examID,
		StudentID:  studentID,
		ImagePaths: imagePaths,
	}

	if err := publishMessage(TopicGradingExam, message); err != nil {
		log.Printf("ERROR: Failed to publish exam task for job %s: %v", jobID, err)
		return err
	}

	log.Printf("Exam task published successfully: job=%s, exam=%s, student=%s", jobID, examID, studentID)
	return nil
}

// IsKafkaAvailable 检查 Kafka 是否可用
func IsKafkaAvailable() bool {
	producer := GetProducer()
	_, _, err := producer.SendMessage(&sarama.ProducerMessage{
		Topic: "health-check",
		Value: sarama.ByteEncoder([]byte("ping")),
	})
	// 我们不关心消息是否发送成功，只关心生产者是否能够正常工作
	return err == nil
}
