package mq

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/IBM/sarama"
	"grading-gateway/internal/rpa"
)

type RPAConsumer struct{}

func (RPAConsumer) Setup(sarama.ConsumerGroupSession) error   { return nil }
func (RPAConsumer) Cleanup(sarama.ConsumerGroupSession) error { return nil }
func (RPAConsumer) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
	for message := range claim.Messages() {
		var task RPAFetchMessage
		if err := json.Unmarshal(message.Value, &task); err != nil {
			session.MarkMessage(message, "invalid task")
			continue
		}
		// MySQL QUEUED tasks form the durable retry outbox. Start is idempotent in the Worker.
		if err := rpa.Start(session.Context(), task.JobID, false); err != nil {
			log.Printf("RPA %s awaits scheduler retry", task.JobID)
		}
		session.MarkMessage(message, "")
		session.Commit()
	}
	return nil
}

func StartRPAConsumer() error {
	config := sarama.NewConfig()
	config.Consumer.Offsets.Initial = sarama.OffsetOldest
	group, err := sarama.NewConsumerGroup([]string{"localhost:9092"}, "rpa-consumer-group", config)
	if err != nil {
		return err
	}
	defer group.Close()
	for {
		if err := group.Consume(context.Background(), []string{TopicRPAFetch}, RPAConsumer{}); err != nil {
			log.Printf("RPA consumer retry: %v", err)
			time.Sleep(3 * time.Second)
		}
	}
}
