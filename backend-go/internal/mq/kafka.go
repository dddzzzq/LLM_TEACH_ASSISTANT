package mq

import (
	"log"
	"sync"

	"github.com/IBM/sarama"
)

var (
	producer      sarama.SyncProducer
	consumerGroup sarama.ConsumerGroup
	once          sync.Once
	mu            sync.RWMutex
	initialized   bool
)

// InitKafka 初始化全局的 SyncProducer 和 ConsumerGroup
func InitKafka(brokers []string) error {
	var initErr error
	once.Do(func() {
		// 配置生产者
		producerConfig := sarama.NewConfig()
		producerConfig.Producer.Return.Successes = true
		producerConfig.Producer.Return.Errors = true
		producerConfig.Producer.RequiredAcks = sarama.WaitForAll
		producerConfig.Producer.Retry.Max = 5

		syncProducer, err := sarama.NewSyncProducer(brokers, producerConfig)
		if err != nil {
			initErr = err
			log.Printf("Failed to create Kafka sync producer: %v", err)
			return
		}
		producer = syncProducer

		// 配置消费者组
		consumerConfig := sarama.NewConfig()
		consumerConfig.Consumer.Return.Errors = true
		consumerConfig.Consumer.Offsets.Initial = sarama.OffsetOldest

		group, err := sarama.NewConsumerGroup(brokers, "grading-consumer-group", consumerConfig)
		if err != nil {
			initErr = err
			log.Printf("Failed to create Kafka consumer group: %v", err)
			// 注意：即使消费者组创建失败，我们仍然保留生产者
			// 可以继续运行，但消费者功能不可用
		} else {
			consumerGroup = group
		}

		mu.Lock()
		initialized = true
		mu.Unlock()

		log.Printf("Kafka initialized with brokers: %v", brokers)
	})

	return initErr
}

// GetProducer 获取 Kafka 生产者实例
func GetProducer() sarama.SyncProducer {
	mu.RLock()
	defer mu.RUnlock()
	if !initialized {
		log.Panic("Kafka not initialized. Call InitKafka first.")
	}
	return producer
}

// GetConsumerGroup 获取 Kafka 消费者组实例
func GetConsumerGroup() sarama.ConsumerGroup {
	mu.RLock()
	defer mu.RUnlock()
	if !initialized {
		log.Panic("Kafka not initialized. Call InitKafka first.")
	}
	return consumerGroup
}

// Close 关闭 Kafka 连接
func Close() {
	mu.Lock()
	defer mu.Unlock()

	if producer != nil {
		if err := producer.Close(); err != nil {
			log.Printf("Error closing Kafka producer: %v", err)
		}
		producer = nil
	}

	if consumerGroup != nil {
		if err := consumerGroup.Close(); err != nil {
			log.Printf("Error closing Kafka consumer group: %v", err)
		}
		consumerGroup = nil
	}

	initialized = false
	log.Println("Kafka connections closed")
}
