package mq

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/IBM/sarama"
)

// Configure the protocol explicitly when deploying a newer broker. Sarama's
// default is Kafka 2.1; Kafka 4 removes several older request versions.
func newKafkaConfig() (*sarama.Config, error) {
	config := sarama.NewConfig()
	if configured := os.Getenv("KAFKA_VERSION"); configured != "" {
		version, err := sarama.ParseKafkaVersion(configured)
		if err != nil {
			return nil, fmt.Errorf("invalid KAFKA_VERSION: %w", err)
		}
		config.Version = version
	}
	return config, nil
}

// BrokerAddresses is shared by the producer and both consumer groups.
func BrokerAddresses() []string {
	configured := os.Getenv("KAFKA_BROKERS")
	if configured == "" {
		return []string{"localhost:9092"}
	}
	var brokers []string
	for _, value := range strings.Split(configured, ",") {
		if address := strings.TrimSpace(value); address != "" {
			brokers = append(brokers, address)
		}
	}
	return brokers
}

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
		producerConfig, err := newKafkaConfig()
		if err != nil {
			initErr = err
			return
		}
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
		consumerConfig, err := newKafkaConfig()
		if err != nil {
			initErr = err
			return
		}
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
