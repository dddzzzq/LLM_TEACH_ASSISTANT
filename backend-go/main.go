package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"time"

	"grading-gateway/internal/agent"
	"grading-gateway/internal/agent/catalog"
	"grading-gateway/internal/agent/modelclient"
	"grading-gateway/internal/browseragent"
	"grading-gateway/internal/database"
	"grading-gateway/internal/grading"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/logbook"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/mq"
	"grading-gateway/internal/portal"
	"grading-gateway/internal/rpa"
	"grading-gateway/internal/server"
)

func main() {
	// 1. 连接数据库
	dsn := "root:123456@tcp(127.0.0.1:3306)/grading_system?charset=utf8mb4&parseTime=True&loc=Local"
	if configured := os.Getenv("DATABASE_DSN"); configured != "" {
		dsn = configured
	}
	database.InitDB(dsn)
	logRoot := os.Getenv("RUNTIME_DIR")
	if logRoot == "" {
		logRoot = filepath.Join("..", "..", "teaching-runtime")
	}
	logService := logbook.New(database.DB, logRoot)
	if err := logService.Init(); err != nil {
		log.Printf("日志索引初始化失败（%T）", err)
	} else {
		go logService.Run(context.Background())
	}

	// 2. 初始化 Redis
	database.InitRedis(nil)
	defer database.CloseRedis()

	// 2.1 初始化工具配置；Markdown Skill 知识包不写入工具表。
	agent.EnsureDefaultToolsSeeded(context.Background())
	agent.EnsureRPATools(context.Background())
	agent.EnsureGradingTools(context.Background())
	agent.EnsurePortalTools(context.Background())
	agent.EnsureTeacherEditionTools(context.Background())
	// Install the credential/session hook before Kafka consumers can restore queued jobs.
	portal.Start(context.Background(), database.DB)
	rpa.PublishFetch = func(id string) error { return mq.PublishRPAFetchTask(mq.RPAFetchMessage{JobID: id}) }

	// 3. 初始化 Kafka
	kafkaBrokers := mq.BrokerAddresses()
	if err := mq.InitKafka(kafkaBrokers); err != nil {
		log.Printf("ERROR: Failed to initialize Kafka: %v", err)
		log.Println("警告: Kafka 初始化失败，异步任务功能将不可用")
	} else {
		defer mq.Close()
		// 启动 Kafka 消费者 - 批改任务
		topics := []string{mq.TopicGradingHomework}
		go func() {
			if err := mq.StartKafkaConsumer(topics); err != nil {
				log.Printf("ERROR: Failed to start Kafka consumer: %v", err)
			}
		}()
		// 启动 RPA 消费者 - 教务系统作业抓取任务
		go func() {
			if err := mq.StartRPAConsumer(); err != nil {
				log.Printf("ERROR: Failed to start RPA consumer: %v", err)
			}
		}()
	}

	// 4. 初始化 Redis 记忆管理器
	if database.DB != nil {
		agent.InitRedisMemoryManager(20, 24*60*time.Minute, database.DB)
	} else {
		log.Println("警告: 数据库连接为空，无法初始化 Redis 记忆管理器")
	}

	// 5. 初始化grpc客户端
	grpcclient.InitGrpcClient()
	defer grpcclient.CloseGrpcClient()
	rpa.DecidePage = func(ctx context.Context, input json.RawMessage, doc catalog.Document, contract json.RawMessage) (json.RawMessage, error) {
		return browseragent.Decide(ctx, input, doc, contract, modelclient.NewPageClient().CallMessages)
	}
	go rpa.RunScheduler(context.Background())
	go materials.Default().Run(context.Background(), grading.ContinuePrepared)
	go grading.RunMaterialOutbox(context.Background())

	// 6. 注册 HTTP 路由并启动服务。
	router := server.NewRouter(database.DB, logService)

	// 默认监听 8000 端口，允许通过 BACKEND_ADDR 覆盖。
	port := backendAddress()
	log.Println("服务器启动，监听在端口", port)
	if err := router.Run(port); err != nil {
		log.Fatalf("服务器启动失败：%v", err)
	}
}

func backendAddress() string {
	if address := os.Getenv("BACKEND_ADDR"); address != "" {
		return address
	}
	return ":8000"
}
