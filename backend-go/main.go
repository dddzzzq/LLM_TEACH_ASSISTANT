package main

import (
	"log"
	"time"

	"grading-gateway/internal/agent"
	"grading-gateway/internal/database"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/handlers"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/mq"

	"github.com/gin-gonic/gin"
)

func main() {
	// 1. 连接数据库
	dsn := "root:123456@tcp(127.0.0.1:3306)/grading_system?charset=utf8mb4&parseTime=True&loc=Local"
	database.InitDB(dsn)

	// 2. 初始化 Redis
	database.InitRedis(nil)
	defer database.CloseRedis()

	// 3. 初始化 Kafka
	kafkaBrokers := []string{"localhost:9092"}
	if err := mq.InitKafka(kafkaBrokers); err != nil {
		log.Printf("ERROR: Failed to initialize Kafka: %v", err)
		log.Println("警告: Kafka 初始化失败，异步任务功能将不可用")
	} else {
		defer mq.Close()
		// 启动 Kafka 消费者
		topics := []string{mq.TopicGradingHomework, mq.TopicGradingExam}
		go func() {
			if err := mq.StartKafkaConsumer(topics); err != nil {
				log.Printf("ERROR: Failed to start Kafka consumer: %v", err)
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

	// 6. 初始化gin框架，定义初始根路由
	router := gin.Default()

	// 7. 配置CORS，使可跨域访问
	router.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	// 8. 创建处理器
	authHandler := handlers.NewAuthHandler(database.DB)
	sessionHandler := handlers.NewSessionHandler(database.DB)

	// 9. 定义公开路由（不需要认证）
	publicGroup := router.Group("/api")
	{
		// 认证相关路由
		publicGroup.POST("/login", authHandler.Login)
		publicGroup.POST("/refresh", authHandler.Refresh)
		publicGroup.POST("/register", authHandler.Register)
	}

	// 10. 定义需要认证的路由组
	protectedGroup := router.Group("/api")
	protectedGroup.Use(middleware.AuthMiddleware())
	{
		// 用户个人信息
		protectedGroup.GET("/profile", authHandler.Profile)

		// 会话管理
		sessionGroup := protectedGroup.Group("/sessions")
		{
			sessionGroup.GET("", sessionHandler.GetUserSessions)
			sessionGroup.POST("", sessionHandler.CreateSession)
			sessionGroup.GET("/:session_id", sessionHandler.GetSessionInfo)
			sessionGroup.GET("/:session_id/history", sessionHandler.GetSessionHistory)
			sessionGroup.PUT("/:session_id", sessionHandler.UpdateSession)
			sessionGroup.DELETE("/:session_id", sessionHandler.DeleteSession)
		}

		// Agent对话路由（需要认证）
		protectedGroup.POST("/agent/chat", handlers.AgentChat)

		// 异步任务状态查询路由
		protectedGroup.GET("/jobs/:job_id", handlers.GetJobStatus)
	}

	// 11. 原有业务路由（保持原有结构）
	// 布置作业路由
	assignmentGroup := router.Group("/assignments")
	assignmentGroup.Use(middleware.AuthMiddleware())
	{
		assignmentGroup.POST("/", middleware.StudentAccessGuard(), handlers.CreateAssignment)                 // 创建作业路由
		assignmentGroup.GET("/", handlers.GetAssignments)                                                     // 获取所有作业路由用于展示
		assignmentGroup.GET("/:id", handlers.GetAssignment)                                                   // 获取单个作业路由
		assignmentGroup.GET("/:id/results", handlers.GetAssignmentSubmissions)                                // 获取单个作业的所有提交
		assignmentGroup.DELETE("/:id", middleware.StudentAccessGuard(), handlers.DeleteAssignment)            // 删除作业路由
		assignmentGroup.POST("/:id/submit", middleware.StudentAccessGuard(), handlers.UploadAssignment)       // 上传学生作业路由
		assignmentGroup.DELETE("/:id/results", middleware.StudentAccessGuard(), handlers.ClearAllSubmissions) // 一键清空路由
		assignmentGroup.GET("/:id/export", handlers.ExportAssignmentExcel)                                    // 导出路由
	}

	// 学生提交路由
	submissionGroup := router.Group("/submissions")
	submissionGroup.Use(middleware.AuthMiddleware())
	{
		submissionGroup.GET("/:id", handlers.GetSubmission)                                        // 获取学生作业情况
		submissionGroup.PUT("/:id", middleware.StudentAccessGuard(), handlers.UpdateSubmission)    // 更新学生分数和评语路由
		submissionGroup.DELETE("/:id", middleware.StudentAccessGuard(), handlers.DeleteSubmission) // 删除单个学生作业路由
	}

	// 试卷主观题批改路由
	examGroup := router.Group("/exams")
	examGroup.Use(middleware.AuthMiddleware())
	{
		examGroup.POST("/", middleware.StudentAccessGuard(), handlers.CreateExam)                            // 步骤1：新建试卷
		examGroup.GET("/", handlers.GetExams)                                                                // 列表
		examGroup.GET("/:id", handlers.GetExam)                                                              // 详情(含题目)
		examGroup.DELETE("/:id", middleware.StudentAccessGuard(), handlers.DeleteExam)                       // 删除整个试卷
		examGroup.POST("/:id/questions", middleware.StudentAccessGuard(), handlers.AddExamQuestion)          // 步骤2：添加考题
		examGroup.POST("/:id/grade_submission", middleware.StudentAccessGuard(), handlers.UploadStudentExam) // 并发交卷核心接口
		examGroup.GET("/:id/results", handlers.GetExamResultsSummary)                                        // 获取某次考试的所有人总分
		examGroup.GET("/:id/results/:student_exam_id", handlers.GetStudentDetailedReport)                    // 获取某个人的详细试卷分析
		examGroup.DELETE("/:id/results/:student_exam_id", middleware.StudentAccessGuard(), handlers.DeleteStudentExamResult)
	}

	// 静态资源服务
	// 供前端渲染图片使用
	router.Static("/uploads", "./uploads")

	// 12. 启动监听在8000端口
	port := ":8000"
	log.Println("服务器启动，监听在端口", port)
	if err := router.Run(port); err != nil {
		log.Fatalf("服务器启动失败：%v", err)
	}
}
