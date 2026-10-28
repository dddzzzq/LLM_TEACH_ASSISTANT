// Package server wires HTTP routes and middleware without starting background services.
package server

import (
	"strings"

	"grading-gateway/internal/handlers"
	"grading-gateway/internal/logbook"
	"grading-gateway/internal/middleware"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

// NewRouter constructs the HTTP API. Dependencies are initialized by the caller.
func NewRouter(db *gorm.DB, logService *logbook.Service) *gin.Engine {
	router := gin.New()
	router.Use(gin.LoggerWithConfig(gin.LoggerConfig{
		SkipQueryString: true,
		Skip: func(c *gin.Context) bool {
			return strings.HasPrefix(c.Request.URL.Path, "/api/logs/")
		},
	}), gin.Recovery())

	// 配置CORS，使可跨域访问
	router.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With, Idempotency-Key")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	// 创建处理器
	authHandler := handlers.NewAuthHandler(db)
	sessionHandler := handlers.NewSessionHandler(db)

	// 定义公开路由（不需要认证）
	publicGroup := router.Group("/api")
	{
		// 认证相关路由
		publicGroup.POST("/login", authHandler.Login)
		publicGroup.POST("/refresh", authHandler.Refresh)
		publicGroup.POST("/register", authHandler.Register)
		// The WebSocket authenticates its first message before accessing a task.
		publicGroup.GET("/rpa/jobs/:id/stream", handlers.RPABrowserStream)
	}

	// 定义需要认证的路由组
	protectedGroup := router.Group("/api")
	protectedGroup.Use(middleware.AuthMiddleware())
	{
		// 用户个人信息
		protectedGroup.GET("/profile", authHandler.Profile)
		portalHandler := handlers.NewPortalHandler(db)
		portalHandler.Register(protectedGroup.Group("/portal", middleware.RBACMiddleware("teacher", "admin")))
		workflows := protectedGroup.Group("/grading/workflows", middleware.RBACMiddleware("teacher", "admin"))
		workflows.Use(func(c *gin.Context) { c.Header("Cache-Control", "no-store"); c.Next() })
		workflows.POST("", portalHandler.CreateWorkflow)
		workflows.GET("/:id", portalHandler.Workflow)
		workflows.POST("/:id/cancel", portalHandler.CancelWorkflow)
		workflows.POST("/:id/retry", portalHandler.RetryWorkflow)
		logGroup := protectedGroup.Group("/logs") // LogHandler centrally enforces admin-only access.
		handlers.LogHandler{Service: logService}.Register(logGroup)
		materialGroup := protectedGroup.Group("/materials", middleware.RBACMiddleware("teacher", "admin"))
		materialGroup.GET("", handlers.ListMaterials)
		materialGroup.POST("", handlers.CreateMaterials)
		materialGroup.GET("/:id", handlers.GetMaterials)
		materialGroup.POST("/:id/retry", handlers.RetryMaterials)
		materialGroup.POST("/:id/grade", handlers.GradeMaterials)
		materialGroup.POST("/:id/cancel", handlers.CancelMaterials)

		// 会话管理
		sessionGroup := protectedGroup.Group("/sessions")
		{
			sessionGroup.GET("", sessionHandler.GetUserSessions)
			sessionGroup.POST("", sessionHandler.CreateSession)
			sessionGroup.GET("/:session_id", sessionHandler.GetSessionInfo)
			sessionGroup.GET("/:session_id/history", sessionHandler.GetSessionHistory)
			sessionGroup.GET("/:session_id/runs", sessionHandler.GetExecutions)
			sessionGroup.GET("/:session_id/runs/:run_id", sessionHandler.GetExecutionEvents)
			sessionGroup.GET("/:session_id/runs/:run_id/context", sessionHandler.GetExecutionContext)
			sessionGroup.PUT("/:session_id", sessionHandler.UpdateSession)
			sessionGroup.DELETE("/:session_id", sessionHandler.DeleteSession)
		}

		// Agent对话路由（需要认证）
		protectedGroup.POST("/agent/chat", handlers.AgentChat)

		rpaGroup := protectedGroup.Group("/rpa/jobs")
		rpaGroup.Use(middleware.RBACMiddleware("teacher", "admin"))
		{
			rpaGroup.POST("", handlers.CreateRPAJob)
			rpaGroup.GET("", handlers.ListRPAJobs)
			rpaGroup.GET("/attention", handlers.GetRPAAttention)
			rpaGroup.GET("/:id", handlers.GetRPAJob)
			rpaGroup.GET("/:id/requirements", handlers.GetRPARequirements)
			rpaGroup.GET("/:id/browser", handlers.RPABrowserFrame)
			rpaGroup.POST("/:id/:action", handlers.ControlRPAJob)
		}

		// 异步任务状态查询路由
		protectedGroup.GET("/jobs/:job_id", handlers.GetJobStatus)

		// 工具配置（教师/管理员），保留旧 /skills 接口。
		adminGroup := protectedGroup.Group("/admin")
		adminGroup.Use(middleware.RBACMiddleware("teacher", "admin"))
		{
			handlers.RegisterToolAdminRoutes(adminGroup)
		}
	}

	// 原有业务路由
	// 布置作业路由
	assignmentGroup := router.Group("/assignments")
	assignmentGroup.Use(middleware.AuthMiddleware())
	assignmentGroup.Use(middleware.AssignmentAccess(db, false))
	{
		assignmentGroup.POST("/", middleware.StudentAccessGuard(), handlers.CreateAssignment) // 创建作业路由
		assignmentGroup.GET("/", handlers.GetAssignments)                                     // 获取所有作业路由用于展示
		assignmentGroup.GET("/:id", handlers.GetAssignment)                                   // 获取单个作业路由
		assignmentGroup.GET("/:id/classes", handlers.GetAssignmentClasses)
		assignmentGroup.GET("/:id/results", handlers.GetAssignmentSubmissions)                                // 获取单个作业的所有提交
		assignmentGroup.DELETE("/:id", middleware.StudentAccessGuard(), handlers.DeleteAssignment)            // 删除作业路由
		assignmentGroup.POST("/:id/submit", middleware.StudentAccessGuard(), handlers.UploadAssignment)       // 上传学生作业路由
		assignmentGroup.DELETE("/:id/results", middleware.StudentAccessGuard(), handlers.ClearAllSubmissions) // 一键清空路由
		assignmentGroup.GET("/:id/export", handlers.ExportAssignmentExcel)                                    // 导出路由
	}

	// 学生提交路由
	submissionGroup := router.Group("/submissions")
	submissionGroup.Use(middleware.AuthMiddleware())
	submissionGroup.Use(middleware.AssignmentAccess(db, true))
	{
		submissionGroup.GET("/:id", handlers.GetSubmission)                                        // 获取学生作业情况
		submissionGroup.PUT("/:id", middleware.StudentAccessGuard(), handlers.UpdateSubmission)    // 更新学生分数和评语路由
		submissionGroup.DELETE("/:id", middleware.StudentAccessGuard(), handlers.DeleteSubmission) // 删除单个学生作业路由
	}

	// 静态资源服务
	// 供前端渲染图片使用
	router.Static("/uploads", "./uploads")

	return router
}
