package main

import (
	"log"

	"grading-gateway/internal/database"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/handlers"

	"github.com/gin-gonic/gin"
)

func main() {
	// 1. 连接数据库
	dsn := "root:123456@tcp(127.0.0.1:3306)/grading_system?charset=utf8mb4&parseTime=True&loc=Local"
	database.InitDB(dsn)

	// 2. 初始化grpc客户端
	grpcclient.InitGrpcClient()
	defer grpcclient.CloseGrpcClient()

	// 3. 初始化gin框架，定义初始根路由
	router := gin.Default()

	// 4. 配置CORS，使可跨域访问
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

	// 5. 定义路由
	// 布置作业路由
	assignmentGroup := router.Group("/assignments")
	{
		assignmentGroup.POST("/", handlers.CreateAssignment)                   // 创建作业路由
		assignmentGroup.GET("/", handlers.GetAssignments)                      // 获取所有作业路由用于展示
		assignmentGroup.GET("/:id", handlers.GetAssignment)                    // 获取单个作业路由
		assignmentGroup.GET("/:id/results", handlers.GetAssignmentSubmissions) // 获取单个作业的所有提交
		assignmentGroup.DELETE("/:id", handlers.DeleteAssignment)              // 删除作业路由
		assignmentGroup.POST("/:id/submit", handlers.UploadAssignment)         // 上传学生作业路由
		assignmentGroup.DELETE("/:id/results", handlers.ClearAllSubmissions)   // 一键清空路由
		assignmentGroup.GET("/:id/export", handlers.ExportAssignmentExcel)     // 导出路由
	}

	// 学生提交路由
	submissionGroup := router.Group("/submissions")
	{
		submissionGroup.GET("/:id", handlers.GetSubmission)       // 获取学生作业情况
		submissionGroup.PUT("/:id", handlers.UpdateSubmission)    // 更新学生分数和评语路由
		submissionGroup.DELETE("/:id", handlers.DeleteSubmission) // 删除单个学生作业路由
	}

	// 2. 试卷主观题批改路由
	examGroup := router.Group("/exams")
	{
		examGroup.POST("/", handlers.CreateExam)                                          // 步骤1：新建试卷
		examGroup.GET("/", handlers.GetExams)                                             // 列表
		examGroup.GET("/:id", handlers.GetExam)                                           // 详情(含题目)
		examGroup.DELETE("/:id", handlers.DeleteExam)                                     // 删除整个试卷
		examGroup.POST("/:id/questions", handlers.AddExamQuestion)                        // 步骤2：添加考题
		examGroup.POST("/:id/grade_submission", handlers.UploadStudentExam)               // 并发交卷核心接口
		examGroup.GET("/:id/results", handlers.GetExamResultsSummary)                     // 获取某次考试的所有人总分
		examGroup.GET("/:id/results/:student_exam_id", handlers.GetStudentDetailedReport) // 获取某个人的详细试卷分析
		examGroup.DELETE("/:id/results/:student_exam_id", handlers.DeleteStudentExamResult)
	}

	// 静态资源服务
	// 供前端渲染图片使用
	router.Static("/uploads", "./uploads")

	// 6. 启动监听在8000端口
	port := ":8000"
	log.Println("服务器启动，监听在端口", port)
	if err := router.Run(port); err != nil {
		log.Fatalf("服务器启动失败：%v", err)
	}
}
