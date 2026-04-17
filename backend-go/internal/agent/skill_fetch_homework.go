package agent

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
	"grading-gateway/internal/mq"

	"github.com/google/uuid"
)

// FetchAndGradeHomeworkSkill 从教务系统下载并批改作业的技能（异步版本）
type FetchAndGradeHomeworkSkill struct{}

// Name 返回技能名称
func (s *FetchAndGradeHomeworkSkill) Name() string {
	return "fetch_and_grade_homework"
}

// Description 返回技能描述
func (s *FetchAndGradeHomeworkSkill) Description() string {
	return "当用户要求从教务系统下载并批改作业时调用此工具。该工具会在后台自动登录教务系统，下载指定课程的作业附件，然后将下载的文件投递到批改流水线进行自动批改。"
}

// Schema 返回 JSON Schema 字符串
func (s *FetchAndGradeHomeworkSkill) Schema() string {
	return `{
		"type": "object",
		"properties": {
			"username": {
				"type": "string",
				"description": "教务系统用户名（学号或工号）"
			},
			"password": {
				"type": "string",
				"description": "教务系统密码"
			},
			"course_name": {
				"type": "string",
				"description": "课程名称（需与教务系统中显示的完全一致）"
			},
			"assignment_name": {
				"type": "string",
				"description": "作业名称（需与教务系统中显示的完全一致）"
			}
		},
		"required": ["username", "password", "course_name", "assignment_name"],
		"additionalProperties": false
	}`
}

// Execute 执行从教务系统下载并批改作业的操作（异步版本）
func (s *FetchAndGradeHomeworkSkill) Execute(args string) (string, error) {
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[Agent Skill] 🚀 开始执行教务系统作业抓取任务")
	log.Printf("[Agent Skill] 📥 接收参数: %s", args)
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 解析参数
	log.Printf("[Agent Skill] 📋 步骤 1/6: 解析请求参数...")
	var params map[string]interface{}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		log.Printf("[Agent Skill] ❌ 参数解析失败: %v", err)
		return "参数解析失败，请确保提供了正确的 JSON 格式，包含 username、password、course_name 和 assignment_name 字段。示例：{\"username\": \"学号\", \"password\": \"密码\", \"course_name\": \"课程名\", \"assignment_name\": \"作业名\"}", nil
	}

	// 获取参数
	username, ok1 := params["username"].(string)
	password, ok2 := params["password"].(string)
	courseName, ok3 := params["course_name"].(string)
	assignmentName, ok4 := params["assignment_name"].(string)

	if !ok1 || username == "" {
		log.Printf("[Agent Skill] ❌ 参数 username 缺失或格式不正确")
		return "参数 username 缺失或格式不正确，请提供教务系统用户名。", nil
	}
	if !ok2 || password == "" {
		log.Printf("[Agent Skill] ❌ 参数 password 缺失或格式不正确")
		return "参数 password 缺失或格式不正确，请提供教务系统密码。", nil
	}
	if !ok3 || courseName == "" {
		log.Printf("[Agent Skill] ❌ 参数 course_name 缺失或格式不正确")
		return "参数 course_name 缺失或格式不正确，请提供课程名称。", nil
	}
	if !ok4 || assignmentName == "" {
		log.Printf("[Agent Skill] ❌ 参数 assignment_name 缺失或格式不正确")
		return "参数 assignment_name 缺失或格式不正确，请提供作业名称。", nil
	}

	log.Printf("[Agent Skill] ✅ 参数校验通过")
	log.Printf("[Agent Skill]    - 用户名: %s", username)
	log.Printf("[Agent Skill]    - 课程名称: %s", courseName)
	log.Printf("[Agent Skill]    - 作业名称: %s", assignmentName)

	// 1. 生成 UUID 作为 JobID
	log.Printf("[Agent Skill] 📋 步骤 2/6: 生成任务ID...")
	jobID := uuid.New().String()
	log.Printf("[Agent Skill] ✅ 已生成 JobID: %s", jobID)

	// 2. 往 AsyncJob 表插入一条记录
	log.Printf("[Agent Skill] 📋 步骤 3/6: 创建异步任务记录到数据库...")
	asyncJob := models.AsyncJob{
		ID:          jobID,
		JobType:     "rpa_fetch_homework",
		ReferenceID: courseName + ":" + assignmentName,
		StudentID:   username,
		Status:      models.JobStatusPending,
		Message:     "任务已创建，等待RPA抓取作业",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	if err := database.DB.Create(&asyncJob).Error; err != nil {
		log.Printf("[Agent Skill] ❌ 创建AsyncJob记录失败: %v", err)
		return "创建异步任务失败，请稍后重试。", nil
	}

	log.Printf("[Agent Skill] ✅ AsyncJob记录已创建到MySQL数据库")
	log.Printf("[Agent Skill]    - JobID: %s", jobID)
	log.Printf("[Agent Skill]    - JobType: %s", asyncJob.JobType)
	log.Printf("[Agent Skill]    - Status: %s", asyncJob.Status)

	// 3. 构造 RPAFetchMessage
	log.Printf("[Agent Skill] 📋 步骤 4/6: 构造Kafka消息...")
	rpaMessage := mq.RPAFetchMessage{
		JobID:          jobID,
		Username:       username,
		Password:       password,
		CourseName:     courseName,
		AssignmentName: assignmentName,
	}
	log.Printf("[Agent Skill] ✅ Kafka消息构造完成")
	log.Printf("[Agent Skill]    - Topic: topic_rpa_fetch")
	log.Printf("[Agent Skill]    - JobID: %s", rpaMessage.JobID)
	log.Printf("[Agent Skill]    - CourseName: %s", rpaMessage.CourseName)
	log.Printf("[Agent Skill]    - AssignmentName: %s", rpaMessage.AssignmentName)

	// 4. 投递到 Kafka 的 topic_rpa_fetch 队列
	log.Printf("[Agent Skill] 📋 步骤 5/6: 投递任务到Kafka消息队列...")
	if err := mq.PublishRPAFetchTask(rpaMessage); err != nil {
		log.Printf("[Agent Skill] ❌ 投递RPA抓取任务到Kafka失败: %v", err)

		// 更新状态为FAILED
		database.DB.Model(&asyncJob).Updates(models.AsyncJob{
			Status:    models.JobStatusFailed,
			Message:   fmt.Sprintf("投递任务到Kafka失败: %v", err),
			UpdatedAt: time.Now(),
		})
		log.Printf("[Agent Skill] ✅ 已更新数据库任务状态为 FAILED")

		return "投递任务到后台队列失败，请联系管理员检查Kafka服务状态。", nil
	}

	log.Printf("[Agent Skill] ✅ RPA抓取任务已成功投递到Kafka消息队列")
	log.Printf("[Agent Skill]    - Topic: topic_rpa_fetch")
	log.Printf("[Agent Skill]    - JobID: %s", jobID)

	// 5. 立即返回一段友好的提示语给LLM
	log.Printf("[Agent Skill] 📋 步骤 6/6: 生成响应消息...")
	resultMsg := fmt.Sprintf(`📥 教务系统作业下载任务已创建成功！

📋 任务ID: %s

📊 任务信息：
- 用户名: %s
- 课程名称: %s
- 作业名称: %s

⏳ 任务状态: 正在后台排队处理中

系统正在自动下载教务系统中的作业附件，下载完成后会自动开始批改。请稍后查看批改结果。`,
		jobID, username, courseName, assignmentName)

	log.Printf("[Agent Skill] ✅ 响应消息生成完成")
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[Agent Skill] 🎉 教务系统作业抓取任务创建流程完成")
	log.Printf("[Agent Skill] 📤 任务已投递到后台队列，等待RPA消费者处理")
	log.Printf("[Agent Skill] 🆔 任务ID: %s", jobID)
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	return resultMsg, nil
}
