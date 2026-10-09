package handlers

import (
	"fmt"
	"log"
	"strings"

	"grading-gateway/internal/agent"
	"grading-gateway/internal/middleware"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// AgentChatRequest 前端发送的对话请求
type AgentChatRequest struct {
	Message   string `json:"message" binding:"required"`
	SessionID string `json:"session_id,omitempty"` // 会话ID，用于记忆管理
}

// AgentChatResponse 返回给前端的响应
type AgentChatResponse struct {
	Reply        string            `json:"reply"`
	Action       string            `json:"action"`
	SessionID    string            `json:"session_id,omitempty"` // 返回会话ID，前端需要保存
	JobID        string            `json:"job_id,omitempty"`
	JobType      string            `json:"job_type,omitempty"`
	ResultURL    string            `json:"result_url,omitempty"`
	LoadedSkills map[string]string `json:"loaded_skills,omitempty"`
}

// AgentChat 处理前端对话请求，由新的 Go Agent 引擎接管
func AgentChat(c *gin.Context) {
	// 获取当前用户ID（需要 AuthMiddleware）
	userID, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		log.Printf("AgentChat: 未授权访问: %v", err)
		c.JSON(401, AgentChatResponse{
			Reply:  "未授权访问，请先登录",
			Action: "none",
		})
		return
	}

	// 获取当前用户角色和用户名
	role, roleErr := middleware.GetRoleFromContext(c)
	username, usernameErr := middleware.GetUsernameFromContext(c)

	var req AgentChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("AgentChat: 无效的请求格式: %v", err)
		c.JSON(400, AgentChatResponse{
			Reply:  "请求格式错误，请提供有效的 message 字段。",
			Action: "none",
		})
		return
	}

	// 生成或使用提供的会话ID
	sessionID := req.SessionID
	if sessionID == "" {
		// 生成新的会话ID（使用UUID）
		sessionID = uuid.New().String()

		// 在 MySQL 中创建新会话
		memoryManager := agent.GetGlobalRedisMemoryManager()
		if memoryManager != nil {
			// 尝试创建新会话
			newSessionID, err := memoryManager.CreateNewSession(userID, "新会话")
			if err != nil {
				log.Printf("AgentChat: 创建新会话失败: %v", err)
			} else {
				sessionID = newSessionID
			}
		}
	}

	// 获取 Redis 记忆管理器
	memoryManager := agent.GetGlobalRedisMemoryManager()
	if memoryManager == nil {
		log.Printf("AgentChat: Redis 记忆管理器未初始化")
		c.JSON(500, AgentChatResponse{
			Reply:     "系统内部错误，请稍后重试。",
			Action:    "none",
			SessionID: sessionID,
		})
		return
	}

	// 添加用户消息到记忆（Redis + MySQL 异步持久化）
	err = memoryManager.AddMessage(userID, sessionID, "user", req.Message)
	if err != nil {
		log.Printf("AgentChat: 添加用户消息失败: %v", err)
		// 继续处理，不影响主要功能
	}

	// 与工具管理共用执行器目录；用户身份由服务端注入。
	registry := agent.NewBuiltinToolRegistry(userID, role)

	// 根据用户角色动态生成系统提示
	systemPrompt := generateSystemPromptWithFetchTool(role, username, roleErr, usernameErr)

	// 获取对话历史（从 Redis）
	history, err := memoryManager.GetFormattedHistory(userID, sessionID)
	if err != nil {
		log.Printf("AgentChat: 获取对话历史失败: %v", err)
		history = ""
	}

	// 构建包含历史对话的用户消息
	userMessageWithHistory := req.Message
	if history != "" {
		userMessageWithHistory = "以下是我们的对话历史：\n" + history + "\n当前问题：" + req.Message
	}

	// 导出所有工具
	tools, toolsErr := agent.BuildToolsForRole(c.Request.Context(), role, registry)
	if toolsErr != nil {
		log.Printf("AgentChat: 加载工具定义失败: %v", toolsErr)
		tools = nil // Fail closed: unavailable configuration cannot expand permissions.
	}

	catalog := agent.SkillCatalog{}
	if role == "teacher" || role == "admin" {
		loaded, loadErr := agent.DefaultGradingSkills()
		if loadErr != nil {
			log.Printf("批改 Skill 加载失败: %v", loadErr)
		} else {
			catalog = loaded
		}
	}
	result, err := agent.RunDialogue(agent.WithRequestKey(c.Request.Context(), c.GetHeader("Idempotency-Key")), systemPrompt, userMessageWithHistory, role, username, registry, tools, catalog, agent.NewDeepSeekClient("").CallMessages)
	if err != nil {
		log.Printf("AgentChat: 执行失败: %v", err)
		c.JSON(502, AgentChatResponse{Reply: "AI 服务暂时不可用，请稍后重试。", SessionID: sessionID, Action: "none"})
		return
	}
	response := formatFinalResponse(result.Reply)
	if err := memoryManager.AddMessage(userID, sessionID, "assistant", response); err != nil {
		log.Printf("保存助手消息失败: %v", err)
	}
	c.JSON(200, AgentChatResponse{Reply: response, Action: determineAction(response), SessionID: sessionID, JobID: result.JobID, JobType: result.JobType, ResultURL: result.ResultURL, LoadedSkills: result.LoadedSkills})

}

// formatFinalResponse 格式化最终回复，确保返回合适的格式
func formatFinalResponse(content string) string {
	if content == "" {
		return "抱歉，我无法处理您的请求。请检查您的输入或稍后重试。"
	}

	// 清理可能的多余空白
	content = strings.TrimSpace(content)

	// 如果内容过长，进行适当截断
	if len(content) > 2000 {
		content = content[:2000] + "..."
	}

	return content
}

// generateUUID 生成UUID作为会话ID
func generateUUID() string {
	// 使用简单的UUID生成（实际项目中应该使用github.com/google/uuid）
	// 这里使用时间戳+随机数模拟UUID
	id := uuid.New()
	return id.String()
}

// determineAction 根据回复内容判断 action 类型
func determineAction(response string) string {
	// 简单规则：如果回复中包含特定关键词，设置相应的 action
	// 这里是一个简化的实现，可以根据实际需求扩展
	lowerResponse := strings.ToLower(response)

	if strings.Contains(lowerResponse, "批改") && strings.Contains(lowerResponse, "触发") {
		return "pipeline_triggered"
	}

	if strings.Contains(lowerResponse, "成绩") || strings.Contains(lowerResponse, "分数") {
		return "score_queried"
	}

	// 默认返回 "none"
	return "none"
}

// generateSystemPromptWithFetchTool 根据用户角色生成不同的系统提示（包含教务系统抓取功能）
func generateSystemPromptWithFetchTool(role string, username string, roleErr error, usernameErr error) string {
	if roleErr != nil || usernameErr != nil {
		return "你是教学助手。用户身份未确认，不要调用业务工具。"
	}
	if role == "student" {
		return fmt.Sprintf("你是教学助手，当前学生学号为 %s。只能查询该学生自己的成绩，使用 query_student_score；不能发起批改或官网下载。", username)
	}
	return fmt.Sprintf(`你是教学助手，当前用户为教师或管理员 %s。
使用 query_student_score 查询学生成绩。作业和试卷批改使用对应 Skill：先 load_skill，再用 inspect_grading_target 检查目标，start_grading_job 提交任务，get_grading_job 查询真实进度。trigger_async_pipeline 仅为旧作业批改入口。
官网下载先用 load_skill 加载 download-homework，再使用 fetch_homework 或兼容入口 fetch_and_grade_homework，仅收集课程名称、作业名称及必要时的学期。禁止索取官网账号密码或验证码；用户在作业抓取控制台的任务浏览器中完成整个登录，再选择班级并交还自动化。
任务创建只表示排队，不能宣称下载或批改完成。用 get_fetch_job 查询真实进度。用户明确要求暂停、继续或取消时调用对应 pause_fetch_job、resume_fetch_job、cancel_fetch_job。缺少目标信息时询问用户。`, username)
}
