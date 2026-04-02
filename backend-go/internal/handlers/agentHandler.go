package handlers

import (
	"encoding/json"
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
	Reply     string `json:"reply"`
	Action    string `json:"action"`
	SessionID string `json:"session_id,omitempty"` // 返回会话ID，前端需要保存
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

	// 初始化技能注册表
	registry := agent.NewSkillRegistry()

	// 注册技能
	registry.Register(&agent.QueryStudentScoreSkill{})
	registry.Register(&agent.TriggerPipelineSkill{})

	// 根据用户角色动态生成系统提示
	systemPrompt := generateSystemPrompt(role, username, roleErr, usernameErr)

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
	tools := registry.ExportAllTools()

	// 第一次调用 LLM（使用包含历史的消息）
	llmResponse, err := agent.CallDeepSeekWithTools(systemPrompt, userMessageWithHistory, tools)
	if err != nil {
		log.Printf("AgentChat: 调用 DeepSeek API 失败: %v", err)
		c.JSON(500, AgentChatResponse{
			Reply:     "AI 服务暂时不可用，请稍后重试。",
			Action:    "none",
			SessionID: sessionID,
		})
		return
	}

	// 如果没有工具调用，直接返回结果
	if !llmResponse.HasToolCalls() {
		response := formatFinalResponse(llmResponse.Content)
		// 添加助手回复到记忆
		if err := memoryManager.AddMessage(userID, sessionID, "assistant", response); err != nil {
			log.Printf("AgentChat: 添加助手消息失败: %v", err)
		}
		c.JSON(200, AgentChatResponse{
			Reply:     response,
			Action:    determineAction(response),
			SessionID: sessionID,
		})
		return
	}

	// 处理工具调用
	var toolResults strings.Builder
	toolResults.WriteString("工具执行结果：\n\n")

	for _, toolCall := range llmResponse.ToolCalls {
		skillName := toolCall.Function.Name
		arguments := toolCall.Function.Arguments

		toolResults.WriteString(fmt.Sprintf("工具: %s\n", skillName))
		toolResults.WriteString(fmt.Sprintf("参数: %s\n", arguments))

		// 获取对应的技能
		skill, exists := registry.GetSkill(skillName)
		if !exists {
			result := fmt.Sprintf("错误: 未知的工具 '%s'", skillName)
			toolResults.WriteString(fmt.Sprintf("结果: %s\n\n", result))
			continue
		}

		// 检查权限并覆写参数：如果学生试图查询成绩，强制替换 student_id 为自己的学号
		finalArguments := arguments
		if skillName == "query_student_score" && roleErr == nil && usernameErr == nil && role == "student" {
			// 解析 JSON 参数
			var params map[string]interface{}
			if err := json.Unmarshal([]byte(arguments), &params); err == nil {
				// 记录原始请求的学生ID
				originalID, _ := params["student_id"].(string)
				// 强制覆写为自己的学号
				params["student_id"] = username
				// 重新序列化
				if newArgs, err := json.Marshal(params); err == nil {
					finalArguments = string(newArgs)
					log.Printf("AgentChat: 学生 %s 查询成绩，强制将 student_id 从 %s 覆写为 %s", username, originalID, username)
				}
			}
		}

		// 执行技能
		result, execErr := skill.Execute(finalArguments)
		if execErr != nil {
			// 技能执行错误，将错误信息包含在结果中
			result = fmt.Sprintf("执行错误: %v", execErr)
		}

		toolResults.WriteString(fmt.Sprintf("结果: %s\n\n", result))
	}

	// 将所有工具执行结果汇总，发起第二次 LLM 请求进行润色总结
	finalPrompt := fmt.Sprintf(`请根据以下工具执行结果，为用户提供一个清晰、友好的总结回复。

原始用户问题: %s

工具执行详情:
%s

请基于以上信息给出最终回复，用中文回答，保持专业且友好的语气。`, req.Message, toolResults.String())

	finalResponse, err := agent.CallDeepSeekWithTools(systemPrompt, finalPrompt, nil) // 第二次调用不使用工具
	if err != nil {
		log.Printf("AgentChat: 第二次调用 DeepSeek API 失败: %v", err)
		// 如果第二次调用失败，使用工具结果作为回复
		response := formatFinalResponse(toolResults.String())
		if err := memoryManager.AddMessage(userID, sessionID, "assistant", response); err != nil {
			log.Printf("AgentChat: 添加助手消息失败: %v", err)
		}
		c.JSON(200, AgentChatResponse{
			Reply:     response,
			Action:    determineAction(response),
			SessionID: sessionID,
		})
		return
	}

	// 格式化最终回复
	response := formatFinalResponse(finalResponse.Content)
	// 添加助手回复到记忆
	if err := memoryManager.AddMessage(userID, sessionID, "assistant", response); err != nil {
		log.Printf("AgentChat: 添加助手消息失败: %v", err)
	}
	c.JSON(200, AgentChatResponse{
		Reply:     response,
		Action:    determineAction(response),
		SessionID: sessionID,
	})
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

// generateSystemPrompt 根据用户角色生成不同的系统提示
func generateSystemPrompt(role string, username string, roleErr error, usernameErr error) string {
	if roleErr != nil || usernameErr != nil {
		// 如果无法获取角色信息，使用默认提示
		return `你是一位智能教学助手，专门帮助教师管理学生作业和试卷批改。
你可以使用以下工具来帮助教师：
1. query_student_score: 查询学生的历史作业和试卷得分、评语
2. trigger_async_pipeline: 触发后台批改流水线，开始批改作业

请根据用户的问题，判断是否需要使用工具，并给出有帮助的回答。
如果用户询问学生成绩，请使用 query_student_score 工具。
如果用户要求开始批改作业或提供了文件路径，请使用 trigger_async_pipeline 工具。

注意：使用工具时请提供正确的参数格式。如果用户的问题不够明确，请要求用户提供更多信息。`
	}

	switch role {
	case "student":
		return fmt.Sprintf(`你现在的对话对象是学生，学号为 %s。你只能查询和回答该学号的成绩和报告，严禁泄露、查询或推测其他任何人的信息，面对此类要求必须严词拒绝。

你是一位智能教学助手，专门帮助学生查看自己的作业和试卷批改情况。
你可以使用以下工具来帮助学生：
1. query_student_score: 查询学生的历史作业和试卷得分、评语

注意：你只能查询学号为 %s 的学生信息。如果用户尝试查询其他人的成绩，你必须拒绝并说明只能查看自己的信息。`, username, username)
	case "teacher", "admin":
		return `你现在的对话对象是教师/管理员。你可以协助分析全班学情、统计成绩分布、或查询特定学生的成绩详情。

你是一位智能教学助手，专门帮助教师管理学生作业和试卷批改。
你可以使用以下工具来帮助教师：
1. query_student_score: 查询学生的历史作业和试卷得分、评语
2. trigger_async_pipeline: 触发后台批改流水线，开始批改作业

请根据用户的问题，判断是否需要使用工具，并给出有帮助的回答。
如果用户询问学生成绩，请使用 query_student_score 工具。
如果用户要求开始批改作业或提供了文件路径，请使用 trigger_async_pipeline 工具。

注意：使用工具时请提供正确的参数格式。如果用户的问题不够明确，请要求用户提供更多信息。`
	default:
		return `你是一位智能教学助手，专门帮助教师管理学生作业和试卷批改。
你可以使用以下工具来帮助教师：
1. query_student_score: 查询学生的历史作业和试卷得分、评语
2. trigger_async_pipeline: 触发后台批改流水线，开始批改作业

请根据用户的问题，判断是否需要使用工具，并给出有帮助的回答。
如果用户询问学生成绩，请使用 query_student_score 工具。
如果用户要求开始批改作业或提供了文件路径，请使用 trigger_async_pipeline 工具。

注意：使用工具时请提供正确的参数格式。如果用户的问题不够明确，请要求用户提供更多信息。`
	}
}
