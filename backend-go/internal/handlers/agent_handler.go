package handlers

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	"gorm.io/gorm"
	"grading-gateway/internal/agent"
	"grading-gateway/internal/database"
	"grading-gateway/internal/execution"
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
	RunID        string            `json:"run_id,omitempty"`
	Reply        string            `json:"reply"`
	Action       string            `json:"action"`
	SessionID    string            `json:"session_id,omitempty"` // 返回会话ID，前端需要保存
	JobID        string            `json:"job_id,omitempty"`
	JobType      string            `json:"job_type,omitempty"`
	ResultURL    string            `json:"result_url,omitempty"`
	LoadedSkills map[string]string `json:"loaded_skills,omitempty"`
}

// AgentChat 处理前端对话请求，由 Go Eino Agent 引擎接管
func AgentChat(c *gin.Context) {
	agentChat(c, database.DB, nil)
}

// agentChat accepts a model seam for isolated HTTP + persistence contract tests.
func agentChat(c *gin.Context, db *gorm.DB, call agent.ModelCall) {
	userID, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		c.JSON(401, AgentChatResponse{Reply: "未授权访问，请先登录", Action: "none"})
		return
	}
	role, roleErr := middleware.GetRoleFromContext(c)
	username, usernameErr := middleware.GetUsernameFromContext(c)
	if roleErr != nil || usernameErr != nil {
		c.JSON(401, AgentChatResponse{Reply: "用户身份未确认", Action: "none"})
		return
	}
	var req AgentChatRequest
	if err := c.ShouldBindJSON(&req); err != nil || strings.TrimSpace(req.Message) == "" {
		c.JSON(400, AgentChatResponse{Reply: "请求格式错误，请提供有效的 message 字段。", Action: "none"})
		return
	}
	key := c.GetHeader("Idempotency-Key")
	if len(key) > 128 || len(req.SessionID) > 36 {
		c.JSON(400, AgentChatResponse{Reply: "请求标识无效", Action: "none"})
		return
	}
	if db == nil {
		c.JSON(503, AgentChatResponse{Reply: "执行记录服务暂不可用", Action: "none"})
		return
	}
	store := execution.Store{DB: db}
	run, reused, err := store.Begin(c.Request.Context(), execution.Start{UserID: userID, SessionID: req.SessionID, RequestKey: key, Input: req.Message, Role: role})
	if err != nil {
		status := 503
		message := "无法保存执行记录，请稍后重试。"
		switch {
		case errors.Is(err, execution.ErrNotFound):
			status = 404
			message = "会话不存在或您无权访问"
		case errors.Is(err, execution.ErrBusy):
			status = 409
			message = "当前会话仍在执行，请等待完成后再发送。"
		case errors.Is(err, execution.ErrConflict):
			status = 409
			message = "请求标识已被另一条消息使用。"
		}
		c.JSON(status, AgentChatResponse{Reply: message, Action: "none", SessionID: req.SessionID})
		return
	}
	if reused {
		if run.Status == execution.Succeeded && run.ResultJSON != "" {
			c.Data(200, "application/json; charset=utf-8", []byte(run.ResultJSON))
			return
		}
		c.JSON(409, AgentChatResponse{Reply: "此请求已有执行记录，请查看执行状态；不会自动重复提交任务。", Action: "none", SessionID: run.SessionID, RunID: run.ID})
		return
	}
	response := AgentChatResponse{Action: "none", SessionID: run.SessionID, RunID: run.ID}
	finish := func(status string, code int) {
		ctx, cancel := context.WithTimeout(context.WithoutCancel(c.Request.Context()), 5*time.Second)
		defer cancel()
		if err := store.Finish(ctx, run.ID, status, response.Reply, response); err != nil {
			log.Printf("AgentChat: execution finalization failed run=%s: %v", run.ID, err)
			response.Reply = "执行结果暂未完整保存，请先查看执行记录和任务状态，避免重复提交。"
			c.JSON(503, response)
			return
		}
		c.JSON(code, response)
	}
	history, err := store.Context(c.Request.Context(), userID, run.SessionID, run.ID)
	if err != nil {
		response.Reply = "无法重建会话上下文，请稍后重试。"
		finish(execution.Failed, 503)
		return
	}
	writer := execution.NewWriter(func(ctx context.Context, kind string, payload any) error {
		return store.Append(ctx, run.ID, kind, payload)
	})
	registry := agent.NewBuiltinToolRegistry(userID, role)
	tools, toolsErr := agent.BuildToolsForRole(c.Request.Context(), role, registry)
	if toolsErr != nil {
		log.Printf("AgentChat: 加载工具定义失败: %v", toolsErr)
		tools = nil
	}
	catalog := agent.SkillCatalog{}
	if role == "teacher" || role == "admin" {
		loaded, err := agent.DefaultGradingSkills()
		if err != nil {
			log.Printf("批改 Skill 加载失败: %v", err)
		} else {
			catalog = loaded
		}
	}
	client := agent.NewDeepSeekClient("")
	if call == nil {
		call = client.CallMessages
	}
	requestID := key
	if requestID == "" {
		requestID = run.ID
	}
	result, err := agent.RunDialogue(agent.WithRequestKey(c.Request.Context(), requestID), generateSystemPromptWithFetchTool(role, username, roleErr, usernameErr), req.Message, role, username, registry, tools, catalog, call, agent.DialogueOptions{History: history, Writer: writer, ModelMetadata: client.Metadata()})
	response.JobID, response.JobType, response.ResultURL, response.LoadedSkills = result.JobID, result.JobType, result.ResultURL, result.LoadedSkills
	if err != nil {
		log.Printf("AgentChat: 执行失败 run=%s: %v", run.ID, err)
		response.Reply = "AI 服务暂时不可用，请查看执行记录后再重试。"
		status := execution.Failed
		code := 502
		if c.Request.Context().Err() != nil {
			status = execution.Interrupted
		}
		if errors.Is(err, execution.ErrRecord) {
			code = 503
			response.Reply = "执行记录写入失败，已停止后续操作；请先核对已有任务，避免重复提交。"
		}
		finish(status, code)
		return
	}
	response.Reply = formatFinalResponse(result.Reply)
	response.Action = determineAction(response.Reply)
	finish(execution.Succeeded, 200)
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
		return "你是教师作业助手。用户身份未确认，不要调用业务工具。"
	}

	return fmt.Sprintf(`你是教师作业助手，当前用户为教师或管理员 %s。
使用 query_student_score 查询学生成绩。作业批改使用 grade-homework Skill：先 load_skill，再用 inspect_grading_target 检查目标，start_grading_job 提交任务，get_grading_job 查询真实进度。trigger_async_pipeline 仅为旧作业批改入口。
官网下载先 load_skill 加载 download-homework，再使用 fetch_homework；默认 after_download=extract，下载解压后停止。仅下载用download；只有老师明确要求批改才用grade，且必须先另行加载grade-homework。output_root使用老师指定位置。禁止索取官网账号密码或验证码。用户在任务浏览器输入账号密码并提交登录，出现滑块后可交还自动化，由页面 Agent 使用专用工具最多尝试三次；失败或其他认证由用户完成，班级范围由用户确认。
独立解压整理使用extract_homework_files；查询材料用get_homework_materials。老师明确指定本机目录批改时用start_grading_job的directory_path；已有材料优先传batch_id。上传现在只导入，补齐标准不会使普通材料自动批改；只有已明确请求的完整流程才接续。参数齐备直接执行，不重复请求阶段切换确认。
任务创建只表示排队，不能宣称下载或批改完成。用 get_fetch_job 查询真实进度。用户明确要求暂停、继续或取消时调用对应 pause_fetch_job、resume_fetch_job、cancel_fetch_job。缺少目标信息时询问用户。`, username)
}
