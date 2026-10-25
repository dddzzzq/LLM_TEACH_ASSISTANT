package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
	"grading-gateway/internal/portal"
)

// PortalTool exposes metadata and explicitly selected work, never credentials or MFA.
type PortalTool struct {
	UserID                     uint
	Role, Operation, RequestID string
}

func (t *PortalTool) Name() string {
	return map[string]string{"connection": "get_portal_connection", "courses": "list_portal_courses", "homeworks": "list_portal_homeworks", "sync": "sync_portal_catalog", "create": "create_portal_grading_workflow", "workflow": "get_portal_grading_workflow"}[t.Operation]
}
func (t *PortalTool) Description() string {
	return map[string]string{"connection": "查询本人教务账号的脱敏连接状态；设置或认证入口 /settings/portal。", "courses": "查询本人已同步课程及稳定目录 ID，不猜测名称。", "homeworks": "按已查询课程 ID 查询官网作业和目录版本，无需选择班级。", "sync": "教师要求更新课程时创建只读目录同步，不下载或批改附件。", "create": "按教师明确选择的官网作业 ID 和目录版本创建批改流程，默认全部班级，自动下载、整理和批改，不再要求班级选择。", "workflow": "查询本人官网作业批改流程及各项进度。"}[t.Operation]
}
func (t *PortalTool) Schema() string {
	switch t.Operation {
	case "homeworks":
		return `{"type":"object","properties":{"course_id":{"type":"string"}},"required":["course_id"],"additionalProperties":false}`
	case "workflow":
		return `{"type":"object","properties":{"workflow_id":{"type":"string"}},"required":["workflow_id"],"additionalProperties":false}`
	case "create":
		return `{"type":"object","properties":{"source":{"type":"string","enum":["portal"]},"snapshot_id":{"type":"string"},"items":{"type":"array","minItems":1,"maxItems":50,"items":{"type":"object","properties":{"homework_id":{"type":"string"}},"required":["homework_id"],"additionalProperties":false}},"question":{"type":"string"},"rubric":{"type":"object"}},"required":["source","snapshot_id","items","question","rubric"],"additionalProperties":false}`
	}
	return `{"type":"object","properties":{},"additionalProperties":false}`
}
func (t *PortalTool) Execute(args string) (string, error) {
	return t.ExecuteContext(context.Background(), args)
}
func (t *PortalTool) ExecuteContext(ctx context.Context, args string) (string, error) {
	if t.UserID == 0 || (t.Role != "teacher" && t.Role != "admin") {
		return "", fmt.Errorf("当前角色无权访问教务连接")
	}
	s := portal.New(database.DB)
	switch t.Operation {
	case "connection":
		value, err := s.View(ctx, t.UserID)
		if err != nil {
			return "", err
		}
		return toolJSON(value)
	case "courses":
		rows, err := s.Courses(ctx, t.UserID)
		if err != nil {
			return "", err
		}
		return toolJSON(rows)
	case "homeworks":
		var body struct {
			CourseID string `json:"course_id"`
		}
		if json.Unmarshal([]byte(args), &body) != nil {
			return "", fmt.Errorf("课程参数无效")
		}
		value, err := s.Homeworks(ctx, t.UserID, body.CourseID)
		if err != nil {
			return "", err
		}
		return toolJSON(value)
	case "sync":
		op, err := s.Queue(ctx, t.UserID, "CATALOG_SYNC")
		if err != nil {
			return "", err
		}
		return toolJSON(map[string]any{"outcome": "accepted", "job_id": op.ID, "job_type": "portal_catalog_sync", "result_url": "/settings/portal", "message": op.Message})
	case "create":
		var body portal.WorkflowRequest
		if json.Unmarshal([]byte(args), &body) != nil {
			return "", fmt.Errorf("官网批改参数无效")
		}
		flow, err := s.CreateWorkflow(ctx, t.UserID, body, t.RequestID)
		if err != nil {
			return "", err
		}
		return toolJSON(map[string]any{"outcome": "accepted", "job_id": flow.ID, "job_type": "portal_grading_workflow", "result_url": "/grade-homework?workflow=" + flow.ID})
	case "workflow":
		var body struct {
			WorkflowID string `json:"workflow_id"`
		}
		if json.Unmarshal([]byte(args), &body) != nil {
			return "", fmt.Errorf("流程参数无效")
		}
		value, err := s.Workflow(ctx, t.UserID, body.WorkflowID)
		if err != nil {
			return "", err
		}
		return toolJSON(value)
	}
	return "", fmt.Errorf("教务操作无效")
}
func (t *PortalTool) SubmitsJob() bool { return t.Operation == "sync" || t.Operation == "create" }
func (t *PortalTool) RequiredSkill(string) string {
	if t.Operation == "create" {
		return "grade-homework"
	}
	return ""
}
func (t *PortalTool) ExecuteBound(ctx context.Context, args string, run ExecutionContext) (string, error) {
	bound := *t
	bound.RequestID = run.RequestID
	return bound.ExecuteContext(ctx, args)
}
func EnsurePortalTools(ctx context.Context) {
	if database.DB == nil {
		return
	}
	for _, op := range []string{"connection", "courses", "homeworks", "sync", "create", "workflow"} {
		tool := &PortalTool{Operation: op}
		row := models.ToolDefinition{Name: tool.Name(), ImplKey: tool.Name(), Description: tool.Description(), SchemaJSON: tool.Schema(), Enabled: true, AllowedRoles: `["teacher","admin"]`}
		database.DB.WithContext(ctx).Where("name = ?", tool.Name()).FirstOrCreate(&row)
	}
	InvalidateToolsCache(ctx)
}
