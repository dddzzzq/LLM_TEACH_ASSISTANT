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
	return map[string]string{"connection": "get_portal_connection", "courses": "list_portal_courses", "homeworks": "list_portal_homeworks", "detail": "get_portal_homework_detail", "read_detail": "read_portal_homework_detail", "sync": "sync_portal_catalog", "create": "create_portal_grading_workflow", "workflow": "get_portal_grading_workflow"}[t.Operation]
}
func (t *PortalTool) Description() string {
	return map[string]string{"connection": "查询本人教务账号的脱敏连接状态；设置或认证入口 /settings/portal。", "courses": "查询本人已同步课程及稳定目录 ID，不猜测名称。", "homeworks": "按已查询课程 ID 查询官网作业、本地 assignment_id、班级稳定 ID 和目录版本。", "detail": "查询已读取的官网作业要求、提取完整性及读取任务状态。", "read_detail": "复用现有页面 Agent 只读获取指定官网作业详情，自动提取文字、图片和文档要求，不下载学生提交或批改。", "sync": "同步官网课程与作业，自动创建本地作业、填写官网要求和评分模板；不自动批改学生提交。", "create": "复用已同步创建的本地作业按班级批改。先查询课程、作业和班级 ID，items 指定 homework_id、class_id，use_saved_assignment=true 沿用已保存的官网题目和教师编辑的评分标准，无需重复填写。只有教师明确要求全部班级时才省略 class_id。", "workflow": "查询本人官网作业批改流程及各项进度。"}[t.Operation]
}
func (t *PortalTool) Schema() string {
	switch t.Operation {
	case "detail":
		return `{"type":"object","properties":{"homework_id":{"type":"string"}},"required":["homework_id"],"additionalProperties":false}`
	case "read_detail":
		return `{"type":"object","properties":{"homework_id":{"type":"string"},"snapshot_id":{"type":"string"}},"required":["homework_id","snapshot_id"],"additionalProperties":false}`
	case "homeworks":
		return `{"type":"object","properties":{"course_id":{"type":"string"}},"required":["course_id"],"additionalProperties":false}`
	case "workflow":
		return `{"type":"object","properties":{"workflow_id":{"type":"string"}},"required":["workflow_id"],"additionalProperties":false}`
	case "create":
		return `{"type":"object","properties":{"source":{"type":"string","enum":["portal"]},"snapshot_id":{"type":"string"},"items":{"type":"array","minItems":1,"maxItems":50,"items":{"type":"object","properties":{"homework_id":{"type":"string"},"class_id":{"type":"string"}},"required":["homework_id"],"additionalProperties":false}},"question":{"type":"string"},"use_saved_assignment":{"type":"boolean"},"use_portal_requirements":{"type":"boolean"},"rubric":{"type":"object"}},"required":["source","snapshot_id","items"],"anyOf":[{"required":["rubric"]},{"properties":{"use_saved_assignment":{"const":true}},"required":["use_saved_assignment"]}],"additionalProperties":false}`
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
	case "detail", "read_detail":
		var body struct {
			HomeworkID string `json:"homework_id"`
			SnapshotID string `json:"snapshot_id"`
		}
		if json.Unmarshal([]byte(args), &body) != nil || body.HomeworkID == "" {
			return "", fmt.Errorf("作业参数无效")
		}
		if t.Operation == "detail" {
			value, err := s.HomeworkDetail(ctx, t.UserID, body.HomeworkID)
			if err != nil {
				return "", err
			}
			return toolJSON(value)
		}
		op, err := s.QueueHomeworkDetail(ctx, t.UserID, body.HomeworkID, body.SnapshotID)
		if err != nil {
			return "", err
		}
		return toolJSON(map[string]any{"outcome": "accepted", "job_id": op.ID, "job_type": "portal_homework_detail", "result_url": "/settings/portal", "message": op.Message})
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
func (t *PortalTool) SubmitsJob() bool {
	return t.Operation == "sync" || t.Operation == "create" || t.Operation == "read_detail"
}
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
	for _, op := range []string{"connection", "courses", "homeworks", "detail", "read_detail", "sync", "create", "workflow"} {
		tool := &PortalTool{Operation: op}
		row := models.ToolDefinition{Name: tool.Name(), ImplKey: tool.Name(), Description: tool.Description(), SchemaJSON: tool.Schema(), Enabled: true, AllowedRoles: `["teacher","admin"]`}
		database.DB.WithContext(ctx).Where("name = ?", tool.Name()).FirstOrCreate(&row)
		legacy := map[string]string{
			"homeworks": "按已查询课程 ID 查询官网作业和目录版本，无需选择班级。",
			"sync":      "教师要求更新课程时创建只读目录同步，不下载或批改附件。",
			"create":    "按教师明确选择的官网作业 ID 和目录版本创建批改流程，默认全部班级；use_portal_requirements=true 使用各份作业已完整读取的官网要求。",
		}
		if row.Description == legacy[op] {
			database.DB.WithContext(ctx).Model(&row).Update("description", tool.Description())
		}
		database.DB.WithContext(ctx).Model(&row).Update("schema_json", tool.Schema())
	}
	InvalidateToolsCache(ctx)
}
