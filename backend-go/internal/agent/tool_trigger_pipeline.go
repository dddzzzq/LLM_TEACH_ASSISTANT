package agent

import (
	"context"
	"encoding/json"
	"grading-gateway/internal/grading"
)

// TriggerPipelineTool retains the old function name and arguments while using the shared job service.
type TriggerPipelineTool struct {
	UserID       uint
	Role         string
	RequestID    string
	SkillVersion string
}

func (t *TriggerPipelineTool) Name() string { return "trigger_async_pipeline" }
func (t *TriggerPipelineTool) Description() string {
	return "兼容的作业批改入口；使用与 start_grading_job 相同的持久任务和队列。需先加载 grade-homework。"
}
func (t *TriggerPipelineTool) Schema() string {
	return `{"type":"object","properties":{"assignment_id":{"type":"string"},"file_path":{"type":"string","description":"用户或上传结果提供的作业 ZIP 路径"}},"required":["assignment_id","file_path"],"additionalProperties":false}`
}
func (t *TriggerPipelineTool) Execute(args string) (string, error) {
	return t.ExecuteContext(context.Background(), args)
}
func (t *TriggerPipelineTool) ExecuteContext(ctx context.Context, args string) (string, error) {
	var p struct {
		AssignmentID string `json:"assignment_id"`
		FilePath     string `json:"file_path"`
	}
	if err := json.Unmarshal([]byte(args), &p); err != nil {
		return "", err
	}
	result, err := grading.Default().Submit(ctx, grading.Actor{UserID: t.UserID, Role: t.Role}, grading.Request{Kind: "homework", ResourceID: p.AssignmentID, FilePath: p.FilePath, SkillName: "grade-homework", SkillVersion: t.SkillVersion}, t.RequestID)
	if err != nil {
		return "", err
	}
	return toolJSON(result)
}

func (t *TriggerPipelineTool) RequiredSkill(string) string { return "grade-homework" }
func (t *TriggerPipelineTool) SubmitsJob() bool            { return true }
func (t *TriggerPipelineTool) ExecuteBound(ctx context.Context, args string, run ExecutionContext) (string, error) {
	bound := *t
	bound.RequestID = run.RequestID
	bound.SkillVersion = run.SkillVersion
	return bound.ExecuteContext(ctx, args)
}
