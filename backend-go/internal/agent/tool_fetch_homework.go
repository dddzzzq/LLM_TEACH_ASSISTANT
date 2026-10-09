package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"grading-gateway/internal/rpa"
)

// Identity is injected by the server, never accepted from model arguments.
type FetchAndGradeHomeworkTool struct {
	RequestID     string
	UserID        uint
	UseFetchAlias bool
}

func (s *FetchAndGradeHomeworkTool) Name() string {
	if s.UseFetchAlias {
		return "fetch_homework"
	}
	return "fetch_and_grade_homework"
}
func (s *FetchAndGradeHomeworkTool) Description() string {
	return "创建官网作业下载任务。仅需课程名和作业名；用户在任务浏览器中完成登录与验证码、选择班级，可随时接管。附件校验后按现有配置自动批改。禁止索取密码。"
}
func (s *FetchAndGradeHomeworkTool) Schema() string {
	return `{"type":"object","properties":{"course_name":{"type":"string","description":"官网课程名称"},"assignment_name":{"type":"string","description":"官网作业名称"},"term":{"type":"string","description":"可选学期，帮助同名课程消歧"}},"required":["course_name","assignment_name"],"additionalProperties":false}`
}
func (s *FetchAndGradeHomeworkTool) Execute(args string) (string, error) {
	return s.ExecuteContext(context.Background(), args)
}
func (s *FetchAndGradeHomeworkTool) ExecuteContext(ctx context.Context, args string) (string, error) {
	var target rpa.Target
	if err := json.Unmarshal([]byte(args), &target); err != nil {
		return "", fmt.Errorf("请提供课程名与作业名，无需提供账号密码")
	}
	job, err := rpa.CreateWithKey(ctx, s.UserID, target, s.RequestID)
	if err != nil {
		return "", err
	}
	result := map[string]any{"outcome": "accepted", "job_type": "rpa_fetch_homework", "job_id": job.ID, "status": job.Status, "stage": job.Stage,
		"message": "请在作业抓取控制台打开任务浏览器并完成登录；选择班级后系统自动导出、下载并继续批改"}
	b, err := json.Marshal(result)
	return string(b), err
}

type FetchJobControlTool struct {
	UserID    uint
	Operation string
}

func (s *FetchJobControlTool) Name() string { return s.Operation + "_fetch_job" }
func (s *FetchJobControlTool) Description() string {
	return map[string]string{"get": "查询本人官网下载任务的真实进度、附件和批改状态", "pause": "暂停本人下载任务，交给用户操作浏览器", "resume": "用户明确要求继续时交还自动化，恢复前核验登录和班级范围", "cancel": "用户明确要求取消时取消本人下载任务"}[s.Operation]
}
func (s *FetchJobControlTool) Schema() string {
	return `{"type":"object","properties":{"job_id":{"type":"string"}},"required":["job_id"],"additionalProperties":false}`
}
func (s *FetchJobControlTool) Execute(args string) (string, error) {
	return s.ExecuteContext(context.Background(), args)
}
func (s *FetchJobControlTool) ExecuteContext(ctx context.Context, args string) (string, error) {
	var params struct {
		JobID string `json:"job_id"`
	}
	if json.Unmarshal([]byte(args), &params) != nil {
		return "", fmt.Errorf("任务参数无效")
	}
	job, err := rpa.Owned(s.UserID, params.JobID)
	if err != nil {
		return "", fmt.Errorf("任务不存在或无权访问")
	}
	if s.Operation == "get" {
		b, err := json.Marshal(rpa.Detail(job))
		return string(b), err
	}
	if s.Operation == "cancel" && (job.Status == "QUEUED" || job.Status == "INTERRUPTED") {
		if err := rpa.CancelQueued(ctx, job); err != nil {
			return "", err
		}
		return `{"status":"CANCELLED"}`, nil
	}
	b, err := rpa.WorkerRequest(ctx, "POST", fmt.Sprintf("/tasks/%s/%s", job.ID, s.Operation), nil)
	if err != nil {
		return "", err
	}
	if err := rpa.SaveState(job.ID, b); err != nil {
		return "", err
	}
	return string(b), nil
}

func (t *FetchAndGradeHomeworkTool) RequiredSkill(string) string { return "download-homework" }
func (t *FetchAndGradeHomeworkTool) SubmitsJob() bool            { return true }
func (t *FetchAndGradeHomeworkTool) ExecuteBound(ctx context.Context, args string, run ExecutionContext) (string, error) {
	bound := *t
	bound.RequestID = run.RequestID
	return bound.ExecuteContext(ctx, args)
}
