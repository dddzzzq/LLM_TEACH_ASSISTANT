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
	return "创建独立的官网作业拉取任务，默认下载并整理材料。已有官网课程目录或需要按班级批改时，优先查询 list_portal_courses、list_portal_homeworks，再用 create_portal_grading_workflow 和 use_saved_assignment=true 复用已创建作业。after_download可选download、extract、grade；仅老师明确要求批改时使用grade。禁止索取密码。"
}
func (s *FetchAndGradeHomeworkTool) Schema() string {
	return `{"type":"object","properties":{"course_name":{"type":"string","description":"官网课程名称"},"assignment_name":{"type":"string","description":"官网作业名称"},"term":{"type":"string","description":"可选学期"},"after_download":{"type":"string","enum":["download","extract","grade"],"description":"默认extract，下载并解压后停止；download仅下载；grade须老师明确要求批改"},"output_root":{"type":"string","description":"老师指定的保存根目录，系统追加课程/作业/班级/批次"}},"required":["course_name","assignment_name"],"additionalProperties":false}`
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
		"message": "任务将默认处理该作业的全部班级，并按要求自动下载、整理或批改；登录状态与进度可在任务浏览器查看", "after_download": job.AfterDownload, "output_root": job.OutputRoot}
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
func (t *FetchAndGradeHomeworkTool) AdditionalSkills(args string) []string {
	var p rpa.Target
	_ = json.Unmarshal([]byte(args), &p)
	if p.AfterDownload == "grade" {
		return []string{"grade-homework"}
	}
	return nil
}
func (t *FetchAndGradeHomeworkTool) SubmitsJob() bool { return true }
func (t *FetchAndGradeHomeworkTool) ExecuteBound(ctx context.Context, args string, run ExecutionContext) (string, error) {
	bound := *t
	bound.RequestID = run.RequestID
	return bound.ExecuteContext(ctx, args)
}
