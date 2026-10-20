package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/models"
)

type MaterialTool struct {
	UserID    uint
	Role      string
	Operation string
	RequestID string
}

func (t *MaterialTool) Name() string {
	if t.Operation == "prepare" {
		return "extract_homework_files"
	}
	if t.Operation == "cancel" {
		return "cancel_homework_materials"
	}
	return "get_homework_materials"
}
func (t *MaterialTool) Description() string {
	switch t.Operation {
	case "prepare":
		return "独立整理作业材料：扫描老师指定目录或官网下载附件，识别学号姓名，安全解压并生成程序/报告文件清单。默认仅整理，不批改。返回材料批次ID。"
	case "cancel":
		return "取消本人材料整理及尚未提交的自动批改，保留已保存文件；不撤回已创建批改任务。"
	}
	return "查询本人材料批次、学生与文件清单、异常及关联批改进度；不提交任务。"
}
func (t *MaterialTool) Schema() string {
	if t.Operation != "prepare" {
		return `{"type":"object","properties":{"batch_id":{"type":"string"}},"required":["batch_id"],"additionalProperties":false}`
	}
	return `{"type":"object","properties":{"source_path":{"type":"string","description":"老师明确指定的目录或ZIP/RAR/7z绝对路径"},"source_job_id":{"type":"string","description":"官网下载任务ID，与source_file_id配合"},"source_file_id":{"type":"string"},"assignment_id":{"type":"integer","description":"已有作业ID"},"course_name":{"type":"string"},"assignment_name":{"type":"string"},"class_name":{"type":"string"},"output_root":{"type":"string","description":"保存根目录，系统追加课程/作业/班级/批次"},"auto_grade":{"type":"boolean","description":"仅老师明确要求整理后批改时设置true；默认false"}},"additionalProperties":false}`
}
func (t *MaterialTool) Execute(args string) (string, error) {
	return t.ExecuteContext(context.Background(), args)
}
func (t *MaterialTool) ExecuteContext(ctx context.Context, args string) (string, error) {
	s := materials.Default()
	actor := materials.Actor{UserID: t.UserID, Role: t.Role}
	if t.Operation == "prepare" {
		var r materials.Request
		if err := json.Unmarshal([]byte(args), &r); err != nil {
			return "", err
		}
		batch, err := s.Create(ctx, actor, r, t.RequestID)
		if err != nil {
			return "", err
		}
		url := "/materials"
		if batch.AssignmentID != 0 {
			url = fmt.Sprintf("/assignments/%d", batch.AssignmentID)
		}
		return toolJSON(map[string]any{"outcome": "accepted", "job_id": batch.ID, "job_type": "homework_materials", "batch_id": batch.ID, "status": batch.Status, "message": batch.Message, "result_url": url})
	}
	var p struct {
		BatchID string `json:"batch_id"`
	}
	if err := json.Unmarshal([]byte(args), &p); err != nil {
		return "", err
	}
	if t.Operation == "cancel" {
		if err := s.Cancel(ctx, actor, p.BatchID); err != nil {
			return "", err
		}
	}
	batch, err := s.Owned(ctx, actor, p.BatchID)
	if err != nil {
		return "", err
	}
	var manifest materials.Manifest
	_ = json.Unmarshal([]byte(batch.ManifestJSON), &manifest)
	var grade models.AsyncJob
	if batch.GradingJobID != "" {
		s.DB.First(&grade, "id = ? AND owner_id = ?", batch.GradingJobID, t.UserID)
	}
	return toolJSON(map[string]any{"batch": batch, "manifest": manifest, "grading_job": grade})
}
func (t *MaterialTool) RequiredSkill(args string) string {
	if t.Operation != "prepare" {
		return ""
	}
	var r materials.Request
	_ = json.Unmarshal([]byte(args), &r)
	if r.SourceJobID != "" && !r.AutoGrade {
		return "download-homework"
	}
	return "grade-homework"
}
func (t *MaterialTool) SubmitsJob() bool { return t.Operation == "prepare" }
func (t *MaterialTool) ExecuteBound(ctx context.Context, args string, run ExecutionContext) (string, error) {
	bound := *t
	bound.RequestID = run.RequestID
	return bound.ExecuteContext(ctx, args)
}
