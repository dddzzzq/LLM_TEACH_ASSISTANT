package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"grading-gateway/internal/database"
	"grading-gateway/internal/grading"
	"grading-gateway/internal/models"
)

type GradingTool struct {
	Operation    string
	Actor        grading.Actor
	RequestID    string
	SkillName    string
	SkillVersion string
}

func (t *GradingTool) Name() string {
	return map[string]string{"inspect": "inspect_grading_target", "submit": "start_grading_job", "status": "get_grading_job"}[t.Operation]
}
func (t *GradingTool) Description() string {
	return map[string]string{
		"inspect": "查找作业或试卷候选，或按资源 ID 检查评分配置；不会启动批改。",
		"submit":  "按已确认目标和材料创建作业或试卷批改任务，返回真实 job_id；需先加载对应批改 Skill。",
		"status":  "查询当前用户发起的批改任务真实状态与结果入口，不重新提交任务。",
	}[t.Operation]
}
func (t *GradingTool) Schema() string {
	switch t.Operation {
	case "inspect":
		return `{"type":"object","properties":{"kind":{"type":"string","enum":["homework","exam"]},"resource_id":{"type":"string"},"query":{"type":"string","description":"课程、作业或试卷名称；没有 ID 时用于查找候选"}},"required":["kind"],"additionalProperties":false}`
	case "status":
		return `{"type":"object","properties":{"job_id":{"type":"string"}},"required":["job_id"],"additionalProperties":false}`
	default:
		return `{"type":"object","properties":{"kind":{"type":"string","enum":["homework","exam"]},"resource_id":{"type":"string"},"file_path":{"type":"string","description":"homework 的用户或上传结果提供的 ZIP 路径"},"student_id":{"type":"string","description":"exam 的学生学号"},"image_paths":{"type":"array","items":{"type":"string"},"description":"exam 的答卷图片，严格按页码排序"}},"required":["kind","resource_id"],"additionalProperties":false}`
	}
}
func (t *GradingTool) Execute(args string) (string, error) {
	return t.ExecuteContext(context.Background(), args)
}
func toolJSON(value any) (string, error) { b, err := json.Marshal(value); return string(b), err }
func (t *GradingTool) ExecuteContext(ctx context.Context, args string) (string, error) {
	if err := t.Actor.Check(); err != nil {
		return "", err
	}
	service := grading.Default()
	switch t.Operation {
	case "status":
		var p struct {
			JobID string `json:"job_id"`
		}
		if err := json.Unmarshal([]byte(args), &p); err != nil {
			return "", err
		}
		result, err := service.Status(ctx, t.Actor, p.JobID)
		if err != nil {
			return "", err
		}
		return toolJSON(result)
	case "inspect":
		var p struct {
			Kind       string `json:"kind"`
			ResourceID string `json:"resource_id"`
			Query      string `json:"query"`
		}
		if err := json.Unmarshal([]byte(args), &p); err != nil {
			return "", err
		}
		if p.ResourceID != "" {
			result, err := service.Check(ctx, t.Actor, p.Kind, p.ResourceID)
			if err != nil {
				return "", err
			}
			return toolJSON(result)
		}
		query := strings.TrimSpace(p.Query)
		if query == "" {
			return toolJSON(grading.Result{Outcome: "needs_input", Missing: []string{"作业/试卷 ID 或名称"}, Message: "请明确需要批改的目标"})
		}
		candidates := []map[string]any{}
		switch p.Kind {
		case "homework":
			var rows []models.Assignment
			if err := database.DB.WithContext(ctx).Select("id", "course_name", "class_name", "task_name").Where("course_name LIKE ? OR task_name LIKE ?", "%"+query+"%", "%"+query+"%").Order("id DESC").Limit(30).Find(&rows).Error; err != nil {
				return "", err
			}
			for _, a := range rows {
				candidates = append(candidates, map[string]any{"resource_id": fmt.Sprint(a.ID), "course_name": a.CourseName, "class_name": a.ClassName, "task_name": a.TaskName})
			}
		case "exam":
			var rows []models.Exam
			if err := database.DB.WithContext(ctx).Select("id", "name").Where("name LIKE ?", "%"+query+"%").Order("id DESC").Limit(30).Find(&rows).Error; err != nil {
				return "", err
			}
			for _, e := range rows {
				candidates = append(candidates, map[string]any{"resource_id": fmt.Sprint(e.ID), "name": e.Name})
			}
		default:
			return "", fmt.Errorf("kind 仅支持 homework 或 exam")
		}
		return toolJSON(map[string]any{"outcome": "candidates", "candidates": candidates, "limit": 30, "message": "确认唯一目标后用 resource_id 检查评分配置；候选过多时缩小查询范围"})
	default:
		var p grading.Request
		if err := json.Unmarshal([]byte(args), &p); err != nil {
			return "", err
		}
		p.SkillName = t.SkillName
		p.SkillVersion = t.SkillVersion
		result, err := service.Submit(ctx, t.Actor, p, t.RequestID)
		if err != nil {
			return "", err
		}
		return toolJSON(result)
	}
}

func (t *GradingTool) RequiredSkill(args string) string {
	if t.Operation != "submit" {
		return ""
	}
	var p struct {
		Kind string `json:"kind"`
	}
	_ = json.Unmarshal([]byte(args), &p)
	if p.Kind == "homework" {
		return "grade-homework"
	}
	if p.Kind == "exam" {
		return "grade-exam"
	}
	return ""
}
func (t *GradingTool) SubmitsJob() bool { return t.Operation == "submit" }
func (t *GradingTool) ExecuteBound(ctx context.Context, args string, run ExecutionContext) (string, error) {
	bound := *t
	bound.RequestID = run.RequestID
	bound.SkillName = run.SkillName
	bound.SkillVersion = run.SkillVersion
	return bound.ExecuteContext(ctx, args)
}
