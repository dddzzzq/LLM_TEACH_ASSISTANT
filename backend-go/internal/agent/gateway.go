package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/google/uuid"
	"grading-gateway/internal/agent/catalog"
	"strings"
)

type contextTool interface {
	ExecuteContext(context.Context, string) (string, error)
}

// ExecutionContext is supplied by the server; model input cannot override it.
type ExecutionContext struct{ RequestID, SkillName, SkillVersion string }
type workflowTool interface{ RequiredSkill(string) string }
type contextualTool interface {
	ExecuteBound(context.Context, string, ExecutionContext) (string, error)
}
type jobSubmittingTool interface{ SubmitsJob() bool }

type toolGateway struct {
	role, username, runID string
	registry              *ToolRegistry
	result                RunResult
	visible               map[string]string
	executed              map[string]string
	submitted             bool
	lastResult            string
}

func newGateway(role, username string, r *ToolRegistry) *toolGateway {
	return &toolGateway{role: role, username: username, runID: uuid.NewString(), registry: r, result: RunResult{LoadedSkills: map[string]string{}}, executed: map[string]string{}}
}
func (g *toolGateway) beforeModel() {
	g.visible = map[string]string{}
	for n, v := range g.result.LoadedSkills {
		g.visible[n] = v
	}
}
func (g *toolGateway) skillLoaded(d catalog.Document) { g.result.LoadedSkills[d.Name] = d.Version }
func (g *toolGateway) execute(ctx context.Context, name, args string) (string, error) {
	fail := func(message string) (string, error) {
		return toolJSON(map[string]any{"outcome": "failed", "message": message})
	}
	if g.submitted {
		return fail("任务已受理，请说明任务 ID，不再执行操作")
	}
	t, ok := g.registry.GetTool(name)
	if !ok {
		return fail("工具执行器不存在")
	}
	if g.role != "teacher" && g.role != "admin" && !(g.role == "student" && name == "query_student_score") {
		return fail("当前角色不能执行该任务")
	}
	if err := validateToolInput(t.Schema(), args); err != nil {
		return fail(err.Error())
	}
	required := ""
	if bound, ok := t.(workflowTool); ok {
		required = bound.RequiredSkill(args)
	}
	if required != "" && g.visible[required] == "" {
		return fail("请先调用 load_skill 加载 " + required + "，再按指导执行")
	}
	if g.role == "student" && name == "query_student_score" {
		args, _ = toolJSON(map[string]string{"student_id": g.username})
	}
	var parsed any
	_ = json.Unmarshal([]byte(args), &parsed)
	canonical, _ := json.Marshal(parsed)
	key := name + string(canonical)
	// Both historical download names represent the same operation.
	if name == "fetch_and_grade_homework" {
		key = "fetch_homework" + string(canonical)
	}
	if prior, ok := g.executed[key]; ok {
		return prior, nil
	}
	var output string
	var err error
	if v, ok := t.(contextualTool); ok {
		output, err = v.ExecuteBound(ctx, args, ExecutionContext{g.runID, required, g.visible[required]})
	} else if v, ok := t.(contextTool); ok {
		output, err = v.ExecuteContext(ctx, args)
	} else {
		output, err = t.Execute(args)
	}
	if err != nil {
		return fail(err.Error())
	}
	g.executed[key] = output
	g.lastResult = output
	var receipt struct {
		JobID     string `json:"job_id"`
		JobType   string `json:"job_type"`
		ResultURL string `json:"result_url"`
		Outcome   string `json:"outcome"`
	}
	if json.Unmarshal([]byte(output), &receipt) == nil && receipt.JobID != "" {
		g.result.JobID = receipt.JobID
		g.result.JobType = receipt.JobType
		g.result.ResultURL = receipt.ResultURL
		if strings.Contains(name, "fetch") {
			g.result.JobType = "rpa_fetch_homework"
		}
		if submit, ok := t.(jobSubmittingTool); ok {
			g.submitted = submit.SubmitsJob()
		}
	}
	return output, nil
}

// Validate the code-owned subset used by built-in contracts before business execution.
func validateToolInput(definition, input string) error {
	var spec map[string]any
	var value any
	if json.Unmarshal([]byte(definition), &spec) != nil || json.Unmarshal([]byte(input), &value) != nil {
		return fmt.Errorf("工具参数不是有效 JSON")
	}
	return validateValue(spec, value)
}
func validateValue(spec map[string]any, value any) error {
	if choices, ok := spec["enum"].([]any); ok {
		match := false
		for _, c := range choices {
			a, _ := json.Marshal(c)
			b, _ := json.Marshal(value)
			if string(a) == string(b) {
				match = true
			}
		}
		if !match {
			return fmt.Errorf("参数不在允许范围内")
		}
	}
	switch spec["type"] {
	case "object":
		obj, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("工具参数必须是对象")
		}
		props, _ := spec["properties"].(map[string]any)
		if required, ok := spec["required"].([]any); ok {
			for _, v := range required {
				if _, exists := obj[v.(string)]; !exists {
					return fmt.Errorf("缺少参数 %s", v)
				}
			}
		}
		for name, v := range obj {
			p, exists := props[name]
			if !exists {
				if spec["additionalProperties"] == false {
					return fmt.Errorf("未知参数 %s", name)
				}
				continue
			}
			if err := validateValue(p.(map[string]any), v); err != nil {
				return err
			}
		}
	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("参数必须是字符串")
		}
	case "array":
		vs, ok := value.([]any)
		if !ok {
			return fmt.Errorf("参数必须是数组")
		}
		if item, ok := spec["items"].(map[string]any); ok {
			for _, v := range vs {
				if err := validateValue(item, v); err != nil {
					return err
				}
			}
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("参数必须是布尔值")
		}
	case "number", "integer":
		v, ok := value.(float64)
		if !ok || spec["type"] == "integer" && v != float64(int64(v)) {
			return fmt.Errorf("参数数字类型无效")
		}
	}
	return nil
}

type requestKeyContext struct{}

func WithRequestKey(ctx context.Context, key string) context.Context {
	return context.WithValue(ctx, requestKeyContext{}, key)
}
