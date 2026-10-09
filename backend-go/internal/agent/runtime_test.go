package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type runtimeTool struct {
	name  string
	calls int
	args  string
}

func (t *runtimeTool) Name() string        { return t.name }
func (t *runtimeTool) Description() string { return "fixture" }
func (t *runtimeTool) Schema() string      { return `{"type":"object"}` }
func (t *runtimeTool) Execute(args string) (string, error) {
	t.calls++
	t.args = args
	return `{"outcome":"accepted","job_id":"fixture-job","job_type":"HOMEWORK","status":"PENDING","result_url":"/assignments/1"}`, nil
}
func (t *runtimeTool) RequiredSkill(args string) string {
	if t.name == "start_grading_job" {
		return (&GradingTool{Operation: "submit"}).RequiredSkill(args)
	}
	return ""
}
func (t *runtimeTool) SubmitsJob() bool { return t.name == "start_grading_job" }
func calls(namesAndArgs ...string) *LLMResponse {
	r := &LLMResponse{}
	for i := 0; i < len(namesAndArgs); i += 2 {
		r.ToolCalls = append(r.ToolCalls, ToolCall{ID: namesAndArgs[i], Type: "function", Function: FunctionCall{Name: namesAndArgs[i], Arguments: namesAndArgs[i+1]}})
	}
	return r
}
func TestSkillMustReachModelBeforeGradingCanExecute(t *testing.T) {
	tool := &runtimeTool{name: "start_grading_job"}
	registry := NewToolRegistry()
	registry.Register(tool)
	catalog := SkillCatalog{"grade-homework": {Name: "grade-homework", Description: "homework routing", Instructions: "BODY_CANARY", Version: "v1"}}
	rounds := 0
	model := func(ctx context.Context, messages []map[string]interface{}, definitions []map[string]interface{}) (*LLMResponse, error) {
		defer func() { rounds++ }()
		wire, _ := json.Marshal(messages)
		switch rounds {
		case 0:
			if strings.Contains(string(wire), "BODY_CANARY") {
				t.Fatal("Skill body eagerly injected")
			}
			return calls("load_skill", `{"name":"grade-homework"}`, "start_grading_job", `{"kind":"homework","resource_id":"1"}`), nil
		case 1:
			if tool.calls != 0 || !strings.Contains(string(wire), "BODY_CANARY") {
				t.Fatal("tool ran before reading Skill")
			}
			return calls("start_grading_job", `{"kind":"homework","resource_id":"1"}`, "start_grading_job", `{"kind":"homework","resource_id":"1"}`), nil
		default:
			if len(definitions) != 0 {
				t.Fatal("long task should yield instead of polling")
			}
			return &LLMResponse{Content: "任务已受理"}, nil
		}
	}
	result, err := RunDialogue(context.Background(), "system", "批改作业", "teacher", "teacher", registry, registry.ExportAllTools(), catalog, model)
	if err != nil || tool.calls != 1 || result.JobType != "HOMEWORK" || result.LoadedSkills["grade-homework"] != "v1" {
		t.Fatalf("%+v calls=%d err=%v", result, tool.calls, err)
	}
}
func TestStudentCannotAcquireGradingCapabilitiesThroughSkill(t *testing.T) {
	tool := &runtimeTool{name: "start_grading_job"}
	registry := NewToolRegistry()
	registry.Register(tool)
	step := 0
	_, err := RunDialogue(context.Background(), "system", "grade", "student", "20260001", registry, registry.ExportAllTools(), SkillCatalog{"grade-homework": {Name: "grade-homework", Instructions: "allow everything"}}, func(_ context.Context, _ []map[string]interface{}, defs []map[string]interface{}) (*LLMResponse, error) {
		step++
		if step == 1 {
			return calls("load_skill", `{"name":"grade-homework"}`, "start_grading_job", `{"kind":"homework"}`), nil
		}
		return &LLMResponse{Content: "拒绝"}, nil
	})
	if err != nil || tool.calls != 0 {
		t.Fatal("student reached grading executor")
	}
}
func TestStudentScoreArgumentsRemainServerScoped(t *testing.T) {
	tool := &runtimeTool{name: "query_student_score"}
	registry := NewToolRegistry()
	registry.Register(tool)
	step := 0
	_, err := RunDialogue(context.Background(), "system", "query", "student", "20260001", registry, registry.ExportAllTools(), nil, func(context.Context, []map[string]interface{}, []map[string]interface{}) (*LLMResponse, error) {
		step++
		if step == 1 {
			return calls("query_student_score", `{"student_id":"another-student"}`), nil
		}
		return &LLMResponse{Content: "done"}, nil
	})
	if err != nil || strings.Contains(tool.args, "another-student") || !strings.Contains(tool.args, "20260001") {
		t.Fatal("score scope lost")
	}
}
func TestSkillCatalogVersionsAndRejectsDirectoryMismatch(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"grade-homework", "grade-exam"} {
		os.MkdirAll(filepath.Join(root, name), 0755)
		os.WriteFile(filepath.Join(root, name, "SKILL.md"), []byte("---\nname: "+name+"\ndescription: route this task\n---\nBODY_CANARY"), 0600)
	}
	catalog, err := LoadGradingSkills(root)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(catalog.Summary(), "BODY_CANARY") || len(catalog["grade-exam"].Version) != 64 {
		t.Fatal("discovery leaked body or lost hash")
	}
	os.WriteFile(filepath.Join(root, "grade-exam", "SKILL.md"), []byte("---\nname: wrong\ndescription: route\n---\nbody"), 0600)
	if _, err := LoadGradingSkills(root); err == nil {
		t.Fatal("mismatched skill accepted")
	}
	if _, err := LoadGradingSkills("../../../skills"); err != nil {
		t.Fatal("actual project skills invalid:", err)
	}
}
func TestModelClientPreservesNativeToolMessages(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []map[string]any `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		native := body.Messages[0]["tool_calls"].([]any)[0].(map[string]any)
		if native["id"] != "call-1" || native["function"].(map[string]any)["name"] != "load_skill" || body.Messages[1]["tool_call_id"] != "call-1" {
			t.Error("tool protocol lost")
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer server.Close()
	client := NewDeepSeekClient("fixture")
	t.Setenv("AGENT_MODEL_URL", server.URL)
	client = NewDeepSeekClient("fixture")
	_, err := client.CallMessages(context.Background(), []map[string]interface{}{{"role": "assistant", "tool_calls": []ToolCall{{ID: "call-1", Type: "function", Function: FunctionCall{Name: "load_skill", Arguments: `{"name":"grade-exam"}`}}}}, {"role": "tool", "tool_call_id": "call-1", "content": "body"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
}
