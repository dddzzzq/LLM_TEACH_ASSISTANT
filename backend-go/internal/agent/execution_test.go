package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"grading-gateway/internal/execution"
	"grading-gateway/internal/models"
)

func TestExecutionRecordsSkillAndToolProtocolForReconstruction(t *testing.T) {
	tool := &runtimeTool{name: "query_student_score"}
	registry := NewToolRegistry()
	registry.Register(tool)
	events := []models.AgentEvent{}
	var mu sync.Mutex
	writer := execution.NewWriter(func(_ context.Context, kind string, data any) error {
		mu.Lock()
		defer mu.Unlock()
		payload, err := json.Marshal(data)
		if err != nil {
			return err
		}
		events = append(events, models.AgentEvent{RunID: "run", Sequence: uint64(len(events) + 1), Version: execution.Version, Type: kind, Payload: payload})
		return nil
	})
	if err := writer.Record(context.Background(), "user.message", execution.Message{Role: "user", Content: "查询"}); err != nil {
		t.Fatal(err)
	}
	round := 0
	call := func(_ context.Context, messages []map[string]any, _ []map[string]any) (*LLMResponse, error) {
		round++
		switch round {
		case 1:
			current := 0
			for _, m := range messages {
				if m["role"] == "user" && m["content"] == "查询" {
					current++
				}
			}
			if current != 1 || messages[1]["content"] != "旧问题" {
				t.Fatalf("history duplicated or flattened: %+v", messages)
			}
			return calls("load_skill", `{"name":"grade-homework"}`), nil
		case 2:
			return calls("query_student_score", `{"student_id":"s1"}`), nil
		default:
			return &LLMResponse{Content: "完成", Usage: map[string]any{"total_tokens": 123}, FinishReason: "stop"}, nil
		}
	}
	result, err := RunDialogue(context.Background(), "policy", "查询", "teacher", "teacher", registry, registry.ExportAllTools(), SkillCatalog{"grade-homework": {Name: "grade-homework", Version: "fixture-version", Instructions: "SKILL_BODY"}}, call, DialogueOptions{Writer: writer, History: []execution.Message{{Role: "user", Content: "旧问题"}, {Role: "assistant", Content: "旧回答"}}})
	if err != nil || tool.calls != 1 || result.Reply != "完成" {
		t.Fatalf("result %+v calls=%d err=%v", result, tool.calls, err)
	}
	if err := writer.Record(context.Background(), "run.finished", execution.Final{Reply: result.Reply, Status: execution.Succeeded}); err != nil {
		t.Fatal(err)
	}
	view, err := execution.Project(events)
	if err != nil {
		t.Fatalf("events: %+v\nprojection: %v", events, err)
	}
	if len(view.Unresolved) != 0 {
		t.Fatalf("completed tools lost results: %+v", view)
	}
	counts := map[string]int{}
	for _, e := range events {
		counts[e.Type]++
		if e.Type == "skill.loaded" && !strings.Contains(string(e.Payload), "SKILL_BODY") {
			t.Fatal("skill version/body lost")
		}
		if e.Type == "tool.bound" && !strings.Contains(string(e.Payload), `"call_id":"query_student_score"`) {
			t.Fatalf("effective args not correlated: %s", e.Payload)
		}
	}
	if counts["model.request"] != 3 || counts["tool.result"] != 2 || counts["tool.started"] != 2 || counts["skill.loaded"] != 1 || counts["tool.bound"] != 1 {
		t.Fatalf("missing execution facts: %+v", counts)
	}
	before := tool.calls
	if _, err := execution.Project(events); err != nil || tool.calls != before {
		t.Fatal("replay executed a tool")
	}
}

func TestExecutionPersistenceFailureStopsBeforeToolSideEffects(t *testing.T) {
	for _, failAt := range []string{"model.request", "model.message", "tool.started", "tool.bound"} {
		t.Run(failAt, func(t *testing.T) {
			tool := &runtimeTool{name: "query_student_score"}
			r := NewToolRegistry()
			r.Register(tool)
			writer := execution.NewWriter(func(_ context.Context, kind string, _ any) error {
				if kind == failAt {
					return errors.New("storage failed")
				}
				return nil
			})
			_, err := RunDialogue(context.Background(), "system", "query", "teacher", "teacher", r, r.ExportAllTools(), nil, func(context.Context, []map[string]any, []map[string]any) (*LLMResponse, error) {
				return calls("query_student_score", `{}`), nil
			}, DialogueOptions{Writer: writer})
			if !errors.Is(err, execution.ErrRecord) || tool.calls != 0 {
				t.Fatalf("record failure allowed effects: calls=%d err=%v", tool.calls, err)
			}
		})
	}
}

func TestExecutionRecordsUnknownAndDeniedTools(t *testing.T) {
	for _, name := range []string{"invented_tool", "start_grading_job"} {
		t.Run(name, func(t *testing.T) {
			tool := &runtimeTool{name: "start_grading_job"}
			r := NewToolRegistry()
			r.Register(tool)
			events := []models.AgentEvent{}
			w := execution.NewWriter(func(_ context.Context, kind string, v any) error {
				data, _ := json.Marshal(v)
				events = append(events, models.AgentEvent{RunID: "r", Sequence: uint64(len(events) + 1), Version: execution.Version, Type: kind, Payload: data})
				return nil
			})
			step := 0
			_, err := RunDialogue(context.Background(), "policy", "query", "student", "s1", r, r.ExportAllTools(), nil, func(context.Context, []map[string]any, []map[string]any) (*LLMResponse, error) {
				step++
				if step == 1 {
					return calls(name, `{}`), nil
				}
				return &LLMResponse{Content: "拒绝"}, nil
			}, DialogueOptions{Writer: w})
			if err != nil || tool.calls != 0 {
				t.Fatalf("denied tool executed: %v", err)
			}
			view, err := execution.Project(events)
			if err != nil || len(view.Unresolved) != 0 {
				t.Fatalf("refusal missing in replay: %+v %v", view, err)
			}
		})
	}
}
