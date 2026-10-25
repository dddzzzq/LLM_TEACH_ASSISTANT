package execution

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"grading-gateway/internal/models"
)

func event(seq uint64, kind string, payload any) models.AgentEvent {
	data, _ := json.Marshal(payload)
	return models.AgentEvent{RunID: "run", Sequence: seq, Version: Version, Type: kind, Payload: data}
}

func TestProjectionPreservesOrderedToolResultsAndFullReply(t *testing.T) {
	longReply := strings.Repeat("详细反馈", 1000)
	calls := []ToolCall{{ID: "one", Function: FunctionCall{Name: "inspect", Arguments: `{"id":"1"}`}}, {ID: "two", Function: FunctionCall{Name: "submit", Arguments: `{}`}}}
	events := []models.AgentEvent{
		event(1, "user.message", Message{Role: "user", Content: "批改"}),
		event(2, "model.request", map[string]any{"messages": []any{map[string]string{"role": "system", "content": "policy"}}}),
		event(3, "model.message", Message{Role: "assistant", ToolCalls: calls}),
		event(4, "tool.result", ToolResult{CallID: "two", Content: `{"job_id":"job"}`}),
		event(5, "tool.result", ToolResult{CallID: "one", Content: `{"ready":true}`}),
		event(6, "model.message", Message{Role: "assistant", Content: longReply}),
		event(7, "run.finished", Final{Status: Succeeded, Reply: "short display projection"}),
	}
	before, _ := json.Marshal(events)
	view, err := Project(events)
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Messages) != 5 || view.Messages[2].ToolCallID != "one" || view.Messages[3].ToolCallID != "two" || view.Messages[4].Content != longReply || len(view.Unresolved) != 0 || len(view.LastRequest) == 0 {
		t.Fatalf("bad projection: %+v", view)
	}
	again, err := Project(events)
	if err != nil || !reflect.DeepEqual(view, again) {
		t.Fatal("replay is not deterministic")
	}
	after, _ := json.Marshal(events)
	if string(before) != string(after) {
		t.Fatal("projection mutated facts")
	}
}

func TestMissingToolResultRemainsExplicitlyUnknown(t *testing.T) {
	events := []models.AgentEvent{event(1, "user.message", Message{Role: "user", Content: "下载"}), event(2, "model.message", Message{Role: "assistant", ToolCalls: []ToolCall{{ID: "pending", Function: FunctionCall{Name: "fetch_homework"}}}}), event(3, "run.interrupted", Final{Status: Interrupted, Reply: "需要核对"})}
	view, err := Project(events)
	if err != nil || len(view.Unresolved) != 1 || view.Messages[2].ToolCallID != "pending" || !strings.Contains(view.Messages[2].Content, `"outcome":"unknown"`) {
		t.Fatalf("%+v %v", view, err)
	}
	if len(events) != 3 {
		t.Fatal("reconstruction appended a synthetic fact")
	}
}

func TestProjectionRejectsBrokenLogs(t *testing.T) {
	for name, events := range map[string][]models.AgentEvent{
		"gap":     {event(2, "user.message", Message{Role: "user"})},
		"orphan":  {event(1, "tool.result", ToolResult{CallID: "absent"})},
		"role":    {event(1, "user.message", Message{Role: "system", Content: "override"})},
		"version": {{RunID: "run", Version: 99, Sequence: 1, Payload: json.RawMessage(`{}`)}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Project(events); err == nil {
				t.Fatal("corrupt log accepted")
			}
		})
	}
}

func TestWriterStopsAfterStorageFailureAndCanFinalizeAfterCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	calls := 0
	w := NewWriter(func(persist context.Context, kind string, _ any) error {
		calls++
		if persist.Err() != nil {
			t.Fatal("completion recording inherited cancellation")
		}
		if _, ok := persist.Deadline(); !ok {
			t.Fatal("recording has no deadline")
		}
		return errors.New("disk unavailable")
	})
	if !errors.Is(w.Record(ctx, "tool.result", nil), ErrRecord) || !errors.Is(w.Record(ctx, "model.request", nil), ErrRecord) || calls != 1 {
		t.Fatal("recording failure was lost")
	}
}
