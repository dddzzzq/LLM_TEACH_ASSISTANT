package execution

import (
	"encoding/json"
	"fmt"

	"grading-gateway/internal/models"
)

type Projection struct {
	Messages    []Message       `json:"messages"`
	Unresolved  []string        `json:"unresolved_tool_calls"`
	LastRequest json.RawMessage `json:"last_model_request,omitempty"`
}

// Project is deterministic and side-effect free. Missing results are explicitly unknown,
// never successful, and never dispatched to an executor during replay.
func Project(events []models.AgentEvent) (Projection, error) {
	view := Projection{Messages: []Message{}, Unresolved: []string{}}
	pending := []ToolCall{}
	results := map[string]ToolResult{}
	flush := func() {
		for _, call := range pending {
			result, ok := results[call.ID]
			if !ok {
				view.Unresolved = append(view.Unresolved, call.ID)
				result.Content = `{"outcome":"unknown","message":"执行记录缺少此工具的结果。请先查询已有任务或请用户核对，不得假定成功或自动重新提交。","source":"reconstruction"}`
			} else if result.Error != "" && result.Content == "" {
				data, _ := json.Marshal(map[string]string{"outcome": "failed", "message": result.Error})
				result.Content = string(data)
			}
			view.Messages = append(view.Messages, Message{Role: "tool", Content: result.Content, ToolCallID: call.ID})
		}
		pending = nil
		results = map[string]ToolResult{}
	}
	for i, event := range events {
		if !json.Valid(event.Payload) {
			return view, fmt.Errorf("invalid execution payload at %d", event.Sequence)
		}
		if event.Version != Version || event.Sequence != uint64(i+1) || i > 0 && event.RunID != events[0].RunID {
			return view, fmt.Errorf("invalid execution event sequence or version at %d", event.Sequence)
		}
		switch event.Type {
		case "model.request":
			view.LastRequest = append(json.RawMessage(nil), event.Payload...)
		case "history.imported":
			var messages []Message
			if err := json.Unmarshal(event.Payload, &messages); err != nil {
				return view, err
			}
			for _, m := range messages {
				if m.Role != "user" && m.Role != "assistant" {
					return view, fmt.Errorf("invalid legacy role")
				}
			}
			view.Messages = append(view.Messages, messages...)
		case "user.message", "model.message":
			flush()
			var m Message
			if err := json.Unmarshal(event.Payload, &m); err != nil {
				return view, err
			}
			if event.Type == "user.message" && m.Role != "user" || event.Type == "model.message" && m.Role != "assistant" {
				return view, fmt.Errorf("invalid event role")
			}
			view.Messages = append(view.Messages, m)
			pending = m.ToolCalls
		case "tool.result":
			var result ToolResult
			if err := json.Unmarshal(event.Payload, &result); err != nil {
				return view, err
			}
			found := false
			for _, call := range pending {
				if call.ID == result.CallID {
					found = true
					break
				}
			}
			if !found {
				return view, fmt.Errorf("orphan tool result %s", result.CallID)
			}
			if _, ok := results[result.CallID]; ok {
				return view, fmt.Errorf("duplicate tool result %s", result.CallID)
			}
			results[result.CallID] = result
		case "run.finished", "run.interrupted":
			flush()
			var final Final
			if err := json.Unmarshal(event.Payload, &final); err != nil {
				return view, err
			}
			// Preserve the untruncated model answer; add a final message only for fallback/error paths.
			if final.Reply != "" && (len(view.Messages) == 0 || view.Messages[len(view.Messages)-1].Role != "assistant" || len(view.Messages[len(view.Messages)-1].ToolCalls) > 0) {
				view.Messages = append(view.Messages, Message{Role: "assistant", Content: final.Reply})
			}
		}
	}
	flush()
	return view, nil
}
