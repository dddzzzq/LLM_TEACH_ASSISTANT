package browseragent

import (
	"context"
	"encoding/json"
	"grading-gateway/internal/agent/catalog"
	"grading-gateway/internal/agent/modelclient"
	"strings"
	"testing"
)

const fixtureContract = `{"oneOf":[{"description":"点击控件","properties":{"tool":{"type":"string","enum":["click"]},"element_ref":{"type":"string"}},"required":["tool","element_ref"]}]}`

func modelResponse(name, args string) *modelclient.LLMResponse {
	return &modelclient.LLMResponse{ToolCalls: []modelclient.ToolCall{{ID: "fixture", Type: "function", Function: modelclient.FunctionCall{Name: name, Arguments: args}}}}
}
func TestEinoPageAgentLoadsSkillBeforeReturningAction(t *testing.T) {
	doc := catalog.Document{Name: "fetch-homework", Description: "navigate", Version: "v1", Instructions: "NAVIGATION_BODY_CANARY"}
	rounds := 0
	action, err := Decide(context.Background(), json.RawMessage(`{"observation":{"elements":[{"ref":"p0f0e0"}]}}`), doc, []byte(fixtureContract), func(_ context.Context, messages []map[string]interface{}, defs []map[string]interface{}) (*modelclient.LLMResponse, error) {
		raw, _ := json.Marshal(messages)
		rounds++
		if rounds == 1 {
			if strings.Contains(string(raw), doc.Instructions) {
				t.Fatal("eager Skill body")
			}
			return modelResponse("load_skill", `{"skill":"fetch-homework"}`), nil
		}
		if !strings.Contains(string(raw), doc.Instructions) {
			t.Fatal("navigation Skill missing")
		}
		return modelResponse("click", `{"element_ref":"p0f0e0"}`), nil
	})
	if err != nil || rounds != 2 || !strings.Contains(string(action), `"tool":"click"`) {
		t.Fatalf("action=%s rounds=%d err=%v", action, rounds, err)
	}
}
func TestPageAgentCannotSkipSkillOrSmuggleExtraActionParameters(t *testing.T) {
	for _, skip := range []bool{true, false} {
		t.Run(map[bool]string{true: "skip-skill", false: "extra-parameter"}[skip], func(t *testing.T) {
			round := 0
			_, err := Decide(context.Background(), []byte(`{}`), catalog.Document{Name: "fetch-homework", Description: "nav", Instructions: "body"}, []byte(fixtureContract), func(context.Context, []map[string]interface{}, []map[string]interface{}) (*modelclient.LLMResponse, error) {
				round++
				if !skip && round == 1 {
					return modelResponse("load_skill", `{"skill":"fetch-homework"}`), nil
				}
				args := `{"element_ref":"p0"}`
				if !skip {
					args = `{"element_ref":"p0","script":"untrusted"}`
				}
				return modelResponse("click", args), nil
			})
			if err == nil {
				t.Fatal("invalid navigation plan accepted")
			}
		})
	}
}
