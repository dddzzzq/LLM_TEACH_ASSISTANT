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

func TestEveryRegisteredPageSkillCanDriveAuthentication(t *testing.T) {
	skills, err := catalog.Default("portal-browser")
	if err != nil {
		t.Fatal(err)
	}
	contract := json.RawMessage(`{"oneOf":[{"type":"object","properties":{"tool":{"type":"string","enum":["fill_portal_credentials"]}},"required":["tool"],"additionalProperties":false}]}`)
	for _, name := range skills.Names() {
		t.Run(name, func(t *testing.T) {
			rounds := 0
			action, err := Decide(context.Background(), json.RawMessage(`{"stage":"AUTHENTICATE","observation":{"authentication":{"login_form":true,"credentials_saved":true,"credentials_filled":false}}}`), skills[name], contract, func(_ context.Context, messages []map[string]interface{}, _ []map[string]interface{}) (*modelclient.LLMResponse, error) {
				rounds++
				if rounds == 1 {
					args, _ := json.Marshal(map[string]string{"skill": name})
					return modelResponse("load_skill", string(args)), nil
				}
				return modelResponse("fill_portal_credentials", `{}`), nil
			})
			if err != nil || string(action) != `{"tool":"fill_portal_credentials"}` || rounds != 2 {
				t.Fatalf("registered Skill rejected before login: action=%s model_calls=%d err=%v", action, rounds, err)
			}
		})
	}
}

func TestHomeworkDetailSkillReachesQuestionReader(t *testing.T) {
	skills, err := catalog.Default("portal-browser")
	if err != nil {
		t.Fatal(err)
	}
	doc := skills["read-homework-detail"]
	contract := json.RawMessage(`{"oneOf":[{"type":"object","properties":{"tool":{"type":"string","enum":["read_homework_detail"]}},"required":["tool"],"additionalProperties":false}]}`)
	rounds := 0
	action, err := Decide(context.Background(), json.RawMessage(`{"stage":"READING_DETAIL","observation":{"task_type":"HOMEWORK_DETAIL","homework_detail":{"ready":true}}}`), doc, contract, func(_ context.Context, messages []map[string]interface{}, _ []map[string]interface{}) (*modelclient.LLMResponse, error) {
		rounds++
		if rounds == 1 {
			return modelResponse("load_skill", `{"skill":"read-homework-detail"}`), nil
		}
		raw, _ := json.Marshal(messages)
		if !strings.Contains(doc.Instructions, "不要自行改写、概括或补全题目") || !strings.Contains(string(raw), "不要自行改写、概括或补全题目") {
			t.Fatal("detail instructions missing from the actual decision")
		}
		return modelResponse("read_homework_detail", `{}`), nil
	})
	if err != nil || string(action) != `{"tool":"read_homework_detail"}` || rounds != 2 {
		t.Fatalf("detail reader decision failed: action=%s model_calls=%d err=%v", action, rounds, err)
	}
}

func TestSamePageAgentLoadsCatalogSkillAndCallsReader(t *testing.T) {
	contract := json.RawMessage(`{"oneOf":[{"description":"读取课程","type":"object","properties":{"tool":{"type":"string","enum":["read_course_list"]}},"required":["tool"],"additionalProperties":false}]}`)
	round := 0
	action, err := Decide(context.Background(), json.RawMessage(`{"stage":"READING_CATALOG","observation":{"catalog":{"courses":[]}}}`), catalog.Document{Name: "collect-homework", Description: "只读获取作业目录", Instructions: "课程→作业，只读当前页面"}, contract, func(_ context.Context, messages []map[string]interface{}, defs []map[string]interface{}) (*modelclient.LLMResponse, error) {
		round++
		if round == 1 {
			return modelResponse("load_skill", `{"skill":"collect-homework"}`), nil
		}
		return modelResponse("read_course_list", `{}`), nil
	})
	if err != nil || string(action) != `{"tool":"read_course_list"}` {
		t.Fatalf("action=%s err=%v", action, err)
	}
}

func TestSamePageAgentLoadsPortalSkillForAuthenticationAndReturn(t *testing.T) {
	skills, err := catalog.Default("portal-browser")
	if err != nil {
		t.Fatal(err)
	}
	doc := skills["sync-teacher-portal"]
	for _, name := range []string{"fill_portal_credentials", "return_to_course_list"} {
		t.Run(name, func(t *testing.T) {
			contract := json.RawMessage(`{"oneOf":[{"type":"object","properties":{"tool":{"type":"string","enum":["` + name + `"]}},"required":["tool"],"additionalProperties":false}]}`)
			round := 0
			action, err := Decide(context.Background(), json.RawMessage(`{"observation":{"catalog":{"return_required":true}}}`), doc, contract, func(_ context.Context, messages []map[string]interface{}, _ []map[string]interface{}) (*modelclient.LLMResponse, error) {
				round++
				if round == 1 {
					return modelResponse("load_skill", `{"skill":"sync-teacher-portal"}`), nil
				}
				raw, _ := json.Marshal(messages)
				if doc.Instructions == "" || !strings.Contains(string(raw), "包括最后一门课和无作业的课程") {
					t.Fatal("portal workflow knowledge missing")
				}
				return modelResponse(name, `{}`), nil
			})
			if err != nil || string(action) != `{"tool":"`+name+`"}` {
				t.Fatalf("action=%s err=%v", action, err)
			}
		})
	}
}

func TestPageAgentCanChooseZeroArgumentSliderTool(t *testing.T) {
	contract := json.RawMessage(`{"oneOf":[{"description":"滑块验证","type":"object","properties":{"tool":{"type":"string","enum":["solve_slider_captcha"]}},"required":["tool"],"additionalProperties":false}]}`)
	round := 0
	action, err := Decide(context.Background(), json.RawMessage(`{"stage":"SOLVE_CAPTCHA","observation":{"captcha":{"visible":true,"attempts_remaining":3}}}`),
		catalog.Document{Name: "fetch-homework", Description: "nav", Instructions: "验证码通过后核验登录"}, contract,
		func(_ context.Context, messages []map[string]interface{}, defs []map[string]interface{}) (*modelclient.LLMResponse, error) {
			round++
			if round == 1 {
				return modelResponse("load_skill", `{"skill":"fetch-homework"}`), nil
			}
			raw, _ := json.Marshal(messages)
			if !strings.Contains(string(raw), "solve_slider_captcha") || !strings.Contains(string(raw), "SOLVE_CAPTCHA") {
				t.Fatal("slider policy or observation missing")
			}
			return modelResponse("solve_slider_captcha", `{}`), nil
		})
	if err != nil || string(action) != `{"tool":"solve_slider_captcha"}` {
		t.Fatalf("action=%s err=%v", action, err)
	}
}

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

func TestPageInformationAndFullSkillReachDecisionModel(t *testing.T) {
	doc := catalog.Document{Name: "fetch-homework", Description: "nav", Instructions: "完整流程：课程、作业、导出"}
	input := json.RawMessage(`{"observation":{"pages":[{"title":"课程空间"}],"frames":[{"frame_id":"p0f1","headings":[{"text":"第一次作业"}]}],"elements":[{"ref":"p0f1e0","role":"button","name":"查看提交","context_labels":["第一次作业"]}],"coverage":{"truncated":false}}}`)
	round := 0
	_, err := Decide(context.Background(), input, doc, []byte(fixtureContract), func(_ context.Context, messages []map[string]interface{}, _ []map[string]interface{}) (*modelclient.LLMResponse, error) {
		round++
		if round == 1 {
			return modelResponse("load_skill", `{"skill":"fetch-homework"}`), nil
		}
		payload, _ := json.Marshal(messages)
		for _, wanted := range []string{"课程空间", "p0f1", "context_labels", "coverage", doc.Instructions, "先整体读取"} {
			if !strings.Contains(string(payload), wanted) {
				t.Fatalf("decision model is missing %s", wanted)
			}
		}
		return modelResponse("click", `{"element_ref":"p0f1e0"}`), nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
