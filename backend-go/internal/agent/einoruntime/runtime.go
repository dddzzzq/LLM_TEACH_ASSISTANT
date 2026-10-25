// Package einoruntime isolates framework types from business services and providers.
package einoruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/adk/middlewares/skill"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	"github.com/eino-contrib/jsonschema"
	"github.com/google/uuid"
	"grading-gateway/internal/agent/catalog"
	"grading-gateway/internal/agent/modelclient"
	"grading-gateway/internal/execution"
)

type Tool struct {
	Name, Description string
	Schema            json.RawMessage
	Execute           func(context.Context, string) (string, error)
	ReturnDirectly    bool
}

type Config struct {
	Name, Instruction string
	Call              modelclient.Call
	Tools             []Tool
	Skills            catalog.Catalog
	SkillLoaded       func(catalog.Document)
	BeforeModel       func()
	DisableTools      func() bool
	MaxIterations     int
	History           []execution.Message
	Record            execution.Record
	ModelMetadata     map[string]any
}

func (c Config) record(ctx context.Context, kind string, payload any) error {
	if c.Record == nil {
		return nil
	}
	return c.Record(ctx, kind, payload)
}

type adapter struct{ Tool }

func (t adapter) Info(context.Context) (*schema.ToolInfo, error) {
	var s jsonschema.Schema
	if err := json.Unmarshal(t.Schema, &s); err != nil {
		return nil, err
	}
	return &schema.ToolInfo{Name: t.Name, Desc: t.Description, ParamsOneOf: schema.NewParamsOneOfByJSONSchema(&s)}, nil
}
func (t adapter) InvokableRun(ctx context.Context, input string, _ ...tool.Option) (string, error) {
	return t.Execute(ctx, input)
}

type skillBackend struct{ cfg Config }

func (b skillBackend) List(context.Context) ([]skill.FrontMatter, error) {
	out := []skill.FrontMatter{}
	for _, name := range b.cfg.Skills.Names() {
		d := b.cfg.Skills[name]
		out = append(out, skill.FrontMatter{Name: name, Description: d.Description})
	}
	return out, nil
}
func (b skillBackend) Get(ctx context.Context, name string) (skill.Skill, error) {
	d, ok := b.cfg.Skills[name]
	if !ok {
		return skill.Skill{}, fmt.Errorf("Skill 不可用: %s", name)
	}
	if err := b.cfg.record(ctx, "skill.loaded", map[string]any{"name": d.Name, "version": d.Version, "instructions": d.Instructions, "call_id": compose.GetToolCallID(ctx)}); err != nil {
		return skill.Skill{}, err
	}
	if b.cfg.SkillLoaded != nil {
		b.cfg.SkillLoaded(d)
	}
	return skill.Skill{FrontMatter: skill.FrontMatter{Name: d.Name, Description: d.Description}, Content: d.Instructions}, nil
}

type chatModel struct{ cfg Config }

func (m *chatModel) Generate(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.Message, error) {
	if m.cfg.BeforeModel != nil {
		m.cfg.BeforeModel()
	}
	messages := make([]map[string]interface{}, 0, len(input))
	for _, v := range input {
		row := map[string]interface{}{"role": string(v.Role), "content": v.Content}
		if len(v.ToolCalls) > 0 {
			row["tool_calls"] = v.ToolCalls
		}
		if v.ToolCallID != "" {
			row["tool_call_id"] = v.ToolCallID
		}
		messages = append(messages, row)
	}
	definitions := []map[string]interface{}{}
	options := model.GetCommonOptions(&model.Options{}, opts...)
	toolsDisabled := m.cfg.DisableTools != nil && m.cfg.DisableTools()
	if toolsDisabled {
		messages = append(messages, map[string]interface{}{"role": "system", "content": "本轮任务已受理，工具调用阶段已结束。请仅根据已有工具回执，用自然语言说明已提交的任务类型、任务 ID 和后续查看入口。不要继续查询或调用工具，不要输出工具调用标记，也不要把任务受理说成任务完成。"})
	} else {
		for _, t := range options.Tools {
			p, err := t.ParamsOneOf.ToJSONSchema()
			if err != nil {
				return nil, err
			}
			definitions = append(definitions, map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": t.Name, "description": t.Desc, "parameters": p}})
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	requestID := uuid.NewString()
	toolChoice := "none"
	if len(definitions) > 0 {
		toolChoice = "auto"
	}
	if err := m.cfg.record(ctx, "model.request", map[string]any{"request_id": requestID, "messages": messages, "tools": definitions, "tool_choice": toolChoice, "model": m.cfg.ModelMetadata}); err != nil {
		return nil, err
	}
	response, err := m.cfg.Call(ctx, messages, definitions)
	if err != nil {
		if recordErr := m.cfg.record(ctx, "model.failed", map[string]any{"request_id": requestID, "error": err.Error()}); recordErr != nil {
			return nil, recordErr
		}
		return nil, err
	}
	if response == nil {
		err := fmt.Errorf("模型返回空响应")
		if recordErr := m.cfg.record(ctx, "model.failed", map[string]any{"request_id": requestID, "error": err.Error()}); recordErr != nil {
			return nil, recordErr
		}
		return nil, err
	}
	// Keep malformed provider output for diagnosis, but out of assistant history.
	if strings.Contains(response.Content, "｜DSML｜") || strings.Contains(response.Content, "|DSML|") || toolsDisabled && len(response.ToolCalls) > 0 {
		if err := m.cfg.record(ctx, "model.completed", map[string]any{"request_id": requestID, "usage": response.Usage, "finish_reason": response.FinishReason, "raw_tool_calls": response.ToolCalls, "raw_content": response.Content}); err != nil {
			return nil, err
		}
		err := fmt.Errorf("模型返回了工具调用内容，无法作为最终回复")
		if recordErr := m.cfg.record(ctx, "model.failed", map[string]any{"request_id": requestID, "error": err.Error()}); recordErr != nil {
			return nil, recordErr
		}
		return nil, err
	}
	out := &schema.Message{Role: schema.Assistant, Content: response.Content}
	used := map[string]bool{}
	for _, msg := range input {
		for _, call := range msg.ToolCalls {
			used[call.ID] = true
		}
	}
	for _, c := range response.ToolCalls {
		if c.ID == "" || used[c.ID] {
			c.ID = uuid.NewString()
		}
		used[c.ID] = true
		out.ToolCalls = append(out.ToolCalls, schema.ToolCall{ID: c.ID, Type: "function", Function: schema.FunctionCall{Name: c.Function.Name, Arguments: c.Function.Arguments}})
	}
	if err := m.cfg.record(ctx, "model.message", toMessage(out)); err != nil {
		return nil, err
	}
	if err := m.cfg.record(ctx, "model.completed", map[string]any{"request_id": requestID, "usage": response.Usage, "finish_reason": response.FinishReason, "raw_tool_calls": response.ToolCalls}); err != nil {
		return nil, err
	}
	return out, nil
}

func toMessage(msg *schema.Message) execution.Message {
	out := execution.Message{Role: string(msg.Role), Content: msg.Content, ToolCallID: msg.ToolCallID}
	for _, c := range msg.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, execution.ToolCall{ID: c.ID, Type: c.Type, Function: execution.FunctionCall{Name: c.Function.Name, Arguments: c.Function.Arguments}})
	}
	return out
}

func recordedTool(cfg Config, next compose.InvokableToolEndpoint) compose.InvokableToolEndpoint {
	return func(ctx context.Context, in *compose.ToolInput) (*compose.ToolOutput, error) {
		ctx = execution.WithCallID(ctx, in.CallID)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := cfg.record(ctx, "tool.started", map[string]any{"call_id": in.CallID, "name": in.Name, "arguments": in.Arguments}); err != nil {
			return nil, err
		}
		out, err := next(ctx, in)
		result := execution.ToolResult{CallID: in.CallID, Name: in.Name}
		if out != nil {
			result.Content = out.Result
		}
		if err != nil {
			result.Error = err.Error()
		}
		if recordErr := cfg.record(ctx, "tool.result", result); recordErr != nil {
			return nil, recordErr
		}
		return out, err
	}
}
func (m *chatModel) Stream(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	v, err := m.Generate(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return schema.StreamReaderFromArray([]*schema.Message{v}), nil
}

// Run delegates the reasoning/tool loop to Eino. Business authorization stays in each executor.
func Run(ctx context.Context, cfg Config, user string) (string, error) {
	if cfg.Call == nil {
		return "", fmt.Errorf("缺少模型调用实现")
	}
	if cfg.MaxIterations == 0 {
		cfg.MaxIterations = 8
	}
	ts := []tool.BaseTool{}
	direct := map[string]bool{}
	for _, t := range cfg.Tools {
		ts = append(ts, adapter{t})
		if t.ReturnDirectly {
			direct[t.Name] = true
		}
	}
	handlers := []adk.ChatModelAgentMiddleware{}
	if len(cfg.Skills) > 0 {
		name := "load_skill"
		h, err := skill.NewMiddleware(ctx, &skill.Config{Backend: skillBackend{cfg}, SkillToolName: &name})
		if err != nil {
			return "", err
		}
		handlers = append(handlers, h)
	}
	a, err := adk.NewChatModelAgent(ctx, &adk.ChatModelAgentConfig{
		Name: cfg.Name, Description: cfg.Name, Instruction: cfg.Instruction, Model: &chatModel{cfg}, MaxIterations: cfg.MaxIterations,
		Handlers: handlers, ToolsConfig: adk.ToolsConfig{ReturnDirectly: direct, ToolsNodeConfig: compose.ToolsNodeConfig{
			Tools: ts, ExecuteSequentially: true,
			ToolCallMiddlewares: []compose.ToolMiddleware{{Invokable: func(next compose.InvokableToolEndpoint) compose.InvokableToolEndpoint { return recordedTool(cfg, next) }}},
			UnknownToolsHandler: func(ctx context.Context, name, args string) (string, error) {
				out, err := recordedTool(cfg, func(context.Context, *compose.ToolInput) (*compose.ToolOutput, error) {
					return &compose.ToolOutput{Result: `{"outcome":"failed","message":"工具不存在或未获准使用"}`}, nil
				})(ctx, &compose.ToolInput{Name: name, Arguments: args, CallID: compose.GetToolCallID(ctx)})
				if err != nil {
					return "", err
				}
				return out.Result, nil
			},
			ToolArgumentsHandler: func(_ context.Context, name, args string) (string, error) {
				if name == "load_skill" {
					var p map[string]any
					if json.Unmarshal([]byte(args), &p) == nil {
						if _, ok := p["skill"]; !ok {
							p["skill"] = p["name"]
						}
						delete(p, "name")
						b, _ := json.Marshal(p)
						return string(b), nil
					}
				}
				return args, nil
			},
		}},
	})
	if err != nil {
		return "", err
	}
	runner := adk.NewRunner(ctx, adk.RunnerConfig{Agent: a})
	messages := make([]*schema.Message, 0, len(cfg.History)+1)
	for _, msg := range cfg.History {
		m := &schema.Message{Role: schema.RoleType(msg.Role), Content: msg.Content, ToolCallID: msg.ToolCallID}
		for _, c := range msg.ToolCalls {
			m.ToolCalls = append(m.ToolCalls, schema.ToolCall{ID: c.ID, Type: c.Type, Function: schema.FunctionCall{Name: c.Function.Name, Arguments: c.Function.Arguments}})
		}
		messages = append(messages, m)
	}
	messages = append(messages, schema.UserMessage(user))
	it := runner.Run(ctx, messages)
	var reply string
	for {
		event, ok := it.Next()
		if !ok {
			break
		}
		if event.Err != nil {
			return reply, event.Err
		}
		if event.Output != nil && event.Output.MessageOutput != nil {
			msg, err := event.Output.MessageOutput.GetMessage()
			if err != nil {
				return reply, err
			}
			if msg != nil && len(msg.ToolCalls) == 0 && msg.Content != "" {
				reply = msg.Content
			}
		}
	}
	return reply, nil
}
