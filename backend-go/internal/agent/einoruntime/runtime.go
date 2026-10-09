// Package einoruntime isolates framework types from business services and providers.
package einoruntime

import (
	"context"
	"encoding/json"
	"fmt"

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
func (b skillBackend) Get(_ context.Context, name string) (skill.Skill, error) {
	d, ok := b.cfg.Skills[name]
	if !ok {
		return skill.Skill{}, fmt.Errorf("Skill 不可用: %s", name)
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
	if m.cfg.DisableTools == nil || !m.cfg.DisableTools() {
		for _, t := range options.Tools {
			p, err := t.ParamsOneOf.ToJSONSchema()
			if err != nil {
				return nil, err
			}
			definitions = append(definitions, map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": t.Name, "description": t.Desc, "parameters": p}})
		}
	}
	response, err := m.cfg.Call(ctx, messages, definitions)
	if err != nil {
		return nil, err
	}
	if response == nil {
		return nil, fmt.Errorf("模型返回空响应")
	}
	out := &schema.Message{Role: schema.Assistant, Content: response.Content}
	for _, c := range response.ToolCalls {
		if c.ID == "" {
			c.ID = uuid.NewString()
		}
		out.ToolCalls = append(out.ToolCalls, schema.ToolCall{ID: c.ID, Type: "function", Function: schema.FunctionCall{Name: c.Function.Name, Arguments: c.Function.Arguments}})
	}
	return out, nil
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
			UnknownToolsHandler: func(_ context.Context, _, _ string) (string, error) {
				return `{"outcome":"failed","message":"工具不存在或未获准使用"}`, nil
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
	it := runner.Query(ctx, user)
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
