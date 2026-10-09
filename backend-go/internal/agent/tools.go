package agent

import (
	"encoding/json"
	"fmt"
	"grading-gateway/internal/grading"
	"sort"
	"sync"
)

// Tool is an executable capability. Its input contract is owned by code.
// A Skill is a Markdown knowledge package; loading one never registers a Tool.
type Tool interface {
	// Name 返回工具的唯一名称
	Name() string
	// Description 返回工具的描述，用于 LLM 理解该工具的功能
	Description() string
	// Schema 返回该工具所需参数的 JSON Schema 字符串
	// 格式遵循 OpenAI Function Calling 规范
	Schema() string
	// Execute 执行工具，传入 JSON 格式的参数字符串
	// 返回执行结果字符串和可能的错误
	Execute(args string) (string, error)
}

// ToolRegistry 工具注册表，用于管理所有可用的工具
type ToolRegistry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

// NewToolRegistry 创建一个新的工具注册表实例
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools: make(map[string]Tool),
	}
}

// NewBuiltinToolRegistry is the shared catalog for chat execution and admin metadata.
// Passing zero is suitable for inspecting metadata, never for executing user tasks.
func NewBuiltinToolRegistry(userID uint, roles ...string) *ToolRegistry {
	role := ""
	if len(roles) > 0 {
		role = roles[0]
	}
	r := NewToolRegistry()
	r.Register(&QueryStudentScoreTool{})
	r.Register(&TriggerPipelineTool{UserID: userID, Role: role})
	for _, op := range []string{"inspect", "submit", "status"} {
		r.Register(&GradingTool{Operation: op, Actor: grading.Actor{UserID: userID, Role: role}})
	}
	r.Register(&FetchAndGradeHomeworkTool{UserID: userID})
	r.Register(&FetchAndGradeHomeworkTool{UserID: userID, UseFetchAlias: true})
	for _, op := range []string{"get", "pause", "resume", "cancel"} {
		r.Register(&FetchJobControlTool{UserID: userID, Operation: op})
	}
	return r
}

// ToolFunction exports the executor's schema, with an optional configured description.
func ToolFunction(tool Tool, description string) (map[string]interface{}, error) {
	var schema map[string]interface{}
	if err := json.Unmarshal([]byte(tool.Schema()), &schema); err != nil {
		return nil, fmt.Errorf("invalid schema for tool %s: %w", tool.Name(), err)
	}
	if schema["type"] != "object" {
		return nil, fmt.Errorf("tool %s must define an object schema", tool.Name())
	}
	if description == "" {
		description = tool.Description()
	}
	return map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name": tool.Name(), "description": description, "parameters": schema,
		},
	}, nil
}

// Register 注册一个新工具
// Duplicate registrations are rejected; aliases must use distinct explicit names.
func (sr *ToolRegistry) Register(tool Tool) error {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	if _, exists := sr.tools[tool.Name()]; exists {
		return fmt.Errorf("工具重复注册: %s", tool.Name())
	}
	sr.tools[tool.Name()] = tool
	return nil
}

// GetTool 根据名称获取工具
// 第二个返回值表示工具是否存在
func (sr *ToolRegistry) GetTool(name string) (Tool, bool) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()
	tool, exists := sr.tools[name]
	return tool, exists
}

// ExportAllTools 将所有注册的工具导出为 OpenAI Tool 格式
// 返回的切片可以直接用于 OpenAI API 的 tools 参数
func (sr *ToolRegistry) ExportAllTools() []map[string]interface{} {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	tools := make([]map[string]interface{}, 0, len(sr.tools))
	for _, tool := range sr.tools {
		if definition, err := ToolFunction(tool, ""); err == nil {
			tools = append(tools, definition)
		}
	}
	sort.Slice(tools, func(i, j int) bool {
		return tools[i]["function"].(map[string]interface{})["name"].(string) < tools[j]["function"].(map[string]interface{})["name"].(string)
	})
	return tools
}

// ListTools 返回所有已注册工具的名称列表
func (sr *ToolRegistry) ListTools() []string {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	names := make([]string, 0, len(sr.tools))
	for name := range sr.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// RemoveTool 移除指定名称的工具
func (sr *ToolRegistry) RemoveTool(name string) {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	delete(sr.tools, name)
}

// Clear 清空所有工具
func (sr *ToolRegistry) Clear() {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	sr.tools = make(map[string]Tool)
}
