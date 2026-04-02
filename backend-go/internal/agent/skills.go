package agent

import (
	"encoding/json"
	"sync"
)

// Skill 接口定义了一个可以被 Agent 调用的技能
type Skill interface {
	// Name 返回技能的唯一名称
	Name() string
	// Description 返回技能的描述，用于 LLM 理解该技能的功能
	Description() string
	// Schema 返回该技能所需参数的 JSON Schema 字符串
	// 格式遵循 OpenAI Function Calling 规范
	Schema() string
	// Execute 执行技能，传入 JSON 格式的参数字符串
	// 返回执行结果字符串和可能的错误
	Execute(args string) (string, error)
}

// SkillRegistry 技能注册表，用于管理所有可用的技能
type SkillRegistry struct {
	mu     sync.RWMutex
	skills map[string]Skill
}

// NewSkillRegistry 创建一个新的技能注册表实例
func NewSkillRegistry() *SkillRegistry {
	return &SkillRegistry{
		skills: make(map[string]Skill),
	}
}

// Register 注册一个新技能
// 如果同名技能已存在，会覆盖旧的技能
func (sr *SkillRegistry) Register(skill Skill) {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	sr.skills[skill.Name()] = skill
}

// GetSkill 根据名称获取技能
// 第二个返回值表示技能是否存在
func (sr *SkillRegistry) GetSkill(name string) (Skill, bool) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()
	skill, exists := sr.skills[name]
	return skill, exists
}

// ExportAllTools 将所有注册的技能导出为 OpenAI Tool 格式
// 返回的切片可以直接用于 OpenAI API 的 tools 参数
func (sr *SkillRegistry) ExportAllTools() []map[string]interface{} {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	tools := make([]map[string]interface{}, 0, len(sr.skills))
	for _, skill := range sr.skills {
		// 解析 Schema 字符串为 map
		var schemaMap map[string]interface{}
		if err := json.Unmarshal([]byte(skill.Schema()), &schemaMap); err != nil {
			// 如果 Schema 解析失败，跳过该技能
			continue
		}

		tool := map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        skill.Name(),
				"description": skill.Description(),
				"parameters":  schemaMap,
			},
		}
		tools = append(tools, tool)
	}
	return tools
}

// ListSkills 返回所有已注册技能的名称列表
func (sr *SkillRegistry) ListSkills() []string {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	names := make([]string, 0, len(sr.skills))
	for name := range sr.skills {
		names = append(names, name)
	}
	return names
}

// RemoveSkill 移除指定名称的技能
func (sr *SkillRegistry) RemoveSkill(name string) {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	delete(sr.skills, name)
}

// Clear 清空所有技能
func (sr *SkillRegistry) Clear() {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	sr.skills = make(map[string]Skill)
}
