package models

import "time"

// ToolDefinition stores tool availability, role policy and description overrides.
// SchemaJSON and ImplKey are legacy metadata; executors and schemas come from the registry.
type ToolDefinition struct {
	ID           uint      `json:"id" gorm:"primaryKey"`
	Name         string    `json:"name" gorm:"type:varchar(128);uniqueIndex;not null"` // tool/function name
	ImplKey      string    `json:"impl_key" gorm:"type:varchar(128);not null"`         // 历史元数据，不用于动态选择执行器
	Description  string    `json:"description" gorm:"type:text;not null"`
	SchemaJSON   string    `json:"schema_json" gorm:"type:longtext;not null"` // JSON Schema 字符串
	Enabled      bool      `json:"enabled" gorm:"type:boolean;default:true;not null;index"`
	AllowedRoles string    `json:"allowed_roles" gorm:"type:text;not null"` // JSON array string, e.g. ["student","teacher"]
	CreatedAt    time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt    time.Time `json:"updated_at" gorm:"autoUpdateTime"`
}

// Keep the existing table and all configured switches/roles during the naming migration.
func (ToolDefinition) TableName() string { return "skill_definitions" }

// Deprecated: use ToolDefinition. Markdown Skills are stored separately under skills/.
type SkillDefinition = ToolDefinition
