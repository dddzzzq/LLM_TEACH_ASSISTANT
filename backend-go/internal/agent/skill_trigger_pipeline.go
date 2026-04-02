package agent

import (
	"encoding/json"
	"fmt"

	"grading-gateway/internal/tools"
)

// TriggerPipelineSkill 触发异步批改流水线的技能
type TriggerPipelineSkill struct{}

// Name 返回技能名称
func (s *TriggerPipelineSkill) Name() string {
	return "trigger_async_pipeline"
}

// Description 返回技能描述
func (s *TriggerPipelineSkill) Description() string {
	return "当用户要求开始批改某个作业或提供了一个本地路径要求批改时调用此工具。"
}

// Schema 返回 JSON Schema 字符串
func (s *TriggerPipelineSkill) Schema() string {
	return `{
		"type": "object",
		"properties": {
			"assignment_id": {
				"type": "string",
				"description": "作业的唯一标识符"
			},
			"file_path": {
				"type": "string",
				"description": "包含学生作业的ZIP或RAR文件的本地路径"
			}
		},
		"required": ["assignment_id", "file_path"],
		"additionalProperties": false
	}`
}

// Execute 执行触发异步批改流水线的操作
func (s *TriggerPipelineSkill) Execute(args string) (string, error) {
	// 解析参数
	var params map[string]interface{}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		// 返回提示字符串，而不是 error
		return "参数解析失败，请确保提供了正确的 JSON 格式，包含 assignment_id 和 file_path 字段。示例：{\"assignment_id\": \"123\", \"file_path\": \"/path/to/submissions.zip\"}", nil
	}

	// 获取 assignment_id
	assignmentID, ok1 := params["assignment_id"].(string)
	// 获取 file_path
	filePath, ok2 := params["file_path"].(string)

	if !ok1 || assignmentID == "" || !ok2 || filePath == "" {
		return "参数 assignment_id 或 file_path 缺失或格式不正确，请提供有效的作业ID和文件路径。", nil
	}

	// 启动异步批改流水线
	go func() {
		// 注意：这里直接调用 tools.ProcessPipeline
		// 该函数内部会处理所有批改逻辑，包括解压、解析、查重、评分等
		tools.ProcessPipeline(assignmentID, filePath)
	}()

	// 立即返回友好提示
	return fmt.Sprintf("后台批改流水线已成功触发！\n作业ID: %s\n文件路径: %s\n系统正在后台处理，请稍后查看结果。", assignmentID, filePath), nil
}
