package agent

import (
	"encoding/json"
	"fmt"
	"strings"

	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

// QueryStudentScoreTool 查询学生成绩的工具
type QueryStudentScoreTool struct {
	UserID uint
	Role   string
}

// Name 返回工具名称
func (s *QueryStudentScoreTool) Name() string {
	return "query_student_score"
}

// Description 返回工具描述
func (s *QueryStudentScoreTool) Description() string {
	return "当需要查询某个学生的历史作业得分、评语时调用此工具。"
}

// Schema 返回 JSON Schema 字符串
func (s *QueryStudentScoreTool) Schema() string {
	return `{
		"type": "object",
		"properties": {
			"student_id": {
				"type": "string",
				"description": "学生的唯一标识符"
			}
		},
		"required": ["student_id"],
		"additionalProperties": false
	}`
}

// Execute 执行查询学生成绩的操作
func (s *QueryStudentScoreTool) Execute(args string) (string, error) {
	// 解析参数
	var params map[string]interface{}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		// 返回提示字符串，而不是 error
		return "参数解析失败，请确保提供了正确的 JSON 格式，包含 student_id 字段。示例：{\"student_id\": \"23009200042\"}", nil
	}

	// 获取 student_id
	studentID, ok := params["student_id"].(string)
	if !ok || studentID == "" {
		return "参数 student_id 缺失或格式不正确，请提供有效的学生 ID。", nil
	}

	// 查询作业提交记录
	var submissions []models.Submission
	if s.UserID == 0 || database.DB == nil || (s.Role != "teacher" && s.Role != "admin") {
		return "当前用户或成绩服务不可用", nil
	}
	query := database.DB.Where("student_id = ?", studentID)
	if s.Role == "teacher" {
		query = query.Where("assignment_id IN (SELECT id FROM assignments WHERE owner_id = ?)", s.UserID)
	}
	if err := query.Find(&submissions).Error; err != nil {
		// 数据库查询错误，返回友好提示
		return fmt.Sprintf("查询学生 %s 的作业记录时发生错误，请稍后重试。", studentID), nil
	}

	// 组装结果
	var result strings.Builder
	result.WriteString(fmt.Sprintf("学生 %s 的成绩查询结果：\n\n", studentID))

	// 作业成绩
	if len(submissions) == 0 {
		result.WriteString("作业记录：暂无\n")
	} else {
		result.WriteString("=== 作业成绩 ===\n")
		for i, sub := range submissions {
			result.WriteString(fmt.Sprintf("%d. 作业ID: %d\n", i+1, sub.AssignmentID))
			result.WriteString(fmt.Sprintf("   得分: %.2f\n", sub.Score))
			if sub.Feedback != "" {
				// 截断过长的评语
				feedback := sub.Feedback
				if len(feedback) > 200 {
					feedback = feedback[:200] + "..."
				}
				result.WriteString(fmt.Sprintf("   评语: %s\n", feedback))
			}
			if sub.IsHumanReviewed {
				result.WriteString(fmt.Sprintf("   教师复查评分: %.2f\n", sub.HumanScore))
				if sub.HumanFeedback != "" {
					result.WriteString(fmt.Sprintf("   教师评语: %s\n", sub.HumanFeedback))
				}
			}
			result.WriteString("\n")
		}
	}

	return result.String(), nil
}
