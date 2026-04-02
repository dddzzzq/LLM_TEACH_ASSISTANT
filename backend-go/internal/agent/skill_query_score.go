package agent

import (
	"encoding/json"
	"fmt"
	"strings"

	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

// QueryStudentScoreSkill 查询学生成绩的技能
type QueryStudentScoreSkill struct{}

// Name 返回技能名称
func (s *QueryStudentScoreSkill) Name() string {
	return "query_student_score"
}

// Description 返回技能描述
func (s *QueryStudentScoreSkill) Description() string {
	return "当需要查询某个学生的历史作业和试卷得分、评语时调用此工具。"
}

// Schema 返回 JSON Schema 字符串
func (s *QueryStudentScoreSkill) Schema() string {
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
func (s *QueryStudentScoreSkill) Execute(args string) (string, error) {
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
	if err := database.DB.Where("student_id = ?", studentID).Find(&submissions).Error; err != nil {
		// 数据库查询错误，返回友好提示
		return fmt.Sprintf("查询学生 %s 的作业记录时发生错误，请稍后重试。", studentID), nil
	}

	// 查询试卷记录
	var studentExams []models.StudentExam
	if err := database.DB.Preload("Answers").Preload("Answers.Question").
		Preload("Report").Where("student_id = ?", studentID).Find(&studentExams).Error; err != nil {
		// 数据库查询错误，返回友好提示
		return fmt.Sprintf("查询学生 %s 的试卷记录时发生错误，请稍后重试。", studentID), nil
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

	// 试卷成绩
	if len(studentExams) == 0 {
		result.WriteString("试卷记录：暂无\n")
	} else {
		result.WriteString("=== 试卷成绩 ===\n")
		for i, exam := range studentExams {
			result.WriteString(fmt.Sprintf("%d. 试卷ID: %d\n", i+1, exam.ExamID))

			// 总分
			totalScore := 0.0
			if exam.Report.TotalScore > 0 {
				totalScore = exam.Report.TotalScore
			} else {
				// 计算各题总分
				for _, answer := range exam.Answers {
					totalScore += answer.Score
				}
			}

			result.WriteString(fmt.Sprintf("   总分: %.2f\n", totalScore))

			// 各题详情
			if len(exam.Answers) > 0 {
				result.WriteString("   各题得分:\n")
				for _, answer := range exam.Answers {
					result.WriteString(fmt.Sprintf("     - 第%d题: %.2f分", answer.Question.QuestionNumber, answer.Score))
					if answer.Feedback != "" {
						feedback := answer.Feedback
						if len(feedback) > 100 {
							feedback = feedback[:100] + "..."
						}
						result.WriteString(fmt.Sprintf(" (评语: %s)", feedback))
					}
					result.WriteString("\n")
				}
			}

			// 整体评语
			if exam.Report.Summary != "" {
				summary := exam.Report.Summary
				if len(summary) > 200 {
					summary = summary[:200] + "..."
				}
				result.WriteString(fmt.Sprintf("   整体评语: %s\n", summary))
			}
			result.WriteString("\n")
		}
	}

	// 统计数据
	totalAssignments := len(submissions)
	totalExams := len(studentExams)
	if totalAssignments == 0 && totalExams == 0 {
		result.WriteString("该学生暂无任何成绩记录。")
	} else {
		result.WriteString(fmt.Sprintf("总计：%d 份作业，%d 份试卷。", totalAssignments, totalExams))
	}

	return result.String(), nil
}
