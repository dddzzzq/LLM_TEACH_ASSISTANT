package tools

import (
	"encoding/json"

	"grading-gateway/internal/models"
)

// FormatSubmission 将数据库模型转换为前端 Vue 兼容的 JSON 格式
func FormatSubmission(sub models.Submission) map[string]interface{} {
	var plag []map[string]interface{}
	if sub.PlagiarismReport != "" && sub.PlagiarismReport != "[]" {
		json.Unmarshal([]byte(sub.PlagiarismReport), &plag)
	} else {
		plag = make([]map[string]interface{}, 0)
	}

	var aigc map[string]interface{}
	if sub.AIGCReport != "" && sub.AIGCReport != "{}" {
		json.Unmarshal([]byte(sub.AIGCReport), &aigc)
	}

	var match map[string]interface{}
	if sub.CodeDocMatchReport != "" && sub.CodeDocMatchReport != "{}" {
		json.Unmarshal([]byte(sub.CodeDocMatchReport), &match)
	}

	return map[string]interface{}{
		"material_batch_id": sub.MaterialBatchID, "grading_job_id": sub.GradingJobID,
		"id":                    sub.ID,
		"student_id":            sub.StudentID,
		"student_name":          sub.StudentName,
		"score":                 sub.Score,
		"feedback":              sub.Feedback,
		"merged_content":        sub.MergeContent,
		"plagiarism_reports":    plag,
		"aigc_report":           aigc,
		"code_doc_match_report": match,
		"assignment_id":         sub.AssignmentID,
		"is_human_reviewed":     sub.IsHumanReviewed,
		"human_feedback":        sub.HumanFeedback,
		"human_score":           sub.HumanScore,
	}
}
