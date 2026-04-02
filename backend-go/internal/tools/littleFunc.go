package tools

import (
	"encoding/json"
	"grading-gateway/internal/models"
	"path/filepath"
	"strings"
)

func IsIgnoredFile(filePath string) bool {
	filePath = filepath.ToSlash(filePath)
	parts := strings.Split(filePath, "/")
	filename := filepath.Base(filePath)

	for _, part := range parts {
		if ignoredDirs[part] {
			return true
		}
	}
	if strings.HasPrefix(filename, ".") {
		return true
	}

	ext := strings.ToLower(filepath.Ext(filename))
	if !allowedExtensions[ext] {
		return true
	}

	// 过滤系统生成的无关文件
	if ext == ".doc" || ext == ".docx" {
		if strings.Count(filename, "-") >= 3 && strings.Contains(filename, "学院") {
			return true
		}
	}
	return false
}

func IsLikelyText(content []byte) bool {
	if len(content) == 0 {
		return true
	}
	nonPrintable := 0
	checkLen := len(content)
	if checkLen > 1000 {
		checkLen = 1000
	}

	for _, b := range content[:checkLen] {
		if b < 32 && b != '\n' && b != '\r' && b != '\t' {
			nonPrintable++
		}
	}
	return float64(nonPrintable)/float64(checkLen) < 0.2
}

// formatSubmission 将数据库模型转换为前端 Vue 兼容的 JSON 格式
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
