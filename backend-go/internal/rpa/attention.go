package rpa

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

var ErrRequirementsChanged = errors.New("附件已开始批改、已取消或尚未匹配作业，请刷新任务")
var ErrInvalidRequirements = errors.New("请填写题目要求和非空 JSON 对象形式的评分标准")

type AttentionNotice struct {
	Key             string `json:"key"`
	Kind            string `json:"kind"`
	JobID           string `json:"job_id"`
	FileID          string `json:"file_id,omitempty"`
	AssignmentID    uint   `json:"assignment_id,omitempty"`
	CourseName      string `json:"course_name"`
	AssignmentName  string `json:"assignment_name"`
	ClassName       string `json:"class_name,omitempty"`
	Reason          string `json:"reason,omitempty"`
	Message         string `json:"message"`
	QuestionMissing bool   `json:"question_missing,omitempty"`
	RubricMissing   bool   `json:"rubric_missing,omitempty"`
}

func missingRequirements(a models.Assignment) (bool, bool) {
	var rubric map[string]any
	return strings.TrimSpace(a.Question) == "", json.Unmarshal([]byte(a.Rubric), &rubric) != nil || len(rubric) == 0
}

// Attention uses the task creator's ownership and batches related files/assignments.
func Attention(userID uint) ([]AttentionNotice, error) {
	var jobs []models.RPAJob
	err := database.DB.Where(`user_id = ? AND status <> ? AND (status = ? OR EXISTS
		(SELECT 1 FROM rpa_files WHERE rpa_files.job_id = rpa_jobs.id AND grading_status = ?))`,
		userID, "CANCELLED", "WAITING_USER", "WAITING_RUBRIC").Order("created_at DESC").Limit(100).Find(&jobs).Error
	if err != nil {
		return nil, err
	}
	notices := make([]AttentionNotice, 0)
	if len(jobs) == 0 {
		return notices, nil
	}
	ids := make([]string, 0, len(jobs))
	for _, job := range jobs {
		ids = append(ids, job.ID)
	}
	var files []models.RPAFile
	if err := database.DB.Where("job_id IN ? AND grading_status = ?", ids, "WAITING_RUBRIC").Order("created_at ASC, id ASC").Find(&files).Error; err != nil {
		return nil, err
	}
	assignmentIDs := []uint{}
	for _, file := range files {
		if file.AssignmentID != 0 {
			assignmentIDs = append(assignmentIDs, file.AssignmentID)
		}
	}
	assignments := map[uint]models.Assignment{}
	if len(assignmentIDs) > 0 {
		var rows []models.Assignment
		if err := database.DB.Select("id", "question", "rubric").Where("id IN ?", assignmentIDs).Find(&rows).Error; err != nil {
			return nil, err
		}
		for _, row := range rows {
			assignments[row.ID] = row
		}
	}
	for _, job := range jobs {
		base := AttentionNotice{JobID: job.ID, CourseName: job.CourseName, AssignmentName: job.AssignmentName}
		var state struct {
			ControlMode   string `json:"control_mode"`
			WaitingReason string `json:"waiting_reason"`
			ControlEpoch  int64  `json:"control_epoch"`
		}
		_ = json.Unmarshal([]byte(job.StateJSON), &state)
		if job.Status == "WAITING_USER" && state.ControlMode == "HUMAN" && state.WaitingReason != "USER_TAKEOVER" {
			n := base
			n.Kind, n.Reason, n.Message = "human", state.WaitingReason, job.Message
			n.Key = fmt.Sprintf("%s:human:%s:%d", job.ID, state.WaitingReason, state.ControlEpoch)
			notices = append(notices, n)
		}
		for _, file := range files {
			if file.JobID != job.ID {
				continue
			}
			n := base
			n.FileID, n.AssignmentID, n.ClassName, n.Message = file.ID, file.AssignmentID, file.ClassName, file.Message
			if strings.TrimSpace(file.ClassName) == "" {
				n.Kind, n.Key = "class", job.ID+":class:"+file.ID
			} else if assignment, ok := assignments[file.AssignmentID]; ok {
				n.QuestionMissing, n.RubricMissing = missingRequirements(assignment)
				if !n.QuestionMissing && !n.RubricMissing {
					continue
				}
				n.Kind = "rubric"
				n.Key = fmt.Sprintf("%s:rubric:%s:%d:%t:%t", job.ID, file.ID, file.AssignmentID, n.QuestionMissing, n.RubricMissing)
			} else {
				continue
			} // The scheduler has not yet matched the local assignment.
			notices = append(notices, n)
		}
	}
	return notices, nil
}

type Requirements struct {
	AssignmentID uint           `json:"assignment_id"`
	Question     string         `json:"question"`
	Rubric       map[string]any `json:"rubric"`
}

func FileRequirements(jobID, fileID string) (Requirements, error) {
	var file models.RPAFile
	if err := database.DB.Where("id = ? AND job_id = ?", fileID, jobID).First(&file).Error; err != nil {
		return Requirements{}, err
	}
	if file.GradingStatus != "WAITING_RUBRIC" || file.AssignmentID == 0 {
		return Requirements{}, ErrRequirementsChanged
	}
	var a models.Assignment
	if err := database.DB.First(&a, file.AssignmentID).Error; err != nil {
		return Requirements{}, err
	}
	out := Requirements{AssignmentID: a.ID, Question: a.Question, Rubric: map[string]any{}}
	_ = json.Unmarshal([]byte(a.Rubric), &out.Rubric)
	return out, nil
}

func SupplementRequirements(jobID, fileID, question string, rubric map[string]any) error {
	question = strings.TrimSpace(question)
	if question == "" || len(rubric) == 0 {
		return ErrInvalidRequirements
	}
	encoded, err := json.Marshal(rubric)
	if err != nil {
		return ErrInvalidRequirements
	}
	return database.DB.Transaction(func(tx *gorm.DB) error {
		var file models.RPAFile
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("id = ? AND job_id = ?", fileID, jobID).First(&file).Error; err != nil {
			return err
		}
		var job models.RPAJob
		if err := tx.First(&job, "id = ?", jobID).Error; err != nil {
			return err
		}
		if job.Status == "CANCELLED" || file.GradingStatus != "WAITING_RUBRIC" || file.AssignmentID == 0 {
			return ErrRequirementsChanged
		}
		var a models.Assignment
		if err := tx.First(&a, file.AssignmentID).Error; err != nil {
			return err
		}
		if a.CourseName != job.CourseName || a.TaskName != job.AssignmentName || a.ClassName != file.ClassName {
			return ErrRequirementsChanged
		}
		return tx.Model(&a).Updates(map[string]any{"question": question, "rubric": string(encoded)}).Error
	})
}
