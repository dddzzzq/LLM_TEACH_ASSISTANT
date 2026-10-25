package handlers

import (
	"context"
	"strconv"
	"time"

	"gorm.io/gorm"
	"grading-gateway/internal/models"
	"grading-gateway/internal/schemas"
)

// assignmentSummaries uses bounded aggregate queries for the already-authorized
// assignments. No student answers, paths or manifests are loaded into the index.
func assignmentSummaries(ctx context.Context, db *gorm.DB, ids []uint) (map[uint]schemas.AssignmentSummary, error) {
	out := make(map[uint]schemas.AssignmentSummary, len(ids))
	if len(ids) == 0 {
		return out, nil
	}
	type submissionCount struct {
		AssignmentID uint
		Graded       int
		Reviewed     int
		Failed       int
	}
	var counts []submissionCount
	if err := db.WithContext(ctx).Model(&models.Submission{}).Select(`assignment_id,
        COUNT(DISTINCT CASE WHEN score >= 0 OR (is_human_reviewed = true AND human_score >= 0) THEN NULLIF(student_id, '') END) AS graded,
        COUNT(DISTINCT CASE WHEN is_human_reviewed = true THEN NULLIF(student_id, '') END) AS reviewed,
        SUM(CASE WHEN score < 0 AND NOT (is_human_reviewed = true AND human_score >= 0) THEN 1 ELSE 0 END) AS failed`).
		Where("assignment_id IN ?", ids).Group("assignment_id").Scan(&counts).Error; err != nil {
		return nil, err
	}
	for _, id := range ids {
		out[id] = schemas.AssignmentSummary{Status: "EMPTY"}
	}
	for _, count := range counts {
		s := out[count.AssignmentID]
		s.GradedCount, s.ReviewedCount, s.FailedCount = count.Graded, count.Reviewed, count.Failed
		out[count.AssignmentID] = s
	}
	var batches []struct {
		AssignmentID                       uint
		Status, GradingState, GradingJobID string
		IssueCount                         int
		UpdatedAt                          time.Time
	}
	if err := db.WithContext(ctx).Model(&models.MaterialBatch{}).Select("assignment_id,status,grading_state,grading_job_id,issue_count,updated_at").Where("assignment_id IN ?", ids).Scan(&batches).Error; err != nil {
		return nil, err
	}
	type stateFlags struct{ preparing, grading, attention, ready bool }
	flags := map[uint]stateFlags{}
	currentJobs := map[string]bool{}
	latest := func(s *schemas.AssignmentSummary, stamp time.Time) {
		if !stamp.IsZero() && (s.UpdatedAt == nil || stamp.After(*s.UpdatedAt)) {
			copy := stamp
			s.UpdatedAt = &copy
		}
	}
	for _, batch := range batches {
		if batch.GradingJobID != "" {
			currentJobs[batch.GradingJobID] = true
		}
		s, f := out[batch.AssignmentID], flags[batch.AssignmentID]
		s.BatchCount++
		s.IssueCount += batch.IssueCount
		f.preparing = f.preparing || batch.Status == "PENDING" || batch.Status == "PROCESSING"
		f.attention = f.attention || batch.Status == "NEEDS_REVIEW" || batch.Status == "FAILED" || batch.Status == "PARTIAL_SUCCESS"
		f.ready = f.ready || (batch.Status == "READY" && batch.GradingJobID == "")
		latest(&s, batch.UpdatedAt)
		out[batch.AssignmentID], flags[batch.AssignmentID] = s, f
	}
	refs := make([]string, 0, len(ids))
	for _, id := range ids {
		refs = append(refs, strconv.FormatUint(uint64(id), 10))
	}
	var jobs []struct {
		ID, ReferenceID, MaterialBatchID string
		Status                           models.AsyncJobStatus
		UpdatedAt                        time.Time
	}
	if err := db.WithContext(ctx).Model(&models.AsyncJob{}).Select("id,reference_id,material_batch_id,status,updated_at").Where("job_type = ? AND reference_id IN ?", models.JobTypeHomework, refs).Order("updated_at DESC, id DESC").Scan(&jobs).Error; err != nil {
		return nil, err
	}
	seenLegacy := map[string]bool{}
	for _, job := range jobs {
		id, err := strconv.ParseUint(job.ReferenceID, 10, 32)
		if err != nil {
			continue
		}
		s, ok := out[uint(id)]
		if !ok {
			continue
		}
		f := flags[uint(id)]
		latest(&s, job.UpdatedAt)
		// Retried jobs remain in history. Only each batch's current attempt (or
		// the latest legacy job without a batch) describes actionable state.
		if job.MaterialBatchID != "" && !currentJobs[job.ID] {
			out[uint(id)] = s
			continue
		}
		if job.MaterialBatchID == "" {
			if seenLegacy[job.ReferenceID] {
				out[uint(id)] = s
				continue
			}
			seenLegacy[job.ReferenceID] = true
		}
		f.grading = f.grading || job.Status == models.JobStatusPending || job.Status == models.JobStatusProcessing
		f.attention = f.attention || job.Status == models.JobStatusFailed
		out[uint(id)], flags[uint(id)] = s, f
	}
	for id, s := range out {
		f := flags[id]
		switch {
		case f.grading:
			s.Status = "GRADING"
		case f.preparing:
			s.Status = "PREPARING"
		case f.attention || s.FailedCount > 0:
			s.Status = "ATTENTION"
		case f.ready:
			s.Status = "READY"
		case s.GradedCount > 0:
			s.Status = "GRADED"
		}
		out[id] = s
	}
	return out, nil
}
