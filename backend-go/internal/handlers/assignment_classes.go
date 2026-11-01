package handlers

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

const unassignedClassID = "__unassigned__"

// AssignmentClass describes a read scope. Only classes in the current portal
// catalog can be sent back as a portal execution/import target.
type AssignmentClass struct {
	ID             string `json:"id"`
	Name           string `json:"name"`
	Source         string `json:"source"`
	CanGrade       bool   `json:"can_grade"`
	CanImport      bool   `json:"can_import"`
	MaterialStatus string `json:"material_status"`
	GradingStatus  string `json:"grading_status"`
	IssueCount     int    `json:"issue_count"`
	GradedCount    int    `json:"graded_count"`
	ReviewedCount  int    `json:"reviewed_count"`
}

func namedClass(name string) bool {
	switch strings.TrimSpace(name) {
	case "", "全部班级", "所有班级", "未分班", "未分班材料", "历史未分班成绩":
		return false
	default:
		return true
	}
}

func localClass(name string) AssignmentClass {
	if !namedClass(name) {
		return AssignmentClass{ID: unassignedClassID, Name: "未分班", Source: "unassigned"}
	}
	return AssignmentClass{ID: "local:" + base64.RawURLEncoding.EncodeToString([]byte(name)), Name: name, Source: "local"}
}

func batchClass(batch models.MaterialBatch) AssignmentClass {
	if batch.PortalClassID != "" {
		name := batch.ClassName
		if !namedClass(name) {
			name = "历史班级 " + batch.PortalClassID
		}
		return AssignmentClass{ID: batch.PortalClassID, Name: name, Source: "history"}
	}
	return localClass(batch.ClassName)
}

// Legacy submissions without a material batch retain the assignment's original
// local class. A portal assignment's aggregate label never assigns a class.
func legacyAssignmentClass(a models.Assignment) AssignmentClass {
	if a.PortalHomeworkID == "" {
		return localClass(a.ClassName)
	}
	return localClass("")
}

func assignmentClassCatalog(ctx context.Context, db *gorm.DB, a models.Assignment) ([]AssignmentClass, error) {
	entries := map[string]AssignmentClass{}
	if a.PortalHomeworkID != "" && a.PortalConnectionID != "" {
		var connection models.PortalConnection
		err := db.WithContext(ctx).First(&connection, "id = ? AND owner_id = ?", a.PortalConnectionID, a.OwnerID).Error
		if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, err
		}
		if err == nil && connection.ActiveSnapshotID != "" {
			var homework models.PortalHomework
			err = db.WithContext(ctx).First(&homework, "id = ? AND connection_id = ? AND snapshot_id = ? AND assignment_id = ?", a.PortalHomeworkID, connection.ID, connection.ActiveSnapshotID, a.ID).Error
			if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
				return nil, err
			}
			if err == nil {
				var classes []models.PortalCourseClass
				if err = db.WithContext(ctx).Where("course_id = ? AND snapshot_id = ?", homework.CourseID, connection.ActiveSnapshotID).Find(&classes).Error; err != nil {
					return nil, err
				}
				for _, class := range classes {
					entries[class.ID] = AssignmentClass{ID: class.ID, Name: class.Name, Source: "portal", CanGrade: true, CanImport: true}
				}
			}
		}
	}
	var batches []models.MaterialBatch
	if err := db.WithContext(ctx).Where("assignment_id = ?", a.ID).Find(&batches).Error; err != nil {
		return nil, err
	}
	for _, batch := range batches {
		class := batchClass(batch)
		if _, exists := entries[class.ID]; !exists {
			entries[class.ID] = class
		}
	}
	var legacyCount int64
	if err := db.WithContext(ctx).Model(&models.Submission{}).Where("assignment_id = ? AND NOT EXISTS (SELECT 1 FROM material_batches WHERE material_batches.id = submissions.material_batch_id AND material_batches.assignment_id = submissions.assignment_id)", a.ID).Count(&legacyCount).Error; err != nil {
		return nil, err
	}
	if legacyCount > 0 || len(entries) == 0 {
		class := legacyAssignmentClass(a)
		entries[class.ID] = class
	}
	classes := make([]AssignmentClass, 0, len(entries))
	for _, class := range entries {
		classes = append(classes, class)
	}
	sort.Slice(classes, func(i, j int) bool {
		if classes[i].ID == unassignedClassID || classes[j].ID == unassignedClassID {
			return classes[j].ID == unassignedClassID && classes[i].ID != unassignedClassID
		}
		if classes[i].Name == classes[j].Name {
			return classes[i].ID < classes[j].ID
		}
		return classes[i].Name < classes[j].Name
	})
	if err := summarizeAssignmentClasses(ctx, db, a, classes, batches); err != nil {
		return nil, err
	}
	return classes, nil
}

func summarizeAssignmentClasses(ctx context.Context, db *gorm.DB, a models.Assignment, classes []AssignmentClass, batches []models.MaterialBatch) error {
	byID := make(map[string]*AssignmentClass, len(classes))
	for i := range classes {
		classes[i].MaterialStatus = "EMPTY"
		classes[i].GradingStatus = "NOT_STARTED"
		byID[classes[i].ID] = &classes[i]
	}
	batchClasses := map[string]string{}
	classBatches := map[string][]models.MaterialBatch{}
	jobIDs := []string{}
	for _, batch := range batches {
		classID := batchClass(batch).ID
		batchClasses[batch.ID] = classID
		classBatches[classID] = append(classBatches[classID], batch)
		if class := byID[classID]; class != nil {
			class.IssueCount += batch.IssueCount
		}
		if batch.GradingJobID != "" {
			jobIDs = append(jobIDs, batch.GradingJobID)
		}
	}
	var jobRows []struct {
		ID     string
		Status models.AsyncJobStatus
	}
	if len(jobIDs) > 0 {
		if err := db.WithContext(ctx).Model(&models.AsyncJob{}).Select("id,status").Where("id IN ?", jobIDs).Scan(&jobRows).Error; err != nil {
			return err
		}
	}
	jobs := map[string]models.AsyncJobStatus{}
	for _, job := range jobRows {
		jobs[job.ID] = job.Status
	}
	var counts []struct {
		MaterialBatchID string
		Graded          int
		Reviewed        int
	}
	if err := db.WithContext(ctx).Model(&models.Submission{}).Select(`material_batch_id,
		SUM(CASE WHEN score >= 0 OR (is_human_reviewed = true AND human_score >= 0) THEN 1 ELSE 0 END) AS graded,
		SUM(CASE WHEN is_human_reviewed = true THEN 1 ELSE 0 END) AS reviewed`).Where("assignment_id = ?", a.ID).Group("material_batch_id").Scan(&counts).Error; err != nil {
		return err
	}
	legacyID := legacyAssignmentClass(a).ID
	for _, count := range counts {
		classID, ok := batchClasses[count.MaterialBatchID]
		if !ok {
			classID = legacyID
		}
		if class := byID[classID]; class != nil {
			class.GradedCount += count.Graded
			class.ReviewedCount += count.Reviewed
		}
	}
	for classID, rows := range classBatches {
		class := byID[classID]
		if class == nil {
			continue
		}
		preparing, materialAttention, grading, gradingAttention, ready := false, false, false, false, false
		for _, batch := range rows {
			preparing = preparing || batch.Status == "PENDING" || batch.Status == "PROCESSING"
			materialAttention = materialAttention || batch.Status == "FAILED" || batch.Status == "NEEDS_REVIEW" || batch.Status == "PARTIAL_SUCCESS" || batch.IssueCount > 0
			gradingAttention = gradingAttention || batch.GradingState == "WAITING_REQUIREMENTS" || jobs[batch.GradingJobID] == models.JobStatusFailed
			grading = grading || batch.GradingState == "PENDING" || batch.GradingState == "PROCESSING" || batch.GradingState == "REQUESTED" || jobs[batch.GradingJobID] == models.JobStatusPending || jobs[batch.GradingJobID] == models.JobStatusProcessing
			ready = ready || (batch.Status == "READY" && batch.GradingJobID == "")
		}
		switch {
		case materialAttention:
			class.MaterialStatus = "ATTENTION"
		case preparing:
			class.MaterialStatus = "PROCESSING"
		default:
			class.MaterialStatus = "READY"
		}
		switch {
		case grading:
			class.GradingStatus = "GRADING"
		case gradingAttention:
			class.GradingStatus = "ATTENTION"
		case class.GradedCount > 0:
			class.GradingStatus = "GRADED"
		case ready:
			class.GradingStatus = "READY"
		}
	}
	return nil
}

func GetAssignmentClasses(c *gin.Context) {
	id, err := strconv.ParseUint(c.Param("id"), 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "无效的作业ID"})
		return
	}
	var a models.Assignment
	if err = database.DB.WithContext(c.Request.Context()).First(&a, id).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"detail": "作业不存在"})
		return
	}
	classes, err := assignmentClassCatalog(c.Request.Context(), database.DB, a)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "班级列表暂时无法读取，请重试"})
		return
	}
	c.Header("Cache-Control", "no-store")
	c.JSON(http.StatusOK, classes)
}

// materialClassScope applies to material_batches, never to a whole assignment.
// Invalid encoded local IDs match nothing rather than widening the scope.
func materialClassScope(q *gorm.DB, classID string) *gorm.DB {
	if classID == unassignedClassID {
		return q.Where("(portal_class_id = '' OR portal_class_id IS NULL) AND (class_name IS NULL OR TRIM(class_name) IN ?)", []string{"", "全部班级", "所有班级", "未分班", "未分班材料", "历史未分班成绩"})
	}
	if strings.HasPrefix(classID, "local:") {
		name, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(classID, "local:"))
		if err != nil || !namedClass(string(name)) {
			return q.Where("1 = 0")
		}
		return q.Where("(portal_class_id = '' OR portal_class_id IS NULL) AND class_name = ?", string(name))
	}
	return q.Where("portal_class_id = ?", classID)
}

func submissionClassScope(q *gorm.DB, db *gorm.DB, assignmentID string, classID string) *gorm.DB {
	if classID == "" {
		return q
	}
	batches := materialClassScope(db.Model(&models.MaterialBatch{}).Select("id").Where("assignment_id = ?", assignmentID), classID)
	if classID != unassignedClassID && !strings.HasPrefix(classID, "local:") {
		return q.Where("material_batch_id IN (?)", batches)
	}
	var a models.Assignment
	if db.Select("id,class_name,portal_homework_id").First(&a, "id = ?", assignmentID).Error != nil || legacyAssignmentClass(a).ID != classID {
		return q.Where("material_batch_id IN (?)", batches)
	}
	return q.Where("(material_batch_id IN (?) OR NOT EXISTS (SELECT 1 FROM material_batches WHERE material_batches.id = submissions.material_batch_id AND material_batches.assignment_id = submissions.assignment_id))", batches)
}
