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
	ID        string `json:"id"`
	Name      string `json:"name"`
	Source    string `json:"source"`
	CanGrade  bool   `json:"can_grade"`
	CanImport bool   `json:"can_import"`
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
	if err := db.WithContext(ctx).Select("id,portal_class_id,class_name").Where("assignment_id = ?", a.ID).Find(&batches).Error; err != nil {
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
	return classes, nil
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
