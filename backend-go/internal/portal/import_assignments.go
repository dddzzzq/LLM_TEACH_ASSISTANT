package portal

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/cache"
	"grading-gateway/internal/grading"
	"grading-gateway/internal/models"
)

const DefaultHomeworkRubric = `{"正确性":{"description":"符合官网题目要求，结果正确","score":60},"完成度":{"description":"覆盖要求的功能，提交材料齐全","score":20},"表达与质量":{"description":"结构清晰，说明充分，格式规范","score":20}}`

// One platform homework owns one local assignment. Class is an execution scope,
// never part of the assignment identity. Caller holds the connection lock.
func ensureImportedAssignment(tx *gorm.DB, c *models.PortalConnection, course models.PortalCourse, work *models.PortalHomework) (models.Assignment, error) {
	var a models.Assignment
	var binding models.PortalAssignmentBinding
	id := stableID(c.ID, "binding", work.ID+":")
	err := tx.First(&binding, "id = ? AND owner_id = ?", id, c.OwnerID).Error
	if err == nil {
		err = tx.First(&a, "id = ? AND owner_id = ?", binding.AssignmentID, c.OwnerID).Error
	}
	if errors.Is(err, gorm.ErrRecordNotFound) {
		a = models.Assignment{OwnerID: c.OwnerID, CourseName: course.Title, ClassName: "全部班级", TaskName: work.Title, Rubric: DefaultHomeworkRubric, RequirementsStatus: "PENDING", PortalConnectionID: c.ID, PortalHomeworkID: work.ID}
		if err = tx.Create(&a).Error; err != nil {
			return a, err
		}
		binding = models.PortalAssignmentBinding{ID: id, OwnerID: c.OwnerID, ConnectionID: c.ID, HomeworkID: work.ID, AssignmentID: a.ID}
		if err = tx.Save(&binding).Error; err != nil {
			return a, err
		}
	} else if err != nil {
		return a, err
	}
	updates := map[string]any{"portal_connection_id": c.ID, "portal_homework_id": work.ID, "course_name": course.Title, "task_name": work.Title, "class_name": "全部班级"}
	if a.RequirementsStatus == "" {
		updates["requirements_status"] = "PENDING"
		a.RequirementsStatus = "PENDING"
	}
	if strings.TrimSpace(a.Rubric) == "" {
		updates["rubric"] = DefaultHomeworkRubric
		a.Rubric = DefaultHomeworkRubric
	}
	if err = tx.Model(&a).Updates(updates).Error; err != nil {
		return a, err
	}
	a.PortalConnectionID, a.PortalHomeworkID, a.CourseName, a.TaskName, a.ClassName = c.ID, work.ID, course.Title, work.Title, "全部班级"
	work.AssignmentID = a.ID
	if err = tx.Model(&models.PortalHomework{}).Where("id = ?", work.ID).Update("assignment_id", a.ID).Error; err != nil {
		return a, err
	}
	return a, nil
}

func (r *Runner) reconcileImportedAssignments(ctx context.Context) {
	var works []models.PortalHomework
	if r.Service.DB.WithContext(ctx).Joins("JOIN portal_connections ON portal_connections.id = portal_homeworks.connection_id AND portal_connections.active_snapshot_id = portal_homeworks.snapshot_id").Where("portal_homeworks.assignment_id = 0").Limit(100).Find(&works).Error != nil {
		return
	}
	for _, work := range works {
		_ = r.Service.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
			var c models.PortalConnection
			if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&c, "id = ?", work.ConnectionID).Error; err != nil {
				return err
			}
			if work.SnapshotID != c.ActiveSnapshotID {
				return nil
			}
			// Existing catalogs gain local records without silently launching
			// browser work for every previously saved teacher account on upgrade.
			if err := tx.Model(&c).Update("imports_paused", true).Error; err != nil {
				return err
			}
			var course models.PortalCourse
			if err := tx.First(&course, "id = ?", work.CourseID).Error; err != nil {
				return err
			}
			a, err := ensureImportedAssignment(tx, &c, course, &work)
			if err != nil {
				return err
			}
			var detail models.PortalHomeworkDetail
			if tx.First(&detail, "homework_id = ? AND owner_id = ? AND snapshot_id = ? AND profile_generation = ? AND complete = true", work.ID, c.OwnerID, c.ActiveSnapshotID, c.ProfileGeneration).Error == nil && strings.TrimSpace(detail.Question) != "" {
				return tx.Model(&a).Updates(map[string]any{"question": detail.Question, "requirements_status": "READY"}).Error
			}
			return nil
		})
	}
}

// Continue the existing detail skill one homework at a time. Pending state is
// durable, so restarting the backend resumes the remaining imports.
func (r *Runner) tickImportedRequirements(ctx context.Context) {
	var candidates []struct {
		OwnerID              uint
		ID, ActiveSnapshotID string
	}
	if r.Service.DB.WithContext(ctx).Model(&models.PortalConnection{}).Where("status = ? AND active_operation_id = '' AND active_snapshot_id <> '' AND imports_paused = false", models.PortalConnectionReady).Find(&candidates).Error != nil {
		return
	}
	for _, c := range candidates {
		var work models.PortalHomework
		err := r.Service.DB.WithContext(ctx).Joins("JOIN assignments ON assignments.id = portal_homeworks.assignment_id AND assignments.owner_id = ?", c.OwnerID).
			Where("portal_homeworks.connection_id = ? AND portal_homeworks.snapshot_id = ? AND assignments.requirements_status = ?", c.ID, c.ActiveSnapshotID, "PENDING").Order("portal_homeworks.course_id,portal_homeworks.id").First(&work).Error
		if err == nil {
			_, _ = r.Service.QueueHomeworkDetail(ctx, c.OwnerID, work.ID, c.ActiveSnapshotID)
		}
	}
}

func (s *Service) AssignmentPortal(ctx context.Context, owner, assignmentID uint) (map[string]any, error) {
	var a models.Assignment
	if s.DB.WithContext(ctx).First(&a, "id = ? AND owner_id = ?", assignmentID, owner).Error != nil {
		return nil, ErrNotFound
	}
	if a.PortalHomeworkID == "" {
		return map[string]any{"linked": false}, nil
	}
	c, err := s.connection(ctx, owner, false)
	if err != nil || a.PortalConnectionID != c.ID {
		return nil, ErrNotFound
	}
	var work models.PortalHomework
	if s.DB.WithContext(ctx).First(&work, "id = ? AND connection_id = ? AND snapshot_id = ? AND assignment_id = ?", a.PortalHomeworkID, c.ID, c.ActiveSnapshotID, a.ID).Error != nil {
		return map[string]any{"linked": true, "available": false}, nil
	}
	catalog, err := s.Homeworks(ctx, owner, work.CourseID)
	if err != nil {
		return nil, err
	}
	catalog["linked"], catalog["available"], catalog["homework"], catalog["requirements_status"] = true, true, work, a.RequirementsStatus
	catalog["question"] = a.Question
	delete(catalog, "homeworks")
	if c.ActiveOperationID != "" {
		var op models.PortalOperation
		if s.DB.First(&op, "id = ? AND owner_id = ? AND type = ?", c.ActiveOperationID, owner, "HOMEWORK_DETAIL").Error == nil {
			var target detailTarget
			if json.Unmarshal([]byte(op.PayloadJSON), &target) == nil && target.HomeworkID == work.ID {
				catalog["operation"] = op
			}
		}
	}
	return catalog, nil
}

func (s *Service) SaveAssignmentRubric(ctx context.Context, owner, id uint, rubric json.RawMessage) error {
	if len(rubric) > 100000 || len(grading.HomeworkMissing(models.Assignment{Question: "task", Rubric: string(rubric)})) > 0 {
		return errors.New("评分标准必须为非空 JSON 对象或数组")
	}
	result := s.DB.WithContext(ctx).Model(&models.Assignment{}).Where("id = ? AND owner_id = ?", id, owner).Update("rubric", string(rubric))
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		var count int64
		if s.DB.Model(&models.Assignment{}).Where("id = ? AND owner_id = ?", id, owner).Count(&count).Error != nil || count == 0 {
			return ErrNotFound
		}
	}
	_ = cache.InvalidateAssignmentCache(ctx, id)
	return nil
}

func (s *Service) GradeAssignmentClasses(ctx context.Context, owner, id uint, classes []string, key string) (models.PortalWorkflow, error) {
	var a models.Assignment
	if s.DB.WithContext(ctx).First(&a, "id = ? AND owner_id = ?", id, owner).Error != nil {
		return models.PortalWorkflow{}, ErrNotFound
	}
	if a.PortalHomeworkID == "" {
		return models.PortalWorkflow{}, errors.New("该作业未关联官网作业")
	}
	c, err := s.connection(ctx, owner, false)
	if err != nil {
		return models.PortalWorkflow{}, err
	}
	if len(classes) == 0 {
		return models.PortalWorkflow{}, errors.New("请选择要批改的班级")
	}
	items := make([]Selection, 0, len(classes))
	for _, class := range classes {
		if class == "" {
			return models.PortalWorkflow{}, errors.New("班级标识不能为空")
		}
		items = append(items, Selection{HomeworkID: a.PortalHomeworkID, ClassID: class})
	}
	return s.CreateWorkflow(ctx, owner, WorkflowRequest{Source: "portal", SnapshotID: c.ActiveSnapshotID, Items: items, UseSavedAssignment: true}, key)
}

type ImportSummary struct {
	Total     int `json:"total"`
	Pending   int `json:"pending"`
	Reading   int `json:"reading"`
	Ready     int `json:"ready"`
	Attention int `json:"attention"`
}

func (s *Service) importSummary(ctx context.Context, c models.PortalConnection) ImportSummary {
	var rows []struct {
		RequirementsStatus string
		Count              int
	}
	summary := ImportSummary{}
	err := s.DB.WithContext(ctx).Model(&models.PortalHomework{}).Select("assignments.requirements_status, COUNT(*) AS count").Joins("JOIN assignments ON assignments.id = portal_homeworks.assignment_id AND assignments.owner_id = ?", c.OwnerID).Where("portal_homeworks.connection_id = ? AND portal_homeworks.snapshot_id = ?", c.ID, c.ActiveSnapshotID).Group("assignments.requirements_status").Scan(&rows).Error
	if err != nil {
		return summary
	}
	for _, row := range rows {
		summary.Total += row.Count
		switch row.RequirementsStatus {
		case "PENDING":
			summary.Pending += row.Count
		case "READING":
			summary.Reading += row.Count
		case "READY":
			summary.Ready += row.Count
		default:
			summary.Attention += row.Count
		}
	}
	return summary
}

func (s *Service) PauseImports(ctx context.Context, owner uint, paused bool) error {
	c, err := s.connection(ctx, owner, false)
	if err != nil {
		return ErrNotFound
	}
	return s.DB.WithContext(ctx).Model(&c).Update("imports_paused", paused).Error
}
