package portal

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/grading"
	"grading-gateway/internal/models"
	"grading-gateway/internal/rpa"
)

type Selection struct {
	HomeworkID string `json:"homework_id"`
	ClassID    string `json:"class_id,omitempty"` // Optional compatibility for existing explicit selections.
}
type WorkflowRequest struct {
	Source     string          `json:"source"`
	SnapshotID string          `json:"snapshot_id"`
	Items      []Selection     `json:"items"`
	Question   string          `json:"question"`
	Rubric     json.RawMessage `json:"rubric"`
}

func (s *Service) CreateWorkflow(ctx context.Context, owner uint, req WorkflowRequest, key string) (models.PortalWorkflow, error) {
	var flow models.PortalWorkflow
	if s.DB == nil {
		return flow, ErrUnavailable
	}
	if strings.TrimSpace(key) == "" {
		return flow, errors.New("请提供本次批改请求的 Idempotency-Key")
	}
	if req.Source != "portal" || req.SnapshotID == "" || len(req.Items) == 0 || len(req.Items) > 50 || strings.TrimSpace(req.Question) == "" || len(req.Question) > 100000 || len(req.Rubric) > 100000 || len(key) > 128 {
		return flow, errors.New("请选择官网作业并填写题目与评分标准")
	}
	if len(grading.HomeworkMissing(models.Assignment{Question: req.Question, Rubric: string(req.Rubric)})) > 0 {
		return flow, errors.New("评分标准必须为非空 JSON 对象或数组")
	}
	raw, _ := json.Marshal(req)
	sum := sha256.Sum256(raw)
	hash := hex.EncodeToString(sum[:])
	id := uuid.NewString()
	if key != "" {
		id = stableID(fmt.Sprint(owner), "workflow", key)
	}
	err := s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		ts := *s
		ts.DB = tx
		c, err := ts.connection(ctx, owner, true)
		if err != nil {
			return ErrNotFound
		}
		var prior models.PortalWorkflow
		if err = tx.First(&prior, "id = ?", id).Error; err == nil {
			if prior.OwnerID != owner || prior.RequestHash != hash {
				return errors.New("同一请求键不能用于不同批改目标")
			}
			flow = prior
			return nil
		} else if !errors.Is(err, gorm.ErrRecordNotFound) {
			return err
		}
		if c.ActiveSnapshotID != req.SnapshotID || c.Status == models.PortalConnectionDisabled {
			return errors.New("课程目录已更新或连接已断开，请重新选择")
		}
		flow = models.PortalWorkflow{ID: id, OwnerID: owner, ConnectionID: c.ID, ProfileGeneration: c.ProfileGeneration, Status: "QUEUED", RequestHash: hash}
		if err = tx.Create(&flow).Error; err != nil {
			return err
		}
		seen := map[string]bool{}
		for _, selected := range req.Items {
			identity := selected.HomeworkID + ":" + selected.ClassID
			if seen[identity] {
				return errors.New("请勿重复选择相同作业")
			}
			seen[identity] = true
			var work models.PortalHomework
			var course models.PortalCourse
			var class models.PortalCourseClass
			if tx.Where("id = ? AND connection_id = ? AND snapshot_id = ?", selected.HomeworkID, c.ID, c.ActiveSnapshotID).First(&work).Error != nil {
				return ErrNotFound
			}
			if selected.ClassID != "" && tx.Where("id = ? AND course_id = ? AND snapshot_id = ?", selected.ClassID, work.CourseID, c.ActiveSnapshotID).First(&class).Error != nil {
				return errors.New("所选班级不属于该课程")
			}
			if tx.First(&course, "id = ?", work.CourseID).Error != nil {
				return ErrNotFound
			}
			className := "全部班级"
			if selected.ClassID != "" {
				className = class.Name
			}
			a := models.Assignment{OwnerID: owner, CourseName: course.Title, ClassName: className, TaskName: work.Title, Question: req.Question, Rubric: string(req.Rubric)}
			if err = tx.Create(&a).Error; err != nil {
				return err
			}
			binding := models.PortalAssignmentBinding{ID: stableID(c.ID, "binding", identity), OwnerID: owner, ConnectionID: c.ID, HomeworkID: work.ID, ClassID: class.ID, AssignmentID: a.ID}
			if err = tx.Clauses(clause.OnConflict{UpdateAll: true}).Create(&binding).Error; err != nil {
				return err
			}
			target := rpa.PortalTarget{CourseID: course.PlatformID, HomeworkID: work.PlatformID, ClassID: class.PlatformID, ClassName: className, CourseURL: course.URL, HomeworkURL: work.URL}
			data, _ := json.Marshal(target)
			item := models.PortalWorkflowItem{ID: uuid.NewString(), WorkflowID: flow.ID, OwnerID: owner, SnapshotID: req.SnapshotID, HomeworkID: work.ID, ClassID: class.ID, AssignmentID: a.ID, CourseName: course.Title, HomeworkName: work.Title, ClassName: className, Question: req.Question, Rubric: string(req.Rubric), TargetJSON: string(data), Status: "QUEUED", Message: "等待官网附件下载"}
			if err = tx.Create(&item).Error; err != nil {
				return err
			}
		}
		return nil
	})
	return flow, err
}
func (s *Service) Workflow(ctx context.Context, owner uint, id string) (map[string]any, error) {
	var flow models.PortalWorkflow
	if s.DB == nil {
		return nil, ErrUnavailable
	}
	if s.DB.WithContext(ctx).Where("id = ? AND owner_id = ?", id, owner).First(&flow).Error != nil {
		return nil, ErrNotFound
	}
	items := []models.PortalWorkflowItem{}
	if err := s.DB.WithContext(ctx).Where("workflow_id = ? AND owner_id = ?", id, owner).Order("created_at,id").Find(&items).Error; err != nil {
		return nil, err
	}
	return map[string]any{"workflow": flow, "items": items}, nil
}
func (s *Service) CancelWorkflow(ctx context.Context, owner uint, id string) error {
	var jobs []string
	err := s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var flow models.PortalWorkflow
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("id = ? AND owner_id = ?", id, owner).First(&flow).Error; err != nil {
			return ErrNotFound
		}
		if terminal(flow.Status) || flow.Status == "PARTIAL_SUCCESS" {
			return nil
		}
		if err := tx.Model(&flow).Update("status", "CANCELLED").Error; err != nil {
			return err
		}
		if err := tx.Model(&models.PortalWorkflowItem{}).Where("workflow_id = ? AND status NOT IN ?", id, []string{"SUCCEEDED", "FAILED", "CANCELLED"}).Pluck("rpa_job_id", &jobs).Error; err != nil {
			return err
		}
		return tx.Model(&models.PortalWorkflowItem{}).Where("workflow_id = ? AND status NOT IN ?", id, []string{"SUCCEEDED", "FAILED", "CANCELLED"}).Updates(map[string]any{"status": "CANCELLED", "message": "流程已取消，已完成结果保留"}).Error
	})
	if err != nil {
		return err
	}
	for _, id := range jobs {
		if id != "" {
			if job, err := rpa.Owned(owner, id); err == nil {
				_ = rpa.CancelQueued(ctx, job)
			}
		}
	}
	return nil
}

// RetryWorkflow queues only failed/interrupted downloads. Completed grading is retained.
// Material/AI failures need explicit review on the assignment page instead of blind regrading.
func (s *Service) RetryWorkflow(ctx context.Context, owner uint, id string) error {
	if s.DB == nil {
		return ErrUnavailable
	}
	return s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		ts := *s
		ts.DB = tx
		c, err := ts.connection(ctx, owner, true)
		if err != nil {
			return ErrNotFound
		}
		if c.Status != models.PortalConnectionReady || c.ActiveOperationID != "" {
			return errors.New("请先完成教务认证或等待目录操作结束")
		}
		var flow models.PortalWorkflow
		if err = tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("id = ? AND owner_id = ? AND connection_id = ?", id, owner, c.ID).First(&flow).Error; err != nil {
			return ErrNotFound
		}
		if flow.Status == "CANCELLED" {
			return errors.New("已取消的流程请重新选择目标创建")
		}
		if flow.ProfileGeneration != c.ProfileGeneration {
			return errors.New("教务账号或浏览器配置已变更，请重新选择目标创建流程")
		}
		var items []models.PortalWorkflowItem
		if err = tx.Where("workflow_id = ? AND status = ?", id, "FAILED").Find(&items).Error; err != nil {
			return err
		}
		eligible := []string{}
		for _, item := range items {
			var job models.RPAJob
			if item.RPAJobID != "" && tx.Where("id = ? AND user_id = ? AND status IN ?", item.RPAJobID, owner, []string{"FAILED", "INTERRUPTED", "PARTIAL_SUCCESS"}).First(&job).Error == nil {
				eligible = append(eligible, item.ID)
			}
		}
		if len(eligible) == 0 {
			return errors.New("没有可重试的中断下载；材料或评分异常请在作业详情核对处理")
		}
		if err = tx.Model(&models.PortalWorkflowItem{}).Where("id IN ?", eligible).Updates(map[string]any{"status": "RETRY_QUEUED", "message": "已排队从原下载检查点重试"}).Error; err != nil {
			return err
		}
		return tx.Model(&flow).Update("status", "RUNNING").Error
	})
}
func (r *Runner) tickWorkflows(ctx context.Context) {
	var flows []models.PortalWorkflow
	if r.Service.DB.WithContext(ctx).Where("status IN ?", []string{"QUEUED", "RUNNING", "WAITING_AUTH", "WAITING_REVIEW"}).Order("created_at").Find(&flows).Error != nil {
		return
	}
	for _, flow := range flows {
		var items []models.PortalWorkflowItem
		if r.Service.DB.WithContext(ctx).Where("workflow_id = ? AND status NOT IN ?", flow.ID, []string{"SUCCEEDED", "FAILED", "CANCELLED"}).Order("created_at,id").Find(&items).Error != nil {
			continue
		}
		for _, item := range items {
			if item.Status == "RETRY_QUEUED" {
				r.retryItem(ctx, flow, item)
				continue
			}
			if item.RPAJobID == "" {
				r.startItem(ctx, flow, item)
				break
			}
			var job models.RPAJob
			if r.Service.DB.First(&job, "id = ?", item.RPAJobID).Error != nil {
				continue
			}
			status, message := "RUNNING", job.Message
			switch job.Status {
			case "FAILED", "PARTIAL_SUCCESS", "INTERRUPTED":
				status = "FAILED"
			case "CANCELLED":
				status = "CANCELLED"
			case "WAITING_USER":
				status = "WAITING_AUTH"
			case "SUCCEEDED":
				var files []models.RPAFile
				if r.Service.DB.Where("job_id = ?", job.ID).Find(&files).Error != nil {
					continue
				}
				if len(files) == 0 {
					message = "正在登记已下载附件"
					if time.Since(job.UpdatedAt) > time.Minute {
						status = "FAILED"
						message = "下载附件未能登记，请检查下载任务"
					}
				} else {
					all := true
					failed := false
					review := false
					for _, f := range files {
						if f.MaterialBatchID == "" {
							all = false
							continue
						}
						var batch models.MaterialBatch
						if r.Service.DB.First(&batch, "id = ?", f.MaterialBatchID).Error != nil {
							all = false
							continue
						}
						if batch.GradingJobID == "" {
							all = false
							if batch.Status == "NEEDS_REVIEW" {
								review = true
							}
							if batch.Status == "FAILED" || batch.Status == "CANCELLED" || batch.Status == "PARTIAL_SUCCESS" {
								failed = true
							}
							continue
						}
						var gradingJob models.AsyncJob
						if r.Service.DB.First(&gradingJob, "id = ?", batch.GradingJobID).Error != nil {
							all = false
							continue
						}
						if gradingJob.Status == models.JobStatusFailed || gradingJob.Status == models.JobStatusCancelled {
							failed = true
						}
						if gradingJob.Status != models.JobStatusSuccess {
							all = false
						}
					}
					if failed {
						status = "FAILED"
						message = "部分材料或评分未完成，请查看材料批次处理异常"
					} else if review {
						status = "WAITING_REVIEW"
						message = "材料存在待核对项，请在作业详情查看异常后继续批改"
					} else if all {
						status = "SUCCEEDED"
						message = "附件整理与批改已完成"
					} else {
						message = "附件已下载，正在整理和批改"
					}
				}
			}
			r.Service.DB.Model(&item).Where("status NOT IN ?", []string{"CANCELLED", "SUCCEEDED", "FAILED"}).Updates(map[string]any{"status": status, "message": message})
		}
		var current []models.PortalWorkflowItem
		if r.Service.DB.Where("workflow_id = ?", flow.ID).Find(&current).Error != nil || len(current) == 0 {
			continue
		}
		pending, success, failed, waiting := 0, 0, 0, false
		for _, item := range current {
			switch item.Status {
			case "SUCCEEDED":
				success++
			case "FAILED", "CANCELLED":
				failed++
			default:
				pending++
				if item.Status == "WAITING_AUTH" {
					waiting = true
				}
			}
		}
		status := "RUNNING"
		if pending == 0 {
			status = "SUCCEEDED"
			if failed > 0 {
				status = "FAILED"
				if success > 0 {
					status = "PARTIAL_SUCCESS"
				}
			}
		} else if waiting {
			status = "WAITING_AUTH"
		}
		r.Service.DB.Model(&flow).Where("status <> ?", "CANCELLED").Update("status", status)
	}
}

func (r *Runner) retryItem(ctx context.Context, flow models.PortalWorkflow, item models.PortalWorkflowItem) {
	c, err := r.Service.connection(ctx, flow.OwnerID, false)
	if err != nil || c.Status != models.PortalConnectionReady || c.ActiveOperationID != "" {
		return
	}
	var active int64
	if r.Service.DB.Model(&models.RPAJob{}).Where("portal_connection_id = ? AND id <> ? AND status NOT IN ?", c.ID, item.RPAJobID, []string{"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED"}).Count(&active).Error != nil || active > 0 {
		return
	}
	// If the previous restart response was lost, observe it instead of starting a second browser.
	raw, readErr := r.Worker(ctx, "GET", "/tasks/"+item.RPAJobID, nil)
	var current struct {
		Status string `json:"status"`
	}
	if readErr == nil && json.Unmarshal(raw, &current) == nil && (current.Status == "RUNNING" || current.Status == "WAITING_USER" || current.Status == "SUCCEEDED") {
		err = rpa.SaveState(item.RPAJobID, raw)
	} else {
		err = rpa.Start(ctx, item.RPAJobID, true)
	}
	if err != nil {
		return
	}
	r.Service.DB.Model(&item).Where("status = ?", "RETRY_QUEUED").Updates(map[string]any{"status": "RUNNING", "message": "正在从原检查点继续下载"})
}
func (r *Runner) startItem(ctx context.Context, flow models.PortalWorkflow, item models.PortalWorkflowItem) {
	_ = r.Service.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var c models.PortalConnection
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&c, "id = ?", flow.ConnectionID).Error; err != nil {
			return err
		}
		var lockedFlow models.PortalWorkflow
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&lockedFlow, "id = ?", flow.ID).Error; err != nil {
			return err
		}
		if lockedFlow.Status == "CANCELLED" {
			return nil
		}
		if c.Status == models.PortalConnectionDisabled {
			return tx.Model(&item).Updates(map[string]any{"status": "FAILED", "message": "教务连接已断开"}).Error
		}
		if lockedFlow.ProfileGeneration != c.ProfileGeneration {
			return tx.Model(&item).Updates(map[string]any{"status": "FAILED", "message": "教务账号或配置已变更，请从当前目录重新选择"}).Error
		}
		if c.Status != models.PortalConnectionReady {
			return tx.Model(&item).Updates(map[string]any{"status": "WAITING_AUTH", "message": "请在教务账号设置中完成认证，完成后自动继续"}).Error
		}
		if c.ActiveOperationID != "" {
			return nil
		}
		var active int64
		if err := tx.Model(&models.RPAJob{}).Where("portal_connection_id = ? AND status NOT IN ?", c.ID, []string{"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED"}).Count(&active).Error; err != nil {
			return err
		}
		if active > 0 {
			return nil
		}
		var locked models.PortalWorkflowItem
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&locked, "id = ?", item.ID).Error; err != nil {
			return err
		}
		if locked.RPAJobID != "" || terminal(locked.Status) {
			return nil
		}
		id := stableID(item.ID, "download", "v1")
		job := models.RPAJob{ID: id, UserID: flow.OwnerID, CourseName: item.CourseName, AssignmentName: item.HomeworkName, AfterDownload: "grade", Status: "QUEUED", Stage: "OPEN_PORTAL", Message: "按所选官网目标下载附件", PortalConnectionID: c.ID, PortalWorkflowItemID: item.ID, AssignmentID: item.AssignmentID, PortalTargetJSON: item.TargetJSON}
		if err := tx.Clauses(clause.OnConflict{DoNothing: true}).Create(&job).Error; err != nil {
			return err
		}
		async := models.AsyncJob{ID: id, OwnerID: flow.OwnerID, JobType: "rpa_fetch_homework", ReferenceID: fmt.Sprint(item.AssignmentID), Status: models.JobStatusPending, Message: job.Message}
		if err := tx.Clauses(clause.OnConflict{DoNothing: true}).Create(&async).Error; err != nil {
			return err
		}
		return tx.Model(&locked).Updates(map[string]any{"rpa_job_id": id, "status": "RUNNING", "message": job.Message}).Error
	})
}
