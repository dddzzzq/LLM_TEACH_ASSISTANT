package grading

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/database"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/models"
	"grading-gateway/internal/mq"
)

var PublishMaterialTask = mq.PublishHomeworkTask

// SubmitBatch shares one durable job per batch; repeated clicks never regrade it.
func SubmitBatch(ctx context.Context, actor Actor, r Request) (Result, error) {
	if err := actor.Check(); err != nil {
		return Result{}, err
	}
	batch, err := materials.Default().Owned(ctx, materials.Actor{UserID: actor.UserID, Role: actor.Role}, r.BatchID)
	if err != nil {
		return Result{}, err
	}
	if batch.AssignmentID == 0 {
		if r.ResourceID == "" {
			return Result{Outcome: "needs_input", Message: "材料尚未关联作业，请指定 resource_id 后批改", Missing: []string{"所属作业ID"}}, nil
		}
		id, err := ResourceID(r.ResourceID)
		if err != nil {
			return Result{}, err
		}
		var a models.Assignment
		if database.DB.WithContext(ctx).Where("owner_id = ?", actor.UserID).First(&a, id).Error != nil {
			return Result{}, fmt.Errorf("作业不存在")
		}
		if a.CourseName != batch.CourseName || a.TaskName != batch.AssignmentName || (batch.ClassName != "" && a.ClassName != batch.ClassName) {
			return Result{}, fmt.Errorf("材料课程、作业或班级与目标不一致")
		}
		if err := database.DB.WithContext(ctx).Model(&batch).Where("assignment_id = 0 AND grading_job_id = ''").Update("assignment_id", id).Error; err != nil {
			return Result{}, err
		}
		batch.AssignmentID = id
	}
	if r.ResourceID != "" && r.ResourceID != fmt.Sprint(batch.AssignmentID) {
		return Result{}, fmt.Errorf("材料批次与指定作业不一致")
	}
	if batch.GradingJobID != "" {
		return Default().Status(ctx, actor, batch.GradingJobID)
	}
	if batch.Status != "READY" && !(batch.Status == "NEEDS_REVIEW" && r.AllowPartial) {
		return Result{Outcome: "needs_input", Message: "材料未就绪或存在待核对项；查看清单，明确允许仅批改已识别学生后再提交", Missing: []string{"完成材料整理或核对异常清单"}}, nil
	}
	checked, err := Default().Check(ctx, actor, "homework", fmt.Sprint(batch.AssignmentID))
	if err != nil || checked.Outcome == "needs_input" {
		return checked, err
	}
	if _, err := materials.LoadManifest(batch.ManifestPath, batch.ManifestSHA256); err != nil {
		return Result{}, err
	}
	var a models.Assignment
	if err := database.DB.WithContext(ctx).First(&a, batch.AssignmentID).Error; err != nil {
		return Result{}, err
	}
	id := uuid.NewSHA1(uuid.NameSpaceURL, []byte("batch-grade:"+batch.ID)).String()
	payload := Request{Kind: "homework", ResourceID: fmt.Sprint(batch.AssignmentID), BatchID: batch.ID, FilePath: batch.ManifestPath, SkillName: r.SkillName, SkillVersion: r.SkillVersion}
	raw, _ := json.Marshal(payload)
	job := models.AsyncJob{ID: id, OwnerID: actor.UserID, JobType: models.JobTypeHomework, ReferenceID: fmt.Sprint(batch.AssignmentID), MaterialBatchID: batch.ID, Status: models.JobStatusPending, Message: "材料已确认，等待批改队列", PayloadJSON: string(raw), DispatchState: "READY", SkillName: r.SkillName, SkillVersion: r.SkillVersion, QuestionSnapshot: a.Question, RubricSnapshot: a.Rubric}
	if batch.SourceJobID != "" {
		var download models.RPAJob
		if database.DB.WithContext(ctx).First(&download, "id = ? AND user_id = ? AND assignment_id = ?", batch.SourceJobID, actor.UserID, a.ID).Error == nil && download.PortalWorkflowItemID != "" {
			var item models.PortalWorkflowItem
			if database.DB.First(&item, "id = ? AND owner_id = ? AND assignment_id = ?", download.PortalWorkflowItemID, actor.UserID, a.ID).Error != nil {
				return Result{}, fmt.Errorf("官网批改配置快照不可用")
			}
			if len(HomeworkMissing(models.Assignment{Question: item.Question, Rubric: item.Rubric})) > 0 {
				return Result{}, fmt.Errorf("官网批改配置快照不完整")
			}
			job.QuestionSnapshot, job.RubricSnapshot = item.Question, item.Rubric
		}
	}
	err = database.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var locked models.MaterialBatch
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&locked, "id = ?", batch.ID).Error; err != nil {
			return err
		}
		if locked.Status == "CANCELLED" {
			return fmt.Errorf("材料批次已取消")
		}
		if locked.AssignmentID != batch.AssignmentID {
			return fmt.Errorf("材料所属作业已变化，请刷新")
		}
		if locked.GradingJobID != "" {
			return tx.First(&job, "id = ?", locked.GradingJobID).Error
		}
		if locked.SourceJobID != "" && locked.AutoGrade {
			var parent models.RPAJob
			if err := tx.First(&parent, "id = ?", locked.SourceJobID).Error; err != nil {
				return err
			}
			if parent.Status == "CANCELLED" {
				return fmt.Errorf("下载流程已取消，停止自动批改")
			}
		}
		if err := tx.Clauses(clause.OnConflict{DoNothing: true}).Create(&job).Error; err != nil {
			return err
		}
		if err := tx.Model(&locked).Updates(map[string]any{"grading_job_id": id, "grading_state": "PENDING", "auto_grade": false}).Error; err != nil {
			return err
		}
		if locked.SourceFileID != "" {
			return tx.Model(&models.RPAFile{}).Where("id = ?", locked.SourceFileID).Updates(map[string]any{"grading_job_id": id, "grading_status": "PENDING", "message": "老师已发起独立批改任务"}).Error
		}
		return nil
	})
	if err != nil {
		return Result{}, err
	}
	_ = DispatchMaterialJob(context.Background(), id)
	if err := database.DB.First(&job, "id = ?", id).Error; err != nil {
		return Result{}, err
	}
	return receipt(job), nil
}

func ContinuePrepared(ctx context.Context, batch models.MaterialBatch) error {
	resource := ""
	if batch.AssignmentID != 0 {
		resource = fmt.Sprint(batch.AssignmentID)
	}
	r, err := SubmitBatch(ctx, Actor{UserID: batch.OwnerID, Role: "teacher"}, Request{Kind: "homework", ResourceID: resource, BatchID: batch.ID})
	if err != nil {
		// A cancelled parent cannot cause an endless resume loop.
		if batch.SourceJobID != "" {
			var parent models.RPAJob
			if database.DB.First(&parent, "id = ?", batch.SourceJobID).Error == nil && parent.Status == "CANCELLED" {
				return database.DB.Model(&batch).Updates(map[string]any{"auto_grade": false, "grading_state": "CANCELLED"}).Error
			}
		}
		return err
	}
	if r.Outcome == "needs_input" {
		return database.DB.Model(&models.MaterialBatch{}).Where("id = ? AND grading_job_id = ''", batch.ID).Updates(map[string]any{"grading_state": "WAITING_REQUIREMENTS", "message": r.Message}).Error
	}
	return nil
}

func DispatchMaterialJob(ctx context.Context, id string) error {
	var job models.AsyncJob
	if err := database.DB.WithContext(ctx).Where("id = ? AND dispatch_state = ?", id, "READY").First(&job).Error; err != nil {
		return err
	}
	if job.Status != models.JobStatusPending {
		return database.DB.Model(&job).Update("dispatch_state", "SENT").Error
	}
	var r Request
	if json.Unmarshal([]byte(job.PayloadJSON), &r) != nil || r.BatchID == "" {
		return fmt.Errorf("材料批改投递信息无效")
	}
	assignmentID, err := ResourceID(r.ResourceID)
	if err != nil {
		return err
	}
	if err := PublishMaterialTask(job.ID, assignmentID, r.FilePath); err != nil {
		return err
	}
	return database.DB.Model(&models.AsyncJob{}).Where("id = ? AND dispatch_state = ?", id, "READY").Update("dispatch_state", "SENT").Error
}
func RunMaterialOutbox(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		var jobs []models.AsyncJob
		if err := database.DB.WithContext(ctx).Where("dispatch_state = ? AND status = ?", "READY", models.JobStatusPending).Limit(20).Find(&jobs).Error; err == nil {
			for _, job := range jobs {
				if err := DispatchMaterialJob(ctx, job.ID); err != nil {
					log.Printf("[task job=%s] 批改队列等待重试 (%T)", job.ID, err)
				}
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func PreparedInput(path string) bool { return filepath.Base(path) == "manifest.json" }
