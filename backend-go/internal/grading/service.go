// Package grading provides a single submission boundary for chat tools and upload APIs.
package grading

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/models"
	"grading-gateway/internal/mq"
)

type Actor struct {
	UserID uint
	Role   string
}

func (a Actor) Check() error {
	if a.UserID == 0 || (a.Role != "teacher" && a.Role != "admin") {
		return fmt.Errorf("只有教师或管理员可以提交或查询批改任务")
	}
	return nil
}

type Request struct {
	BatchID       string `json:"batch_id,omitempty"`
	DirectoryPath string `json:"directory_path,omitempty"`
	OutputRoot    string `json:"output_root,omitempty"`
	AllowPartial  bool   `json:"allow_partial,omitempty"`
	Regrade       bool   `json:"regrade,omitempty"`
	Kind          string `json:"kind"`
	ResourceID    string `json:"resource_id"`
	FilePath      string `json:"file_path,omitempty"`
	SkillName     string `json:"-"`
	SkillVersion  string `json:"-"`
}
type Result struct {
	BatchID   string                `json:"batch_id,omitempty"`
	Outcome   string                `json:"outcome"`
	JobID     string                `json:"job_id,omitempty"`
	JobType   string                `json:"job_type,omitempty"`
	Status    models.AsyncJobStatus `json:"status,omitempty"`
	Message   string                `json:"message"`
	Missing   []string              `json:"missing,omitempty"`
	ResultURL string                `json:"result_url,omitempty"`
}
type Repository interface {
	Assignment(context.Context, uint) (models.Assignment, error)
	Job(context.Context, string) (models.AsyncJob, error)
	Create(context.Context, *models.AsyncJob) error
	Fail(context.Context, string, string) error
}
type Service struct {
	Repo            Repository
	PublishHomework func(string, uint, string) error
}

func Default() Service {
	return Service{Repo: SQLRepository{database.DB}, PublishHomework: mq.PublishHomeworkTask}
}

func ResourceID(value string) (uint, error) {
	id, err := strconv.ParseUint(value, 10, 32)
	if err != nil || id == 0 {
		return 0, fmt.Errorf("请提供有效的资源 ID")
	}
	return uint(id), nil
}
func HomeworkMissing(a models.Assignment) []string {
	missing := []string{}
	if strings.TrimSpace(a.Question) == "" {
		missing = append(missing, "题目要求")
	}
	var rubric any
	err := json.Unmarshal([]byte(a.Rubric), &rubric)
	valid := false
	switch v := rubric.(type) {
	case map[string]any:
		valid = len(v) > 0
	case []any:
		valid = len(v) > 0
	}
	if err != nil || !valid {
		missing = append(missing, "非空 JSON 评分标准")
	}
	return missing
}
func (s Service) Check(ctx context.Context, actor Actor, kind, resourceID string) (Result, error) {
	if err := actor.Check(); err != nil {
		return Result{}, err
	}
	id, err := ResourceID(resourceID)
	if err != nil {
		return Result{}, err
	}
	var missing []string
	var resultURL string
	switch kind {
	case "homework":
		a, err := s.Repo.Assignment(ctx, id)
		if err != nil {
			return Result{}, fmt.Errorf("作业不存在或读取失败")
		}
		if a.OwnerID != actor.UserID {
			return Result{}, fmt.Errorf("作业不存在或无权批改；历史作业需管理员分配归属")
		}
		missing = HomeworkMissing(a)
		resultURL = fmt.Sprintf("/assignments/%d", id)
	default:
		return Result{}, fmt.Errorf("当前版本仅支持 homework 作业批改")
	}
	if len(missing) > 0 {
		return Result{Outcome: "needs_input", Missing: missing, Message: "请在目标详情页补齐：" + strings.Join(missing, "、"), ResultURL: resultURL}, nil
	}
	return Result{Outcome: "ready", Message: "评分配置齐备，请确认本次提交材料", ResultURL: resultURL}, nil
}

// Inputs must exist; no model-generated code or file writes happen here.
func CheckFiles(r Request) ([]string, error) {
	if r.Kind == "homework" {
		if strings.TrimSpace(r.FilePath) == "" {
			return []string{"本次作业 ZIP 附件路径"}, nil
		}
		if !strings.EqualFold(filepath.Ext(r.FilePath), ".zip") {
			return nil, fmt.Errorf("作业仅支持 ZIP 文件")
		}
		archive, err := zip.OpenReader(r.FilePath)
		if err != nil {
			return nil, fmt.Errorf("无法读取有效的作业 ZIP")
		}
		defer archive.Close()
		count := 0
		for _, f := range archive.File {
			// The existing extractor writes archive paths; reject traversal before it runs.
			name := strings.ReplaceAll(f.Name, "\\", "/")
			if strings.HasPrefix(name, "/") || strings.Contains(name, ":") {
				return nil, fmt.Errorf("ZIP 包含无效路径")
			}
			for _, part := range strings.Split(name, "/") {
				if part == ".." {
					return nil, fmt.Errorf("ZIP 包含越界路径")
				}
			}
			if f.Mode()&os.ModeSymlink != 0 {
				return nil, fmt.Errorf("ZIP 不支持符号链接")
			}
			if !f.FileInfo().IsDir() {
				count++
			}
		}
		if count == 0 {
			return nil, fmt.Errorf("ZIP 中没有学生附件")
		}
		return nil, nil
	}
	return nil, fmt.Errorf("当前版本仅支持 homework 作业批改")
}
func (s Service) Submit(ctx context.Context, actor Actor, r Request, requestID string) (Result, error) {
	if r.Kind == "homework" && r.BatchID != "" {
		return SubmitBatch(ctx, actor, r)
	}
	checked, err := s.Check(ctx, actor, r.Kind, r.ResourceID)
	if err != nil || checked.Outcome == "needs_input" {
		return checked, err
	}
	if r.Kind == "homework" && r.DirectoryPath != "" {
		path, err := materials.NormalizePath(r.DirectoryPath)
		if err != nil {
			return Result{}, err
		}
		var existing models.MaterialBatch
		if database.DB.WithContext(ctx).Where("owner_id = ? AND manifest_path = ?", actor.UserID, filepath.Join(path, "manifest.json")).First(&existing).Error == nil {
			r.BatchID = existing.ID
			r.DirectoryPath = ""
			return SubmitBatch(ctx, actor, r)
		}
		id, _ := ResourceID(r.ResourceID)
		batch, err := materials.Default().Create(ctx, materials.Actor{UserID: actor.UserID, Role: actor.Role}, materials.Request{SourcePath: r.DirectoryPath, AssignmentID: id, OutputRoot: r.OutputRoot, AutoGrade: true}, requestID)
		if err != nil {
			return Result{}, err
		}
		return Result{Outcome: "accepted", JobID: batch.ID, JobType: "homework_materials", BatchID: batch.ID, Status: models.JobStatusPending, Message: "材料整理已受理，完成后按本次明确要求继续批改", ResultURL: "/assignments/" + r.ResourceID}, nil
	}
	missing, err := CheckFiles(r)
	if err != nil {
		return Result{}, err
	}
	if len(missing) > 0 {
		checked.Outcome = "needs_input"
		checked.Missing = missing
		checked.Message = "请提供本次批改材料"
		return checked, nil
	}
	if requestID == "" {
		requestID = uuid.NewString()
	}
	data, _ := json.Marshal(r)
	digest := sha256.Sum256(append([]byte(fmt.Sprintf("%d:%s:", actor.UserID, requestID)), data...))
	id := uuid.NewSHA1(uuid.NameSpaceOID, []byte(hex.EncodeToString(digest[:]))).String()
	old, err := s.Repo.Job(ctx, id)
	if err == nil {
		return receipt(old), nil
	}
	if !errors.Is(err, gorm.ErrRecordNotFound) {
		return Result{}, err
	}
	job := models.AsyncJob{ID: id, OwnerID: actor.UserID, JobType: models.JobTypeHomework, ReferenceID: r.ResourceID, Status: models.JobStatusPending, Message: "任务已创建，等待队列处理", SkillName: r.SkillName, SkillVersion: r.SkillVersion}
	if err = s.Repo.Create(ctx, &job); err != nil {
		if old, lookupErr := s.Repo.Job(ctx, id); lookupErr == nil {
			return receipt(old), nil
		}
		return Result{}, err
	}
	// Publishing is intentionally after the durable record. Do not roll back an accepted job on HTTP disconnect.
	assignmentID, _ := ResourceID(r.ResourceID)
	err = s.PublishHomework(job.ID, assignmentID, r.FilePath)
	if err != nil {
		job.Status = models.JobStatusFailed
		job.Message = "消息队列提交失败，请检查队列后重新发起任务"
		if saveErr := s.Repo.Fail(context.Background(), job.ID, job.Message); saveErr != nil {
			return Result{Outcome: "failed", JobID: job.ID, Message: "队列提交失败，任务状态更新失败"}, fmt.Errorf("无法保存任务失败状态: %w", saveErr)
		}
		_ = cache.SetJobStatus(context.Background(), job.ID, string(job.Status), job.Message)
	}
	// Do not cache PENDING here: a fast consumer may already have written PROCESSING/SUCCESS.
	return receipt(job), nil
}
func receipt(job models.AsyncJob) Result {
	outcome := "accepted"
	if job.Status == models.JobStatusFailed {
		outcome = "failed"
	}
	if job.Status == models.JobStatusSuccess {
		outcome = "completed"
	}
	url := "/assignments/" + job.ReferenceID

	return Result{Outcome: outcome, JobID: job.ID, JobType: string(job.JobType), Status: job.Status, Message: job.Message, ResultURL: url, BatchID: job.MaterialBatchID}
}
func (s Service) Status(ctx context.Context, actor Actor, id string) (Result, error) {
	if err := actor.Check(); err != nil {
		return Result{}, err
	}
	job, err := s.Repo.Job(ctx, id)
	if err != nil || job.OwnerID != actor.UserID || job.JobType != models.JobTypeHomework {
		return Result{}, fmt.Errorf("批改任务不存在或不属于当前用户")
	}
	result := receipt(job)
	if job.JobType == models.JobTypeHomework && job.Status == models.JobStatusSuccess && job.MaterialBatchID == "" {
		result.Message += "；成绩池化可能仍在后台执行，请以作业页面为准"
	}
	return result, nil
}

type SQLRepository struct{ DB *gorm.DB }

func (r SQLRepository) Assignment(ctx context.Context, id uint) (models.Assignment, error) {
	var a models.Assignment
	err := r.DB.WithContext(ctx).First(&a, id).Error
	return a, err
}
func (r SQLRepository) Job(ctx context.Context, id string) (models.AsyncJob, error) {
	var j models.AsyncJob
	err := r.DB.WithContext(ctx).First(&j, "id = ?", id).Error
	return j, err
}
func (r SQLRepository) Create(ctx context.Context, j *models.AsyncJob) error {
	return r.DB.WithContext(ctx).Create(j).Error
}
func (r SQLRepository) Fail(ctx context.Context, id, message string) error {
	return r.DB.WithContext(ctx).Model(&models.AsyncJob{}).Where("id = ?", id).Updates(map[string]any{"status": models.JobStatusFailed, "message": message}).Error
}
