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
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
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
	Kind         string   `json:"kind"`
	ResourceID   string   `json:"resource_id"`
	FilePath     string   `json:"file_path,omitempty"`
	StudentID    string   `json:"student_id,omitempty"`
	ImagePaths   []string `json:"image_paths,omitempty"`
	SkillName    string   `json:"-"`
	SkillVersion string   `json:"-"`
}
type Result struct {
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
	Exam(context.Context, uint) (models.Exam, error)
	Job(context.Context, string) (models.AsyncJob, error)
	Create(context.Context, *models.AsyncJob) error
	Fail(context.Context, string, string) error
}
type Service struct {
	Repo            Repository
	PublishHomework func(string, uint, string) error
	PublishExam     func(string, string, string, []string) error
}

func Default() Service {
	return Service{SQLRepository{database.DB}, mq.PublishHomeworkTask, mq.PublishExamTask}
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
func ExamMissing(e models.Exam) []string {
	missing := []string{}
	if len(e.Questions) == 0 {
		return []string{"试卷题目"}
	}
	seen := map[int]bool{}
	for _, q := range e.Questions {
		prefix := fmt.Sprintf("第 %d 题", q.QuestionNumber)
		if q.QuestionNumber <= 0 || seen[q.QuestionNumber] {
			missing = append(missing, "唯一且为正数的题号")
		}
		seen[q.QuestionNumber] = true
		if strings.TrimSpace(q.QuestionText) == "" {
			missing = append(missing, prefix+"题目")
		}
		if strings.TrimSpace(q.StandardAnswer) == "" {
			missing = append(missing, prefix+"标准答案")
		}
		if strings.TrimSpace(q.Rubric) == "" {
			missing = append(missing, prefix+"评分标准")
		}
		if q.MaxScore <= 0 || math.IsNaN(q.MaxScore) || math.IsInf(q.MaxScore, 0) {
			missing = append(missing, prefix+"有效满分")
		}
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
		missing = HomeworkMissing(a)
		resultURL = fmt.Sprintf("/assignments/%d", id)
	case "exam":
		e, err := s.Repo.Exam(ctx, id)
		if err != nil {
			return Result{}, fmt.Errorf("试卷不存在或读取失败")
		}
		missing = ExamMissing(e)
		resultURL = fmt.Sprintf("/exams/%d", id)
	default:
		return Result{}, fmt.Errorf("kind 仅支持 homework 或 exam")
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
	missing := []string{}
	if strings.TrimSpace(r.StudentID) == "" {
		missing = append(missing, "学生学号")
	}
	if len(r.ImagePaths) == 0 {
		missing = append(missing, "按页排序的答卷图片路径")
	}
	if len(missing) > 0 {
		return missing, nil
	}
	for _, p := range r.ImagePaths {
		ext := strings.ToLower(filepath.Ext(p))
		if ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".webp" && ext != ".bmp" {
			return nil, fmt.Errorf("答卷必须是支持的图片文件")
		}
		info, err := os.Stat(p)
		if err != nil || !info.Mode().IsRegular() || info.Size() == 0 {
			return nil, fmt.Errorf("答卷图片不存在或为空")
		}
	}
	return nil, nil
}
func (s Service) Submit(ctx context.Context, actor Actor, r Request, requestID string) (Result, error) {
	checked, err := s.Check(ctx, actor, r.Kind, r.ResourceID)
	if err != nil || checked.Outcome == "needs_input" {
		return checked, err
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
	kind := models.JobTypeHomework
	if r.Kind == "exam" {
		kind = models.JobTypeExam
	}
	job := models.AsyncJob{ID: id, OwnerID: actor.UserID, JobType: kind, ReferenceID: r.ResourceID, StudentID: r.StudentID, Status: models.JobStatusPending, Message: "任务已创建，等待队列处理", SkillName: r.SkillName, SkillVersion: r.SkillVersion}
	if err = s.Repo.Create(ctx, &job); err != nil {
		if old, lookupErr := s.Repo.Job(ctx, id); lookupErr == nil {
			return receipt(old), nil
		}
		return Result{}, err
	}
	// Publishing is intentionally after the durable record. Do not roll back an accepted job on HTTP disconnect.
	if r.Kind == "homework" {
		id, _ := ResourceID(r.ResourceID)
		err = s.PublishHomework(job.ID, id, r.FilePath)
	} else {
		err = s.PublishExam(job.ID, r.ResourceID, r.StudentID, r.ImagePaths)
	}
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
	if job.JobType == models.JobTypeExam {
		url = "/exams/" + job.ReferenceID
	}
	return Result{Outcome: outcome, JobID: job.ID, JobType: string(job.JobType), Status: job.Status, Message: job.Message, ResultURL: url}
}
func (s Service) Status(ctx context.Context, actor Actor, id string) (Result, error) {
	if err := actor.Check(); err != nil {
		return Result{}, err
	}
	job, err := s.Repo.Job(ctx, id)
	if err != nil || job.OwnerID != actor.UserID || (job.JobType != models.JobTypeHomework && job.JobType != models.JobTypeExam) {
		return Result{}, fmt.Errorf("批改任务不存在或不属于当前用户")
	}
	result := receipt(job)
	if job.JobType == models.JobTypeHomework && job.Status == models.JobStatusSuccess {
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
func (r SQLRepository) Exam(ctx context.Context, id uint) (models.Exam, error) {
	var e models.Exam
	err := r.DB.WithContext(ctx).Preload("Questions").First(&e, id).Error
	return e, err
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
