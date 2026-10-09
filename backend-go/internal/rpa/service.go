package rpa

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/agent/catalog"
	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

var PublishFetch func(string) error
var PublishGrade func(string, uint, string) error
var dispatchMu sync.Mutex

type Target struct {
	CourseName     string `json:"course_name"`
	AssignmentName string `json:"assignment_name"`
	Term           string `json:"term,omitempty"`
}

func Create(ctx context.Context, userID uint, target Target) (*models.RPAJob, error) {
	return CreateWithKey(ctx, userID, target, "")
}

func CreateWithKey(ctx context.Context, userID uint, target Target, requestKey string) (*models.RPAJob, error) {
	target.CourseName, target.AssignmentName = strings.TrimSpace(target.CourseName), strings.TrimSpace(target.AssignmentName)
	if userID == 0 || target.CourseName == "" || target.AssignmentName == "" || len([]rune(target.CourseName)) > 255 || len([]rune(target.AssignmentName)) > 255 || len([]rune(target.Term)) > 255 {
		return nil, fmt.Errorf("请提供有效的课程、作业名称和登录用户")
	}
	job := &models.RPAJob{ID: uuid.NewString(), UserID: userID, CourseName: target.CourseName,
		AssignmentName: target.AssignmentName, Term: target.Term, Status: "QUEUED", Stage: "OPEN_PORTAL", Message: "任务已创建，等待打开登录浏览器"}
	if len(requestKey) > 128 {
		return nil, fmt.Errorf("请求幂等键过长")
	}
	if requestKey != "" {
		job.ID = uuid.NewSHA1(uuid.NameSpaceURL, []byte(fmt.Sprintf("rpa-create:%d:%s", userID, requestKey))).String()
	}
	if requestKey != "" {
		var prior models.RPAJob
		err := database.DB.WithContext(ctx).First(&prior, "id = ?", job.ID).Error
		if err == nil {
			if prior.UserID != userID || prior.CourseName != target.CourseName || prior.AssignmentName != target.AssignmentName || prior.Term != target.Term {
				return nil, fmt.Errorf("同一请求键不能用于不同下载目标")
			}
			return &prior, nil
		}
		if !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, err
		}
	}
	skills, err := catalog.Default("portal-browser")
	if err != nil {
		return nil, err
	}
	doc, ok := skills["fetch-homework"]
	if !ok {
		return nil, fmt.Errorf("缺少页面导航 Skill")
	}
	snapshot, _ := json.Marshal(NavigationSnapshot{Skill: doc})
	job.NavigationJSON = string(snapshot)
	err = database.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		created := tx.Clauses(clause.OnConflict{DoNothing: true}).Create(job)
		if created.Error != nil {
			return created.Error
		}
		if created.RowsAffected == 0 {
			var prior models.RPAJob
			if err := tx.First(&prior, "id = ?", job.ID).Error; err != nil {
				return err
			}
			if prior.UserID != userID || prior.CourseName != target.CourseName || prior.AssignmentName != target.AssignmentName || prior.Term != target.Term {
				return fmt.Errorf("同一请求键不能用于不同下载目标")
			}
			*job = prior
			return nil
		}
		return tx.Create(&models.AsyncJob{ID: job.ID, JobType: "rpa_fetch_homework", ReferenceID: target.CourseName + ":" + target.AssignmentName,
			Status: models.JobStatusPending, Message: job.Message}).Error
	})
	if err != nil {
		return nil, err
	}
	// The persisted QUEUED row is also a recoverable dispatch outbox; scheduler retries it.
	if PublishFetch != nil {
		if err := PublishFetch(job.ID); err != nil {
			log.Printf("RPA %s queued for dispatch retry", job.ID)
		}
	}
	return job, nil
}

func Owned(userID uint, jobID string) (*models.RPAJob, error) {
	var job models.RPAJob
	if err := database.DB.Where("id = ? AND user_id = ?", jobID, userID).First(&job).Error; err != nil {
		return nil, err
	}
	return &job, nil
}

func Start(ctx context.Context, jobID string, restart bool) error {
	var job models.RPAJob
	if err := database.DB.First(&job, "id = ?", jobID).Error; err != nil {
		return err
	}
	if !restart && job.Status != "QUEUED" {
		return nil
	}
	if restart && job.Status != "INTERRUPTED" && job.Status != "FAILED" && job.Status != "PARTIAL_SUCCESS" {
		return fmt.Errorf("只有中断或失败的任务可以重新启动")
	}
	var checkpoint map[string]any
	if restart {
		_ = json.Unmarshal([]byte(job.StateJSON), &checkpoint)
	}
	b, err := WorkerRequest(ctx, "POST", "/tasks/"+jobID, map[string]any{
		"target": Target{job.CourseName, job.AssignmentName, job.Term}, "restart": restart, "checkpoint": checkpoint})
	if err != nil {
		return err
	}
	var current models.RPAJob
	if database.DB.First(&current, "id = ?", jobID).Error == nil && current.Status == "CANCELLED" {
		_, _ = WorkerRequest(ctx, "POST", "/tasks/"+jobID+"/cancel", nil)
		return nil
	}
	return SaveState(jobID, b)
}

func CancelQueued(ctx context.Context, job *models.RPAJob) error {
	if err := database.DB.Model(job).Updates(map[string]any{"status": "CANCELLED", "message": "用户取消任务"}).Error; err != nil {
		return err
	}
	_ = database.DB.Model(&models.AsyncJob{}).Where("id = ?", job.ID).Updates(map[string]any{"status": "CANCELLED", "message": "用户取消任务"}).Error
	_ = database.DB.Model(&models.RPAFile{}).Where("job_id = ? AND grading_status IN ?", job.ID, []string{"WAITING_RUBRIC", "READY"}).Update("grading_status", "CANCELLED").Error
	_, _ = WorkerRequest(ctx, "POST", "/tasks/"+job.ID+"/cancel", nil)
	return nil
}

func SaveState(jobID string, raw []byte) error {
	var state struct {
		Status       string `json:"status"`
		Stage        string `json:"stage"`
		Message      string `json:"message"`
		StateVersion int64  `json:"state_version"`
	}
	if err := json.Unmarshal(raw, &state); err != nil {
		return err
	}
	if state.Status == "" {
		return fmt.Errorf("Worker 返回了无效任务状态")
	}
	result := database.DB.Model(&models.RPAJob{}).Where("id = ? AND status <> ? AND state_version <= ?", jobID, "CANCELLED", state.StateVersion).Updates(map[string]any{
		"status": state.Status, "stage": state.Stage, "message": state.Message, "state_json": string(raw), "state_version": state.StateVersion})
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return nil
	}
	if state.Status == "CANCELLED" {
		_ = database.DB.Model(&models.RPAFile{}).Where("job_id = ? AND grading_status IN ?", jobID, []string{"WAITING_RUBRIC", "READY"}).Update("grading_status", "CANCELLED").Error
	}
	_ = database.DB.Model(&models.AsyncJob{}).Where("id = ?", jobID).Updates(map[string]any{"status": state.Status, "message": state.Message}).Error
	_ = cache.SetJobStatus(context.Background(), jobID, state.Status, state.Message)
	return nil
}

func Detail(job *models.RPAJob) map[string]any {
	state := map[string]any{}
	_ = json.Unmarshal([]byte(job.StateJSON), &state)
	state["job_id"], state["status"], state["stage"], state["message"] = job.ID, job.Status, job.Stage, job.Message
	state["target"] = Target{job.CourseName, job.AssignmentName, job.Term}
	var files []models.RPAFile
	database.DB.Where("job_id = ?", job.ID).Find(&files)
	for i := range files {
		if files[i].GradingJobID != "" && files[i].GradingStatus == "PUBLISHED" {
			var grading models.AsyncJob
			if database.DB.First(&grading, "id = ?", files[i].GradingJobID).Error == nil {
				files[i].GradingStatus = string(grading.Status)
			}
		}
	}
	state["grading_files"] = files
	return state
}

func validRubric(assignment models.Assignment) bool {
	var rubric map[string]any
	return strings.TrimSpace(assignment.Question) != "" && json.Unmarshal([]byte(assignment.Rubric), &rubric) == nil && len(rubric) > 0
}

func importFiles(job models.RPAJob) error {
	var state struct {
		Files []struct {
			Status, Path, SHA256 string
			Size                 int64
			ExportRef            string `json:"export_ref"`
			ClassName            string `json:"class_name"`
		} `json:"files"`
	}
	if err := json.Unmarshal([]byte(job.StateJSON), &state); err != nil {
		return err
	}
	for _, f := range state.Files {
		if f.Status != "VERIFIED" {
			continue
		}
		// Go and Python share the configured task directory. Never accept another job's file.
		if !filepath.IsAbs(f.Path) || filepath.Base(filepath.Dir(f.Path)) != job.ID || len(f.SHA256) != 64 {
			continue
		}
		id := sha1.Sum([]byte(job.ID + ":" + f.ExportRef))
		className := f.ClassName
		match := regexp.MustCompile(`班级([A-Za-z0-9]+)-`).FindStringSubmatch(filepath.Base(f.Path))
		if len(match) == 2 {
			className = match[1]
		} else {
			className = strings.TrimSpace(strings.TrimPrefix(className, "班级"))
			if strings.Contains(className, "当前") || strings.Contains(className, "所有") || strings.Contains(className, "全部") {
				className = ""
			}
		}
		file := models.RPAFile{ID: hex.EncodeToString(id[:]), JobID: job.ID, ExportRef: f.ExportRef, ClassName: className,
			Path: f.Path, SHA256: f.SHA256, Size: f.Size, GradingStatus: "WAITING_RUBRIC", Message: "等待匹配作业及评分标准"}
		if err := database.DB.Clauses(clause.OnConflict{DoNothing: true}).Create(&file).Error; err != nil {
			return err
		}
	}
	return nil
}

func dispatchFile(file models.RPAFile) error {
	return database.DB.Transaction(func(tx *gorm.DB) error {
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&file, "id = ?", file.ID).Error; err != nil {
			return err
		}
		if file.GradingStatus != "WAITING_RUBRIC" && file.GradingStatus != "READY" {
			return nil
		}
		var job models.RPAJob
		if err := tx.First(&job, "id = ?", file.JobID).Error; err != nil {
			return err
		}
		if job.Status == "CANCELLED" {
			return tx.Model(&file).Update("grading_status", "CANCELLED").Error
		}
		if file.ClassName == "" {
			return tx.Model(&file).Update("message", "无法确认附件班级，请核对官网导出文件命名").Error
		}
		var assignment models.Assignment
		err := tx.Where("course_name = ? AND class_name = ? AND task_name = ?", job.CourseName, file.ClassName, job.AssignmentName).First(&assignment).Error
		if errors.Is(err, gorm.ErrRecordNotFound) {
			assignment = models.Assignment{CourseName: job.CourseName, ClassName: file.ClassName, TaskName: job.AssignmentName}
			if err := tx.Create(&assignment).Error; err != nil {
				return err
			}
		} else if err != nil {
			return err
		}
		file.AssignmentID = assignment.ID
		if !validRubric(assignment) {
			file.Message = "请在作业管理中补齐题目要求和 JSON 评分标准，系统将自动继续批改"
			return tx.Save(&file).Error
		}
		file.GradingJobID = uuid.NewSHA1(uuid.NameSpaceURL, []byte("rpa-grade:"+file.ID)).String()
		grading := models.AsyncJob{ID: file.GradingJobID, JobType: models.JobTypeHomework, ReferenceID: job.ID,
			Status: models.JobStatusPending, Message: "官网下载完成，等待批改"}
		if err := tx.Clauses(clause.OnConflict{DoNothing: true}).Create(&grading).Error; err != nil {
			return err
		}
		// Commit durable dispatch intent before sending; a later retry reuses the same grading ID.
		file.GradingStatus, file.Message = "READY", "等待投递批改队列"
		return tx.Save(&file).Error
	})
}

func syncJobs(ctx context.Context) {
	dispatchMu.Lock()
	defer dispatchMu.Unlock()
	var jobs []models.RPAJob
	database.DB.Where("status NOT IN ?", []string{"FAILED", "CANCELLED", "INTERRUPTED", "SUCCEEDED", "PARTIAL_SUCCESS"}).Find(&jobs)
	for _, job := range jobs {
		if job.Status == "QUEUED" {
			if err := Start(ctx, job.ID, false); err != nil {
				continue
			}
		} else {
			raw, err := WorkerRequest(ctx, "GET", "/tasks/"+job.ID, nil)
			if err != nil {
				var we *WorkerError
				if (errors.As(err, &we) && we.Status == 404) || time.Since(job.UpdatedAt) > time.Minute {
					database.DB.Model(&job).Updates(map[string]any{"status": "INTERRUPTED", "message": "浏览器 Worker 失联或会话丢失，请重新启动任务并登录"})
				}
				continue
			}
			_ = SaveState(job.ID, raw)
		}
	}
	var active []models.RPAJob
	database.DB.Where("status = ?", "RUNNING").Find(&active)
	for _, job := range active {
		launchNavigation(ctx, job)
	}
	// Import checkpoints even after a Worker failure or a partial download.
	database.DB.Where("state_json <> ? AND status IN ?", "", []string{"SUCCEEDED", "PARTIAL_SUCCESS"}).Find(&jobs)
	for _, job := range jobs {
		if err := importFiles(job); err != nil {
			log.Printf("RPA file import %s failed", job.ID)
		}
	}
	var files []models.RPAFile
	database.DB.Where("grading_status IN ?", []string{"WAITING_RUBRIC", "READY"}).Find(&files)
	for _, file := range files {
		if err := dispatchFile(file); err != nil {
			continue
		}
		if err := database.DB.First(&file, "id = ?", file.ID).Error; err != nil {
			continue
		}
		if file.GradingStatus == "READY" && PublishGrade != nil {
			if err := PublishGrade(file.GradingJobID, file.AssignmentID, file.Path); err == nil {
				database.DB.Model(&file).Updates(map[string]any{"grading_status": "PUBLISHED", "message": "已投递批改队列"})
			}
		}
	}
}

func RunScheduler(ctx context.Context) {
	for {
		syncJobs(ctx)
		select {
		case <-ctx.Done():
			return
		case <-time.After(3 * time.Second):
		}
	}
}
