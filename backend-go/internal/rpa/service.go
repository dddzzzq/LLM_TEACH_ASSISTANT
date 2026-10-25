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
	"grading-gateway/internal/materials"
	"grading-gateway/internal/models"
)

var PublishFetch func(string) error
var dispatchMu sync.Mutex
var portalStarts sync.Map

// Credential tickets are supplied by the portal broker, never stored in jobs or Kafka.
var PortalSession func(context.Context, uint, string, string) (map[string]any, error)

type PortalTarget struct {
	CourseID    string `json:"course_id"`
	HomeworkID  string `json:"homework_id"`
	ClassID     string `json:"class_id"`
	ClassName   string `json:"class_name"`
	CourseURL   string `json:"course_url"`
	HomeworkURL string `json:"homework_url"`
}

type Target struct {
	Portal         *PortalTarget `json:"portal,omitempty"`
	CourseName     string        `json:"course_name"`
	AssignmentName string        `json:"assignment_name"`
	Term           string        `json:"term,omitempty"`
	AfterDownload  string        `json:"after_download,omitempty"`
	OutputRoot     string        `json:"output_root,omitempty"`
}

func Create(ctx context.Context, userID uint, target Target) (*models.RPAJob, error) {
	return CreateWithKey(ctx, userID, target, "")
}

func CreateWithKey(ctx context.Context, userID uint, target Target, requestKey string) (*models.RPAJob, error) {
	if target.Portal != nil {
		return nil, fmt.Errorf("官网目标请通过课程作业选择器创建")
	}
	target.CourseName, target.AssignmentName = strings.TrimSpace(target.CourseName), strings.TrimSpace(target.AssignmentName)
	if target.AfterDownload == "" {
		target.AfterDownload = "extract"
	}
	if target.AfterDownload != "download" && target.AfterDownload != "extract" && target.AfterDownload != "grade" {
		return nil, fmt.Errorf("after_download 仅支持 download、extract 或 grade")
	}
	if target.OutputRoot != "" {
		normalized, err := materials.Default().ValidateOutputRoot(target.OutputRoot)
		if err != nil {
			return nil, err
		}
		target.OutputRoot = normalized
	}
	if userID == 0 || target.CourseName == "" || target.AssignmentName == "" || len([]rune(target.CourseName)) > 255 || len([]rune(target.AssignmentName)) > 255 || len([]rune(target.Term)) > 255 {
		return nil, fmt.Errorf("请提供有效的课程、作业名称和登录用户")
	}
	job := &models.RPAJob{ID: uuid.NewString(), UserID: userID, CourseName: target.CourseName,
		AssignmentName: target.AssignmentName, Term: target.Term, AfterDownload: target.AfterDownload, OutputRoot: target.OutputRoot, Status: "QUEUED", Stage: "OPEN_PORTAL", Message: "任务已创建，等待打开登录浏览器"}
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
			if prior.UserID != userID || prior.CourseName != target.CourseName || prior.AssignmentName != target.AssignmentName || prior.Term != target.Term || prior.AfterDownload != target.AfterDownload || prior.OutputRoot != target.OutputRoot {
				return nil, fmt.Errorf("同一请求键不能用于不同下载目标")
			}
			return &prior, nil
		}
		if !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, err
		}
	}
	var connection models.PortalConnection
	if err := database.DB.WithContext(ctx).Where("owner_id = ? AND provider = ? AND status = ?", userID, "xidian", models.PortalConnectionReady).First(&connection).Error; err == nil {
		if connection.ActiveOperationID != "" {
			return nil, fmt.Errorf("教务账号正在认证或更新课程，请稍后启动下载")
		}
		job.PortalConnectionID, job.ProfileGeneration = connection.ID, connection.ProfileGeneration
	} else if !errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, err
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
			if prior.UserID != userID || prior.CourseName != target.CourseName || prior.AssignmentName != target.AssignmentName || prior.Term != target.Term || prior.AfterDownload != target.AfterDownload || prior.OutputRoot != target.OutputRoot {
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
	if job.PortalConnectionID != "" {
		// Saved profiles are exclusive. Reuse the existing queued-job retry rather
		// than starting a second browser which would fail its profile file lock.
		value, _ := portalStarts.LoadOrStore(job.PortalConnectionID, &sync.Mutex{})
		lock := value.(*sync.Mutex)
		lock.Lock()
		defer lock.Unlock()
		var active int64
		if err := database.DB.WithContext(ctx).Model(&models.RPAJob{}).Where("portal_connection_id = ? AND id <> ? AND status IN ?", job.PortalConnectionID, job.ID, []string{"RUNNING", "WAITING_USER"}).Count(&active).Error; err != nil {
			return err
		}
		if active > 0 {
			return fmt.Errorf("教师专用浏览器正在处理另一个任务，当前任务继续排队")
		}
	}
	var checkpoint map[string]any
	if restart {
		_ = json.Unmarshal([]byte(job.StateJSON), &checkpoint)
	}
	if restart && job.PortalWorkflowItemID == "" && job.TaskType == "" {
		// An explicit new browser session may use the current read-only navigation
		// contract. Export checkpoints retain their Skill and all action receipts.
		var snapshot NavigationSnapshot
		_ = json.Unmarshal([]byte(job.NavigationJSON), &snapshot)
		if exported, _ := checkpoint["export_attempted"].(bool); !exported || snapshot.Skill.Name == "" {
			skills, err := catalog.Default("portal-browser")
			if err != nil {
				return err
			}
			doc := skills["fetch-homework"]
			if doc.Name != "fetch-homework" || doc.Instructions == "" {
				return fmt.Errorf("缺少课程导航 Skill")
			}
			snapshot.Skill = doc
		}
		snapshot.Protocol, snapshot.ActionSchema = "browser.v1", nil
		raw, _ := json.Marshal(snapshot)
		if err := database.DB.Model(&models.RPAJob{}).Where("id = ? AND status IN ?", job.ID, []string{"INTERRUPTED", "FAILED", "PARTIAL_SUCCESS"}).UpdateColumn("navigation_json", string(raw)).Error; err != nil {
			return err
		}
	}
	target := Target{CourseName: job.CourseName, AssignmentName: job.AssignmentName, Term: job.Term}
	body := map[string]any{"restart": restart, "checkpoint": checkpoint}
	if job.PortalConnectionID != "" {
		if PortalSession == nil {
			return fmt.Errorf("教务会话服务未就绪")
		}
		capabilities, err := WorkerRequest(ctx, "GET", "/health", nil)
		var health struct {
			PortalAutomation string `json:"portal_automation"`
		}
		if err != nil || json.Unmarshal(capabilities, &health) != nil || health.PortalAutomation != "teacher.portal.agent.v2" {
			return fmt.Errorf("请升级并重启支持页面 Agent 登录的 RPA Worker")
		}
		if job.PortalTargetJSON != "" {
			var selected PortalTarget
			if json.Unmarshal([]byte(job.PortalTargetJSON), &selected) != nil || selected.HomeworkID == "" {
				return fmt.Errorf("官网目标检查点无效")
			}
			target.Portal = &selected
		}
		session, err := PortalSession(ctx, job.UserID, job.PortalConnectionID, job.ID)
		if err != nil {
			return err
		}
		body["portal_session"] = session
	}
	body["task_type"] = job.TaskType
	body["target"] = target
	b, err := WorkerRequest(ctx, "POST", "/tasks/"+jobID, body)
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
	_ = database.DB.Model(&models.RPAFile{}).Where("job_id = ? AND grading_job_id = ''", job.ID).Update("grading_status", "CANCELLED").Error
	stopPendingMaterials(job.ID)
	_, _ = WorkerRequest(ctx, "POST", "/tasks/"+job.ID+"/cancel", nil)
	return nil
}

func SaveState(jobID string, raw []byte) error {
	var state struct {
		Status       string  `json:"status"`
		Stage        string  `json:"stage"`
		Message      string  `json:"message"`
		StateVersion int64   `json:"state_version"`
		Target       *Target `json:"target"`
	}
	if err := json.Unmarshal(raw, &state); err != nil {
		return err
	}
	if state.Status == "" {
		return fmt.Errorf("Worker 返回了无效任务状态")
	}
	updates := map[string]any{"status": state.Status, "stage": state.Stage, "message": state.Message, "state_json": string(raw), "state_version": state.StateVersion}
	if state.Target != nil {
		if strings.TrimSpace(state.Target.CourseName) == "" || strings.TrimSpace(state.Target.AssignmentName) == "" || len([]rune(state.Target.CourseName)) > 255 || len([]rune(state.Target.AssignmentName)) > 255 || len([]rune(state.Target.Term)) > 255 {
			return fmt.Errorf("Worker 返回了无效课程或作业名称")
		}
		// Accepted website names must drive material folders and local assignment matching.
		updates["course_name"], updates["assignment_name"], updates["term"] = state.Target.CourseName, state.Target.AssignmentName, state.Target.Term
	}
	result := database.DB.Model(&models.RPAJob{}).Where("id = ? AND status <> ? AND state_version <= ?", jobID, "CANCELLED", state.StateVersion).Updates(updates)
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return nil
	}
	if state.Status == "CANCELLED" {
		_ = database.DB.Model(&models.RPAFile{}).Where("job_id = ? AND grading_job_id = ''", jobID).Update("grading_status", "CANCELLED").Error
		stopPendingMaterials(jobID)
	}
	_ = database.DB.Model(&models.AsyncJob{}).Where("id = ?", jobID).Updates(map[string]any{"status": state.Status, "message": state.Message}).Error
	_ = cache.SetJobStatus(context.Background(), jobID, state.Status, state.Message)
	return nil
}

func stopPendingMaterials(jobID string) {
	var ids []string
	database.DB.Model(&models.MaterialBatch{}).Where("source_job_id = ? AND grading_job_id = ''", jobID).Pluck("id", &ids)
	if len(ids) == 0 {
		return
	}
	database.DB.Model(&models.MaterialBatch{}).Where("id IN ? AND grading_job_id = ''", ids).Updates(map[string]any{"status": "CANCELLED", "auto_grade": false, "grading_state": "CANCELLED", "message": "下载流程已取消，已保存文件保留"})
	database.DB.Model(&models.AsyncJob{}).Where("id IN ?", ids).Updates(map[string]any{"status": "CANCELLED", "message": "下载流程已取消"})
}

func Detail(job *models.RPAJob) map[string]any {
	state := map[string]any{}
	_ = json.Unmarshal([]byte(job.StateJSON), &state)
	state["job_id"], state["status"], state["stage"], state["message"] = job.ID, job.Status, job.Stage, job.Message
	state["target"] = Target{CourseName: job.CourseName, AssignmentName: job.AssignmentName, Term: job.Term, AfterDownload: job.AfterDownload, OutputRoot: job.OutputRoot}
	state["task_type"] = job.TaskType
	delete(state, "portal_catalog")
	delete(state, "catalog_progress")
	var files []models.RPAFile
	database.DB.Where("job_id = ?", job.ID).Find(&files)
	for i := range files {
		if files[i].GradingJobID != "" {
			var grading models.AsyncJob
			if database.DB.First(&grading, "id = ?", files[i].GradingJobID).Error == nil {
				files[i].GradingStatus = string(grading.Status)
			}
		}
	}
	state["grading_files"] = files
	var batches []models.MaterialBatch
	database.DB.Where("source_job_id = ? AND owner_id = ?", job.ID, job.UserID).Order("created_at DESC").Find(&batches)
	state["material_batches"] = batches
	return state
}

func validRubric(assignment models.Assignment) bool {
	var rubric map[string]any
	return strings.TrimSpace(assignment.Question) != "" && json.Unmarshal([]byte(assignment.Rubric), &rubric) == nil && len(rubric) > 0
}

// A missing class on a structured target means the selected homework as a
// whole. Explicit historical class targets retain their original boundaries.
func portalAllClasses(job models.RPAJob) bool {
	var target PortalTarget
	return job.PortalTargetJSON != "" && json.Unmarshal([]byte(job.PortalTargetJSON), &target) == nil && target.CourseID != "" && target.HomeworkID != "" && target.ClassID == ""
}

func allClassExport(job models.RPAJob) bool {
	if portalAllClasses(job) {
		return true
	}
	var state struct {
		Scope struct {
			Mode string `json:"mode"`
		} `json:"class_scope"`
	}
	return json.Unmarshal([]byte(job.StateJSON), &state) == nil && state.Scope.Mode == "all"
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
		if portalAllClasses(job) || (className == "" && allClassExport(job)) {
			className = "全部班级"
		}
		file := models.RPAFile{ID: hex.EncodeToString(id[:]), JobID: job.ID, ExportRef: f.ExportRef, ClassName: className,
			Path: f.Path, SHA256: f.SHA256, Size: f.Size, GradingStatus: "DOWNLOADED", Message: "附件已下载，尚未提交批改"}
		if err := database.DB.Clauses(clause.OnConflict{DoNothing: true}).Create(&file).Error; err != nil {
			return err
		}
	}
	return nil
}

func dispatchFile(file models.RPAFile) error {
	var job models.RPAJob
	err := database.DB.Transaction(func(tx *gorm.DB) error {
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&file, "id = ?", file.ID).Error; err != nil {
			return err
		}
		if file.GradingStatus == "CANCELLED" || file.GradingJobID != "" {
			return nil
		}
		if err := tx.First(&job, "id = ?", file.JobID).Error; err != nil {
			return err
		}
		if job.Status == "CANCELLED" {
			return tx.Model(&file).Update("grading_status", "CANCELLED").Error
		}
		if portalAllClasses(job) || (file.ClassName == "" && allClassExport(job)) {
			file.ClassName = "全部班级"
		}
		if file.ClassName == "" {
			return tx.Model(&file).Update("message", "无法确认附件班级，请核对官网导出文件命名").Error
		}
		var assignment models.Assignment
		var err error
		if job.AssignmentID != 0 {
			err = tx.Where("id = ? AND owner_id = ?", job.AssignmentID, job.UserID).First(&assignment).Error
			if err != nil {
				return err
			}
			if assignment.ClassName != file.ClassName {
				return fmt.Errorf("下载班级与选定班级不一致")
			}
		} else {
			err = tx.Where("owner_id = ? AND course_name = ? AND class_name = ? AND task_name = ?", job.UserID, job.CourseName, file.ClassName, job.AssignmentName).First(&assignment).Error
		}
		if errors.Is(err, gorm.ErrRecordNotFound) {
			assignment = models.Assignment{OwnerID: job.UserID, CourseName: job.CourseName, ClassName: file.ClassName, TaskName: job.AssignmentName}
			if err := tx.Create(&assignment).Error; err != nil {
				return err
			}
		} else if err != nil {
			return err
		}
		file.AssignmentID = assignment.ID
		file.GradingStatus, file.Message = "DOWNLOADED", "附件已下载，可整理材料或主动发起批改"
		if job.AfterDownload == "grade" && !validRubric(assignment) {
			file.GradingStatus, file.Message = "WAITING_RUBRIC", "已按要求准备后续批改，等待补齐题目与评分标准"
		}
		return tx.Save(&file).Error
	})
	if err != nil {
		return err
	}
	if file.GradingJobID != "" || file.GradingStatus == "CANCELLED" || job.ID == "" || job.Status == "CANCELLED" {
		return nil
	}
	if file.MaterialBatchID == "" && (job.AfterDownload == "extract" || job.AfterDownload == "grade") && file.ClassName != "" {
		batch, err := materials.Default().Create(context.Background(), materials.Actor{UserID: job.UserID, Role: "teacher"}, materials.Request{SourceJobID: job.ID, SourceFileID: file.ID, AssignmentID: file.AssignmentID, OutputRoot: job.OutputRoot, AutoGrade: job.AfterDownload == "grade"}, "rpa-materials:"+file.ID)
		if err != nil {
			database.DB.Model(&file).Update("message", err.Error())
			return err
		}
		return database.DB.Model(&file).Update("material_batch_id", batch.ID).Error
	}
	return nil
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
	database.DB.Where("grading_status IN ? AND grading_job_id = ''", []string{"DOWNLOADED", "WAITING_RUBRIC", "READY"}).Find(&files)
	for _, file := range files {
		if err := dispatchFile(file); err != nil {
			continue
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
