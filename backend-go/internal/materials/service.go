package materials

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

type Actor struct {
	UserID uint
	Role   string
}

func (a Actor) Check() error {
	if a.UserID == 0 || (a.Role != "teacher" && a.Role != "admin") {
		return fmt.Errorf("只有教师或管理员可以整理作业材料")
	}
	return nil
}

type Request struct {
	SourcePath     string `json:"source_path,omitempty"`
	SourceJobID    string `json:"source_job_id,omitempty"`
	SourceFileID   string `json:"source_file_id,omitempty"`
	AssignmentID   uint   `json:"assignment_id,omitempty"`
	CourseName     string `json:"course_name,omitempty"`
	AssignmentName string `json:"assignment_name,omitempty"`
	ClassName      string `json:"class_name,omitempty"`
	OutputRoot     string `json:"output_root,omitempty"`
	AutoGrade      bool   `json:"auto_grade,omitempty"`
}
type Service struct {
	DB           *gorm.DB
	Root         string
	AllowedRoots []string
}

func Default() *Service {
	root := os.Getenv("HOMEWORK_MATERIALS_ROOT")
	if root == "" {
		root = filepath.Join("..", "homework")
	}
	if drivePath.MatchString(root) {
		if normalized, err := NormalizePath(root); err == nil {
			root = normalized
		}
	}
	root, _ = filepath.Abs(root)
	roots := []string{root}
	if configured := os.Getenv("HOMEWORK_ALLOWED_ROOTS"); configured != "" {
		for _, entry := range strings.Split(configured, "|") {
			if normalized, err := NormalizePath(entry); err == nil {
				roots = append(roots, normalized)
			}
		}
	} else {
		if _, err := os.Stat("/mnt/d"); err == nil {
			roots = append(roots, "/mnt/d/dzq")
		}
		if drivePath.MatchString(root) {
			roots = append(roots, filepath.VolumeName(root)+`\dzq`)
		}
		if project, err := filepath.Abs(".."); err == nil {
			roots = append(roots, project)
		}
	}
	return &Service{DB: database.DB, Root: root, AllowedRoots: roots}
}

func (s *Service) Owned(ctx context.Context, actor Actor, id string) (models.MaterialBatch, error) {
	var batch models.MaterialBatch
	if err := actor.Check(); err != nil {
		return batch, err
	}
	err := s.DB.WithContext(ctx).Where("id = ? AND owner_id = ?", id, actor.UserID).First(&batch).Error
	if err != nil {
		return batch, fmt.Errorf("材料批次不存在或无权访问")
	}
	return batch, nil
}

func (s *Service) Create(ctx context.Context, actor Actor, r Request, key string) (models.MaterialBatch, error) {
	var batch models.MaterialBatch
	expectedSHA := ""
	if err := actor.Check(); err != nil {
		return batch, err
	}
	if len(key) > 128 {
		return batch, fmt.Errorf("请求键过长")
	}
	if r.SourceJobID != "" {
		var job models.RPAJob
		if err := s.DB.WithContext(ctx).Where("id = ? AND user_id = ?", r.SourceJobID, actor.UserID).First(&job).Error; err != nil {
			return batch, fmt.Errorf("下载任务不存在或无权访问")
		}
		var files []models.RPAFile
		query := s.DB.WithContext(ctx).Where("job_id = ?", job.ID)
		if r.SourceFileID != "" {
			query = query.Where("id = ?", r.SourceFileID)
		}
		if err := query.Find(&files).Error; err != nil {
			return batch, err
		}
		if len(files) != 1 {
			return batch, fmt.Errorf("请指定一个已下载附件的 source_file_id；先查询下载任务文件清单")
		}
		f := files[0]
		expectedSHA = f.SHA256
		if job.Status == "CANCELLED" {
			return batch, fmt.Errorf("下载任务已取消，请从保存的本地目录创建独立材料批次")
		}
		r.SourcePath, r.SourceFileID = f.Path, f.ID
		r.CourseName, r.AssignmentName, r.ClassName = job.CourseName, job.AssignmentName, f.ClassName
		if r.AssignmentID == 0 {
			r.AssignmentID = f.AssignmentID
		}
	} else if r.SourceFileID != "" {
		return batch, fmt.Errorf("source_file_id 需要对应的 source_job_id")
	}
	if r.AssignmentID != 0 {
		var a models.Assignment
		if err := s.DB.WithContext(ctx).First(&a, r.AssignmentID).Error; err != nil {
			return batch, fmt.Errorf("作业不存在")
		}
		if r.SourceJobID != "" && (r.CourseName != a.CourseName || r.AssignmentName != a.TaskName || r.ClassName != a.ClassName) {
			return batch, fmt.Errorf("下载附件与指定作业的课程、作业或班级不一致")
		}
		r.CourseName, r.AssignmentName, r.ClassName = a.CourseName, a.TaskName, a.ClassName
	}
	if r.AssignmentID == 0 && r.ClassName != "" {
		var matches []models.Assignment
		if err := s.DB.WithContext(ctx).Where("course_name = ? AND task_name = ? AND class_name = ?", r.CourseName, r.AssignmentName, r.ClassName).Limit(2).Find(&matches).Error; err != nil {
			return batch, err
		}
		if len(matches) == 1 {
			r.AssignmentID = matches[0].ID
		}
	}
	if strings.TrimSpace(r.CourseName) == "" || strings.TrimSpace(r.AssignmentName) == "" {
		return batch, fmt.Errorf("请指定作业ID，或提供课程名和作业名")
	}
	for _, v := range []string{r.CourseName, r.AssignmentName, r.ClassName} {
		if len([]rune(v)) > 255 {
			return batch, fmt.Errorf("课程、作业或班级名称过长")
		}
	}
	source, err := s.allowed(r.SourcePath)
	if err != nil {
		return batch, err
	}
	info, err := os.Lstat(source)
	if err != nil || (!info.IsDir() && !info.Mode().IsRegular()) {
		return batch, fmt.Errorf("作业路径不存在或不是目录/文件")
	}
	if !info.IsDir() && !archive(source) {
		return batch, fmt.Errorf("请提供学生作业目录或ZIP/RAR/7z压缩包")
	}
	r.SourcePath = source
	root := r.OutputRoot
	if root == "" {
		root = s.Root
	}
	root, err = s.allowed(root)
	if err != nil {
		return batch, err
	}
	r.OutputRoot = root
	data, _ := json.Marshal(r)
	hash := sha256.Sum256(data)
	id := uuid.NewString()
	if key != "" {
		id = uuid.NewSHA1(uuid.NameSpaceURL, []byte(fmt.Sprintf("materials:%d:%s", actor.UserID, key))).String()
	}
	batch = models.MaterialBatch{ID: id, OwnerID: actor.UserID, AssignmentID: r.AssignmentID, CourseName: r.CourseName, AssignmentName: r.AssignmentName, ClassName: r.ClassName, SourcePath: source, SourceJobID: r.SourceJobID, SourceFileID: r.SourceFileID, OutputDir: filepath.Join(root, component(r.CourseName), component(r.AssignmentName), component(r.ClassName), id), Status: "PENDING", Message: "材料整理已排队，尚未启动批改", AutoGrade: r.AutoGrade, GradingState: "NOT_REQUESTED", RequestHash: hex.EncodeToString(hash[:])}
	if beneath(source, batch.OutputDir) && info.IsDir() {
		return batch, fmt.Errorf("输出位置不能位于被扫描目录内，请选择其他保存根目录")
	}
	if batch.AutoGrade {
		batch.GradingState = "REQUESTED"
	}
	batch.SourceSHA256 = expectedSHA
	err = s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if r.SourceJobID != "" {
			var parent models.RPAJob
			if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("id = ? AND user_id = ?", r.SourceJobID, actor.UserID).First(&parent).Error; err != nil {
				return err
			}
			if parent.Status == "CANCELLED" {
				return fmt.Errorf("下载流程已取消")
			}
			var current models.RPAFile
			if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("id = ? AND job_id = ?", r.SourceFileID, parent.ID).First(&current).Error; err != nil {
				return err
			}
			if current.ClassName != r.ClassName || current.Path != r.SourcePath || (current.AssignmentID != 0 && current.AssignmentID != r.AssignmentID) {
				return fmt.Errorf("附件关联已变化，请刷新后重试")
			}
		}
		result := tx.Clauses(clause.OnConflict{DoNothing: true}).Create(&batch)
		if result.Error != nil {
			return result.Error
		}
		if result.RowsAffected == 0 {
			var prior models.MaterialBatch
			if err := tx.First(&prior, "id = ?", id).Error; err != nil {
				return err
			}
			if prior.OwnerID != actor.UserID || prior.RequestHash != batch.RequestHash {
				return fmt.Errorf("同一请求键不能用于不同材料或处理方式")
			}
			batch = prior
			return nil
		}
		ref := fmt.Sprint(r.AssignmentID)
		if r.SourceJobID != "" {
			ref = r.SourceJobID
			if err := tx.Model(&models.RPAFile{}).Where("id = ? AND (material_batch_id = '' OR material_batch_id IS NULL)", r.SourceFileID).Update("material_batch_id", batch.ID).Error; err != nil {
				return err
			}
		}
		return tx.Create(&models.AsyncJob{ID: id, OwnerID: actor.UserID, JobType: "homework_materials", ReferenceID: ref, Status: models.JobStatusPending, Message: batch.Message}).Error
	})
	return batch, err
}

// Retry reuses the original source and allocates a new execution directory.
// Retrying extraction never renews a previous request to grade automatically.
func (s *Service) Retry(ctx context.Context, actor Actor, id string) (models.MaterialBatch, error) {
	var batch models.MaterialBatch
	if err := actor.Check(); err != nil {
		return batch, err
	}
	err := s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("id = ? AND owner_id = ?", id, actor.UserID).First(&batch).Error; err != nil {
			return fmt.Errorf("材料批次不存在或无权访问")
		}
		if (batch.Status != "FAILED" && batch.Status != "NEEDS_REVIEW") || batch.GradingJobID != "" {
			return fmt.Errorf("仅可重新解压尚未提交批改的失败或待核对批次")
		}
		if batch.SourceJobID != "" {
			var parent models.RPAJob
			if err := tx.Where("id = ? AND user_id = ?", batch.SourceJobID, actor.UserID).First(&parent).Error; err != nil || parent.Status == "CANCELLED" {
				return fmt.Errorf("原下载任务不可用或已取消，请从保存的本地文件独立导入")
			}
		}
		updates := map[string]any{"status": "PENDING", "message": "已使用原附件重新排队解压，完成后等待老师发起批改", "auto_grade": false, "grading_state": "NOT_REQUESTED", "manifest_json": "", "manifest_path": "", "manifest_sha256": "", "student_count": 0, "issue_count": 0, "lease_owner": "", "lease_until": nil}
		if err := tx.Model(&batch).Updates(updates).Error; err != nil {
			return err
		}
		return tx.Model(&models.AsyncJob{}).Where("id = ?", id).Updates(map[string]any{"status": models.JobStatusPending, "message": updates["message"]}).Error
	})
	if err == nil {
		log.Printf("[materials batch=%s] 重新解压已排队，未发起批改", id)
	}
	return batch, err
}

func (s *Service) Cancel(ctx context.Context, actor Actor, id string) error {
	batch, err := s.Owned(ctx, actor, id)
	if err != nil {
		return err
	}
	if batch.GradingJobID != "" {
		return fmt.Errorf("批改任务已创建，请单独管理批改；取消整理不会撤回已提交批改")
	}
	return s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		r := tx.Model(&models.MaterialBatch{}).Where("id = ? AND grading_job_id = ''", id).Updates(map[string]any{"status": "CANCELLED", "auto_grade": false, "grading_state": "CANCELLED", "message": "已取消后续整理和自动批改，已保存文件保留"})
		if r.Error != nil {
			return r.Error
		}
		if r.RowsAffected == 0 {
			return fmt.Errorf("任务状态已变化，请刷新")
		}
		return tx.Model(&models.AsyncJob{}).Where("id = ?", id).Updates(map[string]any{"status": "CANCELLED", "message": "用户取消材料整理"}).Error
	})
}

func (s *Service) ProcessOne(ctx context.Context, id string) error {
	owner := uuid.NewString()
	now := time.Now()
	until := now.Add(time.Minute)
	r := s.DB.WithContext(ctx).Model(&models.MaterialBatch{}).Where("id = ? AND (status = ? OR (status = ? AND lease_until < ?))", id, "PENDING", "PROCESSING", now).Updates(map[string]any{"status": "PROCESSING", "lease_owner": owner, "lease_until": until, "message": "正在解压、识别学生并整理文件"})
	if r.Error != nil || r.RowsAffected == 0 {
		return r.Error
	}
	var batch models.MaterialBatch
	if err := s.DB.WithContext(ctx).First(&batch, "id = ?", id).Error; err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()
	finished := make(chan struct{})
	defer close(finished)
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-finished:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				r := s.DB.Model(&models.MaterialBatch{}).Where("id = ? AND status = ? AND lease_owner = ?", id, "PROCESSING", owner).Update("lease_until", time.Now().Add(time.Minute))
				if r.Error != nil || r.RowsAffected == 0 {
					cancel()
					return
				}
			}
		}
	}()
	root := filepath.Join(batch.OutputDir, owner)
	source, sourceErr := snapshotSource(ctx, batch, root)
	var manifest Manifest
	prepErr := sourceErr
	if prepErr == nil {
		manifest, prepErr = prepare(ctx, batch.ID, batch.AssignmentID, source, root)
	}
	manifestPath := filepath.Join(root, "manifest.json")
	var hash string
	if prepErr == nil {
		raw, _ := json.MarshalIndent(manifest, "", "  ")
		prepErr = os.WriteFile(manifestPath, raw, 0600)
		if prepErr == nil {
			hash, prepErr = HashFile(manifestPath)
		}
	}
	status, message := "READY", "材料已整理，等待老师发起批改"
	if len(manifest.Issues) > 0 {
		status, message = "NEEDS_REVIEW", "部分材料存在问题，请查看清单；可明确选择批改已识别学生"
	}
	if prepErr != nil {
		status, message = "FAILED", prepErr.Error()
	}
	raw, _ := json.Marshal(manifest)
	update := map[string]any{"status": status, "message": message, "manifest_json": string(raw), "manifest_path": manifestPath, "manifest_sha256": hash, "student_count": len(manifest.Students), "issue_count": len(manifest.Issues), "lease_owner": "", "lease_until": nil}
	r = s.DB.Model(&models.MaterialBatch{}).Where("id = ? AND status = ? AND lease_owner = ?", id, "PROCESSING", owner).Updates(update)
	if r.Error != nil {
		return r.Error
	}
	if r.RowsAffected == 0 {
		return nil
	}
	asyncStatus := models.JobStatusSuccess
	if status == "FAILED" {
		asyncStatus = models.JobStatusFailed
	}
	_ = s.DB.Model(&models.AsyncJob{}).Where("id = ? AND status <> ?", id, "CANCELLED").Updates(map[string]any{"status": asyncStatus, "message": message}).Error
	log.Printf("[materials batch=%s] %s: students=%d issues=%d filtered_entries=%d skipped_symlinks=%d", id, status, len(manifest.Students), len(manifest.Issues), manifest.FilteredEntries, manifest.SkippedSymlinks)
	return nil
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(p)
}
func snapshotSource(ctx context.Context, batch models.MaterialBatch, root string) (string, error) {
	info, err := os.Lstat(batch.SourcePath)
	if err != nil {
		return "", fmt.Errorf("源作业路径无法读取")
	}
	if info.IsDir() {
		return batch.SourcePath, nil
	}
	if !info.Mode().IsRegular() || info.Size() > 2<<30 {
		return "", fmt.Errorf("源压缩包超过2GiB或不是普通文件")
	}
	dir := filepath.Join(root, "original")
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	in, err := os.Open(batch.SourcePath)
	if err != nil {
		return "", err
	}
	defer in.Close()
	actual, err := in.Stat()
	if err != nil || !os.SameFile(info, actual) {
		return "", fmt.Errorf("源压缩包已变化")
	}
	path := filepath.Join(dir, filepath.Base(batch.SourcePath))
	out, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return "", err
	}
	n, err := io.Copy(out, io.LimitReader(contextReader{ctx: ctx, reader: in}, (2<<30)+1))
	out.Close()
	if err != nil {
		return "", err
	}
	if n > 2<<30 {
		return "", fmt.Errorf("源压缩包超过2GiB")
	}
	if batch.SourceSHA256 != "" {
		actual, err := HashFile(path)
		if err != nil || actual != batch.SourceSHA256 {
			return "", fmt.Errorf("下载附件的校验值发生变化，请重新下载")
		}
	}
	return path, nil
}

// Run resumes durable preparations. Grading is injected by composition, never by this package.
func (s *Service) Run(ctx context.Context, continueGrade func(context.Context, models.MaterialBatch) error) {
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	for {
		var jobs []models.MaterialBatch
		if err := s.DB.WithContext(ctx).Where("status = ? OR (status = ? AND lease_until < ?)", "PENDING", "PROCESSING", time.Now()).Order("created_at ASC").Limit(1).Find(&jobs).Error; err == nil {
			for _, b := range jobs {
				if err := s.ProcessOne(ctx, b.ID); err != nil {
					log.Printf("[materials batch=%s] 整理暂未更新 (%T)", b.ID, err)
				}
			}
		}
		if continueGrade != nil {
			var batches []models.MaterialBatch
			if err := s.DB.WithContext(ctx).Where("auto_grade = ? AND status = ? AND grading_state IN ?", true, "READY", []string{"REQUESTED", "WAITING_REQUIREMENTS"}).Limit(20).Find(&batches).Error; err == nil {
				for _, batch := range batches {
					if err := continueGrade(ctx, batch); err != nil && !errors.Is(err, context.Canceled) {
						log.Printf("[materials batch=%s] 后续批改暂未提交 (%T)", batch.ID, err)
					}
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
