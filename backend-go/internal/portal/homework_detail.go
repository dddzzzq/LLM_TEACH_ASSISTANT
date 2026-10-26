package portal

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/agent/modelclient"
	"grading-gateway/internal/models"
	"grading-gateway/internal/rpa"
	"grading-gateway/internal/tools"
)

type DetailSource struct {
	ID     string `json:"id,omitempty"`
	Kind   string `json:"kind"`
	Name   string `json:"name,omitempty"`
	Text   string `json:"text,omitempty"`
	Path   string `json:"path,omitempty"` // Internal checkpoint only; removed by the public view.
	SHA256 string `json:"sha256,omitempty"`
	Size   int64  `json:"size,omitempty"`
	Error  string `json:"error,omitempty"`
}
type HomeworkDetail struct {
	CourseID    string            `json:"course_id"`
	HomeworkID  string            `json:"homework_id"`
	Title       string            `json:"title"`
	Question    string            `json:"question"`
	RawQuestion string            `json:"raw_question,omitempty"`
	Processing  *DetailProcessing `json:"processing,omitempty"`
	Complete    bool              `json:"complete"`
	Sources     []DetailSource    `json:"sources"`
	Warnings    []string          `json:"warnings"`
}
type detailTarget struct {
	HomeworkID   string           `json:"homework_id"`
	SnapshotID   string           `json:"snapshot_id"`
	CourseName   string           `json:"course_name"`
	HomeworkName string           `json:"homework_name"`
	Term         string           `json:"term"`
	Target       rpa.PortalTarget `json:"target"`
}

func (r *Runner) CancelDetailRead(ctx context.Context, owner uint, id string) error {
	if r.Service.DB == nil {
		return ErrUnavailable
	}
	cancelled := false
	err := r.Service.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		ts := *r.Service
		ts.DB = tx
		c, err := ts.connection(ctx, owner, true)
		if err != nil {
			return ErrNotFound
		}
		var op models.PortalOperation
		if tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&op, "id = ? AND owner_id = ? AND connection_id = ? AND type = ?", id, owner, c.ID, "HOMEWORK_DETAIL").Error != nil {
			return ErrNotFound
		}
		if terminal(op.Status) {
			return nil
		}
		if c.ActiveOperationID != op.ID {
			return ErrConflict
		}
		if err = tx.Model(&op).Updates(map[string]any{"status": "CANCELLED", "message": "作业要求读取已取消"}).Error; err != nil {
			return err
		}
		if err = tx.Model(&c).Update("active_operation_id", "").Error; err != nil {
			return err
		}
		if err := appendEvent(tx, &op, "operation.cancelled", map[string]string{"status": "CANCELLED"}); err != nil {
			return err
		}
		cancelled = true
		return nil
	})
	if err != nil {
		return err
	}
	if !cancelled {
		return nil
	}
	if value, ok := r.detailParsers.Load(id); ok {
		value.(context.CancelFunc)()
	}
	if job, err := rpa.Owned(owner, id); err == nil {
		_ = rpa.CancelQueued(ctx, job)
	}
	return nil
}

func (s *Service) QueueHomeworkDetail(ctx context.Context, owner uint, homeworkID, snapshotID string) (models.PortalOperation, error) {
	var op models.PortalOperation
	if s.DB == nil {
		return op, ErrUnavailable
	}
	err := s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		ts := *s
		ts.DB = tx
		c, err := ts.connection(ctx, owner, true)
		if err != nil {
			return ErrNotFound
		}
		var work models.PortalHomework
		var course models.PortalCourse
		if snapshotID == "" || snapshotID != c.ActiveSnapshotID {
			return errors.New("目录已更新，请重新选择作业")
		}
		if tx.Where("id = ? AND connection_id = ? AND snapshot_id = ?", homeworkID, c.ID, snapshotID).First(&work).Error != nil {
			return ErrNotFound
		}
		if tx.First(&course, "id = ? AND connection_id = ? AND snapshot_id = ?", work.CourseID, c.ID, snapshotID).Error != nil {
			return ErrNotFound
		}
		if c.Status != models.PortalConnectionReady {
			return errors.New("请先完成教务账号认证")
		}
		if c.ActiveOperationID != "" {
			if tx.First(&op, "id = ? AND type = ?", c.ActiveOperationID, "HOMEWORK_DETAIL").Error == nil {
				var prior detailTarget
				if json.Unmarshal([]byte(op.PayloadJSON), &prior) == nil && prior.HomeworkID == homeworkID && prior.SnapshotID == snapshotID {
					return nil
				}
			}
			return ErrConflict
		}
		var active int64
		if err = tx.Model(&models.RPAJob{}).Where("portal_connection_id = ? AND status NOT IN ?", c.ID, []string{"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED"}).Count(&active).Error; err != nil {
			return err
		}
		if active > 0 {
			return ErrConflict
		}
		selected := detailTarget{HomeworkID: work.ID, SnapshotID: snapshotID, CourseName: course.Title, HomeworkName: work.Title, Term: course.Term,
			Target: rpa.PortalTarget{CourseID: course.PlatformID, HomeworkID: work.PlatformID, CourseURL: course.URL, HomeworkURL: work.URL}}
		payload, _ := json.Marshal(selected)
		op = newOperation(c, "HOMEWORK_DETAIL")
		op.PayloadJSON = string(payload)
		op.Message = "等待读取官网作业要求"
		if err = tx.Create(&op).Error; err != nil {
			return err
		}
		if err = tx.Model(&c).Update("active_operation_id", op.ID).Error; err != nil {
			return err
		}
		return appendEvent(tx, &op, "operation.queued", map[string]string{"stage": "QUEUED"})
	})
	return op, err
}

func (s *Service) HomeworkDetail(ctx context.Context, owner uint, homeworkID string) (map[string]any, error) {
	c, err := s.connection(ctx, owner, false)
	if err != nil {
		return nil, ErrNotFound
	}
	var work models.PortalHomework
	if s.DB.WithContext(ctx).First(&work, "id = ? AND connection_id = ? AND snapshot_id = ?", homeworkID, c.ID, c.ActiveSnapshotID).Error != nil {
		return nil, ErrNotFound
	}
	out := map[string]any{"homework_id": homeworkID, "snapshot_id": c.ActiveSnapshotID, "detail": nil}
	var operations []models.PortalOperation
	if err := s.DB.WithContext(ctx).Where("connection_id = ? AND owner_id = ? AND type = ?", c.ID, owner, "HOMEWORK_DETAIL").Order("created_at DESC,id DESC").Limit(60).Find(&operations).Error; err != nil {
		return nil, err
	}
	for _, op := range operations {
		var target detailTarget
		if json.Unmarshal([]byte(op.PayloadJSON), &target) == nil && target.HomeworkID == homeworkID && target.SnapshotID == c.ActiveSnapshotID && op.ProfileGeneration == c.ProfileGeneration {
			out["operation"] = op
			break
		}
	}
	var cached models.PortalHomeworkDetail
	err = s.DB.WithContext(ctx).First(&cached, "homework_id = ? AND owner_id = ? AND connection_id = ? AND profile_generation = ? AND snapshot_id = ?", homeworkID, owner, c.ID, c.ProfileGeneration, c.ActiveSnapshotID).Error
	if err == nil {
		var detail HomeworkDetail
		if json.Unmarshal([]byte(cached.ContentJSON), &detail) != nil {
			return nil, ErrUnavailable
		}
		for i := range detail.Sources {
			detail.Sources[i].Path = ""
		}
		out["detail"], out["updated_at"], out["operation_id"] = detail, cached.UpdatedAt, cached.OperationID
	} else if !errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, err
	}
	return out, nil
}

func detailResourcePath(jobID, path string) (string, error) {
	root := os.Getenv("RPA_DOWNLOAD_DIR")
	if root == "" {
		root = filepath.Join("..", "downloads")
	}
	base, err := filepath.Abs(filepath.Join(root, jobID, "requirements"))
	if err != nil {
		return "", err
	}
	abs, err := filepath.Abs(path)
	if err != nil || filepath.Dir(abs) != base {
		return "", errors.New("invalid question resource path")
	}
	actual, err := filepath.EvalSymlinks(abs)
	if err != nil || actual != abs {
		return "", errors.New("question resource is unavailable")
	}
	info, err := os.Stat(abs)
	if err != nil || !info.Mode().IsRegular() || info.Size() > 20<<20 {
		return "", errors.New("invalid question resource")
	}
	return abs, nil
}

func (s *Service) HomeworkDetailFile(ctx context.Context, owner uint, homeworkID, sourceID string) (string, string, error) {
	c, err := s.connection(ctx, owner, false)
	if err != nil {
		return "", "", ErrNotFound
	}
	var cached models.PortalHomeworkDetail
	if s.DB.WithContext(ctx).First(&cached, "homework_id = ? AND owner_id = ? AND connection_id = ? AND profile_generation = ? AND snapshot_id = ?", homeworkID, owner, c.ID, c.ProfileGeneration, c.ActiveSnapshotID).Error != nil {
		return "", "", ErrNotFound
	}
	var detail HomeworkDetail
	if json.Unmarshal([]byte(cached.ContentJSON), &detail) != nil {
		return "", "", ErrUnavailable
	}
	for _, source := range detail.Sources {
		if source.ID == sourceID && source.Path != "" {
			path, err := detailResourcePath(cached.OperationID, source.Path)
			if err != nil {
				return "", "", ErrNotFound
			}
			name := filepath.Base(source.Name)
			if filepath.Ext(name) == "" {
				name += filepath.Ext(path)
			}
			return path, name, nil
		}
	}
	return "", "", ErrNotFound
}

// OCR can take seconds; it runs outside SQL locks and never blocks the portal
// scheduler. A cancelled/replaced operation cannot publish a late result.
func (r *Runner) parseDetail(ctx context.Context, op models.PortalOperation, state operationState) {
	parseCtx, cancel := context.WithTimeout(ctx, 4*time.Minute)
	if _, loaded := r.detailParsers.LoadOrStore(op.ID, cancel); loaded {
		cancel()
		return
	}
	go func() {
		defer r.detailParsers.Delete(op.ID)
		defer cancel()
		result, err := resolveDetail(parseCtx, op, state.HomeworkDetail)
		if err != nil {
			state.Status, state.Stage, state.ErrorCode, state.Message = "FAILED", "READING_DETAIL", "DETAIL_REJECTED", "作业详情核验或解析失败，请重试读取"
		} else {
			state.HomeworkDetail, state.Message = result, "作业要求已读取并整理"
			if !result.Complete {
				state.Message = "作业要求未完整整理，请查看图片、附件或模型处理提示"
			}
		}
		if ctx.Err() != nil {
			return
		} // A restart will resume the persisted browser result.
		if err := r.apply(ctx, op, state); err != nil {
			state.Status, state.ErrorCode, state.Message = "FAILED", "DETAIL_REJECTED", "作业详情已失效，请根据当前目录重新读取"
			_ = r.apply(ctx, op, state)
		}
	}()
}

func resolveDetail(ctx context.Context, op models.PortalOperation, raw *HomeworkDetail) (*HomeworkDetail, error) {
	return resolveDetailWithModel(ctx, op, raw, modelclient.NewDeepSeekClient("").CallJSONMessages)
}

func resolveDetailWithModel(ctx context.Context, op models.PortalOperation, raw *HomeworkDetail, call requirementModel) (*HomeworkDetail, error) {
	var selected detailTarget
	if json.Unmarshal([]byte(op.PayloadJSON), &selected) != nil || raw == nil || raw.CourseID != selected.Target.CourseID || raw.HomeworkID != selected.Target.HomeworkID || raw.Title != selected.HomeworkName || len(raw.Sources) == 0 || len(raw.Sources) > 40 {
		return nil, errors.New("detail target mismatch")
	}
	detail := *raw
	detail.Sources = append([]DetailSource(nil), raw.Sources...)
	detail.Warnings = append([]string(nil), raw.Warnings...)
	detail.Processing = nil
	var images []detailImage
	var imageBytes int
	visionExceeded := false
	var total int64
	for i := range detail.Sources {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		source := &detail.Sources[i]
		source.ID = fmt.Sprint(i + 1)
		if source.Kind == "text" {
			if len(source.Text) > 100000 {
				return nil, errors.New("question too large")
			}
		} else if source.Kind == "image" || source.Kind == "attachment" {
			if source.Path != "" {
				path, err := detailResourcePath(op.ID, source.Path)
				if err != nil {
					return nil, err
				}
				data, err := os.ReadFile(path)
				if err != nil {
					return nil, err
				}
				total += int64(len(data))
				digest := sha256.Sum256(data)
				if total > 60<<20 || hex.EncodeToString(digest[:]) != source.SHA256 || int64(len(data)) != source.Size {
					return nil, errors.New("question resource verification failed")
				}
				text, figures, warnings := tools.ExtractRequirementWithImages(ctx, path, data)
				source.Text = text
				for n, figure := range figures {
					imageBytes += len(figure.Data)
					if imageBytes > 30<<20 || len(images) >= 40 {
						visionExceeded = true
						continue
					}
					images = append(images, detailImage{ID: fmt.Sprintf("%s.%d", source.ID, n+1), SourceID: source.ID, RequirementImage: figure})
				}
				for _, warning := range warnings {
					detail.Warnings = append(detail.Warnings, source.Name+"："+warning)
				}
			} else {
				detail.Warnings = append(detail.Warnings, source.Name+"：资源未下载，未完整提取")
			}
		} else {
			return nil, errors.New("unknown question source")
		}
	}
	detail.RawQuestion = joinDetailSources(detail.Sources)
	detail.Question = detail.RawQuestion
	if len(detail.Question) > 100000 {
		return nil, errors.New("question too large")
	}
	if detail.Question == "" {
		detail.Warnings = append(detail.Warnings, "未提取到可用于批改的题目正文")
	} else if visionExceeded {
		detail.Warnings = append(detail.Warnings, "题目图片超过单次理解上限（40张或30MiB），已保留原始材料，未自动采用不完整要求")
	} else if err := organizeRequirements(ctx, &detail, images, call); err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		// Preserve acquired materials for inspection; a failed/incomplete model
		// response must never silently become automatic grading requirements.
		detail.Warnings = append(detail.Warnings, "作业要求整理失败（模型调用、原图读取或结果校验未完成），请重试读取；已保留原始材料")
	}
	detail.Complete = raw.Complete && len(detail.Warnings) == 0 && detail.Question != "" && detail.Processing != nil
	return &detail, nil
}

func publishDetail(tx *gorm.DB, c *models.PortalConnection, op *models.PortalOperation, detail *HomeworkDetail) error {
	var selected detailTarget
	if json.Unmarshal([]byte(op.PayloadJSON), &selected) != nil || detail == nil || selected.SnapshotID != c.ActiveSnapshotID || detail.HomeworkID != selected.Target.HomeworkID || detail.CourseID != selected.Target.CourseID {
		return errors.New("stale detail")
	}
	var work models.PortalHomework
	if tx.First(&work, "id = ? AND connection_id = ? AND snapshot_id = ? AND platform_id = ?", selected.HomeworkID, c.ID, selected.SnapshotID, detail.HomeworkID).Error != nil {
		return ErrNotFound
	}
	raw, err := json.Marshal(detail)
	if err != nil {
		return err
	}
	row := models.PortalHomeworkDetail{HomeworkID: work.ID, OwnerID: c.OwnerID, ConnectionID: c.ID, ProfileGeneration: c.ProfileGeneration, SnapshotID: selected.SnapshotID, OperationID: op.ID, Question: detail.Question, Complete: detail.Complete, ContentJSON: string(raw)}
	if err := tx.Clauses(clause.OnConflict{UpdateAll: true}).Create(&row).Error; err != nil {
		return err
	}
	if detail.Processing != nil {
		return appendEvent(tx, op, "requirements.organized", map[string]any{"version": detail.Processing.Version, "model": detail.Processing.Model, "used_vision": detail.Processing.UsedVision, "complete": detail.Complete, "source_count": len(detail.Sources)})
	}
	return nil
}
