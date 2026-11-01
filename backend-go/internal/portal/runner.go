package portal

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net"
	"net/url"
	"os"
	"sync"
	"time"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/agent/catalog"
	"grading-gateway/internal/models"
	"grading-gateway/internal/rpa"
)

type WorkerCall func(context.Context, string, string, any) ([]byte, error)
type Runner struct {
	Service       *Service
	Broker        *Broker
	Worker        WorkerCall
	detailParsers sync.Map
}

func NewRunner(s *Service, b *Broker) *Runner {
	return &Runner{Service: s, Broker: b, Worker: rpa.WorkerRequest}
}

type operationState struct {
	Status          string          `json:"status"`
	Stage           string          `json:"stage"`
	Message         string          `json:"message"`
	ErrorCode       string          `json:"error_code"`
	StateVersion    uint64          `json:"state_version"`
	VerifiedSubject string          `json:"verified_subject"`
	Catalog         *Catalog        `json:"catalog"`
	HomeworkDetail  *HomeworkDetail `json:"homework_detail,omitempty"`
}

func (r *Runner) Session(ctx context.Context, owner uint, connectionID, opID, jobID string) (map[string]any, error) {
	if endpoint := os.Getenv("RPA_CONTROL_URL"); endpoint != "" {
		u, err := url.Parse(endpoint)
		if err != nil || u.User != nil {
			return nil, ErrUnavailable
		}
		ip := net.ParseIP(u.Hostname())
		if u.Hostname() != "localhost" && (ip == nil || !ip.IsLoopback()) {
			return nil, ErrUnavailable
		}
	}
	c, err := r.Service.connection(ctx, owner, false)
	if err != nil || c.ID != connectionID || c.Status == models.PortalConnectionDisabled {
		return nil, ErrNotFound
	}
	if jobID != "" {
		if c.Status != models.PortalConnectionReady || c.ActiveOperationID != "" {
			return nil, ErrConflict
		}
		if !r.Service.downloadAllowed(ctx, c, jobID) {
			return nil, ErrNotFound
		}
	}
	ticket, err := r.Broker.Issue(c, opID, jobID)
	if err != nil {
		return nil, err
	}
	return map[string]any{"connection_id": c.ID, "profile_generation": c.ProfileGeneration, "credential_version": c.CredentialVersion, "trust_device": c.TrustDevice, "identity_verified": c.VerifiedSubject != "", "credential_ticket": ticket}, nil
}
func (r *Runner) Run(ctx context.Context) {
	for {
		r.Tick(ctx)
		select {
		case <-ctx.Done():
			return
		case <-time.After(2 * time.Second):
		}
	}
}
func (r *Runner) Tick(ctx context.Context) {
	if r.Service.DB == nil {
		return
	}
	var ops []models.PortalOperation
	if r.Service.DB.WithContext(ctx).Where("status NOT IN ?", []string{"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED", "EXPIRED"}).Order("created_at").Limit(32).Find(&ops).Error != nil {
		return
	}
	for _, op := range ops {
		if ctx.Err() != nil {
			return
		}
		if op.Type == "DISCONNECT" {
			r.cleanup(ctx, op)
			continue
		}
		if err := r.ensureTask(ctx, op); err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				_ = r.apply(ctx, op, operationState{Status: "INTERRUPTED", Stage: "INTERRUPTED", Message: "旧认证操作已停止，请使用原任务浏览器重新认证或更新课程", StateVersion: op.StateVersion + 1})
			}
			continue
		}
		raw, err := r.Worker(ctx, "GET", "/tasks/"+op.ID+"/portal-result", nil)
		if err != nil {
			var we *rpa.WorkerError
			var task models.RPAJob
			if r.Service.DB.WithContext(ctx).First(&task, "id = ?", op.ID).Error == nil && terminal(task.Status) {
				if op.Type == "HOMEWORK_DETAIL" && task.Status == "SUCCEEDED" {
					_ = r.apply(ctx, op, operationState{Status: "INTERRUPTED", Stage: "READING_DETAIL", Message: "作业详情结果暂不可用，请重新读取", StateVersion: op.StateVersion + 1})
					continue
				}
				_ = r.apply(ctx, op, operationState{Status: task.Status, Stage: task.Stage, Message: task.Message, StateVersion: op.StateVersion + 1})
				continue
			}
			if (errors.As(err, &we) && we.Status == 404 && op.Status != "QUEUED") || time.Since(op.CreatedAt) > 30*time.Minute {
				_ = r.apply(ctx, op, operationState{Status: "INTERRUPTED", Stage: "INTERRUPTED", Message: "认证或同步已中断，请重新认证后继续", ErrorCode: "WORKER_LOST", StateVersion: op.StateVersion + 1})
			}
			continue
		}
		var state operationState
		if json.Unmarshal(raw, &state) != nil {
			continue
		}
		if op.Type == "HOMEWORK_DETAIL" && state.Status == "SUCCEEDED" {
			r.parseDetail(ctx, op, state)
			continue
		}
		if (op.Type == "AUTH_BOOTSTRAP" || op.Type == "CATALOG_SYNC") && state.Status == "SUCCEEDED" && catalogHasDetails(state.Catalog) {
			r.parseCatalogDetails(ctx, op, state)
			continue
		}
		if err = r.apply(ctx, op, state); err != nil {
			_ = r.apply(ctx, op, operationState{Status: "FAILED", Stage: "FAILED", Message: "目录或身份核验未通过，已保留上次成功目录", ErrorCode: "RESULT_REJECTED", StateVersion: state.StateVersion + 1})
		}
	}
	r.tickWorkflows(ctx)
	r.reconcileImportedAssignments(ctx)
	r.tickImportedRequirements(ctx)
}

// The operation is a business receipt; its browser is an ordinary RPA job.
// Dispatch, page decisions, action receipts and human controls all use rpa.
func (r *Runner) ensureTask(ctx context.Context, op models.PortalOperation) error {
	skills, err := catalog.Default("portal-browser")
	if err != nil {
		return err
	}
	skill := "sync-teacher-portal"
	if op.Type == "HOMEWORK_DETAIL" {
		skill = "read-homework-detail"
	}
	doc, ok := skills[skill]
	if !ok || doc.Instructions == "" {
		return errors.New("missing portal page skill")
	}
	snapshot, _ := json.Marshal(rpa.NavigationSnapshot{Skill: doc})
	return r.Service.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var c models.PortalConnection
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&c, "id = ?", op.ConnectionID).Error; err != nil {
			return err
		}
		if c.ActiveOperationID != op.ID || c.CredentialVersion != op.CredentialVersion || c.ProfileGeneration != op.ProfileGeneration {
			return ErrConflict
		}
		if op.Status != "QUEUED" {
			var existing models.RPAJob
			return tx.Where("id = ? AND user_id = ? AND task_type = ?", op.ID, op.OwnerID, op.Type).First(&existing).Error
		}
		job := models.RPAJob{ID: op.ID, UserID: op.OwnerID, TaskType: op.Type,
			PortalConnectionID: c.ID, ProfileGeneration: c.ProfileGeneration,
			CourseName: "教务账号", AssignmentName: "首次认证", AfterDownload: "download",
			Status: "QUEUED", Stage: "OPEN_PORTAL", Message: "等待页面 Agent 打开教务平台", NavigationJSON: string(snapshot)}
		if op.Type == "CATALOG_SYNC" {
			job.CourseName, job.AssignmentName = "课程目录", "获取课程与作业"
		}
		if op.Type == "HOMEWORK_DETAIL" {
			var selected detailTarget
			if json.Unmarshal([]byte(op.PayloadJSON), &selected) != nil || selected.HomeworkID == "" {
				return errors.New("invalid detail target")
			}
			job.CourseName, job.AssignmentName, job.Term = selected.CourseName, selected.HomeworkName, selected.Term
			target, _ := json.Marshal(selected.Target)
			job.PortalTargetJSON = string(target)
			job.Message = "等待页面 Agent 读取所选作业要求"
		}
		return tx.Clauses(clause.OnConflict{DoNothing: true}).Create(&job).Error
	})
}
func (r *Runner) apply(ctx context.Context, prior models.PortalOperation, state operationState) error {
	valid := map[string]bool{"RUNNING": true, "WAITING_AUTH": true, "WAITING_OTP": true, "WAITING_IDENTITY": true, "SUCCEEDED": true, "FAILED": true, "INTERRUPTED": true, "EXPIRED": true, "CANCELLED": true}
	if !valid[state.Status] || len(state.Message) > 1024 || len(state.Stage) > 64 || len(state.ErrorCode) > 64 {
		return errors.New("invalid portal state")
	}
	return r.Service.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// Every writer locks connection before operation to keep lock order consistent.
		var c models.PortalConnection
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&c, "id = ?", prior.ConnectionID).Error; err != nil {
			return err
		}
		var op models.PortalOperation
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&op, "id = ?", prior.ID).Error; err != nil {
			return err
		}
		if terminal(op.Status) || c.ActiveOperationID != op.ID || c.CredentialVersion != op.CredentialVersion || c.ProfileGeneration != op.ProfileGeneration || state.StateVersion <= op.StateVersion {
			return nil
		}
		// New workers authenticate and collect in one live browser. Persist the
		// verified identity before collection completes, without ending the job.
		if op.Type == "AUTH_BOOTSTRAP" && (state.VerifiedSubject != "" || state.Status == "SUCCEEDED") {
			creds, err := r.Service.credentials(ctx, c)
			if err != nil {
				return err
			}
			if state.VerifiedSubject != creds.Username {
				return errors.New("identity mismatch")
			}
			c.Status = models.PortalConnectionReady
			digest := sha256.Sum256([]byte(c.ID + "\x00" + state.VerifiedSubject))
			c.VerifiedSubject = hex.EncodeToString(digest[:])
			if c.LastAuthenticatedAt == nil || op.Stage != "READING_CATALOG" {
				c.LastAuthenticatedAt = nowPtr()
			}
			if err = tx.Model(&c).Updates(map[string]any{"status": c.Status, "verified_subject": c.VerifiedSubject, "last_authenticated_at": c.LastAuthenticatedAt}).Error; err != nil {
				return err
			}
		}
		if state.Status == "SUCCEEDED" {
			if op.Type == "HOMEWORK_DETAIL" {
				if err := publishDetail(tx, &c, &op, state.HomeworkDetail); err != nil {
					return err
				}
			}
			if op.Type == "CATALOG_SYNC" || op.Type == "AUTH_BOOTSTRAP" && state.Catalog != nil {
				if state.Catalog == nil {
					return errors.New("missing catalog")
				}
				if err := publishCatalog(tx, &c, *state.Catalog); err != nil {
					return err
				}
				for _, course := range state.Catalog.Courses {
					for _, work := range course.Homeworks {
						if work.Detail != nil {
							selected := catalogDetailTarget(c.ID, c.ActiveSnapshotID, course, work)
							if err := publishSelectedDetail(tx, &c, &op, selected, work.Detail); err != nil {
								return err
							}
						}
					}
				}
			}
		}
		op.Status, op.Stage, op.Message, op.ErrorCode, op.StateVersion = state.Status, state.Stage, state.Message, state.ErrorCode, state.StateVersion
		if op.Type == "HOMEWORK_DETAIL" && terminal(state.Status) && state.Status != "SUCCEEDED" {
			var target detailTarget
			if json.Unmarshal([]byte(op.PayloadJSON), &target) == nil {
				if err := tx.Model(&models.Assignment{}).Where("owner_id = ? AND portal_connection_id = ? AND portal_homework_id = ?", op.OwnerID, c.ID, target.HomeworkID).Update("requirements_status", state.Status).Error; err != nil {
					return err
				}
			}
		}
		if err := tx.Save(&op).Error; err != nil {
			return err
		}
		if err := appendEvent(tx, &op, "operation.state", map[string]any{"status": op.Status, "stage": op.Stage, "message": op.Message, "error_code": op.ErrorCode}); err != nil {
			return err
		}
		if terminal(op.Status) {
			if op.Type == "AUTH_BOOTSTRAP" && op.Status == "SUCCEEDED" && state.Catalog == nil {
				// Compatibility for an older worker finishing authentication only.
				next := newOperation(c, "CATALOG_SYNC")
				if err := tx.Create(&next).Error; err != nil {
					return err
				}
				c.ActiveOperationID = next.ID
				if err := appendEvent(tx, &next, "operation.queued", map[string]string{"stage": "QUEUED"}); err != nil {
					return err
				}
			} else {
				c.ActiveOperationID = ""
				if op.Type == "AUTH_BOOTSTRAP" && c.Status != models.PortalConnectionReady || op.Status == "EXPIRED" {
					c.Status = models.PortalConnectionReauth
				}
			}
			return tx.Model(&c).Updates(map[string]any{"active_operation_id": c.ActiveOperationID, "status": c.Status}).Error
		}
		return nil
	})
}
func (r *Runner) cleanup(ctx context.Context, op models.PortalOperation) {
	var jobs []models.RPAJob
	incomplete := r.Service.DB.WithContext(ctx).Where("portal_connection_id = ? AND status NOT IN ?", op.ConnectionID, []string{"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED"}).Find(&jobs).Error != nil
	for _, job := range jobs {
		if rpa.CancelQueued(ctx, &job) != nil {
			incomplete = true
		}
	}
	if _, err := r.Worker(ctx, "DELETE", "/portal/connections/"+op.ConnectionID, nil); err != nil {
		return
	}
	if incomplete {
		return
	}
	_ = r.Service.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var c models.PortalConnection
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&c, "id = ?", op.ConnectionID).Error; err != nil {
			return err
		}
		if c.ActiveOperationID != op.ID {
			return nil
		}
		if err := tx.Model(&op).Updates(map[string]any{"status": "SUCCEEDED", "stage": "DONE", "message": "凭据与专用浏览器状态已删除"}).Error; err != nil {
			return err
		}
		if err := appendEvent(tx, &op, "operation.state", map[string]string{"status": "SUCCEEDED", "stage": "DONE", "message": "凭据与专用浏览器状态已删除"}); err != nil {
			return err
		}
		return tx.Model(&c).Update("active_operation_id", "").Error
	})
}
func (r *Runner) Control(ctx context.Context, owner uint, id, action string, payload any) ([]byte, error) {
	op, err := r.Service.OwnedOperation(ctx, owner, id)
	if err != nil {
		return nil, err
	}
	c, err := r.Service.connection(ctx, owner, false)
	if err != nil || c.ActiveOperationID != op.ID || terminal(op.Status) {
		return nil, ErrConflict
	}
	switch action {
	case "input", "browser":
	default:
		return nil, errors.New("不支持的认证操作")
	}
	method := "POST"
	if action == "browser" {
		method = "GET"
	}
	if action == "browser" {
		action = "frame"
	}
	return r.Worker(ctx, method, "/tasks/"+url.PathEscape(id)+"/"+action, payload)
}
