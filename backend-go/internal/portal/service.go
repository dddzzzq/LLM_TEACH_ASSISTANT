package portal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/models"
)

const Provider = "xidian"

var ErrNotFound = errors.New("记录不存在或无权访问")
var ErrConflict = errors.New("连接正在处理其他操作，请稍后重试")
var ErrUnavailable = errors.New("教务连接服务尚未就绪")

type Service struct {
	DB     *gorm.DB
	Cipher func() (*CredentialCipher, error)
}

func New(db *gorm.DB) *Service { return &Service{DB: db, Cipher: LoadCipher} }

type SaveRequest struct {
	Username    string `json:"username"`
	Password    string `json:"password"`
	TrustDevice bool   `json:"trust_device"`
}
type ConnectionView struct {
	Import                 ImportSummary            `json:"import"`
	Connection             *models.PortalConnection `json:"connection"`
	Status                 string                   `json:"status"`
	HasPassword            bool                     `json:"has_password"`
	CredentialStorageReady bool                     `json:"credential_storage_ready"`
	Operation              *models.PortalOperation  `json:"operation,omitempty"`
}

func (s *Service) connection(ctx context.Context, owner uint, lock bool) (models.PortalConnection, error) {
	var c models.PortalConnection
	if s.DB == nil || owner == 0 {
		return c, ErrUnavailable
	}
	q := s.DB.WithContext(ctx)
	if lock {
		q = q.Clauses(clause.Locking{Strength: "UPDATE"})
	}
	err := q.Where("owner_id = ? AND provider = ?", owner, Provider).First(&c).Error
	return c, err
}
func binding(c models.PortalConnection) CredentialBinding {
	return CredentialBinding{c.OwnerID, c.ID, c.Provider, c.CredentialVersion}
}
func masked(value string) string {
	r := []rune(value)
	if len(r) <= 4 {
		return strings.Repeat("*", len(r))
	}
	return string(r[:2]) + "****" + string(r[len(r)-2:])
}
func terminal(status string) bool {
	switch status {
	case "SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED", "EXPIRED":
		return true
	}
	return false
}
func (s *Service) View(ctx context.Context, owner uint) (ConnectionView, error) {
	_, keyErr := s.Cipher()
	view := ConnectionView{Status: "UNCONFIGURED", CredentialStorageReady: keyErr == nil}
	c, err := s.connection(ctx, owner, false)
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return view, nil
	}
	if err != nil {
		return view, err
	}
	view.Connection, view.Status, view.HasPassword = &c, string(c.Status), len(c.CredentialCiphertext) > 0
	view.Import = s.importSummary(ctx, c)
	if c.ActiveOperationID != "" {
		var op models.PortalOperation
		if s.DB.WithContext(ctx).Where("id = ? AND owner_id = ?", c.ActiveOperationID, owner).First(&op).Error == nil {
			view.Operation = &op
		}
	} else {
		var op models.PortalOperation
		if s.DB.WithContext(ctx).Where("connection_id = ? AND owner_id = ?", c.ID, owner).Order("created_at DESC,id DESC").First(&op).Error == nil {
			view.Operation = &op
		}
	}
	return view, nil
}
func (s *Service) Save(ctx context.Context, owner uint, req SaveRequest) (models.PortalOperation, error) {
	var op models.PortalOperation
	if owner == 0 || s.DB == nil {
		return op, ErrUnavailable
	}
	cipher, err := s.Cipher()
	if err != nil {
		return op, err
	}
	req.Username = strings.TrimSpace(req.Username)
	if len(req.Username) > 255 || len(req.Password) > 4096 {
		return op, errors.New("教务账号或密码过长")
	}
	err = s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// Lock the existing system user as well, so two first saves cannot race.
		var user models.User
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&user, owner).Error; err != nil {
			return err
		}
		if user.Role != models.RoleTeacher && user.Role != models.RoleAdmin {
			return ErrNotFound
		}
		ts := *s
		ts.DB = tx
		c, err := ts.connection(ctx, owner, true)
		fresh := errors.Is(err, gorm.ErrRecordNotFound)
		if err != nil && !fresh {
			return err
		}
		if !fresh && c.ActiveOperationID != "" {
			return ErrConflict
		}
		if !fresh {
			var count int64
			if err = tx.Model(&models.RPAJob{}).Where("portal_connection_id = ? AND status NOT IN ?", c.ID, []string{"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED"}).Count(&count).Error; err != nil {
				return err
			}
			if count > 0 {
				return ErrConflict
			}
		}
		var old Credentials
		if !fresh && len(c.CredentialCiphertext) > 0 {
			old, err = cipher.Decrypt(binding(c), c.CredentialKeyID, c.CredentialCiphertext, c.CredentialNonce)
			if err != nil {
				return err
			}
		}
		if req.Username == "" {
			req.Username = old.Username
		}
		if req.Password == "" {
			if req.Username != old.Username {
				return errors.New("更换教务账号时请同时填写密码")
			}
			req.Password = old.Password
		}
		if req.Username == "" || req.Password == "" {
			return errors.New("首次连接请填写教务账号和密码")
		}
		accountChanged := fresh || req.Username != old.Username
		if fresh {
			c = models.PortalConnection{ID: uuid.NewString(), OwnerID: owner, Provider: Provider}
		}
		c.CredentialVersion++
		if accountChanged || c.ProfileGeneration == 0 || (c.TrustDevice && !req.TrustDevice) {
			c.ProfileGeneration++
			c.VerifiedSubject = ""
			c.LastAuthenticatedAt = nil
		}
		if accountChanged {
			c.ActiveSnapshotID = ""
			c.LastSyncedAt = nil
			if !fresh {
				if err = cancelConnectionWorkflows(tx, c.ID, "教务账号已更换，请从新账号目录重新选择"); err != nil {
					return err
				}
			}
		}
		c.AccountMask, c.TrustDevice, c.CredentialKeyID = masked(req.Username), req.TrustDevice, cipher.keyID
		c.CredentialCiphertext, c.CredentialNonce, err = cipher.Encrypt(binding(c), Credentials{req.Username, req.Password})
		if err != nil {
			return err
		}
		c.Status = models.PortalConnectionInitializing
		op = newOperation(c, "AUTH_BOOTSTRAP")
		c.ActiveOperationID = op.ID
		if err = tx.Save(&c).Error; err != nil {
			return err
		}
		if err = tx.Create(&op).Error; err != nil {
			return err
		}
		return appendEvent(tx, &op, "operation.queued", map[string]any{"stage": "QUEUED"})
	})
	return op, err
}
func newOperation(c models.PortalConnection, kind string) models.PortalOperation {
	return models.PortalOperation{ID: uuid.NewString(), OwnerID: c.OwnerID, ConnectionID: c.ID, Type: kind, Status: "QUEUED", Stage: "QUEUED", Message: "操作已排队", CredentialVersion: c.CredentialVersion, ProfileGeneration: c.ProfileGeneration}
}
func appendEvent(tx *gorm.DB, op *models.PortalOperation, kind string, payload any) error {
	op.Sequence++
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if err = tx.Model(op).UpdateColumn("sequence", op.Sequence).Error; err != nil {
		return err
	}
	return tx.Create(&models.PortalOperationEvent{OperationID: op.ID, Seq: op.Sequence, Version: 1, Type: kind, PayloadJSON: string(raw)}).Error
}
func (s *Service) Queue(ctx context.Context, owner uint, kind string) (models.PortalOperation, error) {
	var op models.PortalOperation
	if kind != "AUTH_BOOTSTRAP" && kind != "CATALOG_SYNC" {
		return op, errors.New("操作类型无效")
	}
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
		if len(c.CredentialCiphertext) == 0 {
			return errors.New("请先保存教务账号密码")
		}
		if c.ActiveOperationID != "" {
			if err = tx.Where("id = ? AND type = ?", c.ActiveOperationID, kind).First(&op).Error; err == nil {
				return nil
			}
			return ErrConflict
		}
		var count int64
		if err = tx.Model(&models.RPAJob{}).Where("portal_connection_id = ? AND status NOT IN ?", c.ID, []string{"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED"}).Count(&count).Error; err != nil {
			return err
		}
		if count > 0 {
			return ErrConflict
		}
		if kind == "CATALOG_SYNC" && c.Status != models.PortalConnectionReady {
			return errors.New("请先完成教务身份认证")
		}
		op = newOperation(c, kind)
		updates := map[string]any{"active_operation_id": op.ID}
		if kind == "AUTH_BOOTSTRAP" {
			updates["status"] = models.PortalConnectionInitializing
		}
		if err = tx.Model(&c).Updates(updates).Error; err != nil {
			return err
		}
		if err = tx.Create(&op).Error; err != nil {
			return err
		}
		return appendEvent(tx, &op, "operation.queued", map[string]string{"stage": "QUEUED"})
	})
	return op, err
}
func (s *Service) OwnedOperation(ctx context.Context, owner uint, id string) (models.PortalOperation, error) {
	var op models.PortalOperation
	if s.DB == nil {
		return op, ErrUnavailable
	}
	err := s.DB.WithContext(ctx).Where("id = ? AND owner_id = ?", id, owner).First(&op).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		err = ErrNotFound
	}
	return op, err
}
func (s *Service) Events(ctx context.Context, owner uint, id string, after uint64) ([]models.PortalOperationEvent, error) {
	if _, err := s.OwnedOperation(ctx, owner, id); err != nil {
		return nil, err
	}
	events := []models.PortalOperationEvent{}
	err := s.DB.WithContext(ctx).Where("operation_id = ? AND seq > ?", id, after).Order("seq").Limit(200).Find(&events).Error
	return events, err
}
func (s *Service) Disconnect(ctx context.Context, owner uint) error {
	if s.DB == nil {
		return ErrUnavailable
	}
	return s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		ts := *s
		ts.DB = tx
		c, err := ts.connection(ctx, owner, true)
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil
		}
		if err != nil {
			return err
		}
		// A durable cleanup operation revokes the current browser before deleting its profiles.
		if c.ActiveOperationID != "" {
			var previous models.PortalOperation
			if err = tx.First(&previous, "id = ?", c.ActiveOperationID).Error; err == nil {
				if err = tx.Model(&previous).Updates(map[string]any{"status": "CANCELLED", "message": "连接已断开"}).Error; err != nil {
					return err
				}
				if err = appendEvent(tx, &previous, "operation.cancelled", map[string]string{"status": "CANCELLED", "message": "连接已断开"}); err != nil {
					return err
				}
			} else if !errors.Is(err, gorm.ErrRecordNotFound) {
				return err
			}
		}
		c.CredentialVersion++
		c.ProfileGeneration++
		c.Status = models.PortalConnectionDisabled
		c.CredentialCiphertext = []byte{}
		c.CredentialNonce = []byte{}
		c.CredentialKeyID = ""
		c.VerifiedSubject = ""
		c.LastAuthenticatedAt = nil
		c.ActiveSnapshotID = ""
		if err = cancelConnectionWorkflows(tx, c.ID, "连接已断开，后续处理已取消"); err != nil {
			return err
		}
		op := newOperation(c, "DISCONNECT")
		c.ActiveOperationID = op.ID
		if err = tx.Save(&c).Error; err != nil {
			return err
		}
		if err = tx.Create(&op).Error; err != nil {
			return err
		}
		return appendEvent(tx, &op, "operation.queued", map[string]string{"stage": "QUEUED"})
	})
}

func (s *Service) Courses(ctx context.Context, owner uint) ([]models.PortalCourse, error) {
	c, err := s.connection(ctx, owner, false)
	if err != nil {
		return nil, ErrNotFound
	}
	rows := []models.PortalCourse{}
	err = s.DB.WithContext(ctx).Where("connection_id = ? AND snapshot_id = ?", c.ID, c.ActiveSnapshotID).Order("term DESC,title").Find(&rows).Error
	return rows, err
}
func (s *Service) Homeworks(ctx context.Context, owner uint, courseID string) (map[string]any, error) {
	c, err := s.connection(ctx, owner, false)
	if err != nil {
		return nil, ErrNotFound
	}
	var course models.PortalCourse
	if s.DB.WithContext(ctx).Where("id = ? AND connection_id = ? AND snapshot_id = ?", courseID, c.ID, c.ActiveSnapshotID).First(&course).Error != nil {
		return nil, ErrNotFound
	}
	works := []models.PortalHomework{}
	classes := []models.PortalCourseClass{}
	if err = s.DB.WithContext(ctx).Where("course_id = ? AND snapshot_id = ?", courseID, c.ActiveSnapshotID).Order("title").Find(&works).Error; err != nil {
		return nil, err
	}
	if err = s.DB.WithContext(ctx).Where("course_id = ? AND snapshot_id = ?", courseID, c.ActiveSnapshotID).Order("name").Find(&classes).Error; err != nil {
		return nil, err
	}
	return map[string]any{"course": course, "homeworks": works, "classes": classes, "snapshot_id": c.ActiveSnapshotID}, nil
}

func (s *Service) credentials(ctx context.Context, c models.PortalConnection) (Credentials, error) {
	cipher, err := s.Cipher()
	if err != nil {
		return Credentials{}, err
	}
	return cipher.Decrypt(binding(c), c.CredentialKeyID, c.CredentialCiphertext, c.CredentialNonce)
}

func stableID(connection, kind, id string) string {
	return uuid.NewSHA1(uuid.NameSpaceURL, []byte(fmt.Sprintf("portal:%s:%s:%s", connection, kind, id))).String()
}
func nowPtr() *time.Time { v := time.Now(); return &v }

func cancelConnectionWorkflows(tx *gorm.DB, connectionID, message string) error {
	var flows []string
	if err := tx.Model(&models.PortalWorkflow{}).Where("connection_id = ? AND status IN ?", connectionID, []string{"QUEUED", "RUNNING", "WAITING_AUTH", "WAITING_REVIEW"}).Pluck("id", &flows).Error; err != nil {
		return err
	}
	if len(flows) == 0 {
		return nil
	}
	if err := tx.Model(&models.PortalWorkflow{}).Where("id IN ?", flows).Update("status", "CANCELLED").Error; err != nil {
		return err
	}
	return tx.Model(&models.PortalWorkflowItem{}).Where("workflow_id IN ? AND status NOT IN ?", flows, []string{"SUCCEEDED", "FAILED", "CANCELLED"}).Updates(map[string]any{"status": "CANCELLED", "message": message}).Error
}

func (s *Service) downloadAllowed(ctx context.Context, c models.PortalConnection, jobID string) bool {
	var job models.RPAJob
	var item models.PortalWorkflowItem
	var flow models.PortalWorkflow
	if s.DB.WithContext(ctx).Where("id = ? AND user_id = ? AND portal_connection_id = ?", jobID, c.OwnerID, c.ID).First(&job).Error != nil || job.Status == "CANCELLED" || job.Status == "SUCCEEDED" {
		return false
	}
	if job.TaskType != "" {
		return false // Authentication/catalog jobs use their active operation's authorization.
	}
	if job.PortalWorkflowItemID == "" {
		return job.PortalTargetJSON == "" && job.ProfileGeneration == c.ProfileGeneration && job.ProfileGeneration > 0
	}
	if s.DB.WithContext(ctx).Where("id = ? AND owner_id = ?", job.PortalWorkflowItemID, c.OwnerID).First(&item).Error != nil {
		return false
	}
	if item.Status != "RUNNING" && item.Status != "RETRY_QUEUED" && item.Status != "WAITING_AUTH" {
		return false
	}
	if terminal(job.Status) || job.Status == "PARTIAL_SUCCESS" {
		if item.Status != "RETRY_QUEUED" {
			return false
		}
	}
	if s.DB.WithContext(ctx).Where("id = ? AND owner_id = ? AND connection_id = ?", item.WorkflowID, c.OwnerID, c.ID).First(&flow).Error != nil || flow.Status == "CANCELLED" || flow.ProfileGeneration != c.ProfileGeneration {
		return false
	}
	return true
}
