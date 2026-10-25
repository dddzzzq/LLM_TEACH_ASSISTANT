package execution

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"grading-gateway/internal/models"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

type Store struct{ DB *gorm.DB }
type Start struct {
	UserID                             uint
	SessionID, RequestKey, Input, Role string
}

func digest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func (s Store) own(ctx context.Context, user uint, session string) error {
	var row models.ChatSession
	err := s.DB.WithContext(ctx).Where("id = ? AND user_id = ?", session, user).First(&row).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return ErrNotFound
	}
	return err
}

func (s Store) existing(ctx context.Context, req Start) (*models.AgentRun, error) {
	if req.RequestKey == "" {
		return nil, nil
	}
	var run models.AgentRun
	err := s.DB.WithContext(ctx).Where("user_id = ? AND request_hash = ?", req.UserID, digest(req.RequestKey)).First(&run).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if run.InputHash != digest(req.Role+"\n"+req.Input) || req.SessionID != "" && run.SessionID != req.SessionID {
		return nil, ErrConflict
	}
	if err := s.own(ctx, req.UserID, run.SessionID); err != nil {
		return nil, err
	}
	return &run, nil
}

// Begin atomically admits one turn, imports legacy context once, and writes its user message.
// Same-key retries return the original run; they never replay business effects.
func (s Store) Begin(ctx context.Context, req Start) (run models.AgentRun, reused bool, err error) {
	originalRequest := req
	if prior, e := s.existing(ctx, req); prior != nil || e != nil {
		if prior != nil {
			run = *prior
		}
		return run, prior != nil, e
	}
	err = s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if req.SessionID == "" {
			req.SessionID = uuid.NewString()
			if err := tx.Create(&models.ChatSession{ID: req.SessionID, UserID: req.UserID, Title: "新会话"}).Error; err != nil {
				return err
			}
		}
		var session models.ChatSession
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("id = ? AND user_id = ?", req.SessionID, req.UserID).First(&session).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return ErrNotFound
			}
			return err
		}
		if prior, err := (Store{tx}).existing(ctx, req); prior != nil || err != nil {
			if prior != nil {
				run = *prior
				reused = true
			}
			return err
		}
		now := time.Now()
		var active []models.AgentRun
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("session_id = ? AND status = ?", req.SessionID, Running).Find(&active).Error; err != nil {
			return err
		}
		for i := range active {
			if active[i].LeaseUntil.After(now) {
				return ErrBusy
			}
			if err := appendEvent(tx, &active[i], "run.interrupted", Final{Status: Interrupted, Reply: "上次执行中断；已记录结果可查询，未记录的操作结果需要核对，请勿直接重复提交。"}); err != nil {
				return err
			}
			if err := tx.Model(&active[i]).Update("status", Interrupted).Error; err != nil {
				return err
			}
		}
		var lastTurn uint64
		if err := tx.Model(&models.AgentRun{}).Where("session_id = ?", req.SessionID).Select("COALESCE(MAX(turn), 0)").Scan(&lastTurn).Error; err != nil {
			return err
		}
		run = models.AgentRun{ID: uuid.NewString(), Turn: lastTurn + 1, SessionID: req.SessionID, UserID: req.UserID, InputHash: digest(req.Role + "\n" + req.Input), Status: Running, LeaseUntil: now.Add(LeaseDuration)}
		if req.RequestKey != "" {
			hash := digest(req.RequestKey)
			run.RequestHash = &hash
		}
		if err := tx.Create(&run).Error; err != nil {
			return err
		}
		if err := appendEvent(tx, &run, "run.started", map[string]any{"role": req.Role}); err != nil {
			return err
		}
		if lastTurn == 0 {
			var rows []models.ChatMessage
			if err := tx.Where("session_id = ?", req.SessionID).Order("created_at DESC, id DESC").Limit(20).Find(&rows).Error; err != nil {
				return err
			}
			legacy := []Message{}
			for i := len(rows) - 1; i >= 0; i-- {
				if rows[i].Role == "user" || rows[i].Role == "assistant" {
					legacy = append(legacy, Message{Role: rows[i].Role, Content: rows[i].Content, Timestamp: &rows[i].CreatedAt})
				}
			}
			if len(legacy) > 0 {
				if err := appendEvent(tx, &run, "history.imported", legacy); err != nil {
					return err
				}
			}
		}
		if err := appendEvent(tx, &run, "user.message", Message{Role: "user", Content: req.Input}); err != nil {
			return err
		}
		if err := tx.Create(&models.ChatMessage{SessionID: req.SessionID, Role: "user", Content: req.Input}).Error; err != nil {
			return err
		}
		return tx.Model(&session).Update("updated_at", now).Error
	})
	// A simultaneous first turn can race on the global user/request-key unique index.
	if err != nil && req.RequestKey != "" {
		if prior, e := s.existing(ctx, originalRequest); errors.Is(e, ErrConflict) {
			return run, false, e
		} else if e == nil && prior != nil {
			return *prior, true, nil
		}
	}
	return
}

func appendEvent(tx *gorm.DB, run *models.AgentRun, kind string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	run.Sequence++
	event := models.AgentEvent{ID: uuid.NewString(), RunID: run.ID, SessionID: run.SessionID, Sequence: run.Sequence, Version: Version, Type: kind, Payload: data}
	if err := tx.Create(&event).Error; err != nil {
		return err
	}
	run.LeaseUntil = time.Now().Add(LeaseDuration)
	return tx.Model(run).Updates(map[string]any{"sequence": run.Sequence, "lease_until": run.LeaseUntil, "updated_at": time.Now()}).Error
}

func (s Store) mutate(ctx context.Context, id string, fn func(*gorm.DB, *models.AgentRun) error) error {
	return s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var run models.AgentRun
		if err := tx.First(&run, "id = ?", id).Error; err != nil {
			return err
		}
		var session models.ChatSession
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&session, "id = ?", run.SessionID).Error; err != nil {
			return err
		}
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&run, "id = ?", id).Error; err != nil {
			return err
		}
		if run.Status != Running || !run.LeaseUntil.After(time.Now()) {
			return ErrClosed
		}
		return fn(tx, &run)
	})
}
func (s Store) Append(ctx context.Context, id, kind string, payload any) error {
	return s.mutate(ctx, id, func(tx *gorm.DB, run *models.AgentRun) error { return appendEvent(tx, run, kind, payload) })
}
func (s Store) Finish(ctx context.Context, id, status, reply string, response any) error {
	if status != Succeeded && status != Failed && status != Interrupted {
		return ErrConflict
	}
	return s.mutate(ctx, id, func(tx *gorm.DB, run *models.AgentRun) error {
		if err := appendEvent(tx, run, "run.finished", Final{Reply: reply, Status: status, Response: response}); err != nil {
			return err
		}
		data, err := json.Marshal(response)
		if err != nil {
			return err
		}
		if err := tx.Model(run).Updates(map[string]any{"status": status, "result_json": string(data)}).Error; err != nil {
			return err
		}
		if err := tx.Create(&models.ChatMessage{SessionID: run.SessionID, Role: "assistant", Content: reply}).Error; err != nil {
			return err
		}
		return tx.Model(&models.ChatSession{}).Where("id = ?", run.SessionID).Update("updated_at", time.Now()).Error
	})
}

func (s Store) Runs(ctx context.Context, user uint, session string, limit int) ([]models.AgentRun, error) {
	if err := s.own(ctx, user, session); err != nil {
		return nil, err
	}
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	rows := []models.AgentRun{}
	err := s.DB.WithContext(ctx).Where("session_id = ? AND user_id = ?", session, user).Order("turn DESC").Limit(limit).Find(&rows).Error
	return rows, err
}
func (s Store) Events(ctx context.Context, user uint, session, id string, after uint64, limit int) (models.AgentRun, []models.AgentEvent, error) {
	var run models.AgentRun
	if err := s.own(ctx, user, session); err != nil {
		return run, nil, err
	}
	err := s.DB.WithContext(ctx).Where("id = ? AND session_id = ? AND user_id = ?", id, session, user).First(&run).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		err = ErrNotFound
	}
	if err != nil {
		return run, nil, err
	}
	rows := []models.AgentEvent{}
	q := s.DB.WithContext(ctx).Where("run_id = ? AND sequence > ? AND sequence <= ?", id, after, run.Sequence).Order("sequence ASC")
	if limit > 0 {
		q = q.Limit(limit)
	}
	err = q.Find(&rows).Error
	if err == nil && after == 0 && limit <= 0 && uint64(len(rows)) != run.Sequence {
		err = fmt.Errorf("execution %s has an incomplete event log", run.ID)
	}
	return run, rows, err
}

// priorContext keeps complete turns, so context windows cannot orphan tool replies.
func (s Store) priorContext(ctx context.Context, user uint, session string, turn uint64) (Projection, error) {
	var runs []models.AgentRun
	view := Projection{Messages: []Message{}, Unresolved: []string{}}
	if err := s.DB.WithContext(ctx).Where("session_id = ? AND user_id = ? AND turn < ?", session, user, turn).Order("turn DESC").Limit(9).Find(&runs).Error; err != nil {
		return view, err
	}
	for i := len(runs) - 1; i >= 0; i-- {
		_, events, err := s.Events(ctx, user, session, runs[i].ID, 0, 0)
		if err != nil {
			return view, err
		}
		part, err := Project(events)
		if err != nil {
			return view, err
		}
		view.Messages = append(view.Messages, part.Messages...)
		view.Unresolved = append(view.Unresolved, part.Unresolved...)
	}
	return view, nil
}

// Context reconstructs previous turns; Run adds the current user message exactly once.
func (s Store) Context(ctx context.Context, user uint, session, current string) ([]Message, error) {
	run, events, err := s.Events(ctx, user, session, current, 0, 0)
	if err != nil {
		return nil, err
	}
	view, err := s.priorContext(ctx, user, session, run.Turn)
	if err != nil {
		return nil, err
	}
	// On migration, the current run carries the only legacy import.
	for _, e := range events {
		if e.Type == "history.imported" {
			var legacy []Message
			if err := json.Unmarshal(e.Payload, &legacy); err != nil {
				return nil, err
			}
			view.Messages = append(view.Messages, legacy...)
		}
	}
	return view.Messages, nil
}

// Reconstruct returns the selected turn's bounded conversation and exact last model request.
func (s Store) Reconstruct(ctx context.Context, user uint, session, id string) (models.AgentRun, Projection, error) {
	run, events, err := s.Events(ctx, user, session, id, 0, 0)
	if err != nil {
		return run, Projection{}, err
	}
	view, err := Project(events)
	if err != nil {
		return run, view, err
	}
	history, err := s.priorContext(ctx, user, session, run.Turn)
	if err != nil {
		return run, view, err
	}
	view.Messages = append(history.Messages, view.Messages...)
	view.Unresolved = append(history.Unresolved, view.Unresolved...)
	return run, view, nil
}
