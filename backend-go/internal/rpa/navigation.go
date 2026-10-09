package rpa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
	"grading-gateway/internal/agent/catalog"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

// DecidePage is injected at composition time. Business scheduling does not import Eino.
var DecidePage func(context.Context, json.RawMessage, catalog.Document, json.RawMessage) (json.RawMessage, error)
var navigationSlots = make(chan struct{}, 4)
var navigationActive sync.Map

type NavigationSnapshot struct {
	Skill        catalog.Document `json:"skill"`
	Protocol     string           `json:"protocol"`
	ActionSchema json.RawMessage  `json:"action_schema,omitempty"`
}
type pageObservation struct {
	Ready       bool            `json:"ready"`
	ID          string          `json:"observation_id"`
	Revision    int64           `json:"revision"`
	Epoch       int64           `json:"epoch"`
	Target      json.RawMessage `json:"target"`
	Stage       string          `json:"stage"`
	Observation json.RawMessage `json:"observation"`
	Recent      json.RawMessage `json:"recent_actions"`
	State       json.RawMessage `json:"state"`
}
type workerRequest func(context.Context, string, string, any) ([]byte, error)
type pageDecider func(context.Context, json.RawMessage, catalog.Document, json.RawMessage) (json.RawMessage, error)

// actionJournal persists intent before crossing the process boundary.
type actionJournal interface {
	Pending() ([]byte, error)
	Save([]byte) error
	Clear() error
}
type databaseJournal struct {
	ctx           context.Context
	taskID, owner string
}

func (j databaseJournal) Pending() ([]byte, error) {
	var row models.RPAJob
	err := database.DB.WithContext(j.ctx).Select("pending_action").First(&row, "id = ?", j.taskID).Error
	return []byte(row.PendingAction), err
}
func (j databaseJournal) Save(v []byte) error { return j.update(string(v)) }
func (j databaseJournal) Clear() error        { return j.update("") }
func (j databaseJournal) update(v string) error {
	result := database.DB.WithContext(j.ctx).Model(&models.RPAJob{}).Where("id = ? AND decision_owner = ? AND decision_lease_until > ?", j.taskID, j.owner, time.Now()).UpdateColumn("pending_action", v)
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected != 1 {
		return fmt.Errorf("页面任务执行租约已失效")
	}
	return nil
}

// One step reconciles pending intent before asking a model to plan another action.
func drivePage(ctx context.Context, id string, snapshot NavigationSnapshot, request workerRequest, decide pageDecider, journal actionJournal) ([]byte, error) {
	pending, err := journal.Pending()
	if err != nil {
		return nil, err
	}
	if len(pending) > 0 {
		return sendAction(ctx, id, pending, request, journal)
	}
	raw, err := request(ctx, "GET", "/tasks/"+id+"/observation", nil)
	if err != nil {
		return nil, err
	}
	var obs pageObservation
	if err = json.Unmarshal(raw, &obs); err != nil {
		return nil, err
	}
	if !obs.Ready {
		return obs.State, nil
	}
	if obs.ID == "" || obs.Revision <= 0 || obs.Epoch <= 0 {
		return nil, fmt.Errorf("Worker 观察协议不完整")
	}
	input, _ := json.Marshal(map[string]any{"target": obs.Target, "stage": obs.Stage, "observation": obs.Observation, "recent_actions": obs.Recent})
	action, err := decide(ctx, input, snapshot.Skill, snapshot.ActionSchema)
	if err != nil {
		recovery, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer cancel()
		state, notifyErr := request(recovery, "POST", "/tasks/"+id+"/decision-failed", map[string]any{"epoch": obs.Epoch})
		if notifyErr != nil {
			return nil, fmt.Errorf("页面决策失败且交接未完成: %w", notifyErr)
		}
		return state, err
	}
	payload, _ := json.Marshal(map[string]any{"action_id": obs.ID, "observation_id": obs.ID, "revision": obs.Revision, "epoch": obs.Epoch, "action": action, "skill_version": snapshot.Skill.Version})
	if err = journal.Save(payload); err != nil {
		return nil, err
	}
	return sendAction(ctx, id, payload, request, journal)
}
func sendAction(ctx context.Context, id string, payload []byte, request workerRequest, journal actionJournal) ([]byte, error) {
	var response []byte
	var err error
	for attempt := 0; attempt < 2; attempt++ {
		response, err = request(ctx, "POST", "/tasks/"+id+"/action", json.RawMessage(payload))
		if err == nil {
			break
		}
		var we *WorkerError
		if errors.As(err, &we) && we.Status == 409 {
			// Worker checks its receipt before checking the observation. A conflict means
			// this proposal was rejected; new observations may be requested safely.
			if clearErr := journal.Clear(); clearErr != nil {
				return nil, clearErr
			}
			return nil, err
		}
		if errors.As(err, &we) && we.Status >= 400 && we.Status < 500 {
			return nil, err
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}
	if err != nil {
		return nil, err
	} // Keep intent for reconciliation after a restart.
	var receipt struct {
		State   json.RawMessage `json:"state"`
		Outcome string          `json:"outcome"`
	}
	if err = json.Unmarshal(response, &receipt); err != nil {
		return nil, err
	}
	if receipt.Outcome != "completed" && receipt.Outcome != "unknown" {
		return nil, fmt.Errorf("Worker 动作回执无效")
	}
	if err = journal.Clear(); err != nil {
		return receipt.State, err
	}
	return receipt.State, nil
}

func launchNavigation(ctx context.Context, job models.RPAJob) {
	if DecidePage == nil || job.Status != "RUNNING" {
		return
	}
	if _, loaded := navigationActive.LoadOrStore(job.ID, true); loaded {
		return
	}
	select {
	case navigationSlots <- struct{}{}:
	default:
		navigationActive.Delete(job.ID)
		return
	}
	go func() {
		defer func() { <-navigationSlots; navigationActive.Delete(job.ID) }()
		owner := uuid.NewString()
		now := time.Now()
		lease := database.DB.WithContext(ctx).Model(&models.RPAJob{}).Where("id = ? AND status = ? AND (decision_lease_until IS NULL OR decision_lease_until < ?)", job.ID, "RUNNING", now).UpdateColumns(map[string]any{"decision_owner": owner, "decision_lease_until": now.Add(2 * time.Minute)})
		if lease.Error != nil || lease.RowsAffected == 0 {
			return
		}
		defer database.DB.Model(&models.RPAJob{}).Where("id = ? AND decision_owner = ?", job.ID, owner).UpdateColumns(map[string]any{"decision_owner": "", "decision_lease_until": nil})
		stepCtx, cancel := context.WithTimeout(ctx, 90*time.Second)
		defer cancel()
		var snapshot NavigationSnapshot
		if json.Unmarshal([]byte(job.NavigationJSON), &snapshot) != nil || snapshot.Skill.Name == "" {
			skills, err := catalog.Default("portal-browser")
			if err != nil {
				handoffNavigation(stepCtx, job.ID)
				return
			}
			snapshot.Skill = skills["fetch-homework"]
		}
		if snapshot.Skill.Name != "fetch-homework" || snapshot.Skill.Instructions == "" {
			handoffNavigation(stepCtx, job.ID)
			return
		}
		capabilities, err := WorkerRequest(stepCtx, "GET", "/capabilities", nil)
		if err != nil {
			return
		}
		var caps struct {
			Protocol string          `json:"protocol_version"`
			Schema   json.RawMessage `json:"action_schema"`
		}
		if json.Unmarshal(capabilities, &caps) != nil || caps.Protocol != "browser.v1" || len(caps.Schema) == 0 {
			handoffNavigation(stepCtx, job.ID)
			return
		}
		if snapshot.Protocol != "" && snapshot.Protocol != caps.Protocol {
			handoffNavigation(stepCtx, job.ID)
			return
		}
		if len(snapshot.ActionSchema) > 0 && !sameJSON(snapshot.ActionSchema, caps.Schema) {
			// An existing task must retain its original action contract.
			handoffNavigation(stepCtx, job.ID)
			return
		}
		snapshot.Protocol = caps.Protocol
		snapshot.ActionSchema = caps.Schema
		encoded, _ := json.Marshal(snapshot)
		if err = database.DB.WithContext(stepCtx).Model(&models.RPAJob{}).Where("id = ? AND decision_owner = ?", job.ID, owner).UpdateColumn("navigation_json", string(encoded)).Error; err != nil {
			return
		}
		raw, stepErr := drivePage(stepCtx, job.ID, snapshot, WorkerRequest, DecidePage, databaseJournal{stepCtx, job.ID, owner})
		if stepErr != nil {
			log.Printf("RPA page step %s incomplete (%T); pending action retained when necessary", job.ID, stepErr)
		}
		if len(raw) > 0 {
			if err := SaveState(job.ID, raw); err != nil {
				log.Printf("RPA page state %s save failed", job.ID)
			}
		}
	}()
}
func sameJSON(a, b []byte) bool {
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	aa, _ := json.Marshal(x)
	bb, _ := json.Marshal(y)
	return string(aa) == string(bb)
}

func handoffNavigation(ctx context.Context, id string) {
	raw, err := WorkerRequest(ctx, "GET", "/tasks/"+id, nil)
	if err != nil {
		return
	}
	var state struct {
		Epoch int64 `json:"control_epoch"`
	}
	if json.Unmarshal(raw, &state) != nil {
		return
	}
	blocked, err := WorkerRequest(ctx, "POST", "/tasks/"+id+"/decision-failed", map[string]any{"epoch": state.Epoch, "reason": "AGENT_CONFIGURATION"})
	if err == nil {
		_ = SaveState(id, blocked)
	}
}
