package models

import (
	"encoding/json"
	"time"
)

// AgentRun is the indexed lifecycle of one user turn. Events hold the durable facts.
type AgentRun struct {
	ID          string    `json:"id" gorm:"type:char(36);primaryKey"`
	SessionID   string    `json:"session_id" gorm:"type:char(36);not null;uniqueIndex:idx_agent_run_turn,priority:1"`
	Turn        uint64    `json:"turn" gorm:"not null;uniqueIndex:idx_agent_run_turn,priority:2"`
	UserID      uint      `json:"-" gorm:"not null;uniqueIndex:idx_agent_request,priority:1"`
	RequestHash *string   `json:"-" gorm:"size:64;uniqueIndex:idx_agent_request,priority:2"`
	InputHash   string    `json:"-" gorm:"size:64;not null"`
	Status      string    `json:"status" gorm:"size:24;not null;index"`
	Sequence    uint64    `json:"last_seq" gorm:"not null"`
	ResultJSON  string    `json:"-" gorm:"type:longtext"`
	LeaseUntil  time.Time `json:"lease_until"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// AgentEvent is append-only during a session's lifetime. Sequence is local to a run.
type AgentEvent struct {
	ID        string          `json:"id" gorm:"type:char(36);primaryKey"`
	RunID     string          `json:"run_id" gorm:"type:char(36);not null;uniqueIndex:idx_agent_event_seq,priority:1"`
	SessionID string          `json:"session_id" gorm:"type:char(36);not null;index"`
	Sequence  uint64          `json:"seq" gorm:"not null;uniqueIndex:idx_agent_event_seq,priority:2"`
	Version   int             `json:"version" gorm:"not null"`
	Type      string          `json:"type" gorm:"size:48;not null"`
	Payload   json.RawMessage `json:"payload" gorm:"type:longtext;not null"`
	CreatedAt time.Time       `json:"created_at"`
}
