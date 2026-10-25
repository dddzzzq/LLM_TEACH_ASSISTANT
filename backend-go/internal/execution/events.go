// Package execution owns durable agent facts and their read-only projections.
// It never invokes models or business tools while reconstructing a conversation.
package execution

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

const (
	Running       = "RUNNING"
	Succeeded     = "SUCCEEDED"
	Failed        = "FAILED"
	Interrupted   = "INTERRUPTED"
	Version       = 1
	LeaseDuration = 10 * time.Minute
)

var (
	ErrNotFound = errors.New("execution or session not found")
	ErrBusy     = errors.New("session already has an active execution")
	ErrConflict = errors.New("idempotency key belongs to another request")
	ErrClosed   = errors.New("execution is closed or its lease expired")
	ErrRecord   = errors.New("execution record could not be persisted")
)

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Timestamp  *time.Time `json:"timestamp,omitempty"`
}
type ToolResult struct {
	CallID  string `json:"call_id"`
	Name    string `json:"name"`
	Content string `json:"content"`
	Error   string `json:"error,omitempty"`
}
type Final struct {
	Reply    string `json:"reply"`
	Status   string `json:"status"`
	Response any    `json:"response,omitempty"`
}

type Record func(context.Context, string, any) error

type callIDKey struct{}

func WithCallID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, callIDKey{}, id)
}
func CallID(ctx context.Context) string { id, _ := ctx.Value(callIDKey{}).(string); return id }

// Writer remembers a storage failure even if the framework wraps the returned error.
// It serializes event admission and allows bounded completion writes after cancellation.
type Writer struct {
	mu     sync.Mutex
	append Record
	err    error
}

func NewWriter(append Record) *Writer { return &Writer{append: append} }
func (w *Writer) Record(ctx context.Context, kind string, data any) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.err != nil {
		return w.err
	}
	persistCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	if err := w.append(persistCtx, kind, data); err != nil {
		w.err = fmt.Errorf("%w: %v", ErrRecord, err)
	}
	return w.err
}
func (w *Writer) Err() error { w.mu.Lock(); defer w.mu.Unlock(); return w.err }
