package execution

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"grading-gateway/internal/models"
	"grading-gateway/internal/testutil"

	"github.com/google/uuid"
)

func TestStoreAdmissionIdempotencyAndOwnership(t *testing.T) {
	db, user := testutil.ExecutionDB(t)
	s := Store{db}
	ctx := context.Background()
	req := Start{UserID: user, Input: "成绩", Role: "teacher", RequestKey: uuid.NewString()}
	run, reused, err := s.Begin(ctx, req)
	if err != nil || reused {
		t.Fatalf("begin: %v", err)
	}
	prior, reused, err := s.Begin(ctx, req)
	if err != nil || !reused || prior.ID != run.ID {
		t.Fatalf("retry: %+v %v", prior, err)
	}
	conflict := req
	conflict.Input = "其他问题"
	if _, _, err := s.Begin(ctx, conflict); !errors.Is(err, ErrConflict) {
		t.Fatalf("key reused: %v", err)
	}
	if _, _, err := s.Begin(ctx, Start{UserID: user, SessionID: run.SessionID, Input: "并发输入", Role: "teacher"}); !errors.Is(err, ErrBusy) {
		t.Fatalf("concurrent turn admitted: %v", err)
	}
	if _, _, err := s.Events(ctx, user+10000, run.SessionID, run.ID, 0, 0); !errors.Is(err, ErrNotFound) {
		t.Fatalf("foreign read: %v", err)
	}
	if _, _, err := s.Begin(ctx, Start{UserID: user + 10000, SessionID: run.SessionID, Input: "越权", Role: "teacher"}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("foreign write: %v", err)
	}
	if err := s.Finish(ctx, run.ID, Succeeded, "已查询", map[string]string{"reply": "已查询"}); err != nil {
		t.Fatal(err)
	}
	prior, reused, err = s.Begin(ctx, req)
	if err != nil || !reused || prior.Status != Succeeded || prior.ResultJSON == "" {
		t.Fatalf("completed replay: %+v %v", prior, err)
	}
	if err := s.Append(ctx, run.ID, "audit", nil); !errors.Is(err, ErrClosed) {
		t.Fatal("closed run accepted events")
	}
	var count int64
	db.Model(&models.ChatMessage{}).Where("session_id = ?", run.SessionID).Count(&count)
	if count != 2 {
		t.Fatalf("duplicate display messages: %d", count)
	}
}

func TestStoreConcurrentAdmissionAndEventSequence(t *testing.T) {
	db, user := testutil.ExecutionDB(t)
	s := Store{db}
	ctx := context.Background()
	seed, _, err := s.Begin(ctx, Start{UserID: user, Input: "seed", Role: "teacher"})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Finish(ctx, seed.ID, Succeeded, "ok", nil); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	ids := make(chan string, 8)
	errs := make(chan error, 8)
	key := uuid.NewString()
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			run, _, err := s.Begin(ctx, Start{UserID: user, SessionID: seed.SessionID, Input: "same", Role: "teacher", RequestKey: key})
			if err != nil {
				errs <- err
			} else {
				ids <- run.ID
			}
		}()
	}
	wg.Wait()
	close(ids)
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	id := ""
	for value := range ids {
		if id != "" && id != value {
			t.Fatal("same request created multiple runs")
		}
		id = value
	}
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if err := s.Append(ctx, id, "audit.checked", map[string]int{"i": i}); err != nil {
				t.Errorf("append: %v", err)
			}
		}(i)
	}
	wg.Wait()
	run, events, err := s.Events(ctx, user, seed.SessionID, id, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 14 || run.Sequence != 14 {
		t.Fatalf("event count %d seq %d", len(events), run.Sequence)
	}
	for i, e := range events {
		if e.Sequence != uint64(i+1) {
			t.Fatal("event gap")
		}
	}
	page, tail, err := s.Events(ctx, user, seed.SessionID, id, 3, 4)
	if err != nil || page.ID != id || len(tail) != 4 || tail[0].Sequence != 4 || tail[3].Sequence != 7 {
		t.Fatal("cursor pagination changed")
	}
}

func TestStaleRunIsFencedAndLegacyContextImportedOnce(t *testing.T) {
	db, user := testutil.ExecutionDB(t)
	s := Store{db}
	ctx := context.Background()
	session := models.ChatSession{ID: uuid.NewString(), UserID: user, Title: "legacy"}
	if err := db.Create(&session).Error; err != nil {
		t.Fatal(err)
	}
	for _, m := range []models.ChatMessage{{SessionID: session.ID, Role: "user", Content: "旧问题"}, {SessionID: session.ID, Role: "assistant", Content: "旧回复"}} {
		if err := db.Create(&m).Error; err != nil {
			t.Fatal(err)
		}
	}
	run, _, err := s.Begin(ctx, Start{UserID: user, SessionID: session.ID, Input: "下载作业", Role: "teacher"})
	if err != nil {
		t.Fatal(err)
	}
	history, err := s.Context(ctx, user, session.ID, run.ID)
	if err != nil || len(history) != 2 || history[0].Content != "旧问题" {
		t.Fatalf("migration: %+v %v", history, err)
	}
	m := Message{Role: "assistant", ToolCalls: []ToolCall{{ID: "pending", Function: FunctionCall{Name: "fetch_homework"}}}}
	if err := s.Append(ctx, run.ID, "model.message", m); err != nil {
		t.Fatal(err)
	}
	if err := db.Model(&models.AgentRun{}).Where("id = ?", run.ID).Update("lease_until", time.Now().Add(-time.Second)).Error; err != nil {
		t.Fatal(err)
	}
	next, _, err := s.Begin(ctx, Start{UserID: user, SessionID: session.ID, Input: "核对任务状态", Role: "teacher"})
	if err != nil {
		t.Fatal(err)
	}
	if next.Turn != 2 {
		t.Fatal("turn order lost")
	}
	if err := s.Append(ctx, run.ID, "tool.result", ToolResult{CallID: "pending", Content: "late"}); !errors.Is(err, ErrClosed) {
		t.Fatalf("stale writer accepted: %v", err)
	}
	old, view, err := s.Reconstruct(ctx, user, session.ID, run.ID)
	if err != nil || old.Status != Interrupted || len(view.Unresolved) != 1 {
		t.Fatalf("interrupted reconstruction: %+v %v", view, err)
	}
	history, err = s.Context(ctx, user, session.ID, next.ID)
	if err != nil || len(history) != 6 {
		t.Fatalf("next context: %+v %v", history, err)
	}
	var count int64
	db.Model(&models.ChatMessage{}).Where("session_id = ?", session.ID).Count(&count)
	if count != 4 {
		t.Fatalf("legacy import duplicated messages: %d", count)
	}
}

func TestFinishIsAtomicAndMissingTailIsDetected(t *testing.T) {
	db, user := testutil.ExecutionDB(t)
	s := Store{db}
	ctx := context.Background()
	run, _, err := s.Begin(ctx, Start{UserID: user, Input: "atomic", Role: "teacher"})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Finish(ctx, run.ID, Succeeded, "reply", make(chan int)); err == nil {
		t.Fatal("invalid result unexpectedly committed")
	}
	stored, events, err := s.Events(ctx, user, run.SessionID, run.ID, 0, 0)
	if err != nil || stored.Status != Running || stored.Sequence != 2 || len(events) != 2 {
		t.Fatal("failed finalization partially committed")
	}
	if err := s.Finish(ctx, run.ID, Succeeded, "reply", nil); err != nil {
		t.Fatal(err)
	}
	_, events, err = s.Events(ctx, user, run.SessionID, run.ID, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Delete(&models.AgentEvent{}, "id = ?", events[len(events)-1].ID).Error; err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Reconstruct(ctx, user, run.SessionID, run.ID); err == nil {
		t.Fatal("truncated event log accepted")
	}
}
