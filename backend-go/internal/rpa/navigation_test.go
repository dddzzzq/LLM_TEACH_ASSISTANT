package rpa

import (
	"context"
	"encoding/json"
	"errors"
	"grading-gateway/internal/agent/catalog"
	"strings"
	"testing"
)

type memoryJournal struct {
	pending []byte
	fail    bool
}

func (j *memoryJournal) Pending() ([]byte, error) { return j.pending, nil }
func (j *memoryJournal) Save(v []byte) error {
	if j.fail {
		return errors.New("store unavailable")
	}
	j.pending = append([]byte(nil), v...)
	return nil
}
func (j *memoryJournal) Clear() error { j.pending = nil; return nil }
func TestLostActionResponseIsReconciledBeforeAnotherModelCall(t *testing.T) {
	j := &memoryJournal{}
	modelCalls, posts := 0, 0
	lost := true
	var first string
	req := func(_ context.Context, method, path string, body any) ([]byte, error) {
		if method == "GET" {
			return []byte(`{"ready":true,"observation_id":"o1","epoch":3,"revision":2,"target":{},"observation":{}}`), nil
		}
		posts++
		wire, _ := json.Marshal(body)
		if first == "" {
			first = string(wire)
		} else if first != string(wire) {
			t.Fatal("retry changed action identity")
		}
		if lost {
			return nil, errors.New("response lost")
		}
		return []byte(`{"outcome":"completed","state":{"status":"RUNNING"}}`), nil
	}
	decide := func(context.Context, json.RawMessage, catalog.Document, json.RawMessage) (json.RawMessage, error) {
		modelCalls++
		return []byte(`{"tool":"click","element_ref":"p0"}`), nil
	}
	_, err := drivePage(context.Background(), "job", NavigationSnapshot{}, req, decide, j)
	if err == nil || len(j.pending) == 0 || posts != 2 {
		t.Fatal("uncertain intent not retained")
	}
	lost = false
	state, err := drivePage(context.Background(), "job", NavigationSnapshot{}, req, decide, j)
	if err != nil || modelCalls != 1 || len(j.pending) != 0 || !strings.Contains(string(state), "RUNNING") {
		t.Fatalf("calls=%d err=%v", modelCalls, err)
	}
}
func TestNoBrowserActionWithoutDurableIntent(t *testing.T) {
	j := &memoryJournal{fail: true}
	posts := 0
	req := func(_ context.Context, method, _ string, _ any) ([]byte, error) {
		if method == "POST" {
			posts++
		}
		return []byte(`{"ready":true,"observation_id":"o1","revision":1,"epoch":1}`), nil
	}
	_, err := drivePage(context.Background(), "job", NavigationSnapshot{}, req, func(context.Context, json.RawMessage, catalog.Document, json.RawMessage) (json.RawMessage, error) {
		return []byte(`{"tool":"scroll","delta_y":1}`), nil
	}, j)
	if err == nil || posts != 0 {
		t.Fatal("action ran before durable intent")
	}
}
func TestPageModelFailureHandsControlToHuman(t *testing.T) {
	epoch := int64(0)
	req := func(_ context.Context, method, path string, body any) ([]byte, error) {
		if method == "GET" {
			return []byte(`{"ready":true,"observation_id":"o1","revision":1,"epoch":7}`), nil
		}
		if !strings.HasSuffix(path, "/decision-failed") {
			t.Fatal("failure executed browser action")
		}
		epoch = body.(map[string]any)["epoch"].(int64)
		return []byte(`{"status":"WAITING_USER"}`), nil
	}
	state, err := drivePage(context.Background(), "job", NavigationSnapshot{}, req, func(context.Context, json.RawMessage, catalog.Document, json.RawMessage) (json.RawMessage, error) {
		return nil, errors.New("model offline")
	}, &memoryJournal{})
	if err == nil || epoch != 7 || !strings.Contains(string(state), "WAITING_USER") {
		t.Fatal("missing human handoff")
	}
}
