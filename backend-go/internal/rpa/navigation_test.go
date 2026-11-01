package rpa

import (
	"context"
	"encoding/json"
	"errors"
	"grading-gateway/internal/agent/catalog"
	"grading-gateway/internal/models"
	"strings"
	"testing"
)

type memoryJournal struct {
	pending []byte
	fail    bool
}

func TestPortalReturnToolMigrationAndSkillScope(t *testing.T) {
	old := json.RawMessage(`{"oneOf":[{"type":"object","properties":{"tool":{"enum":["click"]},"element_ref":{"type":"string"}},"required":["tool","element_ref"]}]}`)
	added := `{"type":"object","properties":{"tool":{"type":"string","enum":["return_to_course_list"]}},"required":["tool"],"additionalProperties":false}`
	current := json.RawMessage(strings.TrimSuffix(string(old), "]}") + "," + added + "]}")
	if !portalToolExtension(old, current) {
		t.Fatal("known return tool extension rejected")
	}
	for _, incompatible := range []string{
		strings.Replace(string(current), "element_ref", "new_ref", 1),
		strings.Replace(string(current), "return_to_course_list", "goto", 1),
		strings.Replace(string(current), `"additionalProperties":false`, `"additionalProperties":true`, 1),
	} {
		if portalToolExtension(old, json.RawMessage(incompatible)) {
			t.Fatal("incompatible migration accepted")
		}
	}
	for _, kind := range []string{"AUTH_BOOTSTRAP", "CATALOG_SYNC"} {
		if pageSkill(models.RPAJob{TaskType: kind}) != "sync-teacher-portal" {
			t.Fatal("portal job did not select combined skill")
		}
	}
	if pageSkill(models.RPAJob{}) != "fetch-homework" || legacyPortalSkill(models.RPAJob{}, "collect-homework") {
		t.Fatal("download skill scope changed")
	}
	if !legacyPortalSkill(models.RPAJob{TaskType: "CATALOG_SYNC"}, "collect-homework") ||
		!legacyPortalSkill(models.RPAJob{TaskType: "AUTH_BOOTSTRAP"}, "fetch-homework") {
		t.Fatal("known old portal skill rejected")
	}
}

func TestCatalogDetailSelectionMigrationKeepsTargetSchemaStrict(t *testing.T) {
	old := json.RawMessage(`{"oneOf":[{"type":"object","properties":{"tool":{"enum":["click"]},"element_ref":{"type":"string"}},"required":["tool","element_ref"]}]}`)
	added := `{"type":"object","properties":{"tool":{"type":"string","enum":["select_homework_detail"]},"homework_id":{"type":"string"}},"required":["tool","homework_id"],"additionalProperties":false}`
	current := json.RawMessage(strings.TrimSuffix(string(old), "]}") + "," + added + "]}")
	if !portalToolExtension(old, current) {
		t.Fatal("known detail-selection extension rejected")
	}
	for _, invalid := range []string{
		strings.ReplaceAll(string(current), "homework_id", "url"),
		strings.Replace(string(current), `"additionalProperties":false`, `"additionalProperties":true`, 1),
	} {
		if portalToolExtension(old, json.RawMessage(invalid)) {
			t.Fatal("detail selection accepted a changed target contract")
		}
	}
}

func TestSliderMigrationPreservesExistingActionContracts(t *testing.T) {
	old := json.RawMessage(`{"oneOf":[{"type":"object","properties":{"tool":{"enum":["click"]},"element_ref":{"type":"string"}},"required":["tool","element_ref"]}]}`)
	added := `{"type":"object","properties":{"tool":{"type":"string","enum":["solve_slider_captcha"]}},"required":["tool"],"additionalProperties":false}`
	current := json.RawMessage(strings.TrimSuffix(string(old), "]}") + "," + added + "]}")
	if !sliderToolExtension(old, current) {
		t.Fatal("known additive extension rejected")
	}
	for _, incompatible := range []json.RawMessage{
		old, current[:len(current)-1],
		json.RawMessage(strings.Replace(string(current), "element_ref", "new_ref", 1)),
		json.RawMessage(strings.Replace(string(current), "solve_slider_captcha", "run_python", 1)),
		json.RawMessage(strings.Replace(string(current), `"additionalProperties":false`, `"additionalProperties":true`, 1)),
	} {
		if sliderToolExtension(old, incompatible) {
			t.Fatalf("incompatible migration accepted: %s", incompatible)
		}
	}
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

func TestSyncJobUsesDetailSkillForSelectionAndRestoresCatalogSkill(t *testing.T) {
	skills, err := catalog.Default("portal-browser")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"sync-teacher-portal", "collect-homework"} {
		t.Run(name, func(t *testing.T) {
			snapshot := NavigationSnapshot{Skill: skills[name]}
			for _, phase := range []struct {
				observation, want string
			}{
				{`{"task_type":"AUTH_BOOTSTRAP","homework_detail":{"open":false,"ready":false},"catalog":{"courses":[{"complete":false}]}}`, "read-homework-detail"},
				{`{"task_type":"AUTH_BOOTSTRAP","homework_detail":{"open":true,"ready":true}}`, "read-homework-detail"},
				{`{"task_type":"AUTH_BOOTSTRAP","catalog":{"pending_homeworks":[]}}`, name},
			} {
				var sentVersion string
				request := func(_ context.Context, method, path string, payload any) ([]byte, error) {
					if !strings.HasPrefix(path, "/tasks/original-job/") {
						t.Fatal("detail phase created another browser task", path)
					}
					if method == "GET" {
						return []byte(`{"ready":true,"observation_id":"o1","revision":1,"epoch":1,"observation":` + phase.observation + `}`), nil
					}
					var body struct {
						SkillVersion string `json:"skill_version"`
					}
					wire, _ := json.Marshal(payload)
					_ = json.Unmarshal(wire, &body)
					sentVersion = body.SkillVersion
					return []byte(`{"outcome":"completed","state":{"status":"RUNNING"}}`), nil
				}
				decide := func(_ context.Context, _ json.RawMessage, doc catalog.Document, _ json.RawMessage) (json.RawMessage, error) {
					if doc.Name != phase.want {
						t.Fatalf("phase skill = %s, want %s", doc.Name, phase.want)
					}
					return []byte(`{"tool":"click","element_ref":"p0"}`), nil
				}
				if _, err := drivePage(context.Background(), "original-job", snapshot, request, decide, &memoryJournal{}); err != nil {
					t.Fatal(err)
				}
				if sentVersion != skills[phase.want].Version || snapshot.Skill.Name != name {
					t.Fatal("wrong skill receipt or lost the original sync snapshot")
				}
			}
		})
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
