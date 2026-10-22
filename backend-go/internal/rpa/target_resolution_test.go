package rpa

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/uuid"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

func TestIntegrationResolvedTargetPersistsNamesAndAttention(t *testing.T) {
	integrationDB(t)
	job := models.RPAJob{ID: uuid.NewString(), UserID: 98123, CourseName: "移动互连网导论", AssignmentName: "作业1", Status: "RUNNING", AfterDownload: "extract", OutputRoot: "/tmp/teacher-materials"}
	if err := database.DB.Create(&job).Error; err != nil {
		t.Fatal(err)
	}
	pending := json.RawMessage(`{"id":"resolution-a","kind":"course","requested_name":"移动互连网导论","term":"2025-2026第三学期","previous_term":true,"candidates":[{"id":"candidate-a","name":"移动互联网导论"}]}`)
	state := map[string]any{"status": "WAITING_USER", "stage": "LOCATE_COURSE", "state_version": 2, "control_mode": "HUMAN", "control_epoch": 4, "waiting_reason": "TARGET_SUGGESTION", "target_resolution": pending, "target": Target{CourseName: job.CourseName, AssignmentName: job.AssignmentName}}
	raw, _ := json.Marshal(state)
	if err := SaveState(job.ID, raw); err != nil {
		t.Fatal(err)
	}
	notices, err := Attention(job.UserID)
	if err != nil || len(notices) != 1 || notices[0].Kind != "target" || string(notices[0].TargetResolution) != string(pending) {
		t.Fatalf("missing target notice: %+v %v", notices, err)
	}
	foreign, err := Attention(job.UserID + 1)
	if err != nil || len(foreign) != 0 {
		t.Fatal("target candidate leaked to another user")
	}
	state["status"], state["control_mode"], state["target_resolution"], state["state_version"] = "RUNNING", "AUTO", nil, 3
	state["target"] = Target{CourseName: "移动互联网导论", AssignmentName: "作业一", Term: "2025-2026第三学期"}
	raw, _ = json.Marshal(state)
	if err := SaveState(job.ID, raw); err != nil {
		t.Fatal(err)
	}
	database.DB.First(&job, "id = ?", job.ID)
	if job.CourseName != "移动互联网导论" || job.AssignmentName != "作业一" || job.Term != "2025-2026第三学期" || job.AfterDownload != "extract" || job.OutputRoot != "/tmp/teacher-materials" {
		t.Fatalf("accepted names/settings lost: %+v", job)
	}
	if target := Detail(&job)["target"].(Target); target.AssignmentName != "作业一" {
		t.Fatal("detail reverted accepted name")
	}
	// An older scheduler poll cannot overwrite the teacher's accepted name.
	state["state_version"] = 2
	state["target"] = Target{CourseName: "old", AssignmentName: "old"}
	raw, _ = json.Marshal(state)
	if err := SaveState(job.ID, raw); err != nil {
		t.Fatal(err)
	}
	database.DB.First(&job, "id = ?", job.ID)
	if job.CourseName != "移动互联网导论" {
		t.Fatal("stale observation replaced accepted target")
	}
	notices, err = Attention(job.UserID)
	if err != nil || len(notices) != 0 {
		t.Fatal("accepted target still triggers confirmation")
	}
}

func TestIntegrationExplicitRestartRefreshesPreExportSkillButPreservesReceipts(t *testing.T) {
	integrationDB(t)
	token := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(token, []byte("restart-contract-test"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RPA_CONTROL_TOKEN_FILE", token)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-RPA-Token") != "restart-contract-test" {
			t.Error("missing authentication")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"RUNNING","stage":"OPEN_PORTAL","state_version":2}`))
	}))
	defer server.Close()
	t.Setenv("RPA_CONTROL_URL", server.URL)
	for _, exported := range []bool{false, true} {
		state, _ := json.Marshal(map[string]any{"export_attempted": exported})
		old := `{"skill":{"name":"fetch-homework","version":"old","instructions":"old body"},"protocol":"browser.v1","action_schema":{"oneOf":[]}}`
		job := models.RPAJob{ID: uuid.NewString(), UserID: 98124, CourseName: "操作系统", AssignmentName: "作业一", Status: "INTERRUPTED", StateJSON: string(state), NavigationJSON: old, PendingAction: `{"action_id":"old-receipt"}`}
		if err := database.DB.Create(&job).Error; err != nil {
			t.Fatal(err)
		}
		if err := Start(t.Context(), job.ID, true); err != nil {
			t.Fatal(err)
		}
		database.DB.First(&job, "id = ?", job.ID)
		if job.PendingAction != `{"action_id":"old-receipt"}` {
			t.Fatal("restart erased pending receipt")
		}
		var got NavigationSnapshot
		if err := json.Unmarshal([]byte(job.NavigationJSON), &got); err != nil {
			t.Fatal(err)
		}
		if exported && got.Skill.Version != "old" {
			t.Fatal("export checkpoint lost pinned navigation")
		}
		if len(got.ActionSchema) > 0 {
			t.Fatal("explicit new browser session retained obsolete tool contract")
		}
		if !exported && (got.Skill.Version == "old" || got.Skill.Name != "fetch-homework" || len(got.ActionSchema) > 0) {
			t.Fatal("explicit restart did not load current navigation Skill/contract")
		}
	}
}
