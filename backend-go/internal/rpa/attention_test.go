package rpa

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

func TestIntegrationAttentionAndSupplementRequirements(t *testing.T) {
	integrationDB(t)
	owner := uint(81001)
	job := models.RPAJob{ID: uuid.NewString(), UserID: owner, CourseName: "提醒测试课程", AssignmentName: uuid.NewString(), Status: "PARTIAL_SUCCESS"}
	a := models.Assignment{OwnerID: owner, CourseName: job.CourseName, ClassName: "A", TaskName: job.AssignmentName}
	if err := database.DB.Create(&job).Error; err != nil {
		t.Fatal(err)
	}
	if err := database.DB.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	file := models.RPAFile{ID: strings.ReplaceAll(uuid.NewString(), "-", "") + "00000000", JobID: job.ID, ClassName: "A", AssignmentID: a.ID, GradingStatus: "WAITING_RUBRIC"}
	if err := database.DB.Create(&file).Error; err != nil {
		t.Fatal(err)
	}
	foreign := models.RPAJob{ID: uuid.NewString(), UserID: owner + 1, Status: "WAITING_USER", StateJSON: `{"control_mode":"HUMAN","waiting_reason":"LOGIN","control_epoch":2}`}
	if err := database.DB.Create(&foreign).Error; err != nil {
		t.Fatal(err)
	}
	manual := models.RPAJob{ID: uuid.NewString(), UserID: owner, Status: "WAITING_USER", StateJSON: `{"control_mode":"HUMAN","waiting_reason":"USER_TAKEOVER","control_epoch":3}`}
	if err := database.DB.Create(&manual).Error; err != nil {
		t.Fatal(err)
	}
	notices, err := Attention(owner)
	if err != nil || len(notices) != 1 || notices[0].JobID != job.ID || !notices[0].QuestionMissing || !notices[0].RubricMissing {
		t.Fatalf("unexpected notices: %+v %v", notices, err)
	}
	key := notices[0].Key
	database.DB.Model(&job).Update("message", "new poll text")
	notices, _ = Attention(owner)
	if notices[0].Key != key {
		t.Fatal("polling changed reminder identity")
	}
	if _, err := FileRequirements(foreign.ID, file.ID); !errors.Is(err, gorm.ErrRecordNotFound) {
		t.Fatal("cross-task file was readable")
	}
	if err := SupplementRequirements(job.ID, file.ID, " ", map[string]any{"正确性": 100}); !errors.Is(err, ErrInvalidRequirements) {
		t.Fatal("blank question accepted")
	}
	if err := SupplementRequirements(job.ID, file.ID, "task", map[string]any{}); !errors.Is(err, ErrInvalidRequirements) {
		t.Fatal("empty rubric accepted")
	}
	if err := SupplementRequirements(job.ID, file.ID, "task", map[string]any{"正确性": 100}); err != nil {
		t.Fatal(err)
	}
	data, err := FileRequirements(job.ID, file.ID)
	if err != nil || data.Question != "task" || len(data.Rubric) != 1 {
		t.Fatalf("saved requirements missing: %+v %v", data, err)
	}
	notices, _ = Attention(owner)
	if len(notices) != 0 {
		t.Fatal("completed requirements still trigger reminder")
	}
	if err := dispatchFile(file); err != nil {
		t.Fatal(err)
	}
	database.DB.First(&file, "id = ?", file.ID)
	if file.GradingStatus != "DOWNLOADED" || file.GradingJobID != "" {
		t.Fatal("supplement implicitly submitted grading")
	}
	if err := SupplementRequirements(job.ID, file.ID, "overwrite", map[string]any{"x": 1}); !errors.Is(err, ErrRequirementsChanged) {
		t.Fatal("running grading requirements were overwritten")
	}
	state, _ := json.Marshal(map[string]any{"control_mode": "HUMAN", "waiting_reason": "LOGIN", "control_epoch": 4})
	database.DB.Model(&manual).Update("state_json", string(state))
	notices, _ = Attention(owner)
	if len(notices) != 1 || notices[0].Kind != "human" {
		t.Fatalf("missing login reminder: %+v", notices)
	}
	firstKey := notices[0].Key
	database.DB.Model(&manual).Update("state_json", `{"control_mode":"HUMAN","waiting_reason":"LOGIN","control_epoch":5}`)
	notices, _ = Attention(owner)
	if len(notices) != 1 || notices[0].Key == firstKey {
		t.Fatal("new human handoff did not create a fresh reminder")
	}
	database.DB.Model(&manual).Update("status", "CANCELLED")
	notices, _ = Attention(owner)
	if len(notices) != 0 {
		t.Fatal("cancelled task still prompted")
	}
}
