package rpa

import (
	"archive/zip"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/database"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/models"
)

func TestIntegrationDownloadedArchivePreparesOnceWithoutGrading(t *testing.T) {
	integrationDB(t)
	root := t.TempDir()
	t.Setenv("HOMEWORK_ALLOWED_ROOTS", root)
	t.Setenv("HOMEWORK_MATERIALS_ROOT", filepath.Join(root, "prepared"))
	job, err := Create(t.Context(), 8401, Target{CourseName: "流程测试课", AssignmentName: uuid.NewString(), OutputRoot: filepath.Join(root, "out")})
	if err != nil {
		t.Fatal(err)
	}
	if job.AfterDownload != "extract" {
		t.Fatal("default flow is not extraction only")
	}
	dir := filepath.Join(root, job.ID)
	os.MkdirAll(dir, 0700)
	path := filepath.Join(dir, "班级A-作业.zip")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	z := zip.NewWriter(f)
	for name, data := range map[string]string{"20260001_张三/src/main.py": "print(1)", "20260001_张三/报告.md": "design"} {
		w, _ := z.Create(name)
		w.Write([]byte(data))
	}
	z.Close()
	f.Close()
	hash, _ := materials.HashFile(path)
	info, _ := os.Stat(path)
	state, _ := json.Marshal(map[string]any{"files": []map[string]any{{"status": "VERIFIED", "path": path, "sha256": hash, "size": info.Size(), "class_name": "A", "export_ref": "e1"}}})
	job.Status = "SUCCEEDED"
	job.StateJSON = string(state)
	database.DB.Save(job)
	if err := importFiles(*job); err != nil {
		t.Fatal(err)
	}
	var file models.RPAFile
	database.DB.First(&file, "job_id = ?", job.ID)
	if err := dispatchFile(file); err != nil {
		t.Fatal(err)
	}
	database.DB.First(&file, "id = ?", file.ID)
	if file.MaterialBatchID == "" || file.GradingJobID != "" {
		t.Fatal("download did not create an independent material batch")
	}
	if err := materials.Default().ProcessOne(t.Context(), file.MaterialBatchID); err != nil {
		t.Fatal(err)
	}
	if err := dispatchFile(file); err != nil {
		t.Fatal(err)
	}
	var count int64
	database.DB.Model(&models.MaterialBatch{}).Where("source_file_id = ?", file.ID).Count(&count)
	if count != 1 {
		t.Fatal("polling duplicated material batch")
	}
	var batch models.MaterialBatch
	database.DB.First(&batch, "id = ?", file.MaterialBatchID)
	if batch.Status != "READY" || batch.AutoGrade || batch.GradingJobID != "" {
		t.Fatalf("unexpected prepared flow: %+v", batch)
	}
	if err := SaveState(job.ID, []byte(`{"status":"CANCELLED","state_version":99}`)); err != nil {
		t.Fatal(err)
	}
	database.DB.First(&batch, "id = ?", batch.ID)
	if batch.Status != "CANCELLED" || batch.AutoGrade {
		t.Fatal("cancelled flow left continuation active")
	}
}

func TestWorkerRequestAuthenticatesAndPreservesConflict(t *testing.T) {
	token := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(token, []byte("internal-test-token"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RPA_CONTROL_TOKEN_FILE", token)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-RPA-Token") != "internal-test-token" {
			t.Error("missing internal authentication")
		}
		w.WriteHeader(409)
		w.Write([]byte(`{"error":"请先完成登录"}`))
	}))
	defer server.Close()
	t.Setenv("RPA_CONTROL_URL", server.URL)
	_, err := WorkerRequest(context.Background(), "POST", "/tasks/test/resume", nil)
	var we *WorkerError
	if !errors.As(err, &we) || we.Status != 409 || we.Message != "请先完成登录" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRubricGate(t *testing.T) {
	for _, rubric := range []string{"", "null", "{}", "not json"} {
		if validRubric(models.Assignment{Question: "task", Rubric: rubric}) {
			t.Errorf("accepted invalid rubric %q", rubric)
		}
	}
	if !validRubric(models.Assignment{Question: "task", Rubric: `{"correctness":100}`}) {
		t.Fatal("valid rubric rejected")
	}
}

// Run only against a disposable database. The harness creates and drops it outside these tests.
func integrationDB(t *testing.T) {
	t.Helper()
	dsn := os.Getenv("RPA_TEST_DSN")
	if dsn == "" {
		t.Skip("set RPA_TEST_DSN to an isolated MySQL database")
	}
	if !strings.Contains(dsn, "/rpa_test_") {
		t.Fatal("integration tests require an rpa_test_ database")
	}
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		t.Fatal(err)
	}
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old; sql, _ := db.DB(); sql.Close() })
	if err := db.AutoMigrate(&models.RPAJob{}, &models.RPAFile{}, &models.AsyncJob{}, &models.Assignment{}, &models.Submission{}, &models.MaterialBatch{}); err != nil {
		t.Fatal(err)
	}
}

func TestIntegrationOwnershipVersionAndDownloadDoesNotGrade(t *testing.T) {
	integrationDB(t)
	job, err := Create(context.Background(), 1001, Target{CourseName: "测试课程", AssignmentName: "测试作业", AfterDownload: "download"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Owned(1002, job.ID); !errors.Is(err, gorm.ErrRecordNotFound) {
		t.Fatal("cross-user access allowed")
	}
	newState := []byte(`{"status":"WAITING_USER","stage":"AUTHENTICATE","state_version":3,"message":"请登录"}`)
	oldState := []byte(`{"status":"RUNNING","stage":"OPEN_PORTAL","state_version":2}`)
	if err := SaveState(job.ID, newState); err != nil {
		t.Fatal(err)
	}
	if err := SaveState(job.ID, oldState); err != nil {
		t.Fatal(err)
	}
	job, _ = Owned(1001, job.ID)
	if job.Status != "WAITING_USER" {
		t.Fatal("stale response overwrote newer control state")
	}
	state, _ := json.Marshal(map[string]any{"files": []map[string]any{{"status": "VERIFIED", "path": "/downloads/" + job.ID + "/班级A-test.zip", "sha256": strings.Repeat("a", 64), "export_ref": "export-a", "class_name": "A", "size": 100}}})
	job.StateJSON = string(state)
	job.Status = "SUCCEEDED"
	if err := database.DB.Save(job).Error; err != nil {
		t.Fatal(err)
	}
	if err := importFiles(*job); err != nil {
		t.Fatal(err)
	}
	if err := importFiles(*job); err != nil {
		t.Fatal(err)
	}
	var files []models.RPAFile
	database.DB.Where("job_id = ?", job.ID).Find(&files)
	if len(files) != 1 {
		t.Fatal("duplicate completion imported duplicate file")
	}
	if err := dispatchFile(files[0]); err != nil {
		t.Fatal(err)
	}
	database.DB.First(&files[0], "id = ?", files[0].ID)
	if files[0].GradingStatus != "DOWNLOADED" || files[0].GradingJobID != "" || files[0].AssignmentID == 0 {
		t.Fatal("missing rubric was automatically graded")
	}
	database.DB.Model(&models.Assignment{}).Where("id = ?", files[0].AssignmentID).Updates(map[string]any{"question": "task", "rubric": `{"correctness":100}`})
	if err := dispatchFile(files[0]); err != nil {
		t.Fatal(err)
	}
	database.DB.First(&files[0], "id = ?", files[0].ID)
	firstID := files[0].GradingJobID
	if files[0].GradingStatus != "DOWNLOADED" || firstID != "" {
		t.Fatal("adding rubric unexpectedly submitted grading")
	}
	if err := dispatchFile(files[0]); err != nil {
		t.Fatal(err)
	}
	var count int64
	database.DB.Model(&models.AsyncJob{}).Where("reference_id = ? AND job_type = ?", job.ID, models.JobTypeHomework).Count(&count)
	if count != 0 {
		t.Fatal("download created an implicit grading task")
	}
	if err := SaveState(job.ID, []byte(`{"status":"CANCELLED","state_version":4}`)); err != nil {
		t.Fatal(err)
	}
	if err := SaveState(job.ID, []byte(`{"status":"RUNNING","state_version":5}`)); err != nil {
		t.Fatal(err)
	}
	job, _ = Owned(1001, job.ID)
	if job.Status != "CANCELLED" {
		t.Fatal("cancelled job resurrected")
	}
}

func TestIntegrationDownloadIdempotencyAndSkillSnapshot(t *testing.T) {
	integrationDB(t)
	target := Target{CourseName: "幂等课程", AssignmentName: "作业一"}
	key := uuid.NewString()
	first, err := CreateWithKey(context.Background(), 2001, target, key)
	if err != nil {
		t.Fatal(err)
	}
	again, err := CreateWithKey(context.Background(), 2001, target, key)
	if err != nil || first.ID != again.ID {
		t.Fatalf("duplicate creation: %v", err)
	}
	var snapshot NavigationSnapshot
	if json.Unmarshal([]byte(first.NavigationJSON), &snapshot) != nil || snapshot.Skill.Name != "fetch-homework" || snapshot.Skill.Version == "" {
		t.Fatal("missing fixed navigation Skill")
	}
	if _, err := CreateWithKey(context.Background(), 2001, Target{CourseName: "另一课程", AssignmentName: "作业一"}, key); err == nil {
		t.Fatal("idempotency key accepted different target")
	}
	other, err := CreateWithKey(context.Background(), 2002, target, key)
	if err != nil || other.ID == first.ID {
		t.Fatal("idempotency scope crossed users")
	}
}

func TestIntegrationNavigationLeaseAndDurableJournal(t *testing.T) {
	integrationDB(t)
	job, err := Create(context.Background(), 3001, Target{CourseName: "租约课程", AssignmentName: "作业"})
	if err != nil {
		t.Fatal(err)
	}
	if err = database.DB.Model(job).Updates(map[string]any{"status": "RUNNING", "decision_owner": "owner-a", "decision_lease_until": time.Now().Add(time.Minute)}).Error; err != nil {
		t.Fatal(err)
	}
	journal := databaseJournal{context.Background(), job.ID, "owner-a"}
	intent := []byte(`{"action_id":"observation-one","action":{"tool":"scroll","delta_y":10}}`)
	if err = journal.Save(intent); err != nil {
		t.Fatal(err)
	}
	restored := databaseJournal{context.Background(), job.ID, "owner-a"}
	got, err := restored.Pending()
	if err != nil || string(got) != string(intent) {
		t.Fatal("pending action did not survive repository reload")
	}
	stale := databaseJournal{context.Background(), job.ID, "owner-b"}
	if err = stale.Clear(); err == nil {
		t.Fatal("non-owner erased pending intent")
	}
	if err = restored.Clear(); err != nil {
		t.Fatal(err)
	}
	database.DB.Model(job).UpdateColumn("decision_lease_until", time.Now().Add(-time.Minute))
	if err = journal.Save(intent); err == nil {
		t.Fatal("expired owner saved new action")
	}
}
