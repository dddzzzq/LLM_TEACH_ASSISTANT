package materials

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/uuid"
	"grading-gateway/internal/models"
	"grading-gateway/internal/testutil"
)

func TestIntegrationUpgradeUsesOriginalAndPreservesGrades(t *testing.T) {
	db, owner := testutil.ExecutionDB(t)
	if err := db.AutoMigrate(&models.Assignment{}, &models.MaterialBatch{}, &models.AsyncJob{}, &models.Submission{}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	source := filepath.Join(root, "20260001_张三.zip")
	save(t, source, archiveBytes(t, map[string]string{"src/main.ts": "source", "test/cli.test.mjs": "actual test", "report.md": "report"}))
	s := &Service{DB: db, Root: filepath.Join(root, "out"), AllowedRoots: []string{root}}
	a := models.Assignment{OwnerID: owner, CourseName: "课", TaskName: "作业"}
	if err := db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	actor := Actor{UserID: owner, Role: "teacher"}
	batch, err := s.Create(t.Context(), actor, Request{AssignmentID: a.ID, SourcePath: source}, uuid.NewString())
	if err != nil {
		t.Fatal(err)
	}
	if err = s.ProcessOne(t.Context(), batch.ID); err != nil {
		t.Fatal(err)
	}
	batch, _ = s.Owned(t.Context(), actor, batch.ID)
	manifest, err := LoadManifest(batch.ManifestPath, batch.ManifestSHA256)
	if err != nil {
		t.Fatal(err)
	}
	manifest.ReaderVersion = 0
	manifest.Sources = nil
	manifest.Students[0].SourceIDs = nil
	var legacyFiles []File
	for _, file := range manifest.Students[0].Files {
		if !strings.HasSuffix(file.Path, ".mjs") {
			legacyFiles = append(legacyFiles, file)
		}
	}
	manifest.Students[0].Files = legacyFiles
	raw, _ := json.Marshal(manifest)
	save(t, batch.ManifestPath, raw)
	hash, _ := HashFile(batch.ManifestPath)
	job := models.AsyncJob{ID: uuid.NewString(), OwnerID: owner, MaterialBatchID: batch.ID, Status: models.JobStatusProcessing}
	db.Create(&job)
	db.Model(&batch).Updates(map[string]any{"manifest_json": string(raw), "manifest_sha256": hash, "grading_job_id": job.ID})
	grade := models.Submission{AssignmentID: a.ID, MaterialBatchID: batch.ID, GradingJobID: job.ID, StudentID: "20260001", Score: 83, IsHumanReviewed: true, HumanScore: 88, HumanFeedback: "保留教师复核"}
	db.Create(&grade)
	if _, err = s.EnsureCurrentReader(t.Context(), actor, batch.ID); err == nil {
		t.Fatal("upgraded active grading input")
	}
	db.Model(&job).Update("status", models.JobStatusSuccess)
	if _, err = s.EnsureCurrentReader(t.Context(), Actor{UserID: owner + 1, Role: "teacher"}, batch.ID); err == nil {
		t.Fatal("another owner upgraded materials")
	}
	// Browser download locations may disappear; use the retained original ZIP.
	if err = os.Remove(source); err != nil {
		t.Fatal(err)
	}
	updated, err := s.EnsureCurrentReader(t.Context(), actor, batch.ID)
	if err != nil {
		t.Fatal(err)
	}
	if updated.ManifestPath == batch.ManifestPath || updated.GradingJobID != job.ID {
		t.Fatal("lost immutable revision or previous task")
	}
	current, err := LoadManifest(updated.ManifestPath, updated.ManifestSHA256)
	if err != nil || current.ReaderVersion != CurrentReaderVersion || len(current.Students[0].Files) != 3 || len(current.StudentSources(current.Students[0])) != 1 {
		t.Fatal("upgrade did not recover source", current, err)
	}
	if _, err = LoadManifest(batch.ManifestPath, hash); err != nil {
		t.Fatal("old material revision changed", err)
	}
	var saved models.Submission
	db.First(&saved, grade.ID)
	if saved.Score != 83 || !saved.IsHumanReviewed || saved.HumanScore != 88 || saved.HumanFeedback != grade.HumanFeedback {
		t.Fatal("upgrade changed grades")
	}
	again, err := s.EnsureCurrentReader(t.Context(), actor, batch.ID)
	if err != nil || again.ManifestPath != updated.ManifestPath {
		t.Fatal("current reader rebuilt unnecessarily", err)
	}
}
