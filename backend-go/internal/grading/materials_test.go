package grading

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/uuid"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/database"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/models"
)

func TestIntegrationPreparedBatchGradingIsExplicitIdempotentAndRecoverable(t *testing.T) {
	dsn := os.Getenv("RPA_TEST_DSN")
	if dsn == "" {
		t.Skip("requires isolated RPA_TEST_DSN")
	}
	if !strings.Contains(dsn, "/rpa_test_") {
		t.Fatal("refusing production database")
	}
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		t.Fatal(err)
	}
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old; sql, _ := db.DB(); sql.Close() })
	if err := db.AutoMigrate(&models.MaterialBatch{}, &models.AsyncJob{}, &models.Assignment{}, &models.RPAJob{}, &models.RPAFile{}, &models.PortalWorkflowItem{}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	t.Setenv("HOMEWORK_ALLOWED_ROOTS", root+"|"+os.TempDir())
	t.Setenv("HOMEWORK_MATERIALS_ROOT", filepath.Join(root, "prepared"))
	a := models.Assignment{OwnerID: 8201, CourseName: "测试课", ClassName: "A", TaskName: "第一次作业", Question: "编写程序", Rubric: `{"正确性":100}`}
	if err := db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	actor := Actor{UserID: 8201, Role: "teacher"}
	s := materials.Default()
	batch, err := s.Create(ctx, materials.Actor{UserID: actor.UserID, Role: actor.Role}, materials.Request{SourcePath: fixtureZip(t, "20260001_张三/main.py"), AssignmentID: a.ID}, uuid.NewString())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.ProcessOne(ctx, batch.ID); err != nil {
		t.Fatal(err)
	}
	batch, _ = s.Owned(ctx, materials.Actor{UserID: actor.UserID, Role: actor.Role}, batch.ID)
	if batch.GradingJobID != "" {
		t.Fatal("preparation submitted grading")
	}
	// Class grading keeps the configuration selected at submission time even
	// when the teacher edits the shared assignment during material preparation.
	selectedQuestion, selectedRubric := a.Question, a.Rubric
	item := models.PortalWorkflowItem{ID: uuid.NewString(), WorkflowID: uuid.NewString(), OwnerID: actor.UserID, AssignmentID: a.ID, Question: selectedQuestion, Rubric: selectedRubric}
	if err := db.Create(&item).Error; err != nil {
		t.Fatal(err)
	}
	download := models.RPAJob{ID: uuid.NewString(), UserID: actor.UserID, AssignmentID: a.ID, PortalWorkflowItemID: item.ID, Status: "SUCCEEDED"}
	if err := db.Create(&download).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Model(&batch).Update("source_job_id", download.ID).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Model(&a).Updates(map[string]any{"question": "修改后的官网题目", "rubric": `{"新标准":100}`}).Error; err != nil {
		t.Fatal(err)
	}
	oldPublish := PublishMaterialTask
	defer func() { PublishMaterialTask = oldPublish }()
	calls := 0
	PublishMaterialTask = func(id string, assignment uint, path string) error {
		calls++
		var job models.AsyncJob
		if db.First(&job, "id = ?", id).Error != nil || job.DispatchState != "READY" {
			t.Fatal("published without durable outbox")
		}
		if path != batch.ManifestPath || assignment != a.ID {
			t.Fatal("wrong material payload")
		}
		return errors.New("fixture queue offline")
	}
	if _, err := SubmitBatch(ctx, Actor{UserID: 8202, Role: "teacher"}, Request{BatchID: batch.ID}); err == nil {
		t.Fatal("other teacher graded batch")
	}
	first, err := SubmitBatch(ctx, actor, Request{Kind: "homework", ResourceID: fmt.Sprint(a.ID), BatchID: batch.ID, SkillName: "grade-homework", SkillVersion: "fixture-v2"})
	if err != nil || first.JobID == "" || calls != 1 {
		t.Fatalf("%+v %v", first, err)
	}
	var job models.AsyncJob
	db.First(&job, "id = ?", first.JobID)
	if job.DispatchState != "READY" || job.QuestionSnapshot != selectedQuestion || job.RubricSnapshot != selectedRubric || job.MaterialBatchID != batch.ID {
		t.Fatal("snapshot or outbox missing")
	}
	second, err := SubmitBatch(ctx, actor, Request{BatchID: batch.ID})
	if err != nil || second.JobID != first.JobID || calls != 1 {
		t.Fatal("repeated click published twice")
	}
	PublishMaterialTask = func(string, uint, string) error { calls++; return nil }
	if err := DispatchMaterialJob(ctx, first.JobID); err != nil {
		t.Fatal(err)
	}
	db.First(&job, "id = ?", first.JobID)
	if job.DispatchState != "SENT" || calls != 2 {
		t.Fatal("durable outbox did not resume")
	}
	// An explicit auto-grade intent waits for requirements; ordinary batches do not gain one.
	b2, err := s.Create(ctx, materials.Actor{UserID: actor.UserID, Role: actor.Role}, materials.Request{SourcePath: fixtureZip(t, "20260002_李四/main.py"), AssignmentID: a.ID, AutoGrade: true}, uuid.NewString())
	if err != nil {
		t.Fatal(err)
	}
	s.ProcessOne(ctx, b2.ID)
	db.Model(&a).Updates(map[string]any{"question": "", "rubric": "{}"})
	b2, _ = s.Owned(ctx, materials.Actor{UserID: actor.UserID, Role: actor.Role}, b2.ID)
	if err := ContinuePrepared(ctx, b2); err != nil {
		t.Fatal(err)
	}
	b2, _ = s.Owned(ctx, materials.Actor{UserID: actor.UserID, Role: actor.Role}, b2.ID)
	if b2.GradingState != "WAITING_REQUIREMENTS" || b2.GradingJobID != "" {
		t.Fatal("missing requirements did not gate grading")
	}
	db.Model(&a).Updates(map[string]any{"question": "updated requirement", "rubric": `{"功能":100}`})
	if err := ContinuePrepared(ctx, b2); err != nil {
		t.Fatal(err)
	}
	b2, _ = s.Owned(ctx, materials.Actor{UserID: actor.UserID, Role: actor.Role}, b2.ID)
	if b2.GradingJobID == "" {
		t.Fatal("explicit continuation not resumed")
	}
	// Persisted cancellation takes precedence over old queued payloads.
	b3, err := s.Create(ctx, materials.Actor{UserID: actor.UserID, Role: actor.Role}, materials.Request{SourcePath: fixtureZip(t, "20260003_王五/main.py"), AssignmentID: a.ID, AutoGrade: true}, uuid.NewString())
	if err != nil {
		t.Fatal(err)
	}
	s.ProcessOne(ctx, b3.ID)
	s.Cancel(ctx, materials.Actor{UserID: actor.UserID, Role: actor.Role}, b3.ID)
	if _, err := SubmitBatch(ctx, actor, Request{BatchID: b3.ID}); err == nil {
		var now models.MaterialBatch
		db.First(&now, "id = ?", b3.ID)
		if now.GradingJobID != "" {
			t.Fatal("cancelled material submitted")
		}
	}
}
