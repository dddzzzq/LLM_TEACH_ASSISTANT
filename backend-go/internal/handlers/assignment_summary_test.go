package handlers

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/auth"
	"grading-gateway/internal/database"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
	"grading-gateway/internal/schemas"
)

func TestAssignmentWorkspaceSummaryAndOwnership(t *testing.T) {
	dsn := os.Getenv("RPA_TEST_DSN")
	if dsn == "" {
		t.Skip("requires isolated RPA_TEST_DSN")
	}
	if !strings.Contains(dsn, "/rpa_test_") {
		t.Fatal("refusing production DB")
	}
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		t.Fatal(err)
	}
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old; sql, _ := db.DB(); sql.Close() })
	if err := db.AutoMigrate(&models.Assignment{}, &models.Submission{}, &models.MaterialBatch{}, &models.AsyncJob{}); err != nil {
		t.Fatal(err)
	}
	assignments := []models.Assignment{}
	for _, name := range []string{"empty", "graded", "ready", "preparing", "grading", "attention", "retried", "other-owner"} {
		a := models.Assignment{OwnerID: 99181, CourseName: "workspace", ClassName: "全部班级", TaskName: name, Question: "题目", Rubric: `{"功能":100}`}
		if name == "other-owner" {
			a.OwnerID = 99182
		}
		if err := db.Create(&a).Error; err != nil {
			t.Fatal(err)
		}
		assignments = append(assignments, a)
	}
	ids := []uint{}
	for _, a := range assignments {
		ids = append(ids, a.ID)
	}
	defer db.Where("assignment_id IN ?", ids).Delete(&models.Submission{})
	defer db.Where("assignment_id IN ?", ids).Delete(&models.MaterialBatch{})
	refs := []string{}
	for _, id := range ids {
		refs = append(refs, strconv.Itoa(int(id)))
	}
	defer db.Where("reference_id IN ?", refs).Delete(&models.AsyncJob{})
	defer db.Where("id IN ?", ids).Delete(&models.Assignment{})
	for _, sub := range []models.Submission{
		{AssignmentID: ids[1], StudentID: "s1", Score: 90}, {AssignmentID: ids[1], StudentID: "s1", Score: 92},
		{AssignmentID: ids[1], StudentID: "s2", Score: -1, IsHumanReviewed: true, HumanScore: 80},
		{AssignmentID: ids[5], StudentID: "s3", Score: -1},
	} {
		if err := db.Create(&sub).Error; err != nil {
			t.Fatal(err)
		}
	}
	stamp := time.Now().Add(-time.Minute).UTC().Truncate(time.Second)
	for i, status := range map[int]string{2: "READY", 3: "PROCESSING", 4: "READY"} {
		batch := models.MaterialBatch{ID: strconv.Itoa(int(ids[i])), OwnerID: 99181, AssignmentID: ids[i], Status: status, UpdatedAt: stamp}
		if i == 4 {
			batch.GradingJobID = "workspace-active"
		}
		if err := db.Create(&batch).Error; err != nil {
			t.Fatal(err)
		}
	}
	for _, job := range []models.AsyncJob{
		{ID: "workspace-active", JobType: models.JobTypeHomework, ReferenceID: refs[4], MaterialBatchID: strconv.Itoa(int(ids[4])), Status: models.JobStatusProcessing, UpdatedAt: stamp},
		{ID: "workspace-old-failure", JobType: models.JobTypeHomework, ReferenceID: refs[6], Status: models.JobStatusFailed, UpdatedAt: stamp.Add(-time.Hour)},
		{ID: "workspace-retry-success", JobType: models.JobTypeHomework, ReferenceID: refs[6], Status: models.JobStatusSuccess, UpdatedAt: stamp},
	} {
		if err := db.Create(&job).Error; err != nil {
			t.Fatal(err)
		}
	}
	summary, err := assignmentSummaries(context.Background(), db, ids[:7])
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []string{"EMPTY", "GRADED", "READY", "PREPARING", "GRADING", "ATTENTION", "EMPTY"} {
		if summary[ids[i]].Status != want {
			t.Fatalf("%s status=%s want=%s", assignments[i].TaskName, summary[ids[i]].Status, want)
		}
	}
	if s := summary[ids[1]]; s.GradedCount != 2 || s.ReviewedCount != 1 || s.FailedCount != 0 {
		t.Fatalf("incorrect distinct student counts: %+v", s)
	}
	if summary[ids[0]].UpdatedAt != nil || summary[ids[4]].UpdatedAt == nil {
		t.Fatal("invented/missing activity time")
	}
	empty, err := assignmentSummaries(context.Background(), nil, nil)
	if err != nil || len(empty) != 0 {
		t.Fatal("empty list should not query DB")
	}
	t.Setenv("JWT_ACCESS_SECRET", "assignment-workspace-test")
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/assignments/", middleware.AuthMiddleware(), middleware.RBACMiddleware("teacher", "admin"), GetAssignments)
	token, _ := auth.GenerateAccessToken(99181, "teacher", "teacher", nil)
	req := httptest.NewRequest("GET", "/assignments/", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	out := httptest.NewRecorder()
	r.ServeHTTP(out, req)
	if out.Code != 200 {
		t.Fatalf("index failed: %d", out.Code)
	}
	var response []schemas.AssignmentResponse
	if err := json.Unmarshal(out.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if len(response) != 7 || strings.Contains(out.Body.String(), "other-owner") {
		t.Fatal("assignment summaries leaked another teacher's data")
	}
}
