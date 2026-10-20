package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/database"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/models"
	"grading-gateway/pb"
)

type preparedClient struct {
	pb.ComputeServiceClient
	graded atomic.Int32
}

func (c *preparedClient) CheckPlagiarism(context.Context, *pb.PlagiarismRequest, ...grpc.CallOption) (*pb.PlagiarismResponse, error) {
	return &pb.PlagiarismResponse{PlagiarismResultsJson: "{}"}, nil
}
func (c *preparedClient) DetectAIGC(context.Context, *pb.AIGCRequest, ...grpc.CallOption) (*pb.AIGCResponse, error) {
	return &pb.AIGCResponse{AigcReportJson: "{}"}, nil
}
func (c *preparedClient) GradeHomework(_ context.Context, r *pb.GradeRequest, _ ...grpc.CallOption) (*pb.GradeResponse, error) {
	c.graded.Add(1)
	if r.Question != "snapshot requirement" || r.RubricJson != `{"功能":100}` {
		return nil, fmt.Errorf("grading ignored rubric snapshot")
	}
	for _, part := range []string{"alpha code", "beta code", "design report"} {
		if !strings.Contains(r.StudentText, part) {
			return nil, fmt.Errorf("lost file %s", part)
		}
	}
	return &pb.GradeResponse{TotalScore: 91, Feedback: "fixture evidence", MergedContent: r.StudentText, CodeDocMatchReportJson: "{}"}, nil
}

func TestIntegrationPreparedPipelineUsesManifestSnapshotAndPreservesResults(t *testing.T) {
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
	oldDB, oldClient := database.DB, grpcclient.Client
	database.DB = db
	client := &preparedClient{}
	grpcclient.Client = client
	t.Cleanup(func() { database.DB = oldDB; grpcclient.Client = oldClient; sql, _ := db.DB(); sql.Close() })
	if err := db.AutoMigrate(&models.Assignment{}, &models.Submission{}, &models.MaterialBatch{}, &models.AsyncJob{}, &models.RPAJob{}, &models.RPAFile{}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	t.Setenv("HOMEWORK_ALLOWED_ROOTS", root)
	t.Setenv("HOMEWORK_MATERIALS_ROOT", filepath.Join(root, "out"))
	source := filepath.Join(root, "source", "20260001_张三")
	for file, content := range map[string]string{"src/main.py": "alpha code", "test/main.py": "beta code", "设计报告.txt": "design report"} {
		full := filepath.Join(source, file)
		os.MkdirAll(filepath.Dir(full), 0700)
		os.WriteFile(full, []byte(content), 0600)
	}
	a := models.Assignment{CourseName: "课", ClassName: "A", TaskName: "作业", Question: "current changed requirement", Rubric: `{"其他":50}`}
	db.Create(&a)
	s := materials.Default()
	batch, err := s.Create(t.Context(), materials.Actor{UserID: 8501, Role: "teacher"}, materials.Request{SourcePath: source, AssignmentID: a.ID}, uuid.NewString())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.ProcessOne(t.Context(), batch.ID); err != nil {
		t.Fatal(err)
	}
	batch, err = s.Owned(t.Context(), materials.Actor{UserID: 8501, Role: "teacher"}, batch.ID)
	if err != nil {
		t.Fatal(err)
	}
	job := models.AsyncJob{ID: uuid.NewString(), OwnerID: 8501, JobType: models.JobTypeHomework, ReferenceID: fmt.Sprint(a.ID), MaterialBatchID: batch.ID, Status: models.JobStatusPending, QuestionSnapshot: "snapshot requirement", RubricSnapshot: `{"功能":100}`}
	db.Create(&job)
	db.Model(&batch).Update("grading_job_id", job.ID)
	if err := ProcessPipelineForJob(fmt.Sprint(a.ID), batch.ManifestPath, job); err != nil {
		t.Fatal(err)
	}
	var results []models.Submission
	db.Where("grading_job_id = ?", job.ID).Find(&results)
	if len(results) != 1 || results[0].Score != 91 || results[0].MaterialBatchID != batch.ID || client.graded.Load() != 1 {
		t.Fatalf("wrong results %+v", results)
	}
	var m materials.Manifest
	json.Unmarshal([]byte(batch.ManifestJSON), &m)
	os.WriteFile(filepath.Join(filepath.Dir(batch.ManifestPath), filepath.FromSlash(m.Students[0].Files[0].Path)), []byte("tampered"), 0600)
	if err := ProcessPipelineForJob(fmt.Sprint(a.ID), batch.ManifestPath, job); err == nil || client.graded.Load() != 1 {
		t.Fatal("mutated input reached model")
	}
	var count int64
	db.Model(&models.Submission{}).Where("grading_job_id = ?", job.ID).Count(&count)
	if count != 1 {
		t.Fatal("failed retry changed previous results")
	}
}
