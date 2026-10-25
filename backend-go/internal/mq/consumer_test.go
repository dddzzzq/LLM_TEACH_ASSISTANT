package mq

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

func TestIntegrationTerminalGradingMessagesAreAcknowledgedWithoutRunning(t *testing.T) {
	dsn := os.Getenv("RPA_TEST_DSN")
	if dsn == "" {
		t.Skip("requires isolated RPA_TEST_DSN")
	}
	if !strings.Contains(dsn, "/rpa_test_") {
		t.Fatal("refusing non-test database")
	}
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		t.Fatal(err)
	}
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old; sql, _ := db.DB(); sql.Close() })
	if err := db.AutoMigrate(&models.AsyncJob{}); err != nil {
		t.Fatal(err)
	}
	for _, status := range []models.AsyncJobStatus{models.JobStatusCancelled, models.JobStatusSuccess, models.JobStatusFailed} {
		for _, kind := range []models.AsyncJobType{models.JobTypeHomework} {
			t.Run(string(kind)+"/"+string(status), func(t *testing.T) {
				job := models.AsyncJob{ID: uuid.NewString(), JobType: kind, ReferenceID: "test", Status: status, Message: "terminal snapshot"}
				if err := db.Create(&job).Error; err != nil {
					t.Fatal(err)
				}
				message, _ := json.Marshal(map[string]any{"job_id": job.ID, "assignment_id": 0, "zip_path": "/nonexistent/cancelled-task.zip"})
				handler := handleHomeworkTask
				if err := handler(context.Background(), message); err != nil {
					t.Fatal("terminal replay must be acknowledged, not retried:", err)
				}
				var stored models.AsyncJob
				if err := db.First(&stored, "id = ?", job.ID).Error; err != nil {
					t.Fatal(err)
				}
				if stored.Status != status || stored.Message != job.Message {
					t.Fatal("terminal replay restarted or modified the task")
				}
			})
		}
	}
}
