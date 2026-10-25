// Package testutil contains fixtures for tests against a disposable execution database.
package testutil

import (
	"os"
	"strings"
	"testing"

	"grading-gateway/internal/models"

	"github.com/google/uuid"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

func ExecutionDB(t *testing.T) (*gorm.DB, uint) {
	t.Helper()
	dsn := os.Getenv("EXECUTION_TEST_DSN")
	if dsn == "" {
		t.Skip("requires disposable EXECUTION_TEST_DSN")
	}
	if !strings.Contains(dsn, "/rpa_test_execution_") {
		t.Fatal("refusing a non-execution test database")
	}
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sqlDB, _ := db.DB(); _ = sqlDB.Close() })
	if err := db.AutoMigrate(&models.User{}, &models.ChatSession{}, &models.ChatMessage{}, &models.AgentRun{}, &models.AgentEvent{}, &models.ToolDefinition{}); err != nil {
		t.Fatal(err)
	}
	user := models.User{Username: "execution-" + uuid.NewString(), Name: "执行测试", PasswordHash: "test-only", Role: models.RoleTeacher}
	if err := db.Create(&user).Error; err != nil {
		t.Fatal(err)
	}
	return db, user.ID
}
