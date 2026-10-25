package agent

import (
	"testing"
	"time"

	"grading-gateway/internal/models"
	"grading-gateway/internal/testutil"

	"github.com/google/uuid"
)

func TestLegacyCacheRestoreDoesNotRepersistOrChangeMessages(t *testing.T) {
	db, user := testutil.ExecutionDB(t)
	session := models.ChatSession{ID: uuid.NewString(), UserID: user, Title: "legacy cache"}
	if err := db.Create(&session).Error; err != nil {
		t.Fatal(err)
	}
	manager := NewRedisMemoryManager(20, time.Hour, db)
	if err := manager.AddMessage(user, session.ID, "user", "durable without Redis"); err != nil {
		t.Fatal(err)
	}
	before, err := manager.GetRecentMessages(user, session.ID)
	if err != nil || len(before) != 1 {
		t.Fatalf("%+v %v", before, err)
	}
	for i := 0; i < 3; i++ {
		if err := manager.RestoreSessionToRedis(user, session.ID); err != nil {
			t.Fatal(err)
		}
	}
	after, err := manager.GetRecentMessages(user, session.ID)
	if err != nil || len(after) != 1 || !after[0].Timestamp.Equal(before[0].Timestamp) {
		t.Fatal("restoration duplicated a message or changed its timestamp")
	}
	var count int64
	db.Model(&models.ChatMessage{}).Where("session_id = ?", session.ID).Count(&count)
	if count != 1 {
		t.Fatalf("restoration wrote %d copies", count)
	}
	if err := manager.AddMessage(user+10000, session.ID, "assistant", "foreign"); err == nil {
		t.Fatal("foreign message write accepted")
	}
	if err := manager.RestoreSessionToRedis(user+10000, session.ID); err == nil {
		t.Fatal("foreign cache restoration accepted")
	}
}
