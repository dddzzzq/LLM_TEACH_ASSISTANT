package logbook

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/models"
)

func TestRedactSecretsAndSignedLinks(t *testing.T) {
	for _, pair := range [][2]string{
		{`DEEPSEEK_API_KEY=sk-fixturesecret123456789`, "fixturesecret"},
		{`{"password":"fixture-password with spaces","message":"ok"}`, "fixture-password"},
		{`Authorization: Bearer fixture-token.more.parts`, "fixture-token"},
		{`Cookie: session=fixture-session; another=fixture-cookie`, "fixture-cookie"},
		{`https://user:fixture-password@example.test/fixture-path-secret?sign=fixture-sign`, "fixture-"},
		{`mysql://user:fixture-password@localhost/database`, "fixture-password"},
		{`user:fixture-password@tcp(localhost:3306)/db`, "fixture-password"},
	} {
		clean := Redact(pair[0])
		if strings.Contains(clean, pair[1]) {
			t.Fatalf("secret leaked: %s", clean)
		}
		if Redact(clean) != clean {
			t.Fatalf("redaction not idempotent: %q -> %q", clean, Redact(clean))
		}
	}
	if Redact("\x1b[31mfailed\x1b[0m") != "failed" {
		t.Fatal("terminal control codes retained")
	}
}

func TestParseFormatsAndSeverity(t *testing.T) {
	for _, text := range []string{"2026/10/18 12:04:58 message", "2026-10-18 12:04:58,123 - ERROR - message", "[2026-10-18 12:04:58,456] INFO kafka", "2026-10-18T12:04:58+08:00 message"} {
		if _, ok := parseTime(text); !ok {
			t.Fatal("timestamp not recognized:", text)
		}
	}
	if _, ok := parseTime("INFO: time missing"); ok {
		t.Fatal("invented timestamp")
	}
	if classify("WARNING: connection", "") != "WARN" || classify("任务下载失败", "") != "ERROR" || classify("  stack frame", "ERROR") != "ERROR" {
		t.Fatal("severity incorrect")
	}
}

func TestIntegrationIndexSearchContextAndOwnership(t *testing.T) {
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
	t.Cleanup(func() { sql, _ := db.DB(); sql.Close() })
	if err := db.AutoMigrate(&models.RPAJob{}, &models.RPAFile{}, &models.AsyncJob{}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	s := New(db, root)
	if err := s.Init(); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	now := time.Now().Truncate(time.Second)
	owner := models.RPAJob{ID: uuid.NewString(), UserID: 701, Status: "FAILED", CourseName: "课程", AssignmentName: "作业"}
	foreign := models.RPAJob{ID: uuid.NewString(), UserID: 702, Status: "FAILED"}
	state, _ := json.Marshal(map[string]any{"events": []map[string]any{{"time": float64(now.Unix()), "message": "等待人工协助 password=fixture-event-secret"}}})
	owner.StateJSON = string(state)
	for _, job := range []*models.RPAJob{&owner, &foreign} {
		if err := db.Create(job).Error; err != nil {
			t.Fatal(err)
		}
	}
	grading := models.AsyncJob{ID: uuid.NewString(), ReferenceID: owner.ID, JobType: models.JobTypeHomework, Status: models.JobStatusCancelled, Message: "用户取消批改"}
	if err := db.Create(&grading).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&models.RPAFile{ID: strings.ReplaceAll(uuid.NewString(), "-", "") + "00000000", JobID: owner.ID, GradingJobID: grading.ID, ClassName: "A"}).Error; err != nil {
		t.Fatal(err)
	}
	stamp := now.Add(-time.Hour).Format("2006/01/02 15:04:05")
	content := stamp + " INFO job=" + owner.ID + " download started token=fixture-token\n" +
		stamp + " ERROR anonymous_private_context\n" +
		stamp + " ERROR job=" + foreign.ID + " foreign_private_context\n" +
		stamp + " WARN job=" + grading.ID + " download stalled\n" +
		stamp + " INFO job=" + owner.ID + " literal100%_value\n"
	path := filepath.Join(root, "backend.log")
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "rpa.log"), []byte("-----BEGIN PRIVATE KEY-----\nfixture-key-material\n-----END PRIVATE KEY-----\nINFO: no timestamp\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := s.Sync(ctx); err != nil {
		t.Fatal(err)
	}
	var count1, count2 int64
	db.Model(&Entry{}).Count(&count1)
	if err := s.Sync(ctx); err != nil {
		t.Fatal(err)
	}
	db.Model(&Entry{}).Count(&count2)
	if count1 != count2 {
		t.Fatal("re-reading files duplicated entries")
	}
	f := Filter{Source: "all", Level: "all", From: now.Add(-24 * time.Hour), To: now.Add(time.Hour), HideHTTP: true}
	actor := Actor{UserID: 701}
	admin := Actor{UserID: 1, Admin: true}
	p, err := s.Search(ctx, actor, f, "", 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Items) < 4 {
		t.Fatalf("own task and grading history missing: %+v", p)
	}
	for _, entry := range p.Items {
		if strings.Contains(entry.Message, "foreign_private") || strings.Contains(entry.Message, "anonymous_private") || strings.Contains(entry.Message, "fixture-event-secret") || strings.Contains(entry.Message, "fixture-token") {
			t.Fatal("private data leaked to teacher")
		}
	}
	empty, err := s.Search(ctx, Actor{UserID: 799}, f, "", 100)
	if err != nil || len(empty.Items) != 0 {
		t.Fatal("teacher without tasks saw system logs")
	}
	f.TaskID = foreign.ID
	if _, err := s.Search(ctx, actor, f, "", 100); !errors.Is(err, ErrScope) {
		t.Fatal("cross-owner task readable")
	}
	f.TaskID = grading.ID
	if _, err := s.Search(ctx, actor, f, "", 100); err != nil {
		t.Fatal("RPA-created grading task not recognized as owned:", err)
	}
	f.TaskID = ""
	f.Keyword = "literal100%_value"
	literal, err := s.Search(ctx, actor, f, "", 100)
	if err != nil || len(literal.Items) != 1 {
		t.Fatalf("wildcards not treated literally: %v %+v", err, literal)
	}
	f.Keyword = ""
	seen := map[uint64]bool{}
	cursor := ""
	for i := 0; i < 20; i++ {
		page, err := s.Search(ctx, actor, f, cursor, 2)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range page.Items {
			if seen[e.ID] {
				t.Fatal("pagination repeated an entry")
			}
			seen[e.ID] = true
		}
		if !page.HasMore {
			break
		}
		cursor = page.NextCursor
	}
	if len(seen) != len(p.Items) {
		t.Fatal("pagination lost entries")
	}
	f.Keyword = "changed"
	if _, err := s.Search(ctx, actor, f, cursor, 2); !errors.Is(err, ErrQuery) {
		t.Fatal("cursor reused with other filters")
	}
	f.Keyword = ""
	var ownRaw, foreignRaw Entry
	db.Where("source = ? AND message LIKE ?", "backend", "%download stalled%").First(&ownRaw)
	db.Where("source = ? AND message LIKE ?", "backend", "%foreign_private_context%").First(&foreignRaw)
	contextRows, err := s.Context(ctx, actor, ownRaw.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range contextRows {
		if strings.Contains(e.Message, "private_context") {
			t.Fatal("context leaked unowned or unlabelled records")
		}
	}
	if _, err := s.Context(ctx, actor, foreignRaw.ID); !errors.Is(err, ErrScope) {
		t.Fatal("cross-owner context readable")
	}
	contextRows, err = s.Context(ctx, admin, ownRaw.ID)
	if err != nil || len(contextRows) < 5 {
		t.Fatal("admin context missing")
	}
	all, err := s.Export(ctx, admin, f)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range all {
		if strings.Contains(e.Message, "fixture-key-material") {
			t.Fatal("private-key body entered index")
		}
	}
	f.Source = "../../.env"
	if _, err := s.Export(ctx, admin, f); !errors.Is(err, ErrQuery) {
		t.Fatal("arbitrary source accepted")
	}
	f.Source = "all"
	appendFile, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	appendFile.WriteString(stamp + " INFO job=" + owner.ID + " appended_partial")
	appendFile.Close()
	if err := s.Sync(ctx); err != nil {
		t.Fatal(err)
	}
	f.Keyword = "appended_partial"
	page, _ := s.Search(ctx, admin, f, "", 100)
	if len(page.Items) != 0 {
		t.Fatal("partial line indexed too early")
	}
	appendFile, _ = os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	appendFile.WriteString("\n")
	appendFile.Close()
	if err := s.Sync(ctx); err != nil {
		t.Fatal(err)
	}
	page, _ = s.Search(ctx, actor, f, "", 100)
	if len(page.Items) != 1 {
		t.Fatal("completed partial line lost")
	}
	if err := os.WriteFile(path, []byte(stamp+" INFO job="+owner.ID+" after_truncation\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := s.Sync(ctx); err != nil {
		t.Fatal(err)
	}
	f.Keyword = "after_truncation"
	page, _ = s.Search(ctx, actor, f, "", 100)
	if len(page.Items) != 1 {
		t.Fatal("truncated file not re-indexed")
	}
	if err := os.Symlink(path, filepath.Join(root, "ai.log")); err != nil {
		t.Fatal(err)
	}
	sources, err := s.Sources(ctx, admin)
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range sources {
		if source.ID == "ai" && source.Available {
			t.Fatal("followed a log symlink")
		}
	}
	// Retention removes only expired index records and links, preserving originals.
	old := Entry{EventKey: "expired-fixture", Source: "task", At: now.AddDate(0, 0, -31), Level: "INFO", Message: "expired", TaskIDs: "[]"}
	if err := db.Create(&old).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&TaskLink{EntryID: old.ID, TaskID: owner.ID}).Error; err != nil {
		t.Fatal(err)
	}
	if err := s.clean(ctx); err != nil {
		t.Fatal(err)
	}
	var remaining int64
	db.Model(&Entry{}).Where("id = ?", old.ID).Count(&remaining)
	if remaining != 0 {
		t.Fatal("expired index entry retained")
	}
	db.Model(&TaskLink{}).Where("entry_id = ?", old.ID).Count(&remaining)
	if remaining != 0 {
		t.Fatal("expired task link retained")
	}
	if err := db.First(&models.RPAJob{}, "id = ?", owner.ID).Error; err != nil {
		t.Fatal("retention removed business task")
	}
	original, err := os.ReadFile(path)
	if err != nil || !strings.Contains(string(original), "after_truncation") {
		t.Fatal("retention modified original log")
	}
}
