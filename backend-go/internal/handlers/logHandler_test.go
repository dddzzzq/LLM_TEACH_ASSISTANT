package handlers

import (
	"fmt"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/auth"
	"grading-gateway/internal/logbook"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
)

func TestIntegrationLogEndpointsRBACAndScope(t *testing.T) {
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
	job := models.RPAJob{ID: uuid.NewString(), UserID: 63001, CourseName: "private course", Status: "FAILED"}
	if err := db.Create(&job).Error; err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	svc := logbook.New(db, root)
	if err := svc.Init(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "backend.log"), []byte(time.Now().Format("2006/01/02 15:04:05")+" ERROR job="+job.ID+" failure password=fixture-handler-secret\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := svc.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}
	var entry logbook.Entry
	db.Where("source = ? AND message LIKE ?", "backend", "%"+job.ID+"%").First(&entry)
	t.Setenv("JWT_ACCESS_SECRET", "logbook-handler-test-key")
	gin.SetMode(gin.TestMode)
	router := gin.New()
	group := router.Group("/api/logs", middleware.AuthMiddleware())
	LogHandler{Service: svc}.Register(group)
	params := url.Values{"source": {"backend"}, "from": {time.Now().Add(-24 * time.Hour).Format(time.RFC3339)}, "to": {time.Now().Add(time.Hour).Format(time.RFC3339)}, "task_id": {job.ID}}
	for _, tc := range []struct {
		role string
		user uint
		want int
	}{{"", 0, 401}, {"student", job.UserID, 403}, {"teacher", job.UserID + 1, 403}, {"teacher", job.UserID, 403}, {"admin", job.UserID + 2, 200}} {
		for _, path := range []string{"/sources", "/tasks", "/search?" + params.Encode(), "/export?" + params.Encode(), fmt.Sprintf("/entries/%d/context", entry.ID)} {
			req := httptest.NewRequest("GET", "/api/logs"+path, nil)
			if tc.user > 0 {
				token, _ := auth.GenerateAccessToken(tc.user, "test", tc.role, nil)
				req.Header.Set("Authorization", "Bearer "+token)
			}
			out := httptest.NewRecorder()
			router.ServeHTTP(out, req)
			if out.Code != tc.want {
				t.Fatalf("%s %s got %d want %d: %s", tc.role, path, out.Code, tc.want, out.Body.String())
			}
			if strings.Contains(out.Body.String(), "fixture-handler-secret") {
				t.Fatal("secret leaked over HTTP")
			}
			if tc.want == 200 && strings.HasPrefix(path, "/export") && !strings.Contains(out.Header().Get("Content-Type"), "text/plain") {
				t.Fatal("unsafe export MIME type")
			}
		}
	}
	token, _ := auth.GenerateAccessToken(job.UserID, "test", "admin", nil)
	params.Set("source", "../.database.env")
	req := httptest.NewRequest("GET", "/api/logs/search?"+params.Encode(), nil)
	req.Header.Set("Authorization", "Bearer "+token)
	out := httptest.NewRecorder()
	router.ServeHTTP(out, req)
	if out.Code != 400 {
		t.Fatal("arbitrary file source accepted")
	}
}

func TestLogEndpointsRequireAdminBeforeReadingIndex(t *testing.T) {
	t.Setenv("JWT_ACCESS_SECRET", "logbook-admin-only-test-key")
	gin.SetMode(gin.TestMode)
	router := gin.New()
	LogHandler{Service: &logbook.Service{}}.Register(router.Group("/api/logs", middleware.AuthMiddleware()))
	for _, tc := range []struct {
		role string
		want int
	}{{"", 401}, {"student", 403}, {"teacher", 403}, {"admin", 503}} {
		for _, path := range []string{"/sources", "/tasks", "/search", "/export", "/entries/1/context"} {
			req := httptest.NewRequest("GET", "/api/logs"+path, nil)
			if tc.role != "" {
				token, _ := auth.GenerateAccessToken(1, "test", tc.role, nil)
				req.Header.Set("Authorization", "Bearer "+token)
			}
			out := httptest.NewRecorder()
			router.ServeHTTP(out, req)
			if out.Code != tc.want {
				t.Fatalf("%s %s got %d want %d", tc.role, path, out.Code, tc.want)
			}
		}
	}
}
