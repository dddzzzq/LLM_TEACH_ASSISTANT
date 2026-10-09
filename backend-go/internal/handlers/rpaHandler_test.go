package handlers

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/auth"
	"grading-gateway/internal/database"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
)

func TestIntegrationBrowserOwnershipAndRBAC(t *testing.T) {
	dsn := os.Getenv("RPA_TEST_DSN")
	if dsn == "" {
		t.Skip("requires an isolated RPA_TEST_DSN")
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
	if err := db.AutoMigrate(&models.RPAJob{}); err != nil {
		t.Fatal(err)
	}
	job := models.RPAJob{ID: uuid.NewString(), UserID: 501, Status: "RUNNING"}
	if err := db.Create(&job).Error; err != nil {
		t.Fatal(err)
	}
	tokenPath := filepath.Join(t.TempDir(), "token")
	os.WriteFile(tokenPath, []byte("worker-test-token"), 0600)
	t.Setenv("RPA_CONTROL_TOKEN_FILE", tokenPath)
	t.Setenv("JWT_ACCESS_SECRET", "rpa-test-signing-key")
	var called atomic.Int32
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called.Add(1)
		if r.Header.Get("X-RPA-Token") != "worker-test-token" {
			t.Error("missing service authentication")
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"image":"fixture","epoch":1,"revision":1}`))
	}))
	defer worker.Close()
	t.Setenv("RPA_CONTROL_URL", worker.URL)
	gin.SetMode(gin.TestMode)
	router := gin.New()
	group := router.Group("/api/rpa/jobs", middleware.AuthMiddleware(), middleware.RBACMiddleware("teacher", "admin"))
	group.GET("/:id/browser", RPABrowserFrame)
	group.POST("/:id/:action", ControlRPAJob)
	for _, tc := range []struct {
		name, role string
		user       uint
		want       int
	}{
		{"unauthenticated", "", 0, 401}, {"other teacher", "teacher", 502, 404},
		{"student", "student", 501, 403}, {"owner", "teacher", 501, 200},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, method := range []string{"GET", "POST"} {
				endpoint := "/browser"
				if method == "POST" {
					endpoint = "/input"
				}
				req := httptest.NewRequest(method, "/api/rpa/jobs/"+job.ID+endpoint, strings.NewReader(`{"kind":"key","key":"Tab"}`))
				if tc.user != 0 {
					token, err := auth.GenerateAccessToken(tc.user, "test-user", tc.role, nil)
					if err != nil {
						t.Fatal(err)
					}
					req.Header.Set("Authorization", "Bearer "+token)
				}
				response := httptest.NewRecorder()
				before := called.Load()
				router.ServeHTTP(response, req)
				if response.Code != tc.want {
					t.Fatalf("%s got %d: %s", method, response.Code, response.Body.String())
				}
				if tc.want != 200 && called.Load() != before {
					t.Fatal("unauthorized request reached browser worker")
				}
			}
		})
	}
}
