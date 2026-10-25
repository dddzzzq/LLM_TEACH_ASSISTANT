package handlers

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"strings"
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

func TestIntegrationAttentionRequirementsOwnershipAndValidation(t *testing.T) {
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
	if err := db.AutoMigrate(&models.RPAJob{}, &models.RPAFile{}, &models.Assignment{}); err != nil {
		t.Fatal(err)
	}
	job := models.RPAJob{ID: uuid.NewString(), UserID: 91001, CourseName: "test", AssignmentName: uuid.NewString(), Status: "SUCCEEDED"}
	a := models.Assignment{OwnerID: job.UserID, CourseName: job.CourseName, TaskName: job.AssignmentName, ClassName: "A"}
	if err := db.Create(&job).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	file := models.RPAFile{ID: strings.ReplaceAll(uuid.NewString(), "-", "") + "00000000", JobID: job.ID, ClassName: "A", AssignmentID: a.ID, GradingStatus: "WAITING_RUBRIC"}
	if err := db.Create(&file).Error; err != nil {
		t.Fatal(err)
	}
	t.Setenv("JWT_ACCESS_SECRET", "attention-test-signing-key")
	gin.SetMode(gin.TestMode)
	router := gin.New()
	group := router.Group("/api/rpa/jobs", middleware.AuthMiddleware(), middleware.RBACMiddleware("teacher", "admin"))
	group.GET("/attention", GetRPAAttention)
	group.GET("/:id/requirements", GetRPARequirements)
	group.POST("/:id/:action", ControlRPAJob)
	valid, _ := json.Marshal(map[string]any{"file_id": file.ID, "question": "task", "rubric": map[string]any{"correctness": 100}})
	for _, tc := range []struct {
		role      string
		user      uint
		get, post int
	}{
		{"", 0, 401, 401}, {"student", 91001, 403, 403}, {"teacher", 91002, 404, 404}, {"teacher", 91001, 200, 200},
	} {
		for _, method := range []string{"GET", "POST"} {
			url := "/api/rpa/jobs/" + job.ID + "/requirements"
			want := tc.post
			if method == "GET" {
				url += "?file_id=" + file.ID
				want = tc.get
			}
			req := httptest.NewRequest(method, url, strings.NewReader(string(valid)))
			req.Header.Set("Content-Type", "application/json")
			if tc.user != 0 {
				token, err := auth.GenerateAccessToken(tc.user, "test", tc.role, nil)
				if err != nil {
					t.Fatal(err)
				}
				req.Header.Set("Authorization", "Bearer "+token)
			}
			response := httptest.NewRecorder()
			router.ServeHTTP(response, req)
			if response.Code != want {
				t.Fatalf("%s %s: got %d want %d: %s", tc.role, method, response.Code, want, response.Body.String())
			}
		}
	}
	token, _ := auth.GenerateAccessToken(job.UserID, "test", "teacher", nil)
	for _, body := range []string{
		`{"file_id":"` + file.ID + `","question":"task","rubric":[]}`,
		`{"file_id":"` + file.ID + `","question":"task","rubric":{}}`,
		`{"file_id":"` + file.ID + `","question":"   ","rubric":{"x":1}}`,
		`{"file_id":"foreign-file","question":"task","rubric":{"x":1}}`,
	} {
		req := httptest.NewRequest("POST", "/api/rpa/jobs/"+job.ID+"/requirements", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		router.ServeHTTP(response, req)
		if response.Code < 400 {
			t.Fatal("invalid requirements accepted")
		}
	}
}
