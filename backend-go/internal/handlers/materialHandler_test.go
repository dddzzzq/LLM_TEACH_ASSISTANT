package handlers

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/auth"
	"grading-gateway/internal/database"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
)

func TestIntegrationMaterialsAPIRequiresOwnerAndUploadNeverGrades(t *testing.T) {
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
	if err := db.AutoMigrate(&models.MaterialBatch{}, &models.AsyncJob{}, &models.Assignment{}, &models.RPAJob{}, &models.RPAFile{}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	t.Setenv("HOMEWORK_MATERIALS_ROOT", filepath.Join(root, "output"))
	cwd, _ := os.Getwd()
	t.Setenv("HOMEWORK_ALLOWED_ROOTS", root+"|"+cwd)
	a := models.Assignment{CourseName: "课程", ClassName: "A", TaskName: "作业"}
	db.Create(&a)
	source := filepath.Join(root, "input", "20260001_张三")
	os.MkdirAll(source, 0700)
	os.WriteFile(filepath.Join(source, "main.py"), []byte("print(1)"), 0600)
	t.Setenv("JWT_ACCESS_SECRET", "materials-http-test-key")
	gin.SetMode(gin.TestMode)
	router := gin.New()
	g := router.Group("/api/materials", middleware.AuthMiddleware(), middleware.RBACMiddleware("teacher", "admin"))
	g.GET("", ListMaterials)
	g.POST("", CreateMaterials)
	g.GET("/:id", GetMaterials)
	g.POST("/:id/grade", GradeMaterials)
	g.POST("/:id/cancel", CancelMaterials)
	g.POST("/:id/retry", RetryMaterials)
	router.POST("/assignments/:id/submit", middleware.AuthMiddleware(), middleware.RBACMiddleware("teacher", "admin"), UploadAssignment)
	request := func(method, path, role string, user uint, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		if role != "" {
			token, _ := auth.GenerateAccessToken(user, "test", role, nil)
			req.Header.Set("Authorization", "Bearer "+token)
		}
		out := httptest.NewRecorder()
		router.ServeHTTP(out, req)
		return out
	}
	body, _ := json.Marshal(map[string]any{"assignment_id": a.ID, "source_path": source})
	if out := request("POST", "/api/materials", "", 0, string(body)); out.Code != 401 {
		t.Fatal("unauthenticated create allowed")
	}
	if out := request("POST", "/api/materials", "student", 8301, string(body)); out.Code != 403 {
		t.Fatal("student create allowed")
	}
	out := request("POST", "/api/materials", "teacher", 8301, string(body))
	if out.Code != 202 {
		t.Fatalf("create: %d %s", out.Code, out.Body.String())
	}
	var batch models.MaterialBatch
	json.Unmarshal(out.Body.Bytes(), &batch)
	if batch.AutoGrade || batch.GradingJobID != "" {
		t.Fatal("create implicitly graded")
	}
	if out := request("GET", "/api/materials/"+batch.ID, "teacher", 8302, ""); out.Code != 404 {
		t.Fatal("cross-owner detail allowed")
	}
	if out := request("POST", "/api/materials/"+batch.ID+"/grade", "teacher", 8302, "{}"); out.Code != 400 {
		t.Fatal("cross-owner grade allowed")
	}
	if out := request("POST", "/api/materials/"+batch.ID+"/cancel", "teacher", 8302, "{}"); out.Code != 409 {
		t.Fatal("cross-owner cancel allowed")
	}
	if out := request("GET", "/api/materials", "teacher", 8302, ""); strings.Contains(out.Body.String(), batch.ID) {
		t.Fatal("list leaked other owner")
	}
	if out := request("POST", "/api/materials/"+batch.ID+"/cancel", "teacher", 8301, "{}"); out.Code != 200 {
		t.Fatal("owner cancel failed")
	}
	if out := request("POST", "/api/materials/"+batch.ID+"/retry", "teacher", 8301, "{}"); out.Code != 409 {
		t.Fatal("cancelled batch retry allowed")
	}
	out = request("POST", "/api/materials", "teacher", 8301, string(body))
	var failed models.MaterialBatch
	if out.Code != 202 || json.Unmarshal(out.Body.Bytes(), &failed) != nil {
		t.Fatal("retry fixture create failed")
	}
	db.Model(&failed).Updates(map[string]any{"status": "FAILED", "auto_grade": true, "grading_state": "REQUESTED"})
	if out := request("POST", "/api/materials/"+failed.ID+"/retry", "teacher", 8302, "{}"); out.Code != 409 {
		t.Fatal("cross-owner retry allowed")
	}
	if out := request("POST", "/api/materials/"+failed.ID+"/retry", "student", 8301, "{}"); out.Code != 403 {
		t.Fatal("student retry allowed")
	}
	out = request("POST", "/api/materials/"+failed.ID+"/retry", "teacher", 8301, "{}")
	var retried models.MaterialBatch
	if out.Code != 202 || json.Unmarshal(out.Body.Bytes(), &retried) != nil || retried.Status != "PENDING" || retried.AutoGrade || retried.GradingJobID != "" {
		t.Fatalf("retry did not safely requeue: %d %s", out.Code, out.Body.String())
	}
	// Upload remains usable without a rubric, because it prepares rather than grades.
	var compressed bytes.Buffer
	z := zip.NewWriter(&compressed)
	f, _ := z.Create("20260002_李四/main.py")
	f.Write([]byte("print(2)"))
	z.Close()
	var upload bytes.Buffer
	form := multipart.NewWriter(&upload)
	file, _ := form.CreateFormFile("batch_file", "班级作业.zip")
	file.Write(compressed.Bytes())
	form.Close()
	req := httptest.NewRequest("POST", fmt.Sprintf("/assignments/%d/submit", a.ID), &upload)
	req.Header.Set("Content-Type", form.FormDataContentType())
	token, _ := auth.GenerateAccessToken(8301, "test", "teacher", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	out = httptest.NewRecorder()
	router.ServeHTTP(out, req)
	if out.Code != 202 || !strings.Contains(out.Body.String(), "homework_materials") {
		t.Fatalf("upload started grading or required rubric: %d %s", out.Code, out.Body.String())
	}
	var uploaded struct {
		BatchID string `json:"batch_id"`
	}
	json.Unmarshal(out.Body.Bytes(), &uploaded)
	var saved models.MaterialBatch
	db.First(&saved, "id = ?", uploaded.BatchID)
	if saved.AutoGrade || saved.GradingJobID != "" {
		t.Fatal("upload auto graded")
	}
	// Remove only this test's uniquely allocated upload directory, verified beneath cwd.
	parent := filepath.Dir(saved.SourcePath)
	rel, err := filepath.Rel(filepath.Join(cwd, "uploads", "assignments"), parent)
	if err == nil && rel != "." && !strings.HasPrefix(rel, "..") && !filepath.IsAbs(rel) {
		os.RemoveAll(parent)
	}
}
