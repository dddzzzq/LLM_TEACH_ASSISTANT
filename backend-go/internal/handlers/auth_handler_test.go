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
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
)

func TestRegistrationRejectsPasswordInDisplayNameBeforeSaving(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/register", NewAuthHandler(nil).Register)
	for _, name := range []string{"fixture-password", " fixture-password "} {
		body, _ := json.Marshal(RegisterRequest{Username: "fixture-user", Password: "fixture-password", Name: name, Role: "teacher"})
		req := httptest.NewRequest("POST", "/register", strings.NewReader(string(body)))
		req.Header.Set("Content-Type", "application/json")
		out := httptest.NewRecorder()
		router.ServeHTTP(out, req)
		if out.Code != 400 || strings.Contains(out.Body.String(), "fixture-password") {
			t.Fatalf("password accepted as display name or exposed in error: status=%d", out.Code)
		}
	}
}

func TestIntegrationRegistrationLoginAndRefreshReturnCanonicalAccount(t *testing.T) {
	dsn := os.Getenv("RPA_TEST_DSN")
	if dsn == "" {
		t.Skip("requires isolated RPA_TEST_DSN")
	}
	if !strings.Contains(dsn, "/rpa_test_") {
		t.Fatal("refusing production database")
	}
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s, _ := db.DB(); s.Close() })
	if err := db.AutoMigrate(&models.User{}); err != nil {
		t.Fatal(err)
	}
	t.Setenv("JWT_ACCESS_SECRET", "auth-flow-fixture-access")
	t.Setenv("JWT_REFRESH_SECRET", "auth-flow-fixture-refresh")
	gin.SetMode(gin.TestMode)
	router := gin.New()
	handler := NewAuthHandler(db)
	router.POST("/register", handler.Register)
	router.POST("/login", handler.Login)
	router.POST("/refresh", handler.Refresh)
	router.GET("/profile", middleware.AuthMiddleware(), handler.Profile)
	request := func(path string, body any) *httptest.ResponseRecorder {
		raw, _ := json.Marshal(body)
		req := httptest.NewRequest("POST", path, strings.NewReader(string(raw)))
		req.Header.Set("Content-Type", "application/json")
		out := httptest.NewRecorder()
		router.ServeHTTP(out, req)
		return out
	}
	for _, role := range []string{"teacher", "student", "admin"} {
		username := "auth-fixture-" + uuid.NewString()
		password := "fixture-different-secret"
		name := "Fixture Person"
		out := request("/register", RegisterRequest{Username: username, Password: password, Name: name, Role: role})
		if out.Code != 201 {
			t.Fatalf("register: %d", out.Code)
		}
		var registered LoginResponse
		if json.Unmarshal(out.Body.Bytes(), &registered) != nil || registered.Username != username || registered.Name != name || registered.Role != role || strings.Contains(out.Body.String(), password) {
			t.Fatal("registration response mixed account fields")
		}
		var stored models.User
		if err := db.First(&stored, registered.UserID).Error; err != nil {
			t.Fatal(err)
		}
		if stored.Username != username || stored.Name != name || stored.PasswordHash == password || !auth.VerifyPassword(password, stored.PasswordHash) {
			t.Fatal("registration stored account fields incorrectly")
		}
		out = request("/login", LoginRequest{Username: password, Password: username})
		if out.Code != 401 {
			t.Fatal("swapped credentials accepted")
		}
		out = request("/login", LoginRequest{Username: username, Password: password})
		var logged LoginResponse
		if out.Code != 200 || json.Unmarshal(out.Body.Bytes(), &logged) != nil || logged.Username != username || logged.Name != name {
			t.Fatal("login lost canonical account fields")
		}
		claims, err := auth.ParseAccessToken(logged.AccessToken, nil)
		if err != nil || claims.Username != username || claims.UserID != stored.ID {
			t.Fatal("login token identity mismatch")
		}
		out = request("/refresh", RefreshRequest{RefreshToken: logged.RefreshToken})
		var refreshed LoginResponse
		if out.Code != 200 || json.Unmarshal(out.Body.Bytes(), &refreshed) != nil || refreshed.Username != username || refreshed.Name != name {
			t.Fatal("refresh lost canonical account fields")
		}
		req := httptest.NewRequest("GET", "/profile", nil)
		req.Header.Set("Authorization", "Bearer "+refreshed.AccessToken)
		profile := httptest.NewRecorder()
		router.ServeHTTP(profile, req)
		if profile.Code != 200 || strings.Contains(profile.Body.String(), password) || !strings.Contains(profile.Body.String(), username) {
			t.Fatal("profile lost or exposed account fields")
		}
		// An API client cannot bypass the frontend safeguard to store a password in name.
		badUsername := "bad-fixture-" + uuid.NewString()
		out = request("/register", RegisterRequest{Username: badUsername, Password: password, Name: password, Role: role})
		var count int64
		db.Model(&models.User{}).Where("username = ?", badUsername).Count(&count)
		if out.Code != 400 || count != 0 {
			t.Fatal("unsafe display name reached database")
		}
	}
}
