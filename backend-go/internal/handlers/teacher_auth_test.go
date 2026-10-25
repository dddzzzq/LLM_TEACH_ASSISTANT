package handlers

import (
	"encoding/json"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"grading-gateway/internal/auth"
	"grading-gateway/internal/models"
	"grading-gateway/internal/testutil"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestIntegrationTeacherEditionAcceptsExistingTeacherAndAdminButRejectsStudent(t *testing.T) {
	db, _ := testutil.ExecutionDB(t)
	t.Setenv("JWT_ACCESS_SECRET", "teacher-edition-access-test")
	t.Setenv("JWT_REFRESH_SECRET", "teacher-edition-refresh-test")
	gin.SetMode(gin.TestMode)
	handler := NewAuthHandler(db)
	router := gin.New()
	router.POST("/login", handler.Login)
	router.POST("/refresh", handler.Refresh)
	router.POST("/register", handler.Register)
	request := func(path string, payload any) *httptest.ResponseRecorder {
		raw, _ := json.Marshal(payload)
		req := httptest.NewRequest("POST", path, strings.NewReader(string(raw)))
		req.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		router.ServeHTTP(response, req)
		return response
	}
	password := "synthetic-teacher-password"
	hash, _ := auth.HashPassword(password)
	for _, role := range []models.UserRole{models.RoleTeacher, models.RoleAdmin, models.RoleStudent} {
		user := models.User{Username: "teacher-split-" + uuid.NewString(), Name: "Fixture Teacher", PasswordHash: hash, Role: role}
		if err := db.Create(&user).Error; err != nil {
			t.Fatal(err)
		}
		expected := 200
		if role == models.RoleStudent {
			expected = 403
		}
		response := request("/login", LoginRequest{Username: user.Username, Password: password})
		if response.Code != expected {
			t.Fatalf("role %s login: %d", role, response.Code)
		}
		_, refresh, err := auth.GenerateTokenPair(user.ID, user.Username, string(role), nil)
		if err != nil {
			t.Fatal(err)
		}
		if response = request("/refresh", RefreshRequest{RefreshToken: refresh}); response.Code != expected {
			t.Fatalf("role %s refresh: %d", role, response.Code)
		}
	}
	response := request("/register", map[string]string{"username": "new-teacher-" + uuid.NewString(), "password": password, "name": "New Teacher"})
	var result LoginResponse
	if response.Code != 201 || json.Unmarshal(response.Body.Bytes(), &result) != nil || result.Role != "teacher" {
		t.Fatal("registration did not default to teacher")
	}
}
