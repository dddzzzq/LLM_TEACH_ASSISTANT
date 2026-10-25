package server

import (
	"bytes"
	"grading-gateway/internal/auth"
	"grading-gateway/internal/logbook"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestTeacherEditionRejectsStudentTokensAndExamEndpoints(t *testing.T) {
	t.Setenv("JWT_ACCESS_SECRET", "teacher-edition-test-secret")
	router := NewRouter(nil, &logbook.Service{})
	student, err := auth.GenerateAccessToken(1, "fixture", "student", nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/api/profile", "/api/sessions", "/api/jobs/test", "/assignments/"} {
		request := httptest.NewRequest("GET", path, nil)
		request.Header.Set("Authorization", "Bearer "+student)
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		if response.Code != 403 {
			t.Fatalf("student reached %s: %d", path, response.Code)
		}
	}
	teacher, _ := auth.GenerateAccessToken(1, "fixture", "teacher", nil)
	for _, path := range []string{"/exams/", "/exams/1", "/exams/1/grade_submission"} {
		for _, method := range []string{"GET", "POST", "DELETE"} {
			request := httptest.NewRequest(method, path, nil)
			request.Header.Set("Authorization", "Bearer "+teacher)
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			if response.Code != 404 {
				t.Fatalf("exam endpoint remains: %s %s %d", method, path, response.Code)
			}
		}
	}
	for _, role := range []string{"student", "admin"} {
		request := httptest.NewRequest("POST", "/api/register", bytes.NewBufferString(`{"username":"fixture","password":"test-secret","name":"Fixture Teacher","role":"`+role+`"}`))
		request.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		if response.Code != 400 || strings.Contains(response.Body.String(), "test-secret") {
			t.Fatal("registration accepted an unsupported role or exposed a password")
		}
	}
}
