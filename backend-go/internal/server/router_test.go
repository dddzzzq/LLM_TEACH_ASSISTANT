package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"

	"grading-gateway/internal/auth"
	"grading-gateway/internal/logbook"

	"github.com/gin-gonic/gin"
)

type routeContract struct{ Method, Path, Handler string }

// This fixture was captured from main.go before route registration was moved.
func TestRouteContract(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := NewRouter(nil, &logbook.Service{})
	data, err := os.ReadFile("testdata/routes.json")
	if err != nil {
		t.Fatal(err)
	}
	var want []routeContract
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatal(err)
	}
	got := []routeContract{}
	for _, r := range router.Routes() {
		got = append(got, routeContract{r.Method, r.Path, r.Handler})
	}
	sort.Slice(got, func(i, j int) bool { return got[i].Method+got[i].Path < got[j].Method+got[j].Path })
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("route contract changed:\n got: %+v\nwant: %+v", got, want)
	}

	params := regexp.MustCompile(`:[^/]+`)
	for _, route := range want {
		if route.Path == "/api/login" || route.Path == "/api/register" || route.Path == "/api/refresh" || route.Path == "/api/rpa/jobs/:id/stream" || strings.HasPrefix(route.Path, "/uploads/") {
			continue
		}
		t.Run(route.Method+" "+route.Path, func(t *testing.T) {
			response := httptest.NewRecorder()
			router.ServeHTTP(response, httptest.NewRequest(route.Method, params.ReplaceAllString(route.Path, "1"), nil))
			if response.Code != http.StatusUnauthorized {
				t.Fatalf("unauthenticated status = %d, want 401", response.Code)
			}
		})
	}
}

func TestRoleAndPreflightContract(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := NewRouter(nil, &logbook.Service{})
	t.Setenv("JWT_ACCESS_SECRET", "router-contract-test-secret")
	for _, tc := range []struct {
		role, method, path string
		status             int
	}{
		{"student", "GET", "/api/materials", 403},
		{"student", "POST", "/api/materials/1/grade", 403},
		{"student", "GET", "/api/rpa/jobs", 403},
		{"student", "POST", "/api/rpa/jobs/1/resume", 403},
		{"student", "GET", "/api/admin/tools", 403},
		{"student", "GET", "/api/admin/skills", 403},
		{"student", "POST", "/assignments/", 403},
		{"student", "PUT", "/submissions/1", 403},
		{"student", "POST", "/exams/", 403},
		{"teacher", "GET", "/api/logs/sources", 403},
		{"student", "GET", "/api/logs/sources", 403},
		{"admin", "GET", "/api/logs/sources", 503}, // Authorized, but no index was started.
	} {
		t.Run(tc.role+" "+tc.method+" "+tc.path, func(t *testing.T) {
			token, err := auth.GenerateAccessToken(1, "test", tc.role, nil)
			if err != nil {
				t.Fatal(err)
			}
			request := httptest.NewRequest(tc.method, tc.path, nil)
			request.Header.Set("Authorization", "Bearer "+token)
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			if response.Code != tc.status {
				t.Fatalf("status = %d, want %d: %s", response.Code, tc.status, response.Body.String())
			}
		})
	}
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest("OPTIONS", "/api/materials", nil))
	if response.Code != 204 || response.Header().Get("Access-Control-Allow-Origin") != "*" || !strings.Contains(response.Header().Get("Access-Control-Allow-Headers"), "Idempotency-Key") {
		t.Fatalf("preflight contract changed: %d %v", response.Code, response.Header())
	}
}
