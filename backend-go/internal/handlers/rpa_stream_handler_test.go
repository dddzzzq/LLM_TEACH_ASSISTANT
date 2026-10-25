package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"grading-gateway/internal/auth"
)

func TestBrowserStreamOrigin(t *testing.T) {
	t.Setenv("RPA_BROWSER_ALLOWED_ORIGINS", "https://allowed.example")
	for _, tc := range []struct {
		origin string
		want   bool
	}{
		{"https://portal.example", true}, {"", true}, {"https://allowed.example", true},
		{"https://attacker.example", false}, {"https://portal.example.attacker.example", false},
		{"file://portal.example", false},
	} {
		req := httptest.NewRequest("GET", "http://portal.example/api/rpa/jobs/job/stream", nil)
		req.Header.Set("Origin", tc.origin)
		if got := browserStreamOrigin(req); got != tc.want {
			t.Errorf("origin %q got %v, want %v", tc.origin, got, tc.want)
		}
	}
}

func TestBrowserStreamRejectsUnauthenticatedAndStudentBeforeWorker(t *testing.T) {
	t.Setenv("JWT_ACCESS_SECRET", "stream-test-signing-key")
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.GET("/api/rpa/jobs/:id/stream", RPABrowserStream)
	server := httptest.NewServer(router)
	defer server.Close()
	student, err := auth.GenerateAccessToken(501, "fixture", "student", nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, token string
		code        int
	}{
		{"invalid token", "invalid-token", 4401}, {"student", student, 4403},
	} {
		t.Run(tc.name, func(t *testing.T) {
			conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/api/rpa/jobs/job/stream", nil)
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close()
			conn.SetReadDeadline(time.Now().Add(2 * time.Second))
			if err := conn.WriteJSON(map[string]string{"type": "auth", "token": tc.token}); err != nil {
				t.Fatal(err)
			}
			var rejected map[string]any
			if err := conn.ReadJSON(&rejected); err != nil || rejected["type"] != "error" {
				t.Fatalf("missing authentication rejection: %v %v", rejected, err)
			}
			_, _, err = conn.ReadMessage()
			if !websocket.IsCloseError(err, tc.code) {
				t.Fatalf("want close %d, got %v", tc.code, err)
			}
		})
	}
	_, response, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/api/rpa/jobs/job/stream",
		http.Header{"Origin": []string{"https://attacker.example"}})
	if err == nil || response.StatusCode != http.StatusForbidden {
		t.Fatal("cross-origin connection accepted")
	}
	if response != nil {
		response.Body.Close()
	}
}
