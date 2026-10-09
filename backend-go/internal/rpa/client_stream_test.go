package rpa

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestBrowserStreamAuthenticatesWorkerAndPipelinesInput(t *testing.T) {
	tokenPath := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(tokenPath, []byte("stream-worker-token"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RPA_CONTROL_TOKEN_FILE", tokenPath)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tasks/fixture/stream" || r.Header.Get("X-RPA-Token") != "stream-worker-token" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		conn.SetReadDeadline(time.Now().Add(3 * time.Second))
		// Read both inputs before acknowledging either, as a Worker may do.
		for i := 0; i < 2; i++ {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
		conn.WriteJSON(map[string]any{"type": "input_ack", "id": 2})
	}))
	defer server.Close()
	t.Setenv("RPA_CONTROL_URL", server.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, err := OpenBrowserStream(ctx, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	for id := 1; id <= 2; id++ {
		if err := conn.WriteJSON(map[string]any{"type": "input", "id": id}); err != nil {
			t.Fatal(err)
		}
	}
	var ack struct {
		ID int `json:"id"`
	}
	if err := conn.ReadJSON(&ack); err != nil || ack.ID != 2 {
		t.Fatalf("stream did not pipeline input: %+v %v", ack, err)
	}
}
