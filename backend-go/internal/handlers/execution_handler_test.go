package handlers

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"grading-gateway/internal/agent"
	"grading-gateway/internal/execution"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
	"grading-gateway/internal/testutil"

	"github.com/gin-gonic/gin"
)

func TestExecutionHTTPPersistsAndRebuildsStructuredHistoryWithoutRedis(t *testing.T) {
	db, user := testutil.ExecutionDB(t)
	gin.SetMode(gin.TestMode)
	actor := user
	modelCalls := 0
	fullReply := strings.Repeat("detailed answer ", 400)
	call := func(_ context.Context, messages []map[string]any, _ []map[string]any) (*agent.LLMResponse, error) {
		modelCalls++
		if modelCalls == 2 {
			counts := map[string]int{}
			for _, m := range messages {
				if m["role"] == "user" {
					counts[m["content"].(string)]++
				}
			}
			if counts["first"] != 1 || counts["second"] != 1 {
				t.Fatalf("current input duplicated: %+v", messages)
			}
			found := false
			for _, m := range messages {
				if m["role"] == "assistant" && m["content"] == fullReply {
					found = true
				}
			}
			if !found {
				t.Fatal("full persisted model reply was not reconstructed")
			}
		}
		return &agent.LLMResponse{Content: fullReply}, nil
	}
	router := gin.New()
	router.Use(func(c *gin.Context) {
		c.Set(string(middleware.UserIDKey), actor)
		c.Set(string(middleware.RoleKey), "teacher")
		c.Set(string(middleware.UsernameKey), "fixture")
	})
	router.POST("/chat", func(c *gin.Context) { agentChat(c, db, call) })
	h := NewSessionHandler(db)
	router.GET("/sessions/:session_id/history", h.GetSessionHistory)
	router.GET("/sessions/:session_id/runs", h.GetExecutions)
	router.GET("/sessions/:session_id/runs/:run_id", h.GetExecutionEvents)
	router.GET("/sessions/:session_id/runs/:run_id/context", h.GetExecutionContext)
	router.DELETE("/sessions/:session_id", h.DeleteSession)
	request := func(method, path, key string, body any) *httptest.ResponseRecorder {
		data, _ := json.Marshal(body)
		req := httptest.NewRequest(method, path, strings.NewReader(string(data)))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Idempotency-Key", key)
		response := httptest.NewRecorder()
		router.ServeHTTP(response, req)
		return response
	}
	first := request("POST", "/chat", "request-one", AgentChatRequest{Message: "first"})
	if first.Code != 200 {
		t.Fatalf("first %d: %s", first.Code, first.Body.String())
	}
	var reply AgentChatResponse
	if err := json.Unmarshal(first.Body.Bytes(), &reply); err != nil {
		t.Fatal(err)
	}
	if reply.RunID == "" || reply.SessionID == "" {
		t.Fatal("execution identity missing")
	}
	if len(reply.Reply) >= len(fullReply) {
		t.Fatal("fixture no longer distinguishes display from durable output")
	}
	duplicate := request("POST", "/chat", "request-one", AgentChatRequest{Message: "first"})
	if duplicate.Code != 200 || modelCalls != 1 {
		t.Fatalf("retry called model again: %d %d", duplicate.Code, modelCalls)
	}
	conflict := request("POST", "/chat", "request-one", AgentChatRequest{Message: "different", SessionID: reply.SessionID})
	if conflict.Code != 409 {
		t.Fatal("idempotency collision accepted")
	}
	second := request("POST", "/chat", "request-two", AgentChatRequest{Message: "second", SessionID: reply.SessionID})
	if second.Code != 200 || modelCalls != 2 {
		t.Fatalf("second: %d %s", second.Code, second.Body.String())
	}
	base := "/sessions/" + reply.SessionID
	history := request("GET", base+"/history", "", nil)
	var historyBody struct {
		Messages []agent.Message `json:"messages"`
	}
	if history.Code != 200 || json.Unmarshal(history.Body.Bytes(), &historyBody) != nil || len(historyBody.Messages) != 4 {
		t.Fatalf("durable UI history: %s", history.Body.String())
	}
	// Reading/rebuilding repeatedly never writes back messages or dispatches the model.
	for i := 0; i < 2; i++ {
		view := request("GET", base+"/runs/"+reply.RunID+"/context", "", nil)
		if view.Code != 200 || !strings.Contains(view.Body.String(), fullReply) {
			t.Fatalf("context: %s", view.Body.String())
		}
	}
	page := request("GET", base+"/runs/"+reply.RunID+"?after_seq=1&limit=2", "", nil)
	var pageBody struct {
		Events []models.AgentEvent `json:"events"`
		Next   uint64              `json:"next_seq"`
		More   bool                `json:"has_more"`
	}
	if page.Code != 200 || json.Unmarshal(page.Body.Bytes(), &pageBody) != nil || len(pageBody.Events) != 2 || pageBody.Next != 3 || !pageBody.More {
		t.Fatalf("pagination: %s", page.Body.String())
	}
	if request("GET", base+"/runs/"+reply.RunID+"?limit=1000", "", nil).Code != 400 {
		t.Fatal("unbounded event query accepted")
	}
	actor = user + 10000
	for _, path := range []string{base + "/history", base + "/runs", base + "/runs/" + reply.RunID, base + "/runs/" + reply.RunID + "/context"} {
		if response := request("GET", path, "", nil); response.Code != 404 {
			t.Fatalf("foreign access %s: %d", path, response.Code)
		}
	}
	actor = user
	var count int64
	db.Model(&models.ChatMessage{}).Where("session_id = ?", reply.SessionID).Count(&count)
	if count != 4 || modelCalls != 2 {
		t.Fatal("read-only reconstruction produced effects")
	}
	// The event log remains sufficient after the compatibility projection is lost.
	if err := db.Where("session_id = ?", reply.SessionID).Delete(&models.ChatMessage{}).Error; err != nil {
		t.Fatal(err)
	}
	rebuilt := request("GET", base+"/history", "", nil)
	if rebuilt.Code != 200 || json.Unmarshal(rebuilt.Body.Bytes(), &historyBody) != nil || len(historyBody.Messages) != 4 {
		t.Fatalf("display could not be rebuilt from facts: %s", rebuilt.Body.String())
	}
	store := execution.Store{DB: db}
	active, _, err := store.Begin(context.Background(), execution.Start{UserID: user, SessionID: reply.SessionID, Input: "active", Role: "teacher"})
	if err != nil {
		t.Fatal(err)
	}
	if request("DELETE", base, "", nil).Code != 409 {
		t.Fatal("active session deleted")
	}
	if err := store.Finish(context.Background(), active.ID, execution.Succeeded, "done", nil); err != nil {
		t.Fatal(err)
	}
	if deleted := request("DELETE", base, "", nil); deleted.Code != 200 {
		t.Fatalf("delete: %s", deleted.Body.String())
	}
	for _, model := range []any{&models.AgentEvent{}, &models.AgentRun{}, &models.ChatMessage{}} {
		db.Model(model).Where("session_id = ?", reply.SessionID).Count(&count)
		if count != 0 {
			t.Fatal("session deletion left execution data")
		}
	}
}
