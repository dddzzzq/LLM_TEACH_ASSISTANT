package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"grading-gateway/internal/auth"
	"grading-gateway/internal/database"
	"grading-gateway/internal/logbook"
	"grading-gateway/internal/models"
	"grading-gateway/internal/testutil"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestIntegrationPortalHTTPAndAssignmentOwnership(t *testing.T) {
	db, owner := testutil.ExecutionDB(t)
	if err := db.AutoMigrate(&models.PortalConnection{}, &models.PortalOperation{}, &models.PortalOperationEvent{}, &models.PortalWorkflow{}, &models.PortalWorkflowItem{}, &models.RPAJob{}, &models.RPAFile{}, &models.Assignment{}, &models.Submission{}); err != nil {
		t.Fatal(err)
	}
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old })
	t.Setenv("JWT_ACCESS_SECRET", "portal-http-test-only")
	t.Setenv("PORTAL_CREDENTIAL_KEY_BASE64", base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x42}, 32)))
	router := NewRouter(db, &logbook.Service{})
	request := func(method, path, body, role string, user uint) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		if role != "" {
			token, err := auth.GenerateAccessToken(user, "portal-student", role, nil)
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Authorization", "Bearer "+token)
		}
		out := httptest.NewRecorder()
		router.ServeHTTP(out, req)
		return out
	}
	for _, role := range []string{"", "student"} {
		out := request("PUT", "/api/portal/connection", `{"username":"private-teacher-account","password":"private-password","trust_device":true}`, role, owner)
		want := 403
		if role == "" {
			want = 401
		}
		if out.Code != want {
			t.Fatalf("portal role status: %d", out.Code)
		}
	}
	out := request("PUT", "/api/portal/connection", `{"username":"private-teacher-account","password":"private-password","trust_device":true}`, "teacher", owner)
	if out.Code != 202 {
		t.Fatalf("save: %d %s", out.Code, out.Body)
	}
	var saved struct {
		OperationID string `json:"operation_id"`
	}
	if json.Unmarshal(out.Body.Bytes(), &saved) != nil || saved.OperationID == "" {
		t.Fatal("no operation receipt")
	}
	for _, path := range []string{"/api/portal/connection", "/api/portal/operations/" + saved.OperationID, "/api/portal/operations/" + saved.OperationID + "/events"} {
		out = request("GET", path, "", "teacher", owner)
		if out.Code != 200 {
			t.Fatalf("read: %d %s", out.Code, out.Body)
		}
		for _, secret := range []string{"private-teacher-account", "private-password", "credential_ciphertext", "credential_nonce", "credential_key_id"} {
			if strings.Contains(out.Body.String(), secret) {
				t.Fatal("secret in public response")
			}
		}
		if out.Header().Get("Cache-Control") != "no-store" {
			t.Fatal("sensitive response was cacheable")
		}
	}
	if out = request("GET", "/api/portal/operations/"+saved.OperationID, "", "teacher", owner+100000); out.Code != 404 {
		t.Fatal("cross-owner operation readable")
	}
	if out = request("PUT", "/api/portal/connection", `{"owner_id":1,"username":"x","password":"y"}`, "teacher", owner); out.Code != 400 {
		t.Fatal("accepted owner injection")
	}
	if out = request("DELETE", "/api/portal/connection", "", "teacher", owner); out.Code != 202 {
		t.Fatalf("disconnect: %d %s", out.Code, out.Body)
	}
	var connection models.PortalConnection
	db.Where("owner_id = ?", owner).First(&connection)
	if len(connection.CredentialCiphertext) > 0 {
		t.Fatal("disconnect retained credentials")
	}
	out = request("POST", "/assignments/", `{"course_name":"portal-http-course","class_name":"A","task_name":"portal-http-task","question":"question","rubric":{"正确性":100}}`, "teacher", owner)
	if out.Code != 201 {
		t.Fatalf("assignment: %d %s", out.Code, out.Body)
	}
	var created struct {
		ID uint `json:"id"`
	}
	_ = json.Unmarshal(out.Body.Bytes(), &created)
	var assignment models.Assignment
	db.First(&assignment, created.ID)
	if assignment.OwnerID != owner {
		t.Fatal("assignment owner not injected")
	}
	for _, path := range []string{fmt.Sprintf("/assignments/%d", created.ID), fmt.Sprintf("/assignments/%d/results", created.ID), fmt.Sprintf("/assignments/%d/export", created.ID)} {
		if out = request("GET", path, "", "teacher", owner+100000); out.Code != 404 {
			t.Fatalf("foreign teacher read %s: %d", path, out.Code)
		}
	}
	if out = request("GET", "/assignments/", "", "teacher", owner+100000); out.Code != 200 || strings.Contains(out.Body.String(), "portal-http-course") {
		t.Fatal("assignment list leaked teacher data")
	}
}
