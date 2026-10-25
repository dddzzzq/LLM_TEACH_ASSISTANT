package portal

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gorm.io/gorm"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
	"grading-gateway/internal/rpa"
	"grading-gateway/internal/testutil"
)

func TestIntegrationPortalOperationsReuseOrdinaryRPAJobs(t *testing.T) {
	s, owner := portalDB(t)
	ctx := context.Background()
	r := NewRunner(s, NewBroker(s))
	op, err := s.Save(ctx, owner, SaveRequest{"reuse-teacher", "fixture-password", true})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if err = r.ensureTask(ctx, op); err != nil {
			t.Fatal(err)
		}
	}
	var job models.RPAJob
	if err = s.DB.First(&job, "id = ?", op.ID).Error; err != nil {
		t.Fatal(err)
	}
	var count int64
	s.DB.Model(&models.RPAJob{}).Where("id = ?", op.ID).Count(&count)
	var navigation rpa.NavigationSnapshot
	if json.Unmarshal([]byte(job.NavigationJSON), &navigation) != nil || navigation.Skill.Name != "sync-teacher-portal" || count != 1 || job.TaskType != "AUTH_BOOTSTRAP" || job.UserID != owner {
		t.Fatal("initial login did not reuse an idempotent owned RPA job")
	}
	if strings.Contains(job.NavigationJSON, "fixture-password") || strings.Contains(job.NavigationJSON, "reuse-teacher") {
		t.Fatal("credentials leaked to Skill")
	}
	if err = r.apply(ctx, op, operationState{Status: "SUCCEEDED", Stage: "DONE", VerifiedSubject: "reuse-teacher", StateVersion: 1}); err != nil {
		t.Fatal(err)
	}
	view, _ := s.View(ctx, owner)
	if err = r.ensureTask(ctx, *view.Operation); err != nil {
		t.Fatal(err)
	}
	var catalogJob models.RPAJob
	if err = s.DB.First(&catalogJob, "id = ?", view.Operation.ID).Error; err != nil {
		t.Fatal(err)
	}
	if json.Unmarshal([]byte(catalogJob.NavigationJSON), &navigation) != nil || navigation.Skill.Name != "sync-teacher-portal" || catalogJob.TaskType != "CATALOG_SYNC" {
		t.Fatal("catalog did not use the shared page Agent Skill")
	}
	if err = r.apply(ctx, *view.Operation, operationState{Status: "SUCCEEDED", Stage: "DONE", Catalog: ptrCatalog(fixtureCatalog()), StateVersion: 1}); err != nil {
		t.Fatal(err)
	}
	target := rpa.Target{CourseName: "fixture course", AssignmentName: "fixture work"}
	jobPtr, err := rpa.CreateWithKey(ctx, owner, target, "shared-profile-fixture")
	if err != nil {
		t.Fatal(err)
	}
	c, _ := s.connection(ctx, owner, false)
	if jobPtr.PortalConnectionID != c.ID || jobPtr.ProfileGeneration != c.ProfileGeneration || !s.downloadAllowed(ctx, c, jobPtr.ID) {
		t.Fatal("original named download did not reuse saved credentials")
	}
	if err := s.DB.Model(&c).Update("active_operation_id", "fixture-busy").Error; err != nil {
		t.Fatal(err)
	}
	repeated, err := rpa.CreateWithKey(ctx, owner, target, "shared-profile-fixture")
	if err != nil || repeated.ID != jobPtr.ID {
		t.Fatal("an existing request could not be recovered while the connection was busy")
	}
	if err := s.DB.Model(&c).Update("active_operation_id", "").Error; err != nil {
		t.Fatal(err)
	}
	// Use the existing valid UUID task as the occupied browser; no Worker call is needed.
	if err := s.DB.Model(&models.RPAJob{}).Where("id = ?", op.ID).Update("status", "WAITING_USER").Error; err != nil {
		t.Fatal(err)
	}
	if err := rpa.Start(ctx, jobPtr.ID, false); err == nil || !strings.Contains(err.Error(), "继续排队") {
		t.Fatal("a second task attempted to open an occupied teacher profile")
	}
	c.ProfileGeneration++
	if s.downloadAllowed(ctx, c, jobPtr.ID) {
		t.Fatal("old named task can use a replacement teacher profile")
	}
}

func ptrCatalog(value Catalog) *Catalog { return &value }

func portalDB(t *testing.T) (*Service, uint) {
	t.Helper()
	db, owner := testutil.ExecutionDB(t)
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old })
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	t.Cleanup(worker.Close)
	tokenFile := filepath.Join(t.TempDir(), "worker-token")
	if err := os.WriteFile(tokenFile, []byte("test-only-worker-token"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RPA_CONTROL_URL", worker.URL)
	t.Setenv("RPA_CONTROL_TOKEN_FILE", tokenFile)
	if err := db.AutoMigrate(&models.PortalConnection{}, &models.PortalOperation{}, &models.PortalOperationEvent{}, &models.PortalCatalogSnapshot{}, &models.PortalCourse{}, &models.PortalHomework{}, &models.PortalCourseClass{}, &models.PortalAssignmentBinding{}, &models.PortalWorkflow{}, &models.PortalWorkflowItem{}, &models.Assignment{}, &models.RPAJob{}, &models.RPAFile{}, &models.MaterialBatch{}, &models.AsyncJob{}); err != nil {
		t.Fatal(err)
	}
	c := testCipher(t)
	return &Service{DB: db, Cipher: func() (*CredentialCipher, error) { return c, nil }}, owner
}
func fixtureCatalog() Catalog {
	return Catalog{Complete: true, Courses: []CatalogCourse{
		{PlatformID: "11:22", Title: "同名课程", Term: "2025-2026第一学期", URL: "https://mooc1.chaoxing.com/course?courseid=11&clazzid=22", Classes: []CatalogClass{{"22", "A班"}}, Homeworks: []CatalogHomework{{PlatformID: "33", Title: "第一次作业", URL: "https://mooc1.chaoxing.com/work?workid=33&courseid=11"}}},
		{PlatformID: "12:23", Title: "同名课程", Term: "2026-2027第一学期", URL: "https://mooc1.chaoxing.com/course?courseid=12&clazzid=23", Classes: []CatalogClass{{"23", "B班"}}, Homeworks: []CatalogHomework{{PlatformID: "34", Title: "第一次作业", URL: "https://mooc1.chaoxing.com/work?workid=34&courseid=12"}}},
	}}
}

func TestIntegrationWorkflowDefaultsToAllClassesWithoutClassCatalog(t *testing.T) {
	s, owner := portalDB(t)
	ctx := context.Background()
	r := NewRunner(s, NewBroker(s))
	op, err := s.Save(ctx, owner, SaveRequest{"all-class-teacher", "test-password", true})
	if err != nil {
		t.Fatal(err)
	}
	if err = r.apply(ctx, op, operationState{Status: "SUCCEEDED", Stage: "DONE", VerifiedSubject: "all-class-teacher", StateVersion: 1}); err != nil {
		t.Fatal(err)
	}
	view, _ := s.View(ctx, owner)
	catalog := fixtureCatalog()
	for i := range catalog.Courses {
		catalog.Courses[i].Classes = nil
	}
	if err = r.apply(ctx, *view.Operation, operationState{Status: "SUCCEEDED", Stage: "DONE", Catalog: &catalog, StateVersion: 1}); err != nil {
		t.Fatal(err)
	}
	c, _ := s.connection(ctx, owner, false)
	courses, err := s.Courses(ctx, owner)
	if err != nil || len(courses) != 2 {
		t.Fatalf("catalog: %v", err)
	}
	detail, err := s.Homeworks(ctx, owner, courses[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	work := detail["homeworks"].([]models.PortalHomework)[0]
	req := WorkflowRequest{Source: "portal", SnapshotID: c.ActiveSnapshotID, Items: []Selection{{HomeworkID: work.ID}}, Question: "题目", Rubric: json.RawMessage(`{"正确性":100}`)}
	flow, err := s.CreateWorkflow(ctx, owner, req, "all-class-flow")
	if err != nil {
		t.Fatal(err)
	}
	r.tickWorkflows(ctx)
	var item models.PortalWorkflowItem
	if err = s.DB.First(&item, "workflow_id = ?", flow.ID).Error; err != nil {
		t.Fatal(err)
	}
	var target rpa.PortalTarget
	var assignment models.Assignment
	s.DB.First(&assignment, item.AssignmentID)
	if json.Unmarshal([]byte(item.TargetJSON), &target) != nil || target.ClassID != "" || target.HomeworkID == "" || item.ClassID != "" || item.ClassName != "全部班级" || assignment.ClassName != "全部班级" || assignment.Question != req.Question || item.RPAJobID == "" {
		t.Fatal("all-class workflow did not preserve the grading configuration and structured homework target")
	}
	duplicate, err := s.CreateWorkflow(ctx, owner, req, "all-class-flow")
	if err != nil || duplicate.ID != flow.ID {
		t.Fatal("all-class request lost idempotency")
	}
	req.Items = append(req.Items, req.Items[0])
	if _, err = s.CreateWorkflow(ctx, owner, req, "duplicate-all-class"); err == nil {
		t.Fatal("duplicate homework accepted")
	}
}
func TestIntegrationConnectionCatalogAndWorkflowIsolation(t *testing.T) {
	s, owner := portalDB(t)
	ctx := context.Background()
	r := NewRunner(s, NewBroker(s))
	op, err := s.Save(ctx, owner, SaveRequest{"teacher-001", "secret-password", true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = s.Save(ctx, owner, SaveRequest{"", "", true}); !errors.Is(err, ErrConflict) {
		t.Fatal("concurrent credential update allowed")
	}
	c, err := s.connection(ctx, owner, false)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(c.CredentialCiphertext), "secret-password") {
		t.Fatal("plaintext persisted")
	}
	if _, err = s.OwnedOperation(ctx, owner+10000, op.ID); err == nil {
		t.Fatal("other owner read operation")
	}
	if err = r.apply(ctx, op, operationState{Status: "SUCCEEDED", Stage: "DONE", VerifiedSubject: "other-teacher", StateVersion: 1}); err == nil {
		t.Fatal("accepted wrong identity")
	}
	if err = r.apply(ctx, op, operationState{Status: "SUCCEEDED", Stage: "DONE", VerifiedSubject: "teacher-001", StateVersion: 1}); err != nil {
		t.Fatal(err)
	}
	view, err := s.View(ctx, owner)
	if err != nil || view.Status != "READY" || view.Operation == nil || view.Operation.Type != "CATALOG_SYNC" {
		t.Fatalf("bootstrap did not enqueue sync: %+v %v", view, err)
	}
	catalog := fixtureCatalog()
	sync := *view.Operation
	if err = r.apply(ctx, sync, operationState{Status: "SUCCEEDED", Stage: "DONE", Catalog: &catalog, StateVersion: 1}); err != nil {
		t.Fatal(err)
	}
	c, _ = s.connection(ctx, owner, false)
	snapshot := c.ActiveSnapshotID
	if snapshot == "" || c.LastSyncedAt == nil {
		t.Fatal("catalog not published")
	}
	courses, err := s.Courses(ctx, owner)
	if err != nil || len(courses) != 2 || courses[0].ID == courses[1].ID {
		t.Fatal("same-name courses merged")
	}
	detail, err := s.Homeworks(ctx, owner, courses[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	works := detail["homeworks"].([]models.PortalHomework)
	classes := detail["classes"].([]models.PortalCourseClass)
	if _, err = s.Homeworks(ctx, owner+10000, courses[0].ID); err == nil {
		t.Fatal("other owner read catalog")
	}
	req := WorkflowRequest{Source: "portal", SnapshotID: snapshot, Items: []Selection{{works[0].ID, classes[0].ID}}, Question: "题目", Rubric: json.RawMessage(`{"正确性":100}`)}
	flow, err := s.CreateWorkflow(ctx, owner, req, "key-1")
	if err != nil {
		t.Fatal(err)
	}
	duplicate, err := s.CreateWorkflow(ctx, owner, req, "key-1")
	if err != nil || duplicate.ID != flow.ID {
		t.Fatal("workflow not idempotent")
	}
	changed := req
	changed.Question = "changed"
	if _, err = s.CreateWorkflow(ctx, owner, changed, "key-1"); err == nil {
		t.Fatal("idempotency collision allowed")
	}
	foreignDetail, _ := s.Homeworks(ctx, owner, courses[1].ID)
	foreignClasses := foreignDetail["classes"].([]models.PortalCourseClass)
	bad := req
	bad.Items = []Selection{{works[0].ID, foreignClasses[0].ID}}
	if _, err = s.CreateWorkflow(ctx, owner, bad, "bad-class"); err == nil {
		t.Fatal("class from another course accepted")
	}
	r.tickWorkflows(ctx)
	var item models.PortalWorkflowItem
	if err = s.DB.Where("workflow_id = ?", flow.ID).First(&item).Error; err != nil || item.RPAJobID == "" {
		t.Fatal("workflow not dispatched")
	}
	var job models.RPAJob
	s.DB.First(&job, "id = ?", item.RPAJobID)
	if job.AssignmentID != item.AssignmentID || job.PortalConnectionID != c.ID || job.PortalTargetJSON == "" {
		t.Fatal("download lost structured target")
	}
	s.DB.Model(&job).Update("status", "SUCCEEDED")
	r.tickWorkflows(ctx)
	var importing models.PortalWorkflowItem
	s.DB.First(&importing, "id = ?", item.ID)
	if importing.Status == "FAILED" {
		t.Fatal("file import race incorrectly failed workflow")
	}
	s.DB.Model(&job).Update("status", "QUEUED")
	if err = s.DB.Callback().Query().Before("gorm:query").Register("portal-test-query-failure", func(db *gorm.DB) {
		if db.Statement.Table == "portal_workflow_items" {
			_ = db.AddError(errors.New("synthetic query failure"))
		}
	}); err != nil {
		t.Fatal(err)
	}
	r.tickWorkflows(ctx)
	s.DB.Callback().Query().Remove("portal-test-query-failure")
	var stillRunning models.PortalWorkflow
	s.DB.First(&stillRunning, "id = ?", flow.ID)
	if stillRunning.Status == "SUCCEEDED" {
		t.Fatal("query failure was reported as workflow success")
	}
	var assignment models.Assignment
	s.DB.First(&assignment, item.AssignmentID)
	if assignment.OwnerID != owner || assignment.Question != req.Question {
		t.Fatal("wrong grading owner/snapshot")
	}
	if _, err = s.Save(ctx, owner, SaveRequest{"other-account", "other-password", true}); !errors.Is(err, ErrConflict) {
		t.Fatal("account changed during download")
	}
	s.DB.Model(&job).Update("status", "INTERRUPTED")
	s.DB.Model(&item).Update("status", "FAILED")
	s.DB.Model(&flow).Update("status", "FAILED")
	if err = s.RetryWorkflow(ctx, owner+10000, flow.ID); err == nil {
		t.Fatal("other teacher retried workflow")
	}
	if err = s.RetryWorkflow(ctx, owner, flow.ID); err != nil {
		t.Fatal(err)
	}
	var retried models.PortalWorkflowItem
	s.DB.First(&retried, "id = ?", item.ID)
	if retried.Status != "RETRY_QUEUED" || retried.RPAJobID != job.ID {
		t.Fatal("retry lost original checkpoint/job")
	}
	if !s.downloadAllowed(ctx, c, job.ID) {
		t.Fatal("explicit retry could not receive credentials before the restart response")
	}
	private := NewBroker(s)
	private.ready = true
	ticket, err := private.Issue(c, "", job.ID)
	if err != nil {
		t.Fatal(err)
	}
	s.DB.Model(&flow).Update("status", "CANCELLED")
	request := httptest.NewRequest("POST", "/credentials", nil)
	request.Header.Set("X-Portal-Ticket", ticket)
	response := httptest.NewRecorder()
	private.ServeHTTP(response, request)
	if response.Code != 403 {
		t.Fatal("canceled workflow redeemed a download credential ticket")
	}
	s.DB.Model(&flow).Update("status", "RUNNING")
	if err = s.CancelWorkflow(ctx, owner, flow.ID); err != nil {
		t.Fatal(err)
	}
	if _, err = s.Save(ctx, owner, SaveRequest{"other-account", "other-password", true}); err != nil {
		t.Fatal(err)
	}
	current, _ := s.connection(ctx, owner, false)
	if current.ActiveSnapshotID != "" || current.ProfileGeneration <= c.ProfileGeneration {
		t.Fatal("new account reused old catalog/profile")
	}
	if err = s.Disconnect(ctx, owner); err != nil {
		t.Fatal(err)
	}
	current, _ = s.connection(ctx, owner, false)
	if len(current.CredentialCiphertext) != 0 || current.Status != models.PortalConnectionDisabled {
		t.Fatal("disconnect retained credentials")
	}
}
func TestIntegrationCatalogAtomicityAndCredentialTicketRevocation(t *testing.T) {
	s, owner := portalDB(t)
	ctx := context.Background()
	op, err := s.Save(ctx, owner, SaveRequest{"teacher-002", "secret", true})
	if err != nil {
		t.Fatal(err)
	}
	b := NewBroker(s)
	b.ready = true
	c, _ := s.connection(ctx, owner, false)
	ticket, err := b.Issue(c, op.ID, "")
	if err != nil {
		t.Fatal(err)
	}
	request := func(token string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/credentials", nil)
		req.Header.Set("X-Portal-Ticket", token)
		out := httptest.NewRecorder()
		b.ServeHTTP(out, req)
		return out
	}
	if out := request(ticket); out.Code != 200 || !strings.Contains(out.Body.String(), "teacher-002") {
		t.Fatal("ticket not redeemable")
	}
	if out := request(ticket); out.Code != 401 {
		t.Fatal("ticket reused")
	}
	ticket, _ = b.Issue(c, op.ID, "")
	if err = s.Disconnect(ctx, owner); err != nil {
		t.Fatal(err)
	}
	if out := request(ticket); out.Code != 403 {
		t.Fatal("revoked ticket released credentials")
	}
	catalog := fixtureCatalog()
	if err = s.DB.Transaction(func(tx *gorm.DB) error { return publishCatalog(tx, &c, catalog) }); err != nil {
		t.Fatal(err)
	}
	previous := c.ActiveSnapshotID
	catalog.Complete = false
	if err = s.DB.Transaction(func(tx *gorm.DB) error { return publishCatalog(tx, &c, catalog) }); err == nil {
		t.Fatal("partial catalog published")
	}
	var stored models.PortalConnection
	s.DB.First(&stored, "id = ?", c.ID)
	if stored.ActiveSnapshotID != previous {
		t.Fatal("failed refresh erased previous snapshot")
	}
	b.mu.Lock()
	for key, value := range b.tickets {
		value.Until = time.Now().Add(-time.Second)
		b.tickets[key] = value
	}
	b.mu.Unlock()
}

func TestIntegrationLocalCredentialSocketIsPrivateAndSingleUse(t *testing.T) {
	s, owner := portalDB(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	op, err := s.Save(ctx, owner, SaveRequest{"socket-teacher", "socket-password", true})
	if err != nil {
		t.Fatal(err)
	}
	root, err := os.MkdirTemp("/tmp", "portal-ipc-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	path := filepath.Join(root, "credentials.sock")
	t.Setenv("PORTAL_CREDENTIAL_SOCKET", path)
	b := NewBroker(s)
	if err = b.Listen(ctx); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil || info.Mode().Perm() != 0600 {
		t.Fatal("credential socket permissions")
	}
	client := &http.Client{Timeout: 3 * time.Second, Transport: &http.Transport{DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "unix", path)
	}}}
	t.Cleanup(func() { client.CloseIdleConnections() })
	c, _ := s.connection(ctx, owner, false)
	ticket, err := b.Issue(c, op.ID, "")
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < 2; index++ {
		req, _ := http.NewRequestWithContext(ctx, "POST", "http://portal/credentials", nil)
		req.Header.Set("X-Portal-Ticket", ticket)
		response, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		var value Credentials
		if index == 0 {
			if response.StatusCode != 200 || json.NewDecoder(response.Body).Decode(&value) != nil || value.Username != "socket-teacher" {
				t.Fatal("IPC credential handoff failed")
			}
		} else if response.StatusCode != 401 {
			t.Fatal("IPC ticket reused")
		}
		response.Body.Close()
	}
}
