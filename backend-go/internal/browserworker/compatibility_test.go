package browserworker

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	pw "github.com/mxschmitt/playwright-go"
)

func requireIssue(t *testing.T, err error, code string) {
	t.Helper()
	var issue *issue
	if !errors.As(err, &issue) || issue.Code != code {
		t.Fatalf("expected %s, got %v", code, err)
	}
}

func TestAuthenticationKeepsVerificationErrorsAndChangedCodes(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	if _, err := task.page.Goto("https://ids.xidian.edu.cn/login"); err != nil {
		t.Fatal(err)
	}
	auth := &authentication{task: task, credentials: object{"username": "fixture"}, filled: true}
	task.auth = auth
	if err := task.page.SetContent("<input type=password><p>Invalid captcha</p>"); err != nil {
		t.Fatal(err)
	}
	_, err := auth.observation()
	requireIssue(t, err, "LOGIN_VERIFICATION_FAILED")
	if err = task.page.SetContent("<input id=dynamicCode value=123456><p>Invalid verification code</p>"); err != nil {
		t.Fatal(err)
	}
	_, err = auth.observation()
	requireIssue(t, err, "LOGIN_VERIFICATION_FAILED")
	auth.submittedDigest = sha256.Sum256([]byte("654321"))
	view, err := auth.observation()
	if err != nil || !flag(view["otp_entered"]) {
		t.Fatal("replacement code was rejected", err)
	}
	if err = task.page.SetContent("<input id=dynamicCode value=123456><button onclick=\"window.submits=(window.submits||0)+1;setTimeout(()=>document.body.insertAdjacentHTML('beforeend','<p>Invalid verification code</p>'),50)\">登录</button>"); err != nil {
		t.Fatal(err)
	}
	requireIssue(t, auth.submitCode(), "LOGIN_VERIFICATION_FAILED")
	requireIssue(t, auth.submitCode(), "LOGIN_CODE_PENDING")
	count, _ := task.page.Evaluate("window.submits")
	if num(count) != 1 {
		t.Fatal("code submitted twice", count)
	}
	encoded, _ := json.Marshal(task.snapshot())
	if strings.Contains(string(encoded), "123456") {
		t.Fatal("code leaked to task state")
	}
}

func TestDelayedDynamicCodeButtonUsesOriginalDeliveryAction(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	if _, err := task.page.Goto("https://ids.xidian.edu.cn/login"); err != nil {
		t.Fatal(err)
	}
	body := "<h2>Dynamic code (WeChat Work)</h2><input id=dynamicCode><button onclick='window.wrong=true'>Send code</button><script>setTimeout(()=>{const b=document.createElement('button');b.id='getDynamicCode';b.innerText='获取动态码';b.onclick=()=>{window.sent=(window.sent||0)+1;b.innerText='60秒'};document.body.append(b)},150)</script>"
	if err := task.page.SetContent(body); err != nil {
		t.Fatal(err)
	}
	auth := &authentication{task: task}
	if err := auth.requestCode(); err != nil {
		t.Fatal(err)
	}
	if err := auth.requestCode(); err != nil {
		t.Fatal(err)
	}
	values, _ := task.page.Evaluate("({sent:window.sent,wrong:!!window.wrong})")
	if num(obj(values)["sent"]) != 1 || flag(obj(values)["wrong"]) {
		t.Fatal("wrong or repeated delivery action", values)
	}
	view, err := auth.observation()
	if err != nil || !flag(view["code_requested"]) || view["code_method"] != "wechat_work" {
		t.Fatal(view, err)
	}
}

func TestResumePreservesPythonAuthenticationAndSemesterChecks(t *testing.T) {
	for _, name := range []string{"empty-code", "unfinished-login", "identity-confirm", "identity-mismatch", "semester", "already-auto"} {
		t.Run(name, func(t *testing.T) {
			_, task := controlFixture(t)
			if name == "already-auto" {
				if task.Resume() == nil {
					t.Fatal("accepted resume while automatic")
				}
				return
			}
			task.op.Lock()
			task.human("LOGIN", "fixture")
			switch name {
			case "empty-code":
				_, _ = task.page.Goto("https://ids.xidian.edu.cn/login")
				_ = task.page.SetContent("<input id=dynamicCode>")
				task.auth = &authentication{task: task}
				_ = task.update(object{"waiting_reason": "LOGIN_CODE"})
			case "unfinished-login":
				_, _ = task.page.Goto("https://ids.xidian.edu.cn/login")
				_ = task.page.SetContent("<input type=password>")
			case "identity-confirm", "identity-mismatch":
				_, _ = task.page.Goto("https://mooc1.chaoxing.com/base")
				body := "<div cname=课程>课程</div>"
				if name == "identity-mismatch" {
					body += "<span data-teacher-account=other-account></span>"
				}
				_ = task.page.SetContent(body)
				task.auth = &authentication{task: task, credentials: object{"username": "fixture"}}
				_ = task.update(object{"waiting_reason": "LOGIN_IDENTITY"})
			case "semester":
				_ = task.page.SetContent(fixturePortalSession + "<select aria-label='学期'><option>2025-2026第一学期</option><option selected>2026-2027第一学期</option></select>")
				_ = task.update(object{"waiting_reason": "COURSE_OR_TERM"})
			}
			task.op.Unlock()
			err := task.Resume()
			switch name {
			case "empty-code":
				requireIssue(t, err, "LOGIN_CODE_FORMAT")
			case "unfinished-login":
				if err == nil {
					t.Fatal("accepted unfinished login")
				}
			case "identity-mismatch":
				requireIssue(t, err, "IDENTITY_MISMATCH")
				if task.auth.confirmed {
					t.Fatal("mismatched identity confirmed")
				}
			case "identity-confirm":
				if err != nil || !task.auth.confirmed || task.snapshot()["control_mode"] != "AUTO" {
					t.Fatal("teacher confirmation not retained", err)
				}
			case "semester":
				if err != nil || obj(task.snapshot()["target"])["term"] != "2026-2027第一学期" || task.snapshot()["requested_target"] == nil {
					t.Fatal("manual semester selection not retained", task.snapshot(), err)
				}
			}
		})
	}
}

func TestReconciliationRequiresHeadingsOutsideListsAndBoundExportEntry(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	for _, tc := range []struct {
		html         string
		course, work bool
	}{
		{"<ul><li><h2>课程</h2><h3>作业</h3></li></ul><a>导出作业附件</a>", false, false},
		{"<h1>课程</h1><ul><li><h3>作业</h3></li></ul><a>导出作业附件</a>", true, false},
		{"<h1>课程</h1><h2>作业</h2>", true, false},
		{"<h1>课程</h1><h2>作业</h2><a>导出作业附件</a>", true, true},
	} {
		task.courseConfirmed, task.assignmentConfirmed = false, false
		if err := task.page.SetContent(tc.html); err != nil {
			t.Fatal(err)
		}
		task.reconcile()
		if task.courseConfirmed != tc.course || task.assignmentConfirmed != tc.work {
			t.Fatalf("list or unrelated heading confirmed export: %+v", tc)
		}
	}
}

func TestExportCheckpointDoesNotBlockTargetNavigationAfterRestart(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	_ = task.update(object{"export_attempted": true, "export_started_at": now(), "exports": []any{}})
	task.courseConfirmed, task.assignmentConfirmed = false, false
	task.op.Unlock()
	view, err := task.Observe()
	if err != nil || !flag(view["ready"]) {
		t.Fatal("restart checkpoint blocked page navigation", view, err)
	}
}

func TestCatalogPrematureFinishKeepsReadingAndRestoresSameURLList(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	if err := task.page.SetContent(fixturePortalSession + "<div data-courseid=11 cname=课程><a href='/course?courseid=11'>课程</a></div><button>下一页</button>"); err != nil {
		t.Fatal(err)
	}
	c := newCollection(task)
	task.catalog = c
	if err := c.readCourses(); err != nil {
		t.Fatal(err)
	}
	if flag(c.progress["courses_complete"]) {
		t.Fatal("next page was ignored")
	}
	c.pageCourses[task.page] = "11"
	if err := c.finish(); err != nil || task.snapshot()["status"] != "RUNNING" {
		t.Fatal("premature finish became a failure", err)
	}
	delete(c.pageCourses, task.page)
	if err := c.finish(); err != nil || task.snapshot()["status"] != "RUNNING" {
		t.Fatal("incomplete list coverage became a failure", err)
	}
}

func TestReadyExportDownloadsBeforeMissingAssociatedRow(t *testing.T) {
	data := bytes.Buffer{}
	archive := zip.NewWriter(&data)
	entry, _ := archive.Create("answer.txt")
	_, _ = entry.Write([]byte("fixture"))
	_ = archive.Close()
	m := testManager(t, func(c pw.BrowserContext) error {
		return c.Route("**/*", func(r pw.Route) {
			if strings.HasSuffix(r.Request().URL(), "/download") {
				_ = r.Fulfill(pw.RouteFulfillOptions{Body: data.Bytes(), ContentType: pw.String("application/zip"), Headers: map[string]string{"Content-Disposition": "attachment; filename=answer.zip"}})
			} else {
				_ = r.Fulfill(pw.RouteFulfillOptions{Body: "<meta charset=utf-8>" + fixturePortalSession + "<div id=downloadcenter><table><tbody><tr data-id=ready><td>作业 班级B 导出成功 <a class=download_ic href=/download>下载</a></td></tr></tbody></table></div>", ContentType: pw.String("text/html; charset=utf-8")})
			}
		})
	})
	task, err := m.Start(uuid.NewString(), object{"target": object{"course_name": "课程", "assignment_name": "作业"}})
	if err != nil {
		t.Fatal(err)
	}
	await(t, 5*time.Second, func() bool { return !task.starting.Load() })
	if err = task.Resume(); err != nil {
		t.Fatal(err)
	}
	task.op.Lock()
	rows, err := task.exportRows()
	if err != nil || len(rows) != 1 {
		task.op.Unlock()
		t.Fatal(rows, err)
	}
	task.assignmentConfirmed, task.courseConfirmed = true, true
	_ = task.update(object{"export_attempted": true, "export_started_at": now(), "exports": []any{object{"ref": "missing", "text": "作业 班级A 导出成功", "status": "READY"}, rows[0]}})
	err = task.processExports()
	task.op.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	await(t, 8*time.Second, func() bool { return len(objects(task.snapshot()["files"])) == 1 })
	if str(objects(task.snapshot()["files"])[0]["status"]) != "VERIFIED" {
		t.Fatal(task.snapshot())
	}
}

type capturePage struct {
	pw.Page
	entered, release chan struct{}
}

func (p *capturePage) URL() string            { return "https://learning.xidian.edu.cn/portal" }
func (p *capturePage) IsClosed() bool         { return false }
func (p *capturePage) Title() (string, error) { return "fixture", nil }
func (p *capturePage) Screenshot(...pw.PageScreenshotOptions) ([]byte, error) {
	close(p.entered)
	<-p.release
	return []byte("jpeg"), nil
}

type captureContext struct {
	pw.BrowserContext
	page pw.Page
}

func (c *captureContext) Pages() []pw.Page { return []pw.Page{c.page} }

func TestSlowCaptureDoesNotHoldOperationLockOrPublishAcrossEpoch(t *testing.T) {
	m := &Manager{cfg: Config{Runtime: t.TempDir()}, ctx: context.Background()}
	task := newTask(m, object{"job_id": uuid.NewString(), "status": "WAITING_USER", "control_mode": "HUMAN", "control_epoch": float64(1)})
	defer task.cancel()
	page := &capturePage{entered: make(chan struct{}), release: make(chan struct{})}
	task.page, task.context = page, &captureContext{page: page}
	finished := make(chan error, 1)
	go func() { _, err := task.Frame(); finished <- err }()
	<-page.entered
	if !task.op.TryLock() {
		close(page.release)
		t.Fatal("screenshot blocked browser input")
	}
	_ = task.update(object{"control_epoch": float64(2)})
	task.op.Unlock()
	close(page.release)
	if err := <-finished; err == nil || task.frameRevision != 0 {
		t.Fatal("old capture published after control changed")
	}
}

func TestLoginErrorsKeepFrontendCompatibleWaitingReasons(t *testing.T) {
	for code, reason := range map[string]string{"LOGIN_VERIFICATION_FAILED": "LOGIN_CODE", "LOGIN_CODE_FORMAT": "LOGIN_CODE", "LOGIN_CODE_PENDING": "LOGIN_CODE_RESULT", "LOGIN_CODE_SEND_DISABLED": "LOGIN_CODE_REQUEST", "DEVICE_TRUST_UNAVAILABLE": "LOGIN_CODE_REQUEST"} {
		m := &Manager{cfg: Config{Runtime: t.TempDir()}, ctx: context.Background()}
		task := newTask(m, object{"job_id": uuid.NewString(), "status": "RUNNING", "control_mode": "AUTO", "control_epoch": float64(1)})
		if !task.handoffLoginError(problem(code, "fixture")) || task.snapshot()["waiting_reason"] != reason {
			t.Fatal(code, task.snapshot())
		}
		if reason == "LOGIN_CODE" && task.portalResult()["status"] != "WAITING_OTP" {
			t.Fatal("frontend cannot present OTP entry")
		}
		task.cancel()
	}
}

func TestCatalogURLBoundaryMatchesPython(t *testing.T) {
	for _, value := range []string{"https://ids.xidian.edu.cn/course?courseid=1", "https://www.xidian.edu.cn/course?courseid=1", "https://mooc1.chaoxing.com/course?courseid=1&token=secret", "https://mooc1.chaoxing.com:443/course?courseid=1"} {
		if cleanCatalogURL(value) != "" {
			t.Fatal("unsafe catalog URL accepted", value)
		}
	}
	if cleanCatalogURL("https://mooc1.chaoxing.com/course?courseid=1&clazzid=2") == "" {
		t.Fatal("routing URL rejected")
	}
}

func TestPythonActionReceiptSurvivesGoWorkerMigration(t *testing.T) {
	payload := object{"action_id": "legacy", "epoch": float64(1), "revision": float64(2), "observation_id": "old", "skill_version": "版本<>&", "action": object{"tool": "scroll", "delta_y": float64(100)}}
	const digest = "04d11b02e5fe5a856970a65f6621b32a383c8ee5cc9db1ddc5543be132c584c1"
	if pythonActionDigest(payload) != digest {
		t.Fatal("Python digest encoding changed")
	}
	m := &Manager{cfg: Config{Runtime: t.TempDir()}, ctx: context.Background()}
	task := newTask(m, object{"job_id": uuid.NewString(), "status": "INTERRUPTED", "action_receipts": object{"legacy": object{"digest": digest, "outcome": "unknown"}}})
	defer task.cancel()
	result, err := task.Apply(payload)
	if err != nil || result["outcome"] != "unknown" {
		t.Fatal("old durable receipt was not reconciled", result, err)
	}
	obj(payload["action"])["delta_y"] = float64(101)
	if _, err = task.Apply(payload); err == nil {
		t.Fatal("same action ID accepted a different command")
	}
}

func TestCaptchaStageCannotUseGenericActions(t *testing.T) {
	m := &Manager{cfg: Config{Runtime: t.TempDir()}, ctx: context.Background()}
	task := newTask(m, object{"job_id": uuid.NewString(), "status": "RUNNING", "control_mode": "AUTO", "stage": "SOLVE_CAPTCHA"})
	defer task.cancel()
	for _, action := range []object{{"tool": "click", "element_ref": "x"}, {"tool": "fill", "element_ref": "x", "value": "x"}, {"tool": "scroll", "delta_y": float64(10)}, {"tool": "switch_page", "page_id": "0"}} {
		if task.execute(action) == nil {
			t.Fatal("generic action bypassed captcha stage", action)
		}
	}
}

func TestSelectedScopeSurvivesOptionRemovalAndRejectsAmbiguity(t *testing.T) {
	request := object{"mode": "selected", "class_ids": []any{"old-2"}, "option_names": object{"old-2": " 班级 B "}}
	options := []any{object{"id": "new-0", "name": "班级 A", "all": false}, object{"id": "new-1", "name": "班级 B", "all": false}}
	resolved := reconcileScope(request, options)
	if resolved == nil || str(array(resolved["class_ids"])[0]) != "new-1" {
		t.Fatal("named scope was lost", resolved)
	}
	options = append(options, object{"id": "duplicate", "name": "班级 B", "all": false})
	if reconcileScope(request, options) != nil {
		t.Fatal("ambiguous named scope was accepted")
	}
}

func TestLateActionFailureKeepsTeacherTakeover(t *testing.T) {
	_, task := controlFixture(t)
	if err := task.Pause(); err != nil {
		t.Fatal(err)
	}
	before := task.snapshot()
	task.op.Lock()
	task.human("NAVIGATION", "old in-flight action failed")
	task.op.Unlock()
	state := task.snapshot()
	if state["waiting_reason"] != "USER_TAKEOVER" || state["message"] != before["message"] || state["control_epoch"] != before["control_epoch"] {
		t.Fatal("late failure replaced takeover", state)
	}
}

func TestRestartKeepsNamedScopeWhenAnotherOptionIsCurrentClass(t *testing.T) {
	m := testManager(t, func(c pw.BrowserContext) error {
		return c.Route("**/*", func(r pw.Route) {
			_ = r.Fulfill(pw.RouteFulfillOptions{Body: fixturePortalSession, ContentType: pw.String("text/html")})
		})
	})
	for _, selected := range []string{"named", "current"} {
		checkpoint := object{"status": "INTERRUPTED", "captcha_attempts": 3, "last_captcha_result": object{"status": "failed"}, "requested_scope": object{"mode": "selected", "class_ids": []any{selected}, "option_names": object{"named": "班级A", "current": "本班级"}}}
		task, err := m.Start(uuid.NewString(), object{"restart": true, "checkpoint": checkpoint, "target": object{"course_name": "课程", "assignment_name": "作业"}})
		if err != nil {
			t.Fatal(err)
		}
		state := task.snapshot()
		if (state["requested_scope"] != nil) != (selected == "named") {
			t.Fatal("restart changed explicit class intent", selected, state)
		}
		if num(state["captcha_attempts"]) != 0 || state["last_captcha_result"] != nil {
			t.Fatal("new browser inherited the old login's challenge failures", state)
		}
	}
}

func TestUnsubmittedSettingsSwitchToSelectedClassBeforeExport(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	body := fixturePortalSession + "<meta charset=utf-8><select name=clazzid><option value=0 selected>全部班级</option><option value=22>班级A</option><option value=23>班级B</option></select>" +
		"<a onclick=\"window.reopened=(window.reopened||0)+1;setTimeout(()=>document.querySelector('[role=dialog]').hidden=false,100)\">导出作业附件</a>" +
		"<div role=dialog><fieldset><legend>导出格式</legend><label><input type=radio name=format checked>导出完整答题记录</label><label><input type=radio name=format>导出提交附件</label></fieldset>" +
		"<button onclick=\"window.cancelled=(window.cancelled||0)+1;this.parentElement.hidden=true\">取消</button><button onclick=\"window.exports=(window.exports||0)+1;this.parentElement.hidden=true\">确定</button></div>"
	if err := task.page.SetContent(body); err != nil {
		t.Fatal(err)
	}
	_ = task.update(object{"target": object{"course_name": "课程", "assignment_name": "作业", "portal": object{"class_id": "22", "class_name": "班级A"}}})
	task.courseConfirmed, task.assignmentConfirmed = true, true
	if err := task.prepareExport(); err != nil {
		t.Fatal(err)
	}
	values, _ := task.page.Evaluate("[window.cancelled,window.reopened,window.exports]")
	if hash(values) != hash([]any{float64(1), float64(1), float64(1)}) || currentClass(task.page) != "22" || task.snapshot()["stage"] != "WAIT_EXPORT" {
		t.Fatal(values, task.snapshot())
	}
}
