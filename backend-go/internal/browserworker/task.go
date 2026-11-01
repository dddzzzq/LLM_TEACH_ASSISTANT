package browserworker

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	pw "github.com/mxschmitt/playwright-go"
)

type reference struct {
	page pw.Page
	loc  pw.Locator
	item object
}
type navigation struct {
	page    pw.Page
	url     string
	pages   []pw.Page
	frames  map[pw.Frame]string
	started time.Time
}
type frameRecord struct {
	page  pw.Page
	url   string
	epoch float64
	at    time.Time
}
type Task struct {
	manager                              *Manager
	id                                   string
	mu                                   sync.Mutex
	state                                object
	storageErr                           error
	lastErr                              error
	op                                   sync.Mutex
	ctx                                  context.Context
	cancel                               context.CancelFunc
	activity                             atomic.Int64
	starting                             atomic.Bool
	startupPending                       atomic.Bool
	startupMu                            sync.Mutex
	startupCancel                        context.CancelFunc
	startupGeneration                    uint64
	reading                              atomic.Bool
	context                              pw.BrowserContext
	browser                              pw.Browser
	page                                 pw.Page
	profileLock                          *os.File
	ticket                               string
	auth                                 *authentication
	verifiedSubject                      string
	refs                                 map[string]reference
	view                                 object
	courseConfirmed, assignmentConfirmed bool
	approved                             map[string]object
	observation                          object
	revision                             int
	recent                               []any
	marker                               string
	stagnant                             int
	detailPage                           pw.Page
	detailURL                            string
	detailNav                            *navigation
	detailSwitchStreak                   int
	detailSource                         pw.Page
	detailSourceURL                      string
	detailListFrames                     map[pw.Frame]string
	manualPage                           pw.Page
	catalog                              *collection
	frameRevision                        int
	frameHistory                         map[int]frameRecord
	pointerOwner                         string
	pointerPage                          pw.Page
	classLocators                        map[string]pw.Locator
	exportLocators                       map[string]exportLocator
	downloadBusy                         bool
	downloadCDP                          pw.CDPSession
	downloadMu                           sync.Mutex
	downloadBytes                        map[string]float64
	downloadURLs                         map[string]string
	frames                               *frameProducer
}

func newTask(m *Manager, state object) *Task {
	ctx, cancel := context.WithCancel(m.ctx)
	t := &Task{manager: m, id: str(state["job_id"]), state: state, ctx: ctx, cancel: cancel, refs: map[string]reference{}, approved: map[string]object{}, frameHistory: map[int]frameRecord{}, classLocators: map[string]pw.Locator{}, exportLocators: map[string]exportLocator{}, downloadBytes: map[string]float64{}, downloadURLs: map[string]string{}}
	t.activity.Store(time.Now().Unix())
	t.starting.Store(true)
	t.startupPending.Store(true)
	return t
}
func (t *Task) snapshot() object { t.mu.Lock(); defer t.mu.Unlock(); return clone(t.state) }
func (t *Task) update(fields object) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	for k, v := range clone(fields) {
		t.state[k] = v
	}
	t.state["updated_at"] = now()
	t.state["state_version"] = num(t.state["state_version"]) + 1
	t.storageErr = t.manager.persist(t.state)
	return t.storageErr
}
func (t *Task) event(message string) {
	s := t.snapshot()
	events := array(s["events"])
	events = append(events, object{"time": now(), "message": message})
	if len(events) > 60 {
		events = events[len(events)-60:]
	}
	_ = t.update(object{"events": events, "message": message})
}
func (t *Task) automatic() bool {
	s := t.snapshot()
	t.mu.Lock()
	bad := t.storageErr != nil
	t.mu.Unlock()
	return !bad && !t.starting.Load() && t.context != nil && str(s["status"]) == "RUNNING" && str(s["control_mode"]) == "AUTO" && t.ctx.Err() == nil
}
func (t *Task) human(reason, message string) {
	// Pause revokes ownership before waiting for an in-flight browser call.
	// Its late failure must not replace the teacher's takeover reason/message.
	t.mu.Lock()
	if terminal(str(t.state["status"])) || t.ctx != nil && t.ctx.Err() != nil || str(t.state["control_mode"]) == "HUMAN" && str(t.state["waiting_reason"]) == "USER_TAKEOVER" {
		t.mu.Unlock()
		return
	}
	events := append(array(t.state["events"]), object{"message": message, "time": now()})
	if len(events) > 60 {
		events = events[len(events)-60:]
	}
	for key, value := range (object{"status": "WAITING_USER", "control_mode": "HUMAN", "waiting_reason": reason, "control_epoch": num(t.state["control_epoch"]) + 1, "message": message, "events": events, "updated_at": now(), "state_version": num(t.state["state_version"]) + 1}) {
		t.state[key] = value
	}
	t.storageErr = t.manager.persist(t.state)
	t.mu.Unlock()
	t.observation = nil
	t.activity.Store(time.Now().Unix())
}
func (t *Task) failure(err error, operation string) {
	t.lastErr = err
	report := publicError(err)
	report["operation"] = operation
	report["stage"] = t.snapshot()["stage"]
	_ = t.update(object{"last_error": report, "error_code": report["code"]})
	t.event(str(report["message"]))
}
func (t *Task) run() {
	t.op.Lock()
	ctx, err := t.openContext()
	if err == nil {
		t.context = ctx
		ctx.SetDefaultTimeout(8000)
		if hook := t.manager.cfg.ConfigureContext; hook != nil {
			err = hook(ctx)
		}
	}
	if err == nil {
		err = ctx.Route("**/*", func(route pw.Route) {
			request := route.Request()
			allowed := allowedURL(request.URL()) || strings.HasPrefix(request.URL(), "blob:") && resourceURL(strings.TrimPrefix(request.URL(), "blob:"))
			if request.IsNavigationRequest() && !allowed {
				_ = route.Abort()
				s := t.snapshot()
				_ = t.update(object{"status": "WAITING_USER", "control_mode": "HUMAN", "waiting_reason": "DOMAIN", "control_epoch": num(s["control_epoch"]) + 1, "message": "目标页面不在已配置的教学平台域名范围内"})
				return
			}
			_ = route.Fallback()
		})
	}
	if err == nil {
		err = ctx.ExposeBinding("__rpa_manual_export", func(source *pw.BindingSource, _ ...any) any {
			v, e := t.manualExportSubmitted(source.Page)
			if e != nil {
				return false
			}
			return v
		})
	}
	if err == nil {
		err = ctx.AddInitScript(pw.Script{Content: pw.String(script("manual_export_listener"))})
	}
	if err == nil {
		pages := ctx.Pages()
		if len(pages) > 0 {
			t.page = pages[0]
		} else {
			t.page, err = ctx.NewPage()
		}
	}
	if err == nil && len(obj(t.snapshot()["portal_session"])) > 0 {
		var creds object
		creds, err = t.manager.takeCredentials(t.ticket)
		t.ticket = ""
		if err == nil {
			t.auth = &authentication{task: t, credentials: creds, trust: flag(obj(t.snapshot()["portal_session"])["trust_device"]), confirmed: flag(obj(t.snapshot()["portal_session"])["identity_verified"])}
		}
	}
	if err == nil {
		err = t.configureDownloads()
	}
	if err == nil && str(t.snapshot()["task_type"]) == "CATALOG_SYNC" {
		t.catalog = newCollection(t)
	}
	t.op.Unlock()
	if err != nil {
		t.op.Lock()
		t.failure(err, "browser_launch")
		t.human("BROWSER_STARTUP", "浏览器启动未完成，请检查运行环境后重启任务")
		t.starting.Store(false)
		t.op.Unlock()
		return
	}
	t.startup()
	timer := time.NewTicker(500 * time.Millisecond)
	defer timer.Stop()
	for {
		select {
		case <-t.ctx.Done():
			return
		case <-timer.C:
			if !t.op.TryLock() {
				continue
			}
			s := t.snapshot()
			if terminal(str(s["status"])) {
				t.closeBrowser()
				t.op.Unlock()
				return
			}
			if t.automatic() && t.catalog == nil && str(s["task_type"]) != "HOMEWORK_DETAIL" && !flag(s["export_attempted"]) && t.exportArea() != nil {
				if e := t.prepareExport(); e != nil {
					t.failure(e, "prepare_export")
					t.human("SCOPE_SELECTION", str(publicError(e)["message"]))
				}
			}
			if t.automatic() && flag(s["export_attempted"]) && t.assignmentConfirmed && str(s["stage"]) != "SOLVE_CAPTCHA" && !t.downloadBusy {
				if err := t.processExports(); err != nil {
					t.failure(err, "export_download")
					t.human("EXPORT_ASSOCIATION", "导出状态需要核对，请检查下载中心后继续")
				}
			}
			t.op.Unlock()
		}
	}
}
func (t *Task) usable() bool {
	if t.page == nil || t.page.IsClosed() {
		return false
	}
	if t.authenticated() {
		return true
	}
	for _, p := range t.context.Pages() {
		if p.IsClosed() {
			continue
		}
		for _, f := range p.Frames() {
			if !allowedURL(f.URL()) {
				continue
			}
			n, e := f.Locator("input[type=password]:visible,#sliderDiv:visible,input#captcha:visible," + constants["portal_auth.OTP"]).Count()
			if e == nil && n > 0 {
				t.page = p
				return true
			}
		}
	}
	return false
}
func (t *Task) closeBrowser() {
	t.releasePointer("")
	if t.downloadCDP != nil {
		t.downloadCDP.RemoveListeners("Browser.downloadWillBegin")
		t.downloadCDP.RemoveListeners("Browser.downloadProgress")
		_ = t.downloadCDP.Detach()
		t.downloadCDP = nil
	}
	if t.context != nil {
		_ = t.context.Close()
		t.context = nil
	}
	if t.browser != nil {
		_ = t.browser.Close()
		t.browser = nil
	}
	unlockFile(t.profileLock)
	t.profileLock = nil
	if t.auth != nil {
		t.auth.credentials = nil
	}
}
func (t *Task) Stop(status string) {
	t.cancel()
	t.mu.Lock()
	if !terminal(str(t.state["status"])) {
		t.state["status"] = status
		t.state["control_mode"] = "HUMAN"
		t.state["control_epoch"] = num(t.state["control_epoch"]) + 1
		t.state["state_version"] = num(t.state["state_version"]) + 1
		t.storageErr = t.manager.persist(t.state)
	}
	t.mu.Unlock()
	t.op.Lock()
	t.closeBrowser()
	t.op.Unlock()
}
func (t *Task) Pause() error {
	s := t.snapshot()
	if terminal(str(s["status"])) || t.ctx.Err() != nil {
		return errors.New("任务已结束")
	}
	// Revoke ownership before waiting for a browser RPC. In particular, a drag
	// or a slow startup must see the new epoch without holding up the pause API.
	t.activity.Store(time.Now().Unix())
	if e := t.update(object{"status": "WAITING_USER", "control_mode": "HUMAN", "waiting_reason": "USER_TAKEOVER", "control_epoch": num(s["control_epoch"]) + 1}); e != nil {
		return e
	}
	t.cancelStartup()
	cleanup := func() { t.observation = nil; t.releasePointer("") }
	if t.op.TryLock() {
		cleanup()
		t.op.Unlock()
	} else {
		t.manager.wg.Add(1)
		go func() { defer t.manager.wg.Done(); t.op.Lock(); defer t.op.Unlock(); cleanup() }()
	}
	t.event("已暂停自动操作，可在浏览器中手动处理")
	return nil
}
func (t *Task) Resume() error {
	t.op.Lock()
	defer t.op.Unlock()
	s := t.snapshot()
	if terminal(str(s["status"])) || t.context == nil || t.ctx.Err() != nil {
		return errors.New("浏览器会话已结束，请重新启动任务")
	}
	if str(s["control_mode"]) != "HUMAN" {
		return errors.New("当前任务不能交还自动化")
	}
	if t.startupPending.Load() {
		t.releasePointer("")
		if err := t.update(object{"status": "RUNNING", "control_mode": "AUTO", "waiting_reason": "", "control_epoch": num(s["control_epoch"]) + 1}); err != nil {
			return err
		}
		t.starting.Store(true)
		t.manager.wg.Add(1)
		go func() { defer t.manager.wg.Done(); t.startup() }()
		return nil
	}
	if s["target_resolution"] != nil {
		return errors.New("请先处理课程或作业候选确认")
	}
	if t.auth != nil && str(s["waiting_reason"]) == "LOGIN_CODE" {
		view, err := t.auth.observation()
		if err != nil {
			return err
		}
		if flag(view["otp_required"]) && !flag(view["otp_entered"]) && !flag(view["trust_prompt"]) {
			return problem("LOGIN_CODE_FORMAT", "请先输入收到的验证码，再继续登录")
		}
	}
	authenticated := t.authenticated()
	captcha := !authenticated && t.captchaFrame() != nil
	if t.auth != nil && str(s["waiting_reason"]) == "LOGIN_IDENTITY" && authenticated {
		// Returning control is the teacher's explicit confirmation. A concrete
		// mismatched identity still rejects it before enabling automation.
		prior := t.auth.confirmed
		t.auth.confirmed = true
		if _, err := t.auth.identity(); err != nil {
			t.auth.confirmed = prior
			return err
		}
	}
	if !authenticated && !captcha && t.auth == nil {
		return errors.New("请先在浏览器中完成登录并返回教学平台")
	}
	if str(s["waiting_reason"]) == "CLASS_SCOPE" && len(obj(s["requested_scope"])) == 0 {
		return errors.New("请先确认班级范围")
	}
	if str(s["waiting_reason"]) == "EXPORT_ASSOCIATION" && len(array(s["exports"])) == 0 {
		rows, err := t.exportRows()
		if err != nil || len(rows) == 0 {
			return errors.New("请先打开下载中心并确认本次导出记录")
		}
	}
	if str(s["waiting_reason"]) == "COURSE_OR_TERM" && !flag(s["export_attempted"]) {
		view, err := t.inventory()
		if err != nil {
			return err
		}
		terms := map[string]bool{}
		for _, el := range objects(view["elements"]) {
			if value := str(el["selected_option"]); semester.MatchString(value) {
				terms[value] = true
			} else if str(el["semester_kind"]) == "trigger" && semester.MatchString(str(el["name"])) {
				terms[str(el["name"])] = true
			}
		}
		if len(terms) == 1 {
			for term := range terms {
				target := clone(obj(s["target"]))
				if target["term"] != term {
					original := s["requested_target"]
					if original == nil {
						original = clone(target)
					}
					target["term"] = term
					if err := t.update(object{"target": target, "requested_target": original}); err != nil {
						return err
					}
				}
			}
		}
	}
	t.releasePointer("")
	t.observation = nil
	t.marker = ""
	t.stagnant = 0
	t.activity.Store(time.Now().Unix())
	if t.ctx.Err() != nil || terminal(str(t.snapshot()["status"])) || num(t.snapshot()["control_epoch"]) != num(s["control_epoch"]) {
		return errors.New("控制权已变化，请重新观察")
	}
	authStatus, stage := "REQUIRED", str(s["stage"])
	if authenticated {
		authStatus = "AUTHENTICATED"
	}
	attempts := num(s["captcha_attempts"])
	if captcha {
		authStatus, stage = "CAPTCHA_REQUIRED", "SOLVE_CAPTCHA"
	} else {
		attempts = 0
		if t.auth != nil && !authenticated {
			stage = "AUTHENTICATE"
		} else if contains(stage, "AUTHENTICATE", "SOLVE_CAPTCHA") {
			stage = "LOCATE_COURSE"
		}
	}
	if err := t.update(object{"status": "RUNNING", "control_mode": "AUTO", "waiting_reason": "", "control_epoch": num(s["control_epoch"]) + 1, "auto_actions": 0, "navigation_failures": 0, "decision_failures": 0, "auth_status": authStatus, "captcha_attempts": attempts, "stage": stage}); err != nil {
		return err
	}
	t.event("正在重新核验页面并继续自动化")
	return nil
}
func (t *Task) Observe() (object, error) {
	if t.starting.Load() {
		return object{"ready": false, "state": t.snapshot()}, nil
	}
	t.op.Lock()
	defer t.op.Unlock()
	return t.observe()
}
func (t *Task) observe() (object, error) {
	s := t.snapshot()
	if !t.automatic() || t.reading.Load() || flag(s["export_attempted"]) && t.assignmentConfirmed && str(s["stage"]) != "SOLVE_CAPTCHA" {
		return object{"ready": false, "state": s}, nil
	}
	view, err := t.navigationView()
	if err != nil {
		t.failure(err, "observe")
		if !t.handoffLoginError(err) {
			t.human("NAVIGATION", str(publicError(err)["message"]))
		}
		return object{"ready": false, "state": t.snapshot()}, nil
	}
	if view == nil || !t.automatic() {
		return object{"ready": false, "state": t.snapshot()}, nil
	}
	fingerprint := t.fingerprint(view)
	epoch := num(t.snapshot()["control_epoch"])
	if t.observation == nil || str(t.observation["fingerprint"]) != fingerprint || num(t.observation["epoch"]) != epoch {
		t.revision++
		t.observation = object{"id": uuid.NewString(), "revision": t.revision, "epoch": epoch, "fingerprint": fingerprint}
	}
	recent := t.recent
	if len(recent) > 5 {
		recent = recent[len(recent)-5:]
	}
	s = t.snapshot()
	return object{"ready": true, "observation_id": t.observation["id"], "revision": t.revision, "epoch": epoch, "target": s["target"], "stage": s["stage"], "observation": view, "recent_actions": recent, "state": s}, nil
}
func (t *Task) fingerprint(view object) string {
	urls := []string{}
	for _, p := range t.context.Pages() {
		if !p.IsClosed() {
			urls = append(urls, p.URL())
		}
	}
	return hash(object{"view": view, "urls": urls})
}
func (t *Task) Apply(payload object) (object, error) {
	t.op.Lock()
	defer t.op.Unlock()
	action := obj(payload["action"])
	if e := validateAction(action); e != nil {
		return nil, e
	}
	id := str(payload["action_id"])
	if len(id) < 1 || len(id) > 128 {
		return nil, errors.New("动作 ID 无效")
	}
	digest := hash(payload)
	s := t.snapshot()
	receipts := obj(s["action_receipts"])
	prior := obj(receipts[id])
	if len(prior) > 0 {
		if str(prior["digest"]) != digest && str(prior["digest"]) != pythonActionDigest(payload) {
			return nil, errors.New("动作 ID 已用于不同请求")
		}
		return object{"action_id": id, "outcome": prior["outcome"], "state": s}, nil
	}
	current := t.observation
	if !t.automatic() || current == nil || payload["observation_id"] != current["id"] || num(payload["revision"]) != num(current["revision"]) || num(payload["epoch"]) != num(s["control_epoch"]) {
		return nil, errors.New("观察或控制权已变化，请重新观察")
	}
	view, err := t.navigationView()
	if err != nil {
		return nil, err
	}
	if view == nil || !t.automatic() || t.fingerprint(view) != str(current["fingerprint"]) {
		t.observation = nil
		return nil, errors.New("页面已变化，请重新观察")
	}
	if len(receipts) >= 1000 {
		t.human("ACTION_BUDGET", "任务动作预算已到，请核对任务")
		return nil, errors.New("任务动作预算已到")
	}
	receipt := object{"digest": digest, "outcome": "unknown", "tool": action["tool"], "skill_version": bounded(str(payload["skill_version"]), 128)}
	receipts[id] = receipt
	t.observation = nil
	// Persist UNKNOWN before crossing the browser boundary. A repeated ID never
	// replays a possibly completed click, including after a Worker restart.
	if err = t.update(object{"action_receipts": receipts}); err != nil {
		return nil, err
	}
	err = t.execute(action)
	if err == nil {
		receipt["outcome"] = "completed"
		count := num(s["auto_actions"]) + 1
		fields := object{"action_receipts": receipts, "auto_actions": count, "navigation_failures": 0, "decision_failures": 0}
		if t.automatic() {
			fields["last_error"] = nil
			fields["error_code"] = nil
		}
		err = t.update(fields)
		if err != nil {
			return nil, err
		}
		markerAction := clone(action)
		delete(markerAction, "value")
		marker := str(current["fingerprint"]) + hash(markerAction)
		if marker == t.marker {
			t.stagnant++
		} else {
			t.stagnant = 1
		}
		t.marker = marker
		if t.stagnant >= 3 && t.automatic() {
			t.human("NO_PROGRESS", "连续三次相同操作后页面仍没有变化，请协助检查")
		}
		budget := 60
		if t.catalog != nil {
			completed, works := 0, 0
			for _, value := range t.catalog.courses() {
				course := obj(value)
				if flag(course["complete"]) && requirementsComplete(course) {
					completed++
				}
				works += len(obj(course["homeworks"]))
			}
			budget = min(1000, 60+12*completed+min(900, 4*works))
		}
		if count >= float64(budget) && t.automatic() {
			t.human("ACTION_BUDGET", "自动页面操作已达到本轮预算，请检查后交还自动化")
		}
	} else {
		t.failure(err, "navigation:"+str(action["tool"]))
		t.recent = append(t.recent, object{"tool": action["tool"], "outcome": "unknown", "message": publicError(err)["message"]})
		if t.automatic() {
			n := num(s["navigation_failures"]) + 1
			_ = t.update(object{"navigation_failures": n})
			if !t.handoffLoginError(err) && n >= 3 {
				t.human("NAVIGATION", "页面操作连续失败，请核对当前页面")
			}
		}
	}
	if err == nil {
		t.recent = append(t.recent, object{"tool": action["tool"], "element_ref": action["element_ref"]})
	}
	return object{"outcome": receipt["outcome"], "action_id": id, "state": t.snapshot()}, nil
}
func (t *Task) DecisionFailed(p object) object {
	t.op.Lock()
	defer t.op.Unlock()
	s := t.snapshot()
	if t.automatic() && num(p["epoch"]) == num(s["control_epoch"]) {
		t.observation = nil
		n := num(s["decision_failures"]) + 1
		_ = t.update(object{"decision_failures": n})
		if str(p["reason"]) == "AGENT_CONFIGURATION" {
			t.human("AGENT_CONFIGURATION", "导航 Skill 或浏览器协议不兼容，请检查服务版本")
		} else if n >= 3 {
			t.human("AGENT_UNAVAILABLE", "页面决策连续三次失败，请稍后恢复或手动操作")
		}
	}
	return t.snapshot()
}
func (t *Task) portalResult() object {
	s := t.snapshot()
	status := str(s["status"])
	if status == "WAITING_USER" {
		status = "WAITING_AUTH"
		if str(s["waiting_reason"]) == "LOGIN_CODE" {
			status = "WAITING_OTP"
		}
	}
	out := object{"status": status, "stage": s["stage"], "message": s["message"], "error_code": s["error_code"], "state_version": s["state_version"]}
	if t.verifiedSubject != "" {
		out["verified_subject"] = t.verifiedSubject
	}
	for _, k := range []string{"portal_catalog", "homework_detail"} {
		if v, ok := s[k]; ok {
			key := k
			if k == "portal_catalog" {
				key = "catalog"
			}
			out[key] = v
		}
	}
	return out
}
func pageIndex(ctx pw.BrowserContext, page pw.Page) string {
	if ctx != nil {
		for i, p := range ctx.Pages() {
			if p == page {
				return strconv.Itoa(i)
			}
		}
	}
	return ""
}
func count(l pw.Locator) int    { n, _ := l.Count(); return n }
func visible(l pw.Locator) bool { v, e := l.IsVisible(); return e == nil && v }
func enabled(l pw.Locator) bool { v, e := l.IsEnabled(); return e == nil && v }
func (t *Task) pageByID(id string) (pw.Page, error) {
	n, e := strconv.Atoi(id)
	if e != nil || n < 0 || t.context == nil || n >= len(t.context.Pages()) {
		return nil, fmt.Errorf("页面已变化")
	}
	p := t.context.Pages()[n]
	if p.IsClosed() {
		return nil, errors.New("页面已关闭")
	}
	return p, nil
}
