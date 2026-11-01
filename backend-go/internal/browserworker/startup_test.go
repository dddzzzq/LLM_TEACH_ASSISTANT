package browserworker

import (
	"errors"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	pw "github.com/mxschmitt/playwright-go"
)

func fastStartup() *startupPolicy {
	return &startupPolicy{time.Second, 5 * time.Second, []time.Duration{30 * time.Millisecond}, []time.Duration{30 * time.Millisecond}, 10 * time.Millisecond}
}

func TestStartupWaitsForDelayedLoginAndRetriesOnlyTransientFailures(t *testing.T) {
	for _, code := range []int{200, 503, 403} {
		t.Run(strconv.Itoa(code), func(t *testing.T) {
			var portal, login atomic.Int32
			m := testManager(t, func(c pw.BrowserContext) error {
				return c.Route("**/*", func(r pw.Route) {
					body, status := "", 200
					if strings.Contains(r.Request().URL(), "ids.xidian.edu.cn") {
						login.Add(1)
						body = "<script>setTimeout(()=>document.body.innerHTML='<input type=password>',80)</script>"
					} else {
						count := portal.Add(1)
						if code != 200 && count == 1 {
							status = code
						}
						body = "<a href='/course?courseid=99'>公开课程</a><a>个人空间</a><script>setTimeout(()=>{const a=document.createElement('a');a.href='https://ids.xidian.edu.cn/login';a.textContent='登录';document.body.append(a)},80)</script>"
						if status != 200 {
							body = "temporarily unavailable"
						}
					}
					_ = r.Fulfill(pw.RouteFulfillOptions{Status: pw.Int(status), ContentType: pw.String("text/html; charset=utf-8"), Body: body})
				})
			})
			m.cfg.startup = fastStartup()
			task, err := m.Start(uuid.NewString(), object{"target": object{"course_name": "课程", "assignment_name": "作业"}})
			if err != nil {
				t.Fatal(err)
			}
			await(t, 8*time.Second, func() bool { return !task.starting.Load() })
			state := task.snapshot()
			if code == 403 {
				report := obj(state["last_error"])
				if portal.Load() != 1 || login.Load() != 0 || state["waiting_reason"] != "BROWSER_STARTUP" || flag(obj(report["evidence"])["retryable"]) {
					t.Fatalf("client error retried: %+v", state)
				}
			} else {
				expected := int32(1)
				if code == 503 {
					expected = 2
				}
				if portal.Load() != expected || login.Load() != 1 || state["waiting_reason"] != "LOGIN" || num(obj(state["startup_result"])["attempts"]) != float64(expected) {
					t.Fatalf("login was reloaded or not awaited: portal=%d login=%d state=%+v", portal.Load(), login.Load(), state)
				}
			}
		})
	}
}

func TestStartupTimeoutAfterLoginAppearsKeepsExistingPage(t *testing.T) {
	var requests atomic.Int32
	released := make(chan struct{})
	m := testManager(t, func(c pw.BrowserContext) error {
		return c.Route("**/*", func(r pw.Route) {
			if strings.HasSuffix(r.Request().URL(), "/slow.js") {
				<-released
				_ = r.Fulfill(pw.RouteFulfillOptions{Body: ""})
				return
			}
			requests.Add(1)
			_ = r.Fulfill(pw.RouteFulfillOptions{Body: "<input type=password><script src='/slow.js'></script>", ContentType: pw.String("text/html")})
		})
	})
	t.Cleanup(func() { close(released) })
	m.cfg.startup = fastStartup()
	m.cfg.startup.attempt = 200 * time.Millisecond
	task, err := m.Start(uuid.NewString(), object{"target": object{"course_name": "课程", "assignment_name": "作业"}})
	if err != nil {
		t.Fatal(err)
	}
	await(t, 4*time.Second, func() bool { return !task.starting.Load() })
	if task.snapshot()["waiting_reason"] != "LOGIN" || requests.Load() != 1 {
		t.Fatalf("ready login was reloaded: %+v", task.snapshot())
	}
}

func TestStartupExhaustionResumeReusesSameBrowser(t *testing.T) {
	var requests atomic.Int32
	m := testManager(t, func(c pw.BrowserContext) error {
		return c.Route("**/*", func(r pw.Route) {
			requests.Add(1)
			_ = r.Fulfill(pw.RouteFulfillOptions{Status: pw.Int(503), Body: "temporarily unavailable"})
		})
	})
	m.cfg.startup = fastStartup()
	m.cfg.startup.backoff, m.cfg.startup.cooldown = nil, nil
	task, err := m.Start(uuid.NewString(), object{"target": object{"course_name": "课程", "assignment_name": "作业"}})
	if err != nil {
		t.Fatal(err)
	}
	await(t, 4*time.Second, func() bool { return !task.starting.Load() })
	if task.snapshot()["waiting_reason"] != "BROWSER_STARTUP" || requests.Load() != 1 {
		t.Fatal(task.snapshot())
	}
	task.op.Lock()
	page := task.page
	err = page.SetContent("<input type=password>")
	task.op.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if err = task.Resume(); err != nil {
		t.Fatal(err)
	}
	await(t, 4*time.Second, func() bool { return !task.starting.Load() })
	if task.page != page || task.snapshot()["waiting_reason"] != "LOGIN" || requests.Load() != 1 {
		t.Fatalf("resume did not reuse ready browser: %+v", task.snapshot())
	}
}

func TestStartupErrorSanitizationAndRetryability(t *testing.T) {
	for _, tc := range []struct {
		err    error
		status int
		retry  bool
	}{
		{errors.New("net::ERR_CONNECTION_RESET at https://host/?token=private"), 0, true},
		{errors.New("net::ERR_CERT_AUTHORITY_INVALID"), 0, false},
		{errors.New("http"), 429, true},
		{errors.New("http"), 401, false},
		{errors.New("net::ERR_CONNECTION_RESET"), 200, true},
		{pw.ErrTimeout, 0, true},
	} {
		if startupRetryable(tc.err, tc.status) != tc.retry {
			t.Fatal(tc)
		}
	}
}

func TestStartupReusesExistingPortalLoginWithoutReloading(t *testing.T) {
	_, task := controlFixture(t)
	var reloads, logins atomic.Int32
	if err := task.context.Route("https://learning.xidian.edu.cn/**", func(r pw.Route) {
		reloads.Add(1)
		_ = r.Fulfill(pw.RouteFulfillOptions{Body: "unexpected portal reload"})
	}); err != nil {
		t.Fatal(err)
	}
	if err := task.context.Route("https://ids.xidian.edu.cn/**", func(r pw.Route) {
		logins.Add(1)
		_ = r.Fulfill(pw.RouteFulfillOptions{Body: "<input type=password>", ContentType: pw.String("text/html")})
	}); err != nil {
		t.Fatal(err)
	}
	task.op.Lock()
	err := task.page.SetContent("<a class=denglu href='https://ids.xidian.edu.cn/login'>登录</a>")
	task.human("BROWSER_STARTUP", "fixture")
	task.startupPending.Store(true)
	task.op.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if err = task.Resume(); err != nil {
		t.Fatal(err)
	}
	await(t, 4*time.Second, func() bool { return !task.starting.Load() })
	if reloads.Load() != 0 || logins.Load() != 1 || task.snapshot()["waiting_reason"] != "LOGIN" {
		t.Fatal("existing login entry was reloaded", reloads.Load(), logins.Load(), task.snapshot())
	}
}
