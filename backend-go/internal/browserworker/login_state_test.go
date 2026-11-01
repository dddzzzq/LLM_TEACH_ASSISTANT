package browserworker

import (
	"testing"

	pw "github.com/mxschmitt/playwright-go"
)

func TestPortalLoginStateRequiresSessionEvidence(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	public := `<meta charset=utf-8><a href=/space>个人空间</a><a href="/course?courseid=123">公开课程</a>`
	for _, tc := range []struct {
		name, url, html string
		auth, required  bool
	}{
		{"public-legacy-login", "https://learning.xidian.edu.cn/portal", public + `<a class=denglu href="https://ids.xidian.edu.cn/login">登录</a>`, false, true},
		{"public-semantic-login", "https://learning.xidian.edu.cn/portal", public + `<a class=new-login href="https://ids.xidian.edu.cn/login">登录</a>`, false, true},
		{"public-icon-login", "https://learning.xidian.edu.cn/portal", public + `<button aria-label="登录"></button>`, false, true},
		{"public-before-login-control-loads", "https://learning.xidian.edu.cn/portal", public, false, false},
		{"public-hidden-login", "https://learning.xidian.edu.cn/portal", public + `<a class=denglu hidden>登录</a>`, false, false},
		{"public-empty-identity", "https://learning.xidian.edu.cn/portal", public + `<span data-login-account=""></span>`, false, false},
		{"portal-account", "https://learning.xidian.edu.cn/portal", public + fixturePortalSession, true, false},
		{"portal-logout", "https://learning.xidian.edu.cn/portal", public + `<button>退出登录</button>`, true, false},
		{"login-overrides-stale-identity", "https://learning.xidian.edu.cn/portal", public + fixturePortalSession + `<a href="https://ids.xidian.edu.cn/login">登录</a>`, false, true},
		{"password-overrides-identity", "https://learning.xidian.edu.cn/portal", fixturePortalSession + `<input type=password>`, false, true},
		{"sso-otp", "https://ids.xidian.edu.cn/login", `<input id=dynamicCode autocomplete=one-time-code>`, false, true},
		{"course-list", "https://mooc1.chaoxing.com/base", `<div cname=课程><a href="/course?courseid=123">课程</a></div>`, true, false},
		{"other-campus-page", "https://www.xidian.edu.cn/", public + fixturePortalSession, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := task.page.Goto(tc.url); err != nil {
				t.Fatal(err)
			}
			if err := task.page.SetContent(tc.html); err != nil {
				t.Fatal(err)
			}
			if got := task.authenticated(); got != tc.auth {
				t.Fatalf("authenticated=%v, want %v", got, tc.auth)
			}
			if got := task.loginRequired(); got != tc.required {
				t.Fatalf("loginRequired=%v, want %v", got, tc.required)
			}
		})
	}
}

func TestPublicPortalDoesNotRequestTeacherIdentity(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	if err := task.page.SetContent(`<a href=/space>个人空间</a><a href="/course?courseid=123">公开课程</a><a href="https://ids.xidian.edu.cn/login">登录</a>`); err != nil {
		t.Fatal(err)
	}
	task.auth = &authentication{task: task, credentials: object{"username": "teacher-fixture"}}
	// Even a prior verified subject must not override the visible login entry.
	task.verifiedSubject = "teacher-fixture"
	view, err := task.navigationView()
	if err != nil {
		t.Fatal(err)
	}
	if view == nil || flag(obj(view["authentication"])["authenticated"]) || str(task.snapshot()["waiting_reason"]) == "LOGIN_IDENTITY" {
		t.Fatalf("public portal was sent to identity verification: view=%+v state=%+v", view, task.snapshot())
	}
}

func TestAuthenticatedCourseTabCanCoexistWithPublicPortal(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	if err := task.page.SetContent(`<a href="https://ids.xidian.edu.cn/login">登录</a><a href="/course?courseid=123">公开课程</a>`); err != nil {
		t.Fatal(err)
	}
	private, err := task.context.NewPage()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = private.Goto("https://mooc1.chaoxing.com/base", pw.PageGotoOptions{WaitUntil: pw.WaitUntilStateDomcontentloaded}); err != nil {
		t.Fatal(err)
	}
	if !task.authenticated() || task.loginRequired() || task.page != private {
		t.Fatal("stale public portal interrupted the authenticated course tab")
	}
}
