package browserworker

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	pw "github.com/mxschmitt/playwright-go"
)

func fixtureBroker(t *testing.T, socket string, username, password string) *atomic.Int32 {
	t.Helper()
	listener, e := net.Listen("unix", socket)
	if e != nil {
		t.Fatal(e)
	}
	calls := &atomic.Int32{}
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(r.Header.Get("X-Portal-Ticket")) != 64 {
			w.WriteHeader(401)
			return
		}
		calls.Add(1)
		_ = json.NewEncoder(w).Encode(object{"username": username, "password": password})
	})}
	go server.Serve(listener)
	t.Cleanup(func() { server.Close() })
	t.Setenv("PORTAL_CREDENTIAL_SOCKET", socket)
	return calls
}

func TestGoPortalAuthenticationCatalogAndQuestionResources(t *testing.T) {
	if os.Getenv("RPA_GO_BROWSER_TESTS") != "1" {
		t.Skip("requires installed Go Playwright Chromium")
	}
	socketDir, e := os.MkdirTemp("", "go-portal-")
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { os.RemoveAll(socketDir) })
	brokerCalls := fixtureBroker(t, filepath.Join(socketDir, "credentials.sock"), "teacher-fixture", "password-canary-not-for-model")
	var codeRequests, trustDecisions atomic.Int32
	m := testManager(t, func(ctx pw.BrowserContext) error {
		return ctx.Route("**/*", func(route pw.Route) {
			u, _ := url.Parse(route.Request().URL())
			cookie := route.Request().Headers()["cookie"]
			body := ""
			switch {
			case u.Hostname() == "ids.xidian.edu.cn":
				body = `<meta charset=utf-8><input id=username name=username autocomplete=username><input id=password type=password><button id=login_submit onclick="document.body.innerHTML=document.querySelector('#otp-template').innerHTML">登录</button><template id=otp-template><h2>企业微信动态验证码</h2><input id=dynamicCode autocomplete=one-time-code><button id=getDynamicCode onclick="fetch('/sent');this.innerText='60秒'">获取验证码</button><button onclick="if(document.querySelector('#dynamicCode').value==='123456')document.body.innerHTML=document.querySelector('#trust-template').innerHTML">登录</button><template id=trust-template><h2>可信浏览器</h2><button onclick="fetch('/trusted');document.cookie='fixture_auth=yes; domain=.xidian.edu.cn; path=/; Secure';location.href='https://learning.xidian.edu.cn/portal?ok=1'">仅本次登录</button><button onclick="throw Error('wrong trust choice')">信任此设备</button></template></template>`
				if u.Path == "/sent" {
					codeRequests.Add(1)
					body = "ok"
				}
				if u.Path == "/trusted" {
					trustDecisions.Add(1)
					body = "ok"
				}
			case u.Path == "/portal" && !strings.Contains(cookie, "fixture_auth=yes"):
				body = `<meta charset=utf-8><a href="/space">个人空间</a><a href="/course?courseid=999">公开课程</a><a class="login-entry" href="https://ids.xidian.edu.cn/login">登录</a>`
			case u.Path == "/portal":
				body = `<meta charset=utf-8><p>个人空间</p><span data-teacher-account=teacher-fixture>teacher-fixture</span><ul><li data-term="2026-2027第一学期" data-class-name="A班"><a title="同名课程" target=_blank href="https://learning.xidian.edu.cn/course?courseid=11&clazzid=22">同名课程</a></li><li data-term="2026-2027第一学期" data-class-name="B班"><a title="同名课程" target=_blank href="https://learning.xidian.edu.cn/course?courseid=12&clazzid=23">同名课程</a></li></ul>`
			case u.Path == "/course":
				body = fmt.Sprintf(`<meta charset=utf-8><h1 class=courseName>同名课程</h1><a title="作业" href="https://learning.xidian.edu.cn/work/list?%s">作业</a>`, u.RawQuery)
			case u.Path == "/question-resource.pdf":
				_ = route.Fulfill(pw.RouteFulfillOptions{Body: "%PDF-1.4 fixture", ContentType: pw.String("application/pdf"), Headers: map[string]string{"Content-Disposition": "attachment; filename=requirements.pdf"}})
				return
			case u.Path == "/work/list":
				body = fmt.Sprintf(`<meta charset=utf-8><h1>作业</h1><ul><li data-workid=33><h3 class=workName><a title="编程作业" target=_blank href="https://learning.xidian.edu.cn/question?%s&workid=33">编程作业</a></h3></li><li data-workid=34><h3 class=workName><a title="实验报告" target=_blank href="https://learning.xidian.edu.cn/question?%s&workid=34">实验报告</a></h3></li></ul>`, u.RawQuery, u.RawQuery)
			case u.Path == "/question":
				title := "编程作业"
				if u.Query().Get("workid") == "34" {
					title = "实验报告"
				}
				body = fmt.Sprintf(`<meta charset=utf-8><h2>%s</h2><main data-homework-detail data-workid="%s" data-question-content><p>原始题目：保留这个句子。</p><img alt="题目图片" width=30 height=30 src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aU1sAAAAASUVORK5CYII="><a href="javascript:;" onclick="this.href='https://learning.xidian.edu.cn/question-resource.pdf'">题目附件.pdf</a><div class=student-answer>STUDENT_ANSWER_MUST_NOT_APPEAR</div><div class=analysis>ANSWER_ANALYSIS_MUST_NOT_APPEAR</div></main>`, title, u.Query().Get("workid"))
			default:
				t.Errorf("unexpected browser fixture request: %s", publicURL(u.String()))
				_ = route.Abort()
				return
			}
			_ = route.Fulfill(pw.RouteFulfillOptions{Body: body, ContentType: pw.String("text/html; charset=utf-8")})
		})
	})
	session := object{"connection_id": uuid.NewString(), "profile_generation": float64(1), "credential_version": float64(1), "trust_device": false, "identity_verified": false, "credential_ticket": strings.Repeat("a", 64)}
	task, e := m.Start(uuid.NewString(), object{"target": object{"course_name": "教务课程目录", "assignment_name": "课程与作业"}, "task_type": "AUTH_BOOTSTRAP", "portal_session": session})
	if e != nil {
		t.Fatal(e)
	}
	await(t, 15*time.Second, func() bool { return !task.starting.Load() })
	deadline := time.Now().Add(60 * time.Second)
	steps := 0
	for time.Now().Before(deadline) {
		s := task.snapshot()
		if s["status"] == "SUCCEEDED" {
			result := obj(s["portal_catalog"])
			courses := objects(result["courses"])
			if !flag(result["details_included"]) || len(courses) != 2 || brokerCalls.Load() != 1 || codeRequests.Load() != 1 || trustDecisions.Load() != 1 {
				t.Fatalf("invalid catalog or repeated authentication: broker=%d codes=%d trust=%d result=%+v", brokerCalls.Load(), codeRequests.Load(), trustDecisions.Load(), result)
			}
			for _, c := range courses {
				works := objects(c["homeworks"])
				if len(works) != 2 {
					t.Fatalf("missing homework: %+v", c)
				}
				for _, w := range works {
					detail := obj(w["detail"])
					if !flag(detail["complete"]) || len(array(detail["sources"])) != 3 {
						task.op.Lock()
						cause := task.lastErr
						task.op.Unlock()
						t.Fatalf("missing question resources: %v detail=%+v", cause, detail)
					}
					for _, source := range objects(detail["sources"]) {
						if str(source["kind"]) != "text" {
							data, e := os.ReadFile(str(source["path"]))
							if e != nil || len(data) == 0 || textDigest(data) != str(source["sha256"]) {
								t.Fatal("invalid original question resource")
							}
						}
					}
				}
			}
			encoded, _ := json.Marshal(s)
			for _, canary := range []string{"password-canary-not-for-model", "123456", "STUDENT_ANSWER_MUST_NOT_APPEAR", "ANSWER_ANALYSIS_MUST_NOT_APPEAR", strings.Repeat("a", 64)} {
				if strings.Contains(string(encoded), canary) {
					t.Fatal("private data leaked to the task journal:", canary)
				}
			}
			return
		}
		if s["status"] == "WAITING_USER" {
			if s["waiting_reason"] != "LOGIN_CODE" {
				task.op.Lock()
				cause := task.lastErr
				task.op.Unlock()
				t.Fatalf("unexpected human wait: %v state=%+v", cause, s)
			}
			frame, e := task.Frame()
			if e != nil {
				t.Fatal(e)
			}
			if e = task.Input(object{"kind": "login_code", "text": "123456", "page_id": frame["page_id"], "revision": frame["revision"], "epoch": frame["epoch"]}, "teacher"); e != nil {
				t.Fatal(e)
			}
			if e = task.Resume(); e != nil {
				t.Fatal(e)
			}
		}
		view, e := task.Observe()
		if e != nil {
			t.Fatal(e)
		}
		if !flag(view["ready"]) {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		o := obj(view["observation"])
		a := object{}
		auth := obj(o["authentication"])
		switch {
		case str(view["stage"]) == "AUTHENTICATE":
			switch {
			case flag(auth["login_form"]) && !flag(auth["credentials_filled"]):
				a["tool"] = "fill_portal_credentials"
			case flag(auth["otp_required"]) && !flag(auth["code_requested"]):
				a["tool"] = "request_login_code"
			case flag(auth["otp_required"]) && flag(auth["otp_entered"]):
				a["tool"] = "submit_login_code"
			case flag(auth["trust_prompt"]):
				a["tool"] = "apply_device_trust"
			}
		case len(obj(o["homework_detail"])) > 0:
			detail := obj(o["homework_detail"])
			if flag(detail["ready"]) {
				a["tool"] = "read_homework_detail"
			} else {
				title := str(obj(view["target"])["assignment_name"])
				for _, item := range objects(o["elements"]) {
					if str(item["name"]) == title && str(item["tag"]) == "a" {
						a = object{"tool": "click", "element_ref": item["ref"]}
						break
					}
				}
			}
		default:
			c := obj(o["catalog"])
			switch str(c["next_step"]) {
			case "FINISH_CATALOG":
				a["tool"] = "finish_catalog"
			case "READ_HOMEWORK_LIST":
				a = object{"tool": "read_homework_list", "course_id": c["current_course_id"]}
			case "SELECT_HOMEWORK_DETAIL":
				a = object{"tool": "select_homework_detail", "homework_id": c["next_homework_id"]}
			case "RETURN_TO_COURSE_LIST":
				a["tool"] = "return_to_course_list"
			case "OPEN_HOMEWORK_LIST":
				for _, item := range objects(o["elements"]) {
					if str(item["name"]) == "作业" && str(item["tag"]) == "a" {
						a = object{"tool": "click", "element_ref": item["ref"]}
						break
					}
				}
			case "VISIT_NEXT_COURSE":
				if !flag(c["courses_complete"]) {
					a["tool"] = "read_course_list"
				} else {
					expected := strings.Split(str(c["next_course_id"]), ":")
					for _, item := range objects(o["elements"]) {
						ids := obj(item["platform_ids"])
						if str(item["tag"]) == "a" && str(ids["course_id"]) == expected[0] && (len(expected) == 1 || str(ids["class_id"]) == expected[1]) {
							a = object{"tool": "click", "element_ref": item["ref"]}
							break
						}
					}
				}
			}
		}
		if len(a) == 0 {
			t.Fatalf("no fixture action for state=%+v observation=%+v", s, o)
		}
		out := applyObserved(t, task, view, a)
		if out["outcome"] != "completed" {
			task.op.Lock()
			cause := task.lastErr
			task.op.Unlock()
			t.Fatalf("action failed: %v state=%+v", cause, out)
		}
		steps++
		if steps > 60 {
			t.Fatal("catalog stopped making progress")
		}
	}
	t.Fatalf("catalog timed out: %+v", task.snapshot())
}

func TestProfileFencesAndWorkerLock(t *testing.T) {
	dir := t.TempDir()
	m := &Manager{cfg: Config{Runtime: dir}, ctx: context.Background()}
	session := object{"connection_id": uuid.NewString(), "profile_generation": float64(2), "credential_version": float64(3)}
	if e := m.fence(session); e != nil {
		t.Fatal(e)
	}
	old := clone(session)
	old["profile_generation"] = float64(1)
	if e := m.fence(old); e == nil {
		t.Fatal("reopened a revoked profile")
	}
	f, e := lockFile(filepath.Join(dir, "rpa-worker.lock"))
	if e != nil {
		t.Fatal(e)
	}
	defer unlockFile(f)
	if second, e := lockFile(filepath.Join(dir, "rpa-worker.lock")); e == nil {
		unlockFile(second)
		t.Fatal("two workers acquired the same runtime")
	}
}

func TestPersistentBrowserCanDownloadQuestionAttachment(t *testing.T) {
	m := testManager(t, nil)
	c, e := m.pw.Chromium.LaunchPersistentContext(t.TempDir(), pw.BrowserTypeLaunchPersistentContextOptions{Channel: pw.String("chromium"), Headless: pw.Bool(true), AcceptDownloads: pw.Bool(true), DownloadsPath: pw.String(t.TempDir()), Args: []string{"--no-sandbox", "--disable-dev-shm-usage", "--no-proxy-server"}})
	if e != nil {
		t.Fatal(e)
	}
	defer c.Close()
	if e = c.Route("**/*", func(r pw.Route) {
		if strings.HasSuffix(r.Request().URL(), "/download") {
			_ = r.Fulfill(pw.RouteFulfillOptions{Body: "question content", ContentType: pw.String("text/plain"), Headers: map[string]string{"Content-Disposition": "attachment; filename=question.txt"}})
		} else {
			_ = r.Fulfill(pw.RouteFulfillOptions{Body: `<a href=/download>attachment</a>`, ContentType: pw.String("text/html")})
		}
	}); e != nil {
		t.Fatal(e)
	}
	if e = c.SetOffline(true); e != nil {
		t.Fatal(e)
	}
	p := c.Pages()[0]
	if _, e = p.Goto("https://learning.xidian.edu.cn/question"); e != nil {
		t.Fatal(e)
	}
	d, e := p.ExpectDownload(func() error { return p.Locator("a").Click() })
	if e != nil {
		t.Fatal(e)
	}
	path, e := d.Path()
	if e != nil {
		t.Fatal(e)
	}
	data, e := os.ReadFile(path)
	if e != nil || string(data) != "question content" {
		t.Fatal("attachment content mismatch", e)
	}
}
