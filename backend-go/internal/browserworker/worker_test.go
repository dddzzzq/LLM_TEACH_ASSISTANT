package browserworker

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	pw "github.com/mxschmitt/playwright-go"
	"grading-gateway/internal/agent/catalog"
	"grading-gateway/internal/agent/modelclient"
	"grading-gateway/internal/browseragent"
)

const fixturePortalSession = `<span data-login-account="teacher-fixture" hidden></span>`

func testManager(t *testing.T, hook func(pw.BrowserContext) error) *Manager {
	t.Helper()
	if os.Getenv("RPA_GO_BROWSER_TESTS") != "1" {
		t.Skip("set RPA_GO_BROWSER_TESTS=1 with the pinned Go Playwright Chromium installed")
	}
	dir := t.TempDir()
	pythonApp, _ := filepath.Abs("../../../ai_engine_python")
	m, e := New(context.Background(), Config{PythonApp: pythonApp, Runtime: filepath.Join(dir, "runtime"), Downloads: filepath.Join(dir, "downloads"), TokenFile: filepath.Join(dir, "token"), StartURL: "https://learning.xidian.edu.cn/portal?ok=1", MaxTasks: 4, ConfigureContext: func(c pw.BrowserContext) error {
		if hook != nil {
			if e := hook(c); e != nil {
				return e
			}
		}
		return c.SetOffline(true)
	}})
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		if e := m.Close(); e != nil {
			t.Error(e)
		}
	})
	return m
}
func await(t *testing.T, timeout time.Duration, check func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if check() {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatal("timed out waiting for browser fixture")
}
func actionObject(raw json.RawMessage) object { var v object; _ = json.Unmarshal(raw, &v); return v }
func applyObserved(t *testing.T, task *Task, view object, a object) object {
	t.Helper()
	out, e := task.Apply(object{"action_id": view["observation_id"], "observation_id": view["observation_id"], "revision": view["revision"], "epoch": view["epoch"], "skill_version": "fixture", "action": a})
	if e != nil {
		t.Fatal(e)
	}
	return out
}
func toolResponse(name, args string) *modelclient.LLMResponse {
	return &modelclient.LLMResponse{ToolCalls: []modelclient.ToolCall{{ID: uuid.NewString(), Type: "function", Function: modelclient.FunctionCall{Name: name, Arguments: args}}}}
}

func TestGoEinoBrowserDownloadsVerifiedArchive(t *testing.T) {
	for _, variant := range []string{"legacy", "semantic", "custom", "nested", "portal-export", "english-export", "english-export-hover"} {
		t.Run(variant, func(t *testing.T) {
			var submitted atomic.Int32
			archive := bytes.Buffer{}
			bundle := zip.NewWriter(&archive)
			entry, _ := bundle.Create("20260001/report.txt")
			entry.Write([]byte("integration student answer"))
			bundle.Close()
			fixture := variant
			if variant == "english-export-hover" {
				fixture = "english-export"
			}
			html, e := os.ReadFile("testdata/" + fixture + ".html")
			if e != nil {
				t.Fatal(e)
			}
			if variant == "english-export-hover" {
				html = []byte(strings.Replace(string(html), "<button onclick=", `<button onmouseleave="document.querySelector('#menu').hidden=true" onclick=`, 1))
			}
			m := testManager(t, func(ctx pw.BrowserContext) error {
				return ctx.Route("**/*", func(r pw.Route) {
					u, _ := url.Parse(r.Request().URL())
					switch u.Path {
					case "/download":
						_ = r.Fulfill(pw.RouteFulfillOptions{Body: archive.Bytes(), ContentType: pw.String("application/zip"), Headers: map[string]string{"Content-Disposition": "attachment; filename=homework.zip"}})
					case "/export":
						submitted.Add(1)
						_ = r.Fulfill(pw.RouteFulfillOptions{Body: "ok"})
					case "/course":
						_ = r.Fulfill(pw.RouteFulfillOptions{Body: string(html), ContentType: pw.String("text/html; charset=utf-8")})
					default:
						body := `<meta charset=utf-8>` + fixturePortalSession + `<a>个人空间</a><div cname="操作系统"><a target=_blank href="https://learning.xidian.edu.cn/course">操作系统</a></div>`
						if variant == "semantic" {
							body = `<meta charset=utf-8>` + fixturePortalSession + `<nav><div role=link>个人空间</div></nav><section role=list><article role=listitem><h3>操作系统</h3><span>2026秋季</span><button aria-label="进入课程" onclick="window.open('https://learning.xidian.edu.cn/course','_blank')">进入</button></article></section>`
						}
						_ = r.Fulfill(pw.RouteFulfillOptions{Body: body, ContentType: pw.String("text/html; charset=utf-8")})
					}
				})
			})
			task, e := m.Start(uuid.NewString(), object{"target": object{"course_name": "操作系统", "assignment_name": "第一次作业"}})
			if e != nil {
				t.Fatal(e)
			}
			await(t, 10*time.Second, func() bool { return !task.starting.Load() })
			if task.snapshot()["waiting_reason"] == "LOGIN_COMPLETE" {
				if e = task.Resume(); e != nil {
					t.Fatal(e)
				}
			}
			if task.snapshot()["status"] != "RUNNING" {
				task.op.Lock()
				startupErr := task.lastErr
				task.op.Unlock()
				t.Fatalf("startup: %v state=%+v", startupErr, task.snapshot())
			}
			root, _ := filepath.Abs("../../../skills")
			t.Setenv("TEACH_SKILLS_DIR", root)
			skills, e := catalog.Default("portal-browser")
			if e != nil {
				t.Fatal(e)
			}
			contract, _ := assets.ReadFile("assets/actions.json")
			deadline := time.Now().Add(30 * time.Second)
			decisions := 0
			for time.Now().Before(deadline) {
				s := task.snapshot()
				if s["status"] == "SUCCEEDED" {
					files := objects(s["files"])
					if len(files) != 1 || str(files[0]["status"]) != "VERIFIED" || len(str(files[0]["sha256"])) != 64 || submitted.Load() != 1 || decisions < 3 {
						t.Fatalf("invalid completion: submissions=%d decisions=%d state=%+v", submitted.Load(), decisions, s)
					}
					data, e := os.ReadFile(str(files[0]["path"]))
					if e != nil || !bytes.Equal(data, archive.Bytes()) {
						t.Fatal("download artifact differs from fixture")
					}
					return
				}
				if s["status"] != "RUNNING" {
					t.Fatalf("unexpected state: %+v", s)
				}
				view, e := task.Observe()
				if e != nil {
					t.Fatal(e)
				}
				if !flag(view["ready"]) {
					time.Sleep(50 * time.Millisecond)
					continue
				}
				observation := obj(view["observation"])
				a := object{}
				if flag(observation["can_open_export"]) {
					a = object{"tool": "open_export_settings"}
				} else {
					for _, item := range objects(observation["elements"]) {
						name := str(item["name"])
						if (name == "批阅" || name == "查看提交") && itemMatches(item, "第一次作业") {
							a = object{"tool": "click", "element_ref": item["ref"]}
							break
						}
						if name == "操作系统" && flag(item["opens_new_page"]) || name == "进入课程" {
							a = object{"tool": "click", "element_ref": item["ref"]}
						}
					}
				}
				if len(a) == 0 {
					t.Fatalf("no fixture action for observation: %+v", observation)
				}
				round := 0
				input, _ := json.Marshal(object{"target": view["target"], "stage": view["stage"], "observation": observation, "recent_actions": view["recent_actions"]})
				raw, e := browseragent.Decide(t.Context(), input, skills["fetch-homework"], contract, func(context.Context, []map[string]interface{}, []map[string]interface{}) (*modelclient.LLMResponse, error) {
					round++
					if round == 1 {
						return toolResponse("load_skill", `{"skill":"fetch-homework"}`), nil
					}
					args := clone(a)
					name := str(args["tool"])
					delete(args, "tool")
					encoded, _ := json.Marshal(args)
					return toolResponse(name, string(encoded)), nil
				})
				if e != nil {
					t.Fatal(e)
				}
				payload := object{"action_id": view["observation_id"], "observation_id": view["observation_id"], "revision": view["revision"], "epoch": view["epoch"], "skill_version": skills["fetch-homework"].Version, "action": actionObject(raw)}
				out, e := task.Apply(payload)
				if e != nil {
					t.Fatal(e)
				}
				if out["outcome"] != "completed" {
					t.Fatalf("action failed: %+v", out)
				}
				again, e := task.Apply(payload)
				if e != nil || again["outcome"] != "completed" {
					t.Fatal("duplicate action lost its durable receipt")
				}
				decisions++
			}
			t.Fatalf("download timed out: %+v", task.snapshot())
		})
	}
}
func TestActionContractsAndInternalAuth(t *testing.T) {
	for _, raw := range []string{`{"tool":"run_python","code":"x"}`, `{"tool":"click","element_ref":"a","url":"https://example.com"}`, `{"tool":"scroll","delta_y":true}`, `{"tool":"switch_page","page_id":""}`} {
		if e := validateAction(actionObject(json.RawMessage(raw))); e == nil {
			t.Fatal("accepted invalid action", raw)
		}
	}
	m := &Manager{token: "fixture-auth-token", tasks: map[string]*Task{}}
	for _, path := range []string{"/health", "/capabilities", "/tasks/" + uuid.NewString() + "/stream"} {
		r := httptest.NewRequest("GET", path, nil)
		w := httptest.NewRecorder()
		m.Handler().ServeHTTP(w, r)
		if w.Code != 401 {
			t.Fatalf("unprotected endpoint %s: %d", path, w.Code)
		}
	}
	r := httptest.NewRequest("GET", "/health", nil)
	r.Header.Set("X-RPA-Token", m.token)
	w := httptest.NewRecorder()
	m.Handler().ServeHTTP(w, r)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"execution_driver":"go-playwright"`) {
		t.Fatal(w.Body.String())
	}
}
func TestExportAssociationRequiresFreshCompleteScope(t *testing.T) {
	row := func(ref, text string) object {
		return object{"ref": ref, "text": text, "status": "READY", "ambiguous": false}
	}
	scope := object{"mode": "all", "baseline_observed": true, "classes": []any{object{"name": "全部班级", "all": true}}, "expected_classes": []any{object{"name": "A班"}, object{"name": "B班"}}}
	rows := []any{row("a", "第一次作业 A班 导出成功"), row("b", "第一次作业 B班 导出成功")}
	if len(matchExports(rows, map[string]bool{}, "第一次作业", scope)) != 2 {
		t.Fatal("all-class exports did not match")
	}
	for _, variant := range []struct {
		rows     []any
		baseline map[string]bool
		scope    object
	}{{rows[:1], map[string]bool{}, scope}, {rows, map[string]bool{"a": true}, scope}, {rows, map[string]bool{}, object{"mode": "all", "baseline_observed": false}}} {
		if len(matchExports(variant.rows, variant.baseline, "第一次作业", variant.scope)) != 0 {
			t.Fatal("associated an old or incomplete export")
		}
	}
	if rowMatchesAssignment(row("a", "第一次作业补交 全部班级 导出成功"), "第一次作业") {
		t.Fatal("accepted a different homework title")
	}
}
func TestRecoveryRetainsUnknownActionAndExportCheckpoint(t *testing.T) {
	dir := t.TempDir()
	id := uuid.NewString()
	state := object{"job_id": id, "target": object{"course_name": "课程", "assignment_name": "作业"}, "status": "RUNNING", "state_version": float64(7), "export_attempted": true, "action_receipts": object{"a": object{"digest": "same", "outcome": "unknown"}}}
	m := &Manager{cfg: Config{Runtime: dir}, tasks: map[string]*Task{}, ctx: context.Background()}
	if e := m.persist(state); e != nil {
		t.Fatal(e)
	}
	if e := m.recover(); e != nil {
		t.Fatal(e)
	}
	s := m.tasks[id].snapshot()
	if s["status"] != "INTERRUPTED" || !flag(s["export_attempted"]) || num(s["state_version"]) != 8 || str(obj(obj(s["action_receipts"])["a"])["outcome"]) != "unknown" {
		t.Fatalf("lost restart checkpoint: %+v", s)
	}
}
func TestZIPVerificationChecksCRC(t *testing.T) {
	b := bytes.Buffer{}
	w := zip.NewWriter(&b)
	entry, _ := w.CreateHeader(&zip.FileHeader{Name: "answer.txt", Method: zip.Store})
	entry.Write([]byte("student-answer"))
	w.Close()
	path := filepath.Join(t.TempDir(), "archive.zip")
	os.WriteFile(path, b.Bytes(), 0600)
	result, e := verifyZIP(path)
	if e != nil || len(str(result["sha256"])) != 64 {
		t.Fatal(result, e)
	}
	corrupt := append([]byte{}, b.Bytes()...)
	i := bytes.Index(corrupt, []byte("student-answer"))
	if i < 0 {
		t.Fatal("invalid fixture")
	}
	corrupt[i] = 'X'
	os.WriteFile(path, corrupt, 0600)
	if _, e = verifyZIP(path); e == nil {
		t.Fatal("accepted a corrupt ZIP")
	}
}
