package browserworker

import (
	"archive/zip"
	"bytes"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	pw "github.com/mxschmitt/playwright-go"
)

func controlFixture(t *testing.T) (*Manager, *Task) {
	t.Helper()
	m := testManager(t, func(c pw.BrowserContext) error {
		return c.Route("**/*", func(r pw.Route) {
			_ = r.Fulfill(pw.RouteFulfillOptions{ContentType: pw.String("text/html; charset=utf-8"), Body: `<meta charset=utf-8>` + fixturePortalSession + `<style>input{position:absolute;left:20px;top:40px;width:200px;height:30px}</style><a>个人空间</a><div cname=课程><a href=/course>课程</a></div><input aria-label="搜索" id=search><button id=action onclick="window.clicked=(window.clicked||0)+1">目标控件</button><input type=password value=PASSWORD_MUST_NOT_LEAK hidden><input name=otp value=CODE_MUST_NOT_LEAK hidden>`})
		})
	})
	task, e := m.Start(uuid.NewString(), object{"target": object{"course_name": "课程", "assignment_name": "作业"}})
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
		t.Fatalf("fixture startup failed: %+v", task.snapshot())
	}
	return m, task
}

func TestTakeoverDuringStartupKeepsTeacherControl(t *testing.T) {
	entered, released := make(chan struct{}), make(chan struct{})
	var enter, release sync.Once
	m := testManager(t, func(c pw.BrowserContext) error {
		return c.Route("**/*", func(r pw.Route) {
			enter.Do(func() { close(entered) })
			<-released
			_ = r.Fulfill(pw.RouteFulfillOptions{ContentType: pw.String("text/html; charset=utf-8"), Body: `<a>个人空间</a><div cname=课程>课程</div>`})
		})
	})
	t.Cleanup(func() { release.Do(func() { close(released) }) })
	task, e := m.Start(uuid.NewString(), object{"target": object{"course_name": "课程", "assignment_name": "作业"}})
	if e != nil {
		t.Fatal(e)
	}
	select {
	case <-entered:
	case <-time.After(10 * time.Second):
		t.Fatal("browser did not start navigation")
	}
	paused := make(chan error, 1)
	go func() { paused <- task.Pause() }()
	select {
	case e = <-paused:
		if e != nil {
			t.Fatal(e)
		}
	case <-time.After(time.Second):
		t.Fatal("an in-flight navigation blocked teacher takeover")
	}
	release.Do(func() { close(released) })
	await(t, 10*time.Second, func() bool { return !task.starting.Load() })
	s := task.snapshot()
	if s["control_mode"] != "HUMAN" || s["waiting_reason"] != "USER_TAKEOVER" || s["status"] != "WAITING_USER" {
		t.Fatalf("startup replaced teacher control: %+v", s)
	}
}

func TestManualExportImmediateResumeDoesNotSubmitAgain(t *testing.T) {
	var submitted atomic.Int32
	archive := bytes.Buffer{}
	w := zip.NewWriter(&archive)
	f, _ := w.Create("answer.txt")
	f.Write([]byte("answer"))
	w.Close()
	html, e := os.ReadFile("testdata/legacy.html")
	if e != nil {
		t.Fatal(e)
	}
	html = []byte(strings.Replace(string(html), "<a class=confirmDown", `<label class=export-range><input type=checkbox>全部班级</label><a class=confirmDown`, 1))
	m := testManager(t, func(c pw.BrowserContext) error {
		return c.Route("**/*", func(r pw.Route) {
			u, _ := url.Parse(r.Request().URL())
			switch u.Path {
			case "/export":
				submitted.Add(1)
				_ = r.Fulfill(pw.RouteFulfillOptions{Body: "ok"})
			case "/download":
				_ = r.Fulfill(pw.RouteFulfillOptions{Body: archive.Bytes(), ContentType: pw.String("application/zip"), Headers: map[string]string{"Content-Disposition": "attachment; filename=homework.zip"}})
			default:
				_ = r.Fulfill(pw.RouteFulfillOptions{Body: fixturePortalSession + `<a>个人空间</a>` + string(html), ContentType: pw.String("text/html; charset=utf-8")})
			}
		})
	})
	task, e := m.Start(uuid.NewString(), object{"target": object{"course_name": "操作系统", "assignment_name": "第一次作业"}})
	if e != nil {
		t.Fatal(e)
	}
	await(t, 10*time.Second, func() bool { return !task.starting.Load() })
	if e = task.Pause(); e != nil {
		t.Fatal(e)
	}
	task.op.Lock()
	_, e = task.page.Evaluate(`document.querySelector('#work').hidden=true;document.querySelector('#grade').hidden=false;document.querySelector('[role=dialog]').hidden=false;document.querySelectorAll('input[type=checkbox]').forEach(e=>e.checked=true)`)
	task.reconcile()
	button := task.page.Locator("a.confirmDown")
	box, e2 := button.BoundingBox()
	task.op.Unlock()
	if e != nil || e2 != nil || box == nil {
		t.Fatal("manual export dialog unavailable", e, e2)
	}
	frame, e := task.Frame()
	if e != nil {
		t.Fatal(e)
	}
	action := func(kind string) object {
		return object{"kind": kind, "x": box.X + box.Width/2, "y": box.Y + box.Height/2, "page_id": frame["page_id"], "epoch": frame["epoch"], "revision": frame["revision"]}
	}
	if e = task.Input(object{"kind": "batch", "actions": []any{action("pointer_down"), action("pointer_up")}}, "teacher"); e != nil {
		t.Fatal(e)
	}
	if !flag(task.snapshot()["export_attempted"]) {
		t.Fatal("manual export intent was not durable before handing over")
	}
	if e = task.Resume(); e != nil {
		t.Fatal(e)
	}
	await(t, 10*time.Second, func() bool { return terminal(str(task.snapshot()["status"])) })
	s := task.snapshot()
	if s["status"] != "SUCCEEDED" || submitted.Load() != 1 || len(array(s["files"])) != 1 {
		t.Fatalf("manual handover replayed or lost the export: count=%d state=%+v", submitted.Load(), s)
	}
}
func elementRef(view object, name string) string {
	for _, item := range objects(obj(view["observation"])["elements"]) {
		if str(item["name"]) == name {
			return str(item["ref"])
		}
	}
	return ""
}
func TestGoBrowserRejectsStaleObservationAndControlEpoch(t *testing.T) {
	_, task := controlFixture(t)
	view, e := task.Observe()
	if e != nil || !flag(view["ready"]) {
		t.Fatal(view, e)
	}
	ref := elementRef(view, "目标控件")
	if ref == "" {
		t.Fatal("fixture control missing")
	}
	task.op.Lock()
	_, e = task.page.Evaluate(`document.querySelector('#action').textContent='变化后的控件'`)
	task.op.Unlock()
	if e != nil {
		t.Fatal(e)
	}
	payload := object{"action_id": view["observation_id"], "observation_id": view["observation_id"], "revision": view["revision"], "epoch": view["epoch"], "action": object{"tool": "click", "element_ref": ref}}
	if _, e = task.Apply(payload); e == nil {
		t.Fatal("executed a stale element reference")
	}
	task.op.Lock()
	clicked, e := task.page.Evaluate("window.clicked||0")
	task.op.Unlock()
	if e != nil || num(clicked) != 0 {
		t.Fatal("stale action clicked the page")
	}
	fresh, e := task.Observe()
	if e != nil {
		t.Fatal(e)
	}
	if e = task.Pause(); e != nil {
		t.Fatal(e)
	}
	if _, e = task.Apply(object{"action_id": fresh["observation_id"], "observation_id": fresh["observation_id"], "revision": fresh["revision"], "epoch": fresh["epoch"], "action": object{"tool": "click", "element_ref": elementRef(fresh, "变化后的控件")}}); e == nil {
		t.Fatal("Agent action ran after human takeover")
	}
	frame, e := task.Frame()
	if e != nil {
		t.Fatal(e)
	}
	jpeg, e := base64.StdEncoding.DecodeString(str(frame["image"]))
	if e != nil || len(jpeg) < 4 || jpeg[0] != 0xff || jpeg[1] != 0xd8 {
		t.Fatal("invalid browser frame")
	}
	if e = task.Resume(); e != nil {
		t.Fatal(e)
	}
	if e = task.Input(object{"kind": "text", "text": "stale-input", "epoch": frame["epoch"], "page_id": frame["page_id"], "revision": frame["revision"]}, "teacher"); e == nil {
		t.Fatal("human input ran after control was returned")
	}
}

func readStream(t *testing.T, c *websocket.Conn, kind string) object {
	t.Helper()
	c.SetReadDeadline(time.Now().Add(10 * time.Second))
	for {
		var msg object
		if e := c.ReadJSON(&msg); e != nil {
			t.Fatal(e)
		}
		if str(msg["type"]) == kind {
			return msg
		}
		if str(msg["type"]) == "error" {
			t.Fatalf("stream error: %+v", msg)
		}
	}
}
func TestGoBrowserStreamAndDisconnectReleasePointer(t *testing.T) {
	m, task := controlFixture(t)
	if e := task.Pause(); e != nil {
		t.Fatal(e)
	}
	server := httptest.NewServer(m.Handler())
	defer server.Close()
	endpoint := "ws" + strings.TrimPrefix(server.URL, "http") + "/tasks/" + task.id + "/stream"
	if c, response, e := websocket.DefaultDialer.Dial(endpoint, nil); e == nil {
		c.Close()
		t.Fatal("unauthorized stream accepted")
	} else if response == nil || response.StatusCode != 401 {
		t.Fatal("unexpected unauthorized stream status")
	}
	conn, _, e := websocket.DefaultDialer.Dial(endpoint, http.Header{"X-RPA-Token": []string{m.token}})
	if e != nil {
		t.Fatal(e)
	}
	defer conn.Close()
	readStream(t, conn, "ready")
	frame := readStream(t, conn, "frame")
	if len(str(frame["image"])) < 100 {
		t.Fatal("empty CDP frame")
	}
	if e = conn.WriteJSON(object{"type": "ping"}); e != nil {
		t.Fatal(e)
	}
	readStream(t, conn, "pong")
	base := func(kind string) object {
		return object{"kind": kind, "x": float64(50), "y": float64(55), "page_id": frame["page_id"], "epoch": frame["epoch"], "revision": frame["revision"]}
	}
	if e = conn.WriteJSON(object{"type": "input", "id": 1, "action": base("pointer_down")}); e != nil {
		t.Fatal(e)
	}
	readStream(t, conn, "input_ack")
	task.op.Lock()
	held := task.pointerPage != nil
	task.op.Unlock()
	if !held {
		t.Fatal("manual pointer was not held")
	}
	conn.Close()
	await(t, 5*time.Second, func() bool { task.op.Lock(); defer task.op.Unlock(); return task.pointerPage == nil })
	if e = task.Resume(); e != nil {
		t.Fatal(e)
	}
	await(t, 5*time.Second, func() bool {
		task.mu.Lock()
		producer := task.frames
		task.mu.Unlock()
		producer.mu.Lock()
		defer producer.mu.Unlock()
		return len(producer.viewers) == 0
	})
}
