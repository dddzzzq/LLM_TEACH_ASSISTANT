package browserworker

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	pw "github.com/mxschmitt/playwright-go"
)

func (t *Task) recordFrame(data string) object {
	epoch := num(t.snapshot()["control_epoch"])
	t.frameRevision++
	t.frameHistory[t.frameRevision] = frameRecord{t.page, t.page.URL(), epoch, time.Now()}
	for id := range t.frameHistory {
		if id < t.frameRevision-128 {
			delete(t.frameHistory, id)
		}
	}
	pages := []any{}
	for _, p := range t.context.Pages() {
		if !p.IsClosed() {
			title, _ := p.Title()
			pages = append(pages, object{"id": pageIndex(t.context, p), "title": bounded(title, 100)})
		}
	}
	return object{"type": "frame", "image": data, "revision": t.frameRevision, "page_id": pageIndex(t.context, t.page), "epoch": epoch, "width": 1440, "height": 1000, "pages": pages}
}
func (t *Task) Frame() (object, error) {
	t.op.Lock()
	if t.page == nil || t.page.IsClosed() || t.context == nil {
		t.op.Unlock()
		return nil, errors.New("浏览器当前不可用")
	}
	page, url, epoch := t.page, t.page.URL(), num(t.snapshot()["control_epoch"])
	t.op.Unlock()
	data, e := page.Screenshot(pw.PageScreenshotOptions{Type: pw.ScreenshotTypeJpeg, Quality: pw.Int(55), Timeout: pw.Float(8000)})
	if e != nil {
		return nil, errors.New("浏览器画面暂时不可用")
	}
	t.op.Lock()
	defer t.op.Unlock()
	if t.context == nil || page != t.page || page.IsClosed() || page.URL() != url || num(t.snapshot()["control_epoch"]) != epoch {
		return nil, errors.New("页面或控制权已变化，请刷新画面")
	}
	return t.recordFrame(base64.StdEncoding.EncodeToString(data)), nil
}
func (t *Task) releasePointer(owner string) {
	if t.pointerPage != nil && (owner == "" || t.pointerOwner == owner) {
		p := t.pointerPage
		t.pointerPage = nil
		t.pointerOwner = ""
		if !p.IsClosed() {
			_ = p.Mouse().Move(-1, -1)
			_ = p.Mouse().Up()
		}
	}
}
func (t *Task) Input(payload object, owner string) error {
	t.op.Lock()
	defer t.op.Unlock()
	actions := []object{payload}
	if str(payload["kind"]) == "batch" {
		actions = objects(payload["actions"])
		if len(actions) != len(array(payload["actions"])) || len(actions) < 1 || len(actions) > 32 {
			return errors.New("批量操作参数无效")
		}
	}
	for _, a := range actions {
		if str(a["kind"]) == "batch" {
			return errors.New("不允许嵌套批量操作")
		}
		if e := t.input(a, owner); e != nil {
			return e
		}
	}
	return nil
}
func (t *Task) input(a object, owner string) error {
	s := t.snapshot()
	if terminal(str(s["status"])) || str(s["control_mode"]) != "HUMAN" || t.context == nil {
		return errors.New("请先暂停并接管浏览器")
	}
	if num(a["epoch"]) != num(s["control_epoch"]) {
		return errors.New("控制权已变化，请刷新画面")
	}
	kind := str(a["kind"])
	if kind == "switch_page" {
		p, e := t.pageByID(str(a["page_id"]))
		if e != nil {
			return e
		}
		t.releasePointer("")
		t.page = p
		t.frameHistory = map[int]frameRecord{}
		return nil
	}
	meta, ok := t.frameHistory[t.frameRevision]
	if !ok || t.page == nil || meta.page != t.page || meta.url != t.page.URL() || meta.epoch != num(s["control_epoch"]) || str(a["page_id"]) != pageIndex(t.context, t.page) || !allowedURL(t.page.URL()) {
		return errors.New("页面已变化，请刷新画面")
	}
	if t.pointerPage != nil && t.pointerOwner != owner {
		return errors.New("另一连接正在拖动，请稍后重试")
	}
	x, y := num(a["x"]), num(a["y"])
	if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) || x < 0 || x > 1440 || y < 0 || y > 1000 {
		return errors.New("坐标越界")
	}
	var e error
	switch kind {
	case "pointer_down":
		displayed, ok := t.frameHistory[int(num(a["revision"]))]
		if !ok || time.Since(displayed.at) > 3*time.Second || displayed.page != meta.page || displayed.url != meta.url || displayed.epoch != meta.epoch {
			return errors.New("画面已刷新，请重试")
		}
		if e = t.page.Mouse().Move(x, y); e == nil {
			e = t.page.Mouse().Down()
			if e == nil {
				t.pointerPage, t.pointerOwner = t.page, owner
			}
		}
	case "pointer_move":
		e = t.page.Mouse().Move(x, y)
	case "pointer_up":
		if !flag(s["export_attempted"]) {
			if area := t.exportArea(); area != nil {
				if e = t.checkpointManualConfirm(area, a); e != nil {
					return e
				}
			}
		}
		if e = t.page.Mouse().Move(x, y); e == nil {
			e = t.page.Mouse().Up()
		}
		t.pointerPage, t.pointerOwner = nil, ""
	case "pointer_cancel":
		t.releasePointer(owner)
	case "text":
		e = t.page.Keyboard().InsertText(bounded(str(a["text"]), 2000))
	case "login_code":
		if t.auth == nil || str(s["waiting_reason"]) != "LOGIN_CODE" {
			return errors.New("当前任务没有等待验证码")
		}
		e = t.auth.fillCode(str(a["text"]), t.page)
	case "key":
		key := str(a["key"])
		allowed := map[string]bool{"Enter": true, "Tab": true, "Backspace": true, "Delete": true, "Escape": true, "ArrowLeft": true, "ArrowRight": true, "ArrowUp": true, "ArrowDown": true, "Control+A": true}
		if !allowed[key] {
			return errors.New("不支持的按键")
		}
		if key == "Enter" && !flag(s["export_attempted"]) {
			if area := t.exportArea(); area != nil {
				if e = t.checkpointManualConfirm(area, a); e != nil {
					return e
				}
			}
		}
		e = t.page.Keyboard().Press(key)
	case "scroll":
		delta := math.Max(-1000, math.Min(1000, num(a["delta_y"])))
		e = t.page.Mouse().Wheel(0, delta)
	default:
		return errors.New("不支持的浏览器输入")
	}
	if e != nil {
		return errors.New("浏览器输入未确认，请核对画面")
	}
	t.activity.Store(time.Now().Unix())
	return nil
}

type frameProducer struct {
	task    *Task
	mu      sync.Mutex
	viewers map[chan object]bool
	cancel  context.CancelFunc
	done    chan struct{}
}

func (p *frameProducer) subscribe() (chan object, func()) {
	p.mu.Lock()
	defer p.mu.Unlock()
	ch := make(chan object, 1)
	p.viewers[ch] = true
	if p.cancel == nil {
		ctx, cancel := context.WithCancel(p.task.ctx)
		p.cancel = cancel
		previous := p.done
		finished := make(chan struct{})
		p.done = finished
		p.task.manager.wg.Add(1)
		go func() {
			defer p.task.manager.wg.Done()
			defer close(finished)
			if previous != nil {
				select {
				case <-previous:
				case <-ctx.Done():
					return
				}
			}
			p.run(ctx)
		}()
	}
	return ch, func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		delete(p.viewers, ch)
		if len(p.viewers) == 0 && p.cancel != nil {
			p.cancel()
			p.cancel = nil
		}
	}
}
func (p *frameProducer) publish(value object) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for ch := range p.viewers {
		select {
		case ch <- value:
		default:
			select {
			case <-ch:
			default:
			}
			select {
			case ch <- value:
			default:
			}
		}
	}
}
func (p *frameProducer) run(ctx context.Context) {
	for ctx.Err() == nil {
		if terminal(str(p.task.snapshot()["status"])) {
			p.publish(object{"type": "ended", "message": "浏览器任务已结束"})
			return
		}
		if e := p.capture(ctx); e != nil && ctx.Err() == nil {
			p.publish(object{"type": "error", "message": "浏览器暂未就绪，正在重新连接"})
			select {
			case <-ctx.Done():
				return
			case <-time.After(time.Second):
			}
		}
	}
}
func (p *frameProducer) capture(ctx context.Context) error {
	t := p.task
	t.op.Lock()
	page, browserContext := t.page, t.context
	if page == nil || browserContext == nil || page.IsClosed() {
		t.op.Unlock()
		return errors.New("browser unavailable")
	}
	epoch := num(t.snapshot()["control_epoch"])
	url := page.URL()
	session, e := browserContext.NewCDPSession(page)
	t.op.Unlock()
	if e != nil {
		return e
	}
	events := make(chan object, 1)
	ackCtx, stopAcks := context.WithCancel(ctx)
	acks := make(chan any, 128)
	ackDone := make(chan struct{})
	overflow := make(chan struct{}, 1)
	go func() {
		defer close(ackDone)
		for {
			select {
			case <-ackCtx.Done():
				return
			case id := <-acks:
				_, _ = session.Send("Page.screencastFrameAck", object{"sessionId": id})
			}
		}
	}()
	defer func() {
		session.RemoveListeners("Page.screencastFrame")
		stopAcks()
		_, _ = session.Send("Page.stopScreencast", nil)
		_ = session.Detach()
		<-ackDone
	}()
	session.On("Page.screencastFrame", func(v object) {
		select {
		case acks <- v["sessionId"]:
		default:
			select {
			case overflow <- struct{}{}:
			default:
			}
		}
		select {
		case events <- v:
		default:
			select {
			case <-events:
			default:
			}
			select {
			case events <- v:
			default:
			}
		}
	})
	_, e = session.Send("Page.startScreencast", object{"format": "jpeg", "quality": 55, "maxWidth": 1440, "maxHeight": 1000, "everyNthFrame": 1})
	if e != nil {
		return e
	}
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	last := time.Time{}
	for {
		var data string
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-overflow:
			return errors.New("screencast acknowledgement backlog")
		case event := <-events:
			data = str(event["data"])
			if time.Since(last) < time.Second/30 {
				continue
			}
		case <-ticker.C:
			if time.Since(last) < time.Second {
				continue
			}
			if !t.op.TryLock() {
				continue
			}
			changed := t.page != page || page.IsClosed() || page.URL() != url || num(t.snapshot()["control_epoch"]) != epoch
			t.op.Unlock()
			if changed {
				return nil
			}
			frame, e := t.Frame()
			if e == nil {
				p.publish(frame)
				last = time.Now()
			}
			continue
		}
		if !t.op.TryLock() {
			continue
		}
		if t.page != page || page.IsClosed() || page.URL() != url || num(t.snapshot()["control_epoch"]) != epoch {
			t.op.Unlock()
			return nil
		}
		frame := t.recordFrame(data)
		t.op.Unlock()
		p.publish(frame)
		last = time.Now()
	}
}
func (m *Manager) serveStream(w http.ResponseWriter, r *http.Request, t *Task) {
	upgrader := websocket.Upgrader{HandshakeTimeout: 5 * time.Second, CheckOrigin: func(*http.Request) bool { return true }}
	socket, e := upgrader.Upgrade(w, r, nil)
	if e != nil {
		return
	}
	defer socket.Close()
	socket.SetReadLimit(16 * 1024)
	owner := uuid.NewString()
	defer func() { t.op.Lock(); t.releasePointer(owner); t.op.Unlock() }()
	t.mu.Lock()
	if t.frames == nil {
		t.frames = &frameProducer{task: t, viewers: map[chan object]bool{}}
	}
	producer := t.frames
	t.mu.Unlock()
	frames, unsubscribe := producer.subscribe()
	defer unsubscribe()
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	inputs := make(chan object, 64)
	replies := make(chan object, 128)
	failed := make(chan error, 2)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		defer cancel()
		last := float64(0)
		for {
			socket.SetReadDeadline(time.Now().Add(40 * time.Second))
			var msg object
			if e := socket.ReadJSON(&msg); e != nil {
				failed <- e
				return
			}
			if str(msg["type"]) == "ping" {
				select {
				case replies <- object{"type": "pong"}:
				default:
					failed <- errors.New("reply backlog")
					return
				}
				continue
			}
			id := num(msg["id"])
			if str(msg["type"]) != "input" || id <= last || id > 9007199254740991 || id != math.Trunc(id) || len(obj(msg["action"])) == 0 {
				failed <- errors.New("invalid input order")
				return
			}
			last = id
			select {
			case inputs <- msg:
			default:
				failed <- errors.New("input backlog")
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():
				return
			case msg := <-inputs:
				if e := t.Input(obj(msg["action"]), owner); e != nil {
					select {
					case failed <- e:
					default:
					}
					cancel()
					return
				}
				select {
				case replies <- object{"type": "input_ack", "id": msg["id"]}:
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	write := func(v object) error {
		socket.SetWriteDeadline(time.Now().Add(5 * time.Second))
		return socket.WriteJSON(v)
	}
	_ = write(object{"type": "ready"})
	for running := true; running; {
		select {
		case <-ctx.Done():
			running = false
		case <-t.ctx.Done():
			_ = write(object{"type": "ended", "message": "浏览器任务已结束"})
			running = false
		case <-failed:
			_ = write(object{"type": "error", "message": "页面或控制权已变化，请核对画面后重新连接"})
			running = false
		case msg := <-replies:
			if write(msg) != nil {
				running = false
			}
		case frame := <-frames:
			if write(frame) != nil || str(frame["type"]) == "ended" {
				running = false
			}
		}
	}
	cancel()
	socket.Close()
	wg.Wait()
}
func decodeObject(r *http.Request) (object, error) {
	var value object
	decoder := json.NewDecoder(http.MaxBytesReader(nil, r.Body, 1024*1024))
	if e := decoder.Decode(&value); e != nil || value == nil {
		return nil, errors.New("请求参数无效")
	}
	var extra any
	if e := decoder.Decode(&extra); e != io.EOF {
		return nil, errors.New("请求只能包含一个 JSON 对象")
	}
	return value, nil
}
