package browserworker

import (
	"context"
	"encoding/base64"
	"errors"
	"math/rand/v2"
	"strings"
	"time"

	pw "github.com/mxschmitt/playwright-go"
)

func (t *Task) captchaFrame() pw.Frame {
	if t.context == nil {
		return nil
	}
	for _, p := range t.context.Pages() {
		if p.IsClosed() || !allowedURL(p.URL()) {
			continue
		}
		for _, f := range p.Frames() {
			if allowedURL(f.URL()) && count(f.Locator("#sliderDiv:visible")) == 1 {
				return f
			}
		}
	}
	return nil
}

func (t *Task) solveCaptcha() error {
	epoch := num(t.snapshot()["control_epoch"])
	deadline := time.Now().Add(20 * time.Second)
	ctx, cancel := context.WithDeadline(t.ctx, deadline)
	defer cancel()
	owned := func() bool { return t.automatic() && num(t.snapshot()["control_epoch"]) == epoch }
	active := func() bool { return owned() && time.Now().Before(deadline) }
	f := t.captchaFrame()
	if f == nil {
		if active() {
			t.human("LOGIN", "未发现可处理的滑块，请核对当前登录页面")
		}
		return nil
	}
	container := f.Locator("#sliderDiv")
	t.page = f.Page()
	failed := func(phase, code string) {
		_ = t.update(object{"last_captcha_result": object{"driver": "python-opencv", "status": "failed", "attempt": t.snapshot()["captcha_attempts"], "phase": phase, "code": code}})
	}
	for num(t.snapshot()["captcha_attempts"]) < 3 && active() {
		if !visible(container) {
			return t.captchaFinished(epoch)
		}
		attempt := num(t.snapshot()["captcha_attempts"]) + 1
		_ = t.update(object{"captcha_attempts": attempt, "message": "正在调用原 OpenCV 工具自动完成滑块验证", "last_captcha_result": object{"driver": "python-opencv", "status": "running", "attempt": attempt}})
		v, e := container.Evaluate(script("slider_canvases"), nil)
		if e != nil {
			failed("read_canvases", "CAPTCHA_CANVAS_UNAVAILABLE")
			continue
		}
		data := obj(v)
		decode := func(s string) ([]byte, error) {
			parts := strings.SplitN(s, ",", 2)
			if len(parts) != 2 {
				return nil, errors.New("invalid image")
			}
			return base64.StdEncoding.DecodeString(parts[1])
		}
		bg, e := decode(str(data["bg"]))
		if e != nil {
			failed("decode_background", "CAPTCHA_IMAGE_INVALID")
			continue
		}
		piece, e := decode(str(data["block"]))
		if e != nil {
			failed("decode_piece", "CAPTCHA_IMAGE_INVALID")
			continue
		}
		distance, e := captchaDistance(ctx, t.manager.cfg, bg, piece)
		if e != nil {
			failed("opencv_match", str(publicError(e)["code"]))
			continue
		}
		distance *= num(data["scale"])
		box, e := container.Locator(".slider").BoundingBox()
		bounds, boundsErr := container.BoundingBox()
		if e != nil || boundsErr != nil || box == nil || bounds == nil || distance <= 0 || distance > bounds.Width {
			failed("geometry", "CAPTCHA_GEOMETRY_INVALID")
			continue
		}
		x, y := box.X+box.Width/2, box.Y+box.Height/2
		mouse := t.page.Mouse()
		if !active() {
			if owned() {
				t.human("LOGIN", "滑块处理超时，请手动完成验证")
			}
			return nil
		}
		if e = mouse.Move(x, y); e != nil {
			failed("drag", "CAPTCHA_MOUSE_UNAVAILABLE")
			continue
		}
		if e = mouse.Down(); e != nil {
			failed("drag", "CAPTCHA_MOUSE_UNAVAILABLE")
			continue
		}
		for step := 1; step <= 10 && active(); step++ {
			_ = mouse.Move(x+distance*float64(step)/10, y+rand.Float64()*2-1)
			time.Sleep(10 * time.Millisecond)
		}
		_ = mouse.Up()
		if !active() {
			if owned() {
				t.human("LOGIN", "滑块处理超时，请手动完成验证")
			}
			return nil
		}
		if e = container.WaitFor(pw.LocatorWaitForOptions{State: pw.WaitForSelectorStateHidden, Timeout: pw.Float(3000)}); e == nil {
			return t.captchaFinished(epoch)
		}
		failed("verify_challenge", "CAPTCHA_NOT_ACCEPTED")
	}
	if owned() {
		t.human("LOGIN", "原 OpenCV 滑块工具自动验证未通过，请在任务浏览器核查")
	}
	return nil
}

func (t *Task) captchaFinished(epoch float64) error {
	if !t.automatic() || num(t.snapshot()["control_epoch"]) != epoch {
		return nil
	}
	_ = t.update(object{"last_captcha_result": object{"driver": "python-opencv", "status": "solved", "attempt": t.snapshot()["captcha_attempts"]}, "message": "滑块验证工具已完成，正在继续核验登录"})
	if t.auth != nil {
		return nil
	}
	for i := 0; i < 20 && t.automatic() && num(t.snapshot()["control_epoch"]) == epoch; i++ {
		if t.authenticated() {
			stage := "LOCATE_COURSE"
			if flag(t.snapshot()["export_attempted"]) {
				stage = "WAIT_EXPORT"
			}
			return t.update(object{"auth_status": "AUTHENTICATED", "stage": stage, "captcha_attempts": 0})
		}
		select {
		case <-t.ctx.Done():
			return nil
		case <-time.After(200 * time.Millisecond):
		}
	}
	if t.automatic() && num(t.snapshot()["control_epoch"]) == epoch {
		t.human("LOGIN", "滑块已处理，请检查并完成后续登录验证")
	}
	return nil
}
