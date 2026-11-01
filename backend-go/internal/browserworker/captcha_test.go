package browserworker

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/google/uuid"
	pw "github.com/mxschmitt/playwright-go"
)

func opencvTestConfig(t *testing.T) Config {
	t.Helper()
	if os.Getenv("RPA_GO_BROWSER_TESTS") != "1" {
		t.Skip("set RPA_GO_BROWSER_TESTS=1 with the original Python/OpenCV environment")
	}
	cfg := DefaultConfig()
	cfg.PythonApp, _ = filepath.Abs("../../../ai_engine_python")
	return cfg
}

func TestCaptchaDistanceMatchesAlphaCrop(t *testing.T) {
	cfg := opencvTestConfig(t)
	bg, block := image.NewNRGBA(image.Rect(0, 0, 220, 80)), image.NewNRGBA(image.Rect(0, 0, 220, 80))
	for y := 20; y < 45; y++ {
		for x := 0; x < 25; x++ {
			value := uint8(255)
			if y >= 24 && y < 37 && x >= 4 && x < 11 || y >= 32 && y < 40 && x >= 14 && x < 21 {
				value = 0
			}
			pixel := color.NRGBA{value, value, value, 255}
			bg.SetNRGBA(x+100, y, pixel)
			block.SetNRGBA(x+5, y, pixel)
		}
	}
	encode := func(im image.Image) []byte {
		b := bytes.Buffer{}
		if e := png.Encode(&b, im); e != nil {
			t.Fatal(e)
		}
		return b.Bytes()
	}
	distance, e := captchaDistance(context.Background(), cfg, encode(bg), encode(block))
	if e != nil || distance != 95 {
		t.Fatalf("distance=%v error=%v", distance, e)
	}
	if _, e = captchaDistance(context.Background(), cfg, []byte("broken"), encode(block)); e == nil {
		t.Fatal("accepted a broken challenge image")
	}
}

func TestCaptchaMatchesOriginalPythonImageVectors(t *testing.T) {
	cfg := opencvTestConfig(t)
	data, err := os.ReadFile("testdata/captcha_python_vectors.json")
	if err != nil {
		t.Fatal(err)
	}
	var vectors []struct {
		Name, Background, Piece string
		Distance                float64
	}
	if err = json.Unmarshal(data, &vectors); err != nil {
		t.Fatal(err)
	}
	for _, vector := range vectors {
		t.Run(vector.Name, func(t *testing.T) {
			background, err := base64.StdEncoding.DecodeString(vector.Background)
			if err != nil {
				t.Fatal(err)
			}
			piece, err := base64.StdEncoding.DecodeString(vector.Piece)
			if err != nil {
				t.Fatal(err)
			}
			distance, err := captchaDistance(context.Background(), cfg, background, piece)
			if err != nil || distance != vector.Distance {
				t.Fatalf("OpenCV=%v, expected=%v, error=%v", distance, vector.Distance, err)
			}
		})
	}
}
func TestGoCaptchaUsesExistingPageAndBoundedAction(t *testing.T) {
	html, e := os.ReadFile("testdata/slider.html")
	if e != nil {
		t.Fatal(e)
	}
	m := testManager(t, func(c pw.BrowserContext) error {
		return c.Route("**/*", func(r pw.Route) {
			_ = r.Fulfill(pw.RouteFulfillOptions{Body: string(html), ContentType: pw.String("text/html; charset=utf-8")})
		})
	})
	task, e := m.Start(uuid.NewString(), object{"target": object{"course_name": "课程", "assignment_name": "作业"}})
	if e != nil {
		t.Fatal(e)
	}
	await(t, 10*time.Second, func() bool { return !task.starting.Load() })
	if e = task.Resume(); e != nil {
		t.Fatal(e)
	}
	view, e := task.Observe()
	if e != nil || !flag(view["ready"]) || view["stage"] != "SOLVE_CAPTCHA" {
		t.Fatal(view, e)
	}
	out := applyObserved(t, task, view, object{"tool": "solve_slider_captcha"})
	if out["outcome"] != "completed" {
		t.Fatal(out)
	}
	task.op.Lock()
	visible := task.captchaFrame() != nil
	released, _ := task.page.Evaluate("window.releases")
	pages := len(task.context.Pages())
	task.op.Unlock()
	result := obj(task.snapshot()["last_captcha_result"])
	if visible || num(released) != 1 || pages != 1 || num(task.snapshot()["captcha_attempts"]) != 1 || result["driver"] != "python-opencv" || result["status"] != "solved" {
		t.Fatalf("challenge did not use the original page: visible=%v releases=%v pages=%d state=%+v", visible, released, pages, task.snapshot())
	}
}

func TestOpenCVSliderFailureKeepsOriginalAttemptLimit(t *testing.T) {
	_, task := controlFixture(t)
	func() {
		task.op.Lock()
		defer task.op.Unlock()
		html, err := os.ReadFile("testdata/slider.html")
		if err != nil {
			t.Fatal(err)
		}
		if err = task.page.SetContent(string(html)); err != nil {
			t.Fatal(err)
		}
		// A missing interpreter must report the original tool's failure, never
		// silently use another image algorithm or move the mouse with a guess.
		task.manager.cfg.Python = filepath.Join(t.TempDir(), "missing-python")
		_ = task.update(object{"stage": "SOLVE_CAPTCHA"})
		if err = task.execute(object{"tool": "solve_slider_captcha"}); err != nil {
			t.Fatal(err)
		}
		state := task.snapshot()
		result := obj(state["last_captcha_result"])
		released, _ := task.page.Evaluate("window.releases")
		if state["control_mode"] != "HUMAN" || num(state["captcha_attempts"]) != 3 || num(released) != 0 || result["driver"] != "python-opencv" || result["code"] != "CAPTCHA_OPENCV_UNAVAILABLE" {
			t.Fatal("unexpected tool failure or hidden fallback", state, released)
		}
	}()
	if err := task.Resume(); err != nil {
		t.Fatal(err)
	}
	if num(task.snapshot()["captcha_attempts"]) != 3 {
		t.Fatal("resuming an exhausted challenge reset its attempt budget")
	}
}

func TestOpenCVImageBridgeHonorsCancellation(t *testing.T) {
	cfg := opencvTestConfig(t)
	data, err := os.ReadFile("testdata/captcha_python_vectors.json")
	if err != nil {
		t.Fatal(err)
	}
	var vectors []struct{ Background, Piece string }
	if err = json.Unmarshal(data, &vectors); err != nil {
		t.Fatal(err)
	}
	background, _ := base64.StdEncoding.DecodeString(vectors[0].Background)
	piece, _ := base64.StdEncoding.DecodeString(vectors[0].Piece)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	started := time.Now()
	_, err = captchaDistance(ctx, cfg, background, piece)
	requireIssue(t, err, "CAPTCHA_OPENCV_TIMEOUT")
	if time.Since(started) > time.Second {
		t.Fatal("image helper ignored cancellation")
	}
}
