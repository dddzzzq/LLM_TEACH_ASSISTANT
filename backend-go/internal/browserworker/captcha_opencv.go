package browserworker

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"os/exec"
	"time"
)

// Reuse the original OpenCV implementation. Go retains browser ownership;
// the local Python helper receives only the two canvas images over stdin.
func captchaDistance(ctx context.Context, cfg Config, background, piece []byte) (float64, error) {
	for _, data := range [][]byte{background, piece} {
		if len(data) == 0 || len(data) > 2<<20 {
			return 0, problem("CAPTCHA_IMAGE_INVALID", "滑块图片大小无效")
		}
		size, _, err := image.DecodeConfig(bytes.NewReader(data))
		if err != nil || size.Width < 1 || size.Height < 1 || size.Width > 4096 || size.Height > 4096 {
			return 0, problem("CAPTCHA_IMAGE_INVALID", "滑块图片尺寸无效")
		}
	}
	payload, _ := json.Marshal(object{"background": background, "piece": piece})
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, cfg.Python, "-m", "app.rpa.slider_captcha")
	cmd.Dir = cfg.PythonApp
	cmd.Stdin = bytes.NewReader(payload)
	// Do not forward the Worker's service tokens or other application secrets.
	cmd.Env = []string{"PYTHONDONTWRITEBYTECODE=1", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1"}
	for _, key := range []string{"PATH", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL"} {
		if value, ok := os.LookupEnv(key); ok {
			cmd.Env = append(cmd.Env, key+"="+value)
		}
	}
	out, err := cmd.Output()
	if err != nil {
		if ctx.Err() != nil {
			return 0, problem("CAPTCHA_OPENCV_TIMEOUT", "OpenCV 滑块识别超时")
		}
		return 0, problem("CAPTCHA_OPENCV_UNAVAILABLE", "原 OpenCV 滑块工具不可用，请检查 Python 环境")
	}
	var result struct {
		Status, Code, Driver string
		Distance             float64
	}
	if len(out) > 1024 || json.Unmarshal(out, &result) != nil || result.Driver != "python-opencv" {
		return 0, problem("CAPTCHA_OPENCV_INVALID_RESULT", "OpenCV 滑块工具返回结果无效")
	}
	if result.Status != "ok" {
		if result.Code == "CAPTCHA_OPENCV_UNAVAILABLE" {
			return 0, problem(result.Code, "原 OpenCV 滑块工具依赖不可用")
		}
		return 0, problem("CAPTCHA_IMAGE_UNCERTAIN", "OpenCV 未能可靠识别当前滑块")
	}
	if math.IsNaN(result.Distance) || math.IsInf(result.Distance, 0) || result.Distance <= 0 || result.Distance > 4096 {
		return 0, problem("CAPTCHA_OPENCV_INVALID_RESULT", "OpenCV 滑块距离无效")
	}
	return result.Distance, nil
}
