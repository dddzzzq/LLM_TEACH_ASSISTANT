package rpa

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

var workerClient = &http.Client{Timeout: 15 * time.Second}

type WorkerError struct {
	Status  int
	Message string
}

func (e *WorkerError) Error() string { return e.Message }

func WorkerRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	var reader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(b)
	}
	base, token, err := workerConnection()
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, base+path, reader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-RPA-Token", token)
	req.Header.Set("Content-Type", "application/json")
	resp, err := workerClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("无法连接浏览器 Worker")
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 8*1024*1024))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		var result struct {
			Error  string `json:"error"`
			Detail string `json:"detail"`
		}
		_ = json.Unmarshal(b, &result)
		if result.Error == "" {
			result.Error = result.Detail
		}
		if result.Error == "" {
			result.Error = "浏览器操作未完成"
		}
		return nil, &WorkerError{resp.StatusCode, result.Error}
	}
	return b, nil
}

// OpenBrowserStream only connects to the configured internal Worker. User
// authentication and job ownership must be checked by the caller first.
func OpenBrowserStream(ctx context.Context, jobID string) (*websocket.Conn, error) {
	base, token, err := workerConnection()
	if err != nil {
		return nil, err
	}
	endpoint, err := url.Parse(base + "/tasks/" + url.PathEscape(jobID) + "/stream")
	if err != nil {
		return nil, fmt.Errorf("浏览器 Worker 地址无效")
	}
	switch endpoint.Scheme {
	case "http":
		endpoint.Scheme = "ws"
	case "https":
		endpoint.Scheme = "wss"
	default:
		return nil, fmt.Errorf("浏览器 Worker 地址无效")
	}
	dialer := websocket.Dialer{HandshakeTimeout: 5 * time.Second}
	conn, response, err := dialer.DialContext(ctx, endpoint.String(), http.Header{"X-Rpa-Token": []string{token}})
	if response != nil && response.Body != nil {
		response.Body.Close()
	}
	if err != nil {
		return nil, fmt.Errorf("浏览器流式连接失败，请检查 Worker 版本及日志")
	}
	return conn, nil
}

func workerConnection() (string, string, error) {
	runtime := os.Getenv("RUNTIME_DIR")
	if runtime == "" {
		runtime = "../../teaching-runtime"
	}
	tokenPath := os.Getenv("RPA_CONTROL_TOKEN_FILE")
	if tokenPath == "" {
		tokenPath = filepath.Join(runtime, ".rpa-control-token")
	}
	token, err := os.ReadFile(tokenPath)
	if err != nil {
		return "", "", fmt.Errorf("浏览器 Worker 未就绪")
	}
	base := os.Getenv("RPA_CONTROL_URL")
	if base == "" {
		base = "http://127.0.0.1:8765"
	}
	return strings.TrimRight(base, "/"), strings.TrimSpace(string(token)), nil
}
