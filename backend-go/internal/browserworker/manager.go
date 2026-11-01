package browserworker

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	pw "github.com/mxschmitt/playwright-go"
)

type Config struct {
	Runtime, Downloads, TokenFile, Proxy, StartURL string
	Python, PythonApp                              string
	MaxTasks                                       int
	startup                                        *startupPolicy
	// ConfigureContext is an in-process fixture hook, never an HTTP option.
	ConfigureContext func(pw.BrowserContext) error
}

func DefaultConfig() Config {
	runtime := env("RUNTIME_DIR", filepath.Join("..", "..", "teaching-runtime"))
	n, _ := strconv.Atoi(env("RPA_MAX_TASKS", "4"))
	if n < 1 {
		n = 4
	}
	return Config{Runtime: runtime, Downloads: env("RPA_DOWNLOAD_DIR", filepath.Join("..", "downloads")), TokenFile: env("RPA_CONTROL_TOKEN_FILE", filepath.Join(runtime, ".rpa-control-token")), Proxy: os.Getenv("RPA_PROXY_URL"), StartURL: constants["portal_adapter.TARGET_URL"], MaxTasks: n,
		Python: env("PYTHON", filepath.Join(runtime, "python-env", "bin", "python")), PythonApp: filepath.Join(env("ROOT", ".."), "ai_engine_python")}
}

type Manager struct {
	cfg        Config
	mu         sync.Mutex
	tasks      map[string]*Task
	pw         *pw.Playwright
	token      string
	lock       *os.File
	sessionsMu sync.Mutex
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

func New(ctx context.Context, cfg Config) (*Manager, error) {
	defaults := DefaultConfig()
	if cfg.Python == "" {
		cfg.Python = defaults.Python
	}
	if cfg.PythonApp == "" {
		cfg.PythonApp = defaults.PythonApp
	}
	if cfg.MaxTasks < 1 {
		cfg.MaxTasks = 4
	}
	if cfg.StartURL == "" {
		cfg.StartURL = constants["portal_adapter.TARGET_URL"]
	}
	var err error
	for _, p := range []*string{&cfg.Runtime, &cfg.Downloads, &cfg.TokenFile, &cfg.PythonApp} {
		*p, err = filepath.Abs(*p)
		if err != nil {
			return nil, err
		}
	}
	if !filepath.IsAbs(cfg.Python) && strings.ContainsAny(cfg.Python, `/\`) {
		cfg.Python, err = filepath.Abs(cfg.Python)
		if err != nil {
			return nil, err
		}
	}
	if err = os.MkdirAll(cfg.Runtime, 0700); err != nil {
		return nil, err
	}
	lock, err := lockFile(filepath.Join(cfg.Runtime, "rpa-worker.lock"))
	if err != nil {
		return nil, errors.New("已有浏览器 Worker 持有运行锁")
	}
	token, err := loadToken(cfg.TokenFile)
	if err != nil {
		unlockFile(lock)
		return nil, err
	}
	driver, err := pw.Run()
	if err != nil {
		unlockFile(lock)
		return nil, errors.New("Go Playwright 驱动不可用，请在 backend-go 执行 go run ./cmd/browser-worker --install")
	}
	ctx, cancel := context.WithCancel(ctx)
	m := &Manager{cfg: cfg, tasks: map[string]*Task{}, pw: driver, token: token, lock: lock, ctx: ctx, cancel: cancel}
	if err = m.recover(); err != nil {
		cancel()
		driver.Stop()
		unlockFile(lock)
		return nil, err
	}
	m.wg.Add(1)
	go m.watchdog()
	return m, nil
}
func loadToken(path string) (string, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return "", err
	}
	f, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if e == nil {
		b := make([]byte, 32)
		_, e = rand.Read(b)
		if e == nil {
			_, e = f.WriteString(hex.EncodeToString(b))
		}
		if e == nil {
			e = f.Sync()
		}
		f.Close()
		if e != nil {
			return "", e
		}
	} else if !os.IsExist(e) {
		return "", e
	}
	b, e := os.ReadFile(path)
	if e != nil {
		return "", e
	}
	s := strings.TrimSpace(string(b))
	if len(s) < 32 {
		return "", errors.New("浏览器控制令牌无效")
	}
	return s, nil
}
func (m *Manager) recover() error {
	dir := filepath.Join(m.cfg.Runtime, "rpa-tasks")
	if e := os.MkdirAll(dir, 0700); e != nil {
		return e
	}
	files, e := filepath.Glob(filepath.Join(dir, "*.json"))
	if e != nil {
		return e
	}
	for _, p := range files {
		b, e := os.ReadFile(p)
		var state object
		if e != nil || json.Unmarshal(b, &state) != nil || !validID(str(state["job_id"])) || str(state["job_id"])+".json" != filepath.Base(p) || len(obj(state["target"])) == 0 || str(state["status"]) == "" {
			if e = os.Rename(p, p+".corrupt"); e != nil {
				return e
			}
			continue
		}
		t := newTask(m, state)
		if !terminal(str(state["status"])) {
			state["status"] = "INTERRUPTED"
			state["control_mode"] = "HUMAN"
			state["state_version"] = num(state["state_version"]) + 1
			state["message"] = "Worker 已重启，浏览器会话丢失；请重新启动并登录，导出检查点和动作回执已保留"
			if e = m.persist(state); e != nil {
				return e
			}
		}
		m.tasks[str(state["job_id"])] = t
	}
	return nil
}
func validID(s string) bool { u, e := uuid.Parse(s); return e == nil && u.String() == s }
func terminal(s string) bool {
	return s == "SUCCEEDED" || s == "PARTIAL_SUCCESS" || s == "FAILED" || s == "CANCELLED" || s == "INTERRUPTED"
}
func (m *Manager) persist(s object) error {
	return atomicJSON(filepath.Join(m.cfg.Runtime, "rpa-tasks", str(s["job_id"])+".json"), s)
}
func (m *Manager) task(id string) (*Task, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	t := m.tasks[id]
	if t == nil {
		return nil, problem("TASK_NOT_FOUND", "任务不存在")
	}
	return t, nil
}
func (m *Manager) Start(id string, payload object) (*Task, error) {
	if !validID(id) {
		return nil, errors.New("任务 ID 无效")
	}
	target := obj(payload["target"])
	for _, k := range []string{"course_name", "assignment_name"} {
		v := strings.TrimSpace(str(target[k]))
		if v == "" || len([]rune(v)) > 255 {
			return nil, errors.New("课程和作业名称不能为空")
		}
	}
	kind := str(payload["task_type"])
	session := obj(payload["portal_session"])
	if kind != "" && kind != "AUTH_BOOTSTRAP" && kind != "CATALOG_SYNC" && kind != "HOMEWORK_DETAIL" || kind != "" && len(session) == 0 {
		return nil, errors.New("教务任务类型或会话无效")
	}
	if len(session) > 0 {
		if e := m.fence(session); e != nil {
			return nil, e
		}
		m.mu.Lock()
		others := []*Task{}
		for _, t := range m.tasks {
			if t.id != id {
				others = append(others, t)
			}
		}
		m.mu.Unlock()
		for _, t := range others {
			s := obj(t.snapshot()["portal_session"])
			if s["connection_id"] == session["connection_id"] && (num(s["profile_generation"]) < num(session["profile_generation"]) || num(s["credential_version"]) < num(session["credential_version"])) {
				t.Stop("CANCELLED")
			}
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	old := m.tasks[id]
	restart := flag(payload["restart"])
	if old != nil && !restart {
		if hash(old.snapshot()["target"]) != hash(target) {
			return nil, errors.New("任务 ID 已用于其他目标")
		}
		return old, nil
	}
	prior := object{}
	if old != nil {
		prior = old.snapshot()
	} else if restart {
		prior = obj(payload["checkpoint"])
	}
	if restart && !contains(str(prior["status"]), "INTERRUPTED", "FAILED", "PARTIAL_SUCCESS") {
		return nil, errors.New("只有中断或失败的任务可以重新启动")
	}
	active := 0
	for _, t := range m.tasks {
		if !terminal(str(t.snapshot()["status"])) {
			active++
		}
	}
	if active >= m.cfg.MaxTasks {
		return nil, errors.New("浏览器任务已满，请等待已有任务结束")
	}
	state := clone(prior)
	// Start/restart creates a new browser and login session. Its challenge
	// budget must not inherit failures from the browser that was closed.
	// Resume on the same browser deliberately keeps its existing budget.
	state["captcha_attempts"], state["last_captcha_result"] = 0, nil
	for k, v := range (object{"job_id": id, "target": target, "task_type": kind, "status": "RUNNING", "stage": "OPEN_PORTAL", "control_mode": "AUTO", "control_epoch": num(prior["control_epoch"]) + 1, "state_version": num(prior["state_version"]) + 1, "updated_at": now(), "session_started_at": now(), "auth_status": "UNKNOWN", "waiting_reason": "", "login_code_method": "", "error_code": nil, "startup_attempt": 0, "startup_result": nil, "startup_round": 0, "portal_navigation_attempts": 0, "semester_lookup": object{}, "message": "正在启动 Go 任务浏览器", "last_error": nil, "target_resolution": nil, "startup_retry_kind": "", "startup_retry_at": nil}) {
		state[k] = v
	}
	if state["created_at"] == nil {
		state["created_at"] = now()
	}
	files := []any{}
	for _, f := range objects(prior["files"]) {
		if str(f["status"]) == "VERIFIED" {
			files = append(files, f)
		}
	}
	state["files"] = files
	if state["exports"] == nil {
		state["exports"] = []any{}
	}
	// A new default class cannot inherit a generic previous "this class" scope.
	scope := obj(state["requested_scope"])
	if str(scope["mode"]) == "selected" {
		for _, ref := range array(scope["class_ids"]) {
			if contains(str(obj(scope["option_names"])[str(ref)]), "本班级", "当前班级") {
				state["requested_scope"] = nil
				break
			}
		}
	}
	t := newTask(m, state)
	if len(session) > 0 {
		meta := object{}
		for _, k := range []string{"connection_id", "profile_generation", "credential_version", "trust_device", "identity_verified"} {
			meta[k] = session[k]
		}
		state["portal_session"] = meta
		t.ticket = str(session["credential_ticket"])
	}
	if e := m.persist(state); e != nil {
		return nil, e
	}
	m.tasks[id] = t
	m.wg.Add(1)
	go func() { defer m.wg.Done(); t.run() }()
	return t, nil
}
func (m *Manager) watchdog() {
	defer m.wg.Done()
	timer := time.NewTicker(5 * time.Second)
	defer timer.Stop()
	for {
		select {
		case <-m.ctx.Done():
			return
		case <-timer.C:
			m.mu.Lock()
			tasks := []*Task{}
			for _, t := range m.tasks {
				tasks = append(tasks, t)
			}
			m.mu.Unlock()
			for _, t := range tasks {
				s := t.snapshot()
				if terminal(str(s["status"])) {
					continue
				}
				if now()-num(s["session_started_at"]) > 7200 || str(s["control_mode"]) == "HUMAN" && now()-float64(t.activity.Load()) > 900 {
					t.Stop("INTERRUPTED")
				}
			}
		}
	}
}
func (m *Manager) Close() error {
	m.cancel()
	m.mu.Lock()
	tasks := []*Task{}
	for _, t := range m.tasks {
		tasks = append(tasks, t)
	}
	m.mu.Unlock()
	for _, t := range tasks {
		t.Stop("INTERRUPTED")
	}
	m.wg.Wait()
	err := m.pw.Stop()
	unlockFile(m.lock)
	return err
}
func (m *Manager) Authorized(r *http.Request) bool {
	return subtle.ConstantTimeCompare([]byte(r.Header.Get("X-RPA-Token")), []byte(m.token)) == 1
}
func (m *Manager) takeCredentials(ticket string) (object, error) {
	if len(ticket) != 64 {
		return nil, problem("CREDENTIAL_TICKET_INVALID", "认证票据无效，请重新认证")
	}
	socket := env("PORTAL_CREDENTIAL_SOCKET", filepath.Join(m.cfg.Runtime, "portal-credentials.sock"))
	transport := &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return (&net.Dialer{Timeout: 10 * time.Second}).DialContext(ctx, "unix", socket)
	}}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, Timeout: 10 * time.Second}
	r, _ := http.NewRequestWithContext(m.ctx, "POST", "http://localhost/credentials", nil)
	r.Header.Set("X-Portal-Ticket", ticket)
	res, e := client.Do(r)
	if e != nil {
		return nil, problem("CREDENTIAL_TICKET_EXPIRED", "认证票据不可用，请重新认证")
	}
	defer res.Body.Close()
	value := object{}
	if res.StatusCode != 200 || json.NewDecoder(http.MaxBytesReader(nil, res.Body, 16384)).Decode(&value) != nil || str(value["username"]) == "" || str(value["password"]) == "" {
		return nil, problem("CREDENTIAL_UNAVAILABLE", "教务凭据不可用，请重新认证")
	}
	return value, nil
}
