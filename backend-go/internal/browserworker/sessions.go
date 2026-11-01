package browserworker

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"time"

	pw "github.com/mxschmitt/playwright-go"
)

func (m *Manager) fence(session object) error {
	id := str(session["connection_id"])
	g, v := num(session["profile_generation"]), num(session["credential_version"])
	if !validID(id) || g < 1 || v < 1 || g != float64(int64(g)) || v != float64(int64(v)) {
		return errors.New("教务会话版本无效")
	}
	m.sessionsMu.Lock()
	defer m.sessionsMu.Unlock()
	dir := filepath.Join(m.cfg.Runtime, "portal-profiles", id)
	if e := os.MkdirAll(dir, 0700); e != nil {
		return e
	}
	path := filepath.Join(dir, ".fence")
	prior := object{}
	b, e := os.ReadFile(path)
	if e == nil {
		if e = json.Unmarshal(b, &prior); e != nil {
			return e
		}
	} else if !os.IsNotExist(e) {
		return e
	}
	if g < num(prior["generation"]) || v < num(prior["version"]) {
		return problem("SESSION_REVOKED", "教务会话已更新，旧操作不能继续")
	}
	return atomicJSON(path, object{"generation": g, "version": v})
}
func (t *Task) openContext() (pw.BrowserContext, error) {
	m := t.manager
	s := obj(t.snapshot()["portal_session"])
	downloads := filepath.Join(m.cfg.Downloads, t.id, ".browser")
	if e := os.MkdirAll(downloads, 0700); e != nil {
		return nil, e
	}
	args := []string{"--no-sandbox", "--disable-dev-shm-usage"}
	var proxy *pw.Proxy
	if m.cfg.Proxy != "" {
		proxy = &pw.Proxy{Server: m.cfg.Proxy}
	} else {
		args = append(args, "--no-proxy-server")
	}
	if len(s) == 0 {
		browser, e := m.pw.Chromium.Launch(pw.BrowserTypeLaunchOptions{Channel: pw.String("chromium"), Headless: pw.Bool(true), Args: args, Proxy: proxy, DownloadsPath: pw.String(downloads)})
		if e != nil {
			return nil, e
		}
		t.browser = browser
		return browser.NewContext(pw.BrowserNewContextOptions{AcceptDownloads: pw.Bool(true), Viewport: &pw.Size{Width: 1440, Height: 1000}})
	}
	if e := m.fence(s); e != nil {
		return nil, e
	}
	dir := filepath.Join(m.cfg.Runtime, "portal-profiles", str(s["connection_id"]))
	deadline := time.Now().Add(15 * time.Second)
	for {
		lock, e := lockFile(filepath.Join(dir, ".lease"))
		if e == nil {
			t.profileLock = lock
			break
		}
		if time.Now().After(deadline) {
			return nil, problem("PORTAL_BUSY", "该教务账号正在处理其他操作，请稍后重试")
		}
		select {
		case <-t.ctx.Done():
			return nil, t.ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
	if e := m.fence(s); e != nil {
		return nil, e
	}
	generation := strconv.FormatInt(int64(num(s["profile_generation"])), 10)
	// Only numeric generations inside this validated UUID directory are removed.
	entries, e := os.ReadDir(dir)
	if e != nil {
		return nil, e
	}
	for _, entry := range entries {
		if _, e := strconv.ParseUint(entry.Name(), 10, 64); e == nil && entry.Name() != generation && entry.IsDir() {
			if e = os.RemoveAll(filepath.Join(dir, entry.Name())); e != nil {
				return nil, e
			}
		}
	}
	profile := filepath.Join(dir, generation)
	if e = os.MkdirAll(profile, 0700); e != nil {
		return nil, e
	}
	return m.pw.Chromium.LaunchPersistentContext(profile, pw.BrowserTypeLaunchPersistentContextOptions{Channel: pw.String("chromium"), Headless: pw.Bool(true), Viewport: &pw.Size{Width: 1440, Height: 1000}, AcceptDownloads: pw.Bool(true), DownloadsPath: pw.String(downloads), Args: args, Proxy: proxy})
}
func (m *Manager) deleteProfile(id string) error {
	if !validID(id) {
		return errors.New("教务连接 ID 无效")
	}
	m.mu.Lock()
	tasks := []*Task{}
	for _, t := range m.tasks {
		if str(obj(t.snapshot()["portal_session"])["connection_id"]) == id {
			tasks = append(tasks, t)
		}
	}
	m.mu.Unlock()
	for _, t := range tasks {
		t.Stop("CANCELLED")
	}
	dir := filepath.Join(m.cfg.Runtime, "portal-profiles", id)
	if _, e := os.Stat(dir); os.IsNotExist(e) {
		return nil
	}
	f, e := lockFile(filepath.Join(dir, ".lease"))
	if e != nil {
		return problem("PORTAL_BUSY", "专用浏览器仍在使用，请稍后重试")
	}
	defer unlockFile(f)
	entries, e := os.ReadDir(dir)
	if e != nil {
		return e
	}
	for _, entry := range entries {
		if entry.Name() == ".lease" || entry.Name() == ".fence" {
			continue
		}
		if e = os.RemoveAll(filepath.Join(dir, entry.Name())); e != nil {
			return e
		}
	}
	return nil
}
