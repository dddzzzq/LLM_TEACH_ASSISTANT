package browserworker

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"time"

	pw "github.com/mxschmitt/playwright-go"
)

type startupPolicy struct {
	attempt, total    time.Duration
	backoff, cooldown []time.Duration
	poll              time.Duration
}

func defaultStartupPolicy() startupPolicy {
	return startupPolicy{25 * time.Second, 420 * time.Second, []time.Duration{5 * time.Second, 15 * time.Second, 30 * time.Second}, []time.Duration{60 * time.Second}, 200 * time.Millisecond}
}

var networkCode = regexp.MustCompile("net::(ERR_[A-Z_]+)")

func startupRetryable(err error, status int) bool {
	if status == 408 || status == 429 || status >= 500 {
		return true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	if errors.Is(err, pw.ErrTimeout) {
		return true
	}
	for _, match := range networkCode.FindAllStringSubmatch(err.Error(), -1) {
		if containsExact(match[1], "ERR_EMPTY_RESPONSE", "ERR_CONNECTION_RESET", "ERR_CONNECTION_CLOSED", "ERR_CONNECTION_TIMED_OUT", "ERR_TIMED_OUT", "ERR_NAME_NOT_RESOLVED", "ERR_NETWORK_CHANGED", "ERR_INTERNET_DISCONNECTED", "ERR_CONNECTION_REFUSED", "ERR_PROXY_CONNECTION_FAILED", "ERR_TUNNEL_CONNECTION_FAILED", "ERR_ABORTED") {
			return true
		}
	}
	return false
}

func (t *Task) cancelStartup() {
	t.startupMu.Lock()
	defer t.startupMu.Unlock()
	if t.startupCancel != nil {
		t.startupCancel()
	}
}

type startupRun struct {
	task        *Task
	ctx         context.Context
	epoch       float64
	phase       string
	httpStatus  int
	detailTried bool
	policy      startupPolicy
}

func (s *startupRun) owned() bool {
	state := s.task.snapshot()
	return s.ctx.Err() == nil && str(state["status"]) == "RUNNING" && str(state["control_mode"]) == "AUTO" && num(state["control_epoch"]) == s.epoch
}

func (s *startupRun) probe() string {
	if !s.owned() {
		return ""
	}
	t := s.task
	t.op.Lock()
	defer t.op.Unlock()
	if !s.owned() {
		return ""
	}
	if t.authenticated() {
		return "AUTHENTICATED"
	}
	if t.usable() {
		return "REQUIRED"
	}
	return ""
}

func (s *startupRun) sleep(delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-s.ctx.Done():
		return false
	case <-timer.C:
		return s.owned()
	}
}

func startupTimeout(ctx context.Context, limit time.Duration) float64 {
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < limit {
		limit = time.Until(deadline)
	}
	if limit < time.Millisecond {
		limit = time.Millisecond
	}
	return float64(limit.Milliseconds())
}

// Navigation RPCs do not hold the task operation mutex. Pause, frames and
// cancellation remain available while the school is slow or replaces a frame.
// Ownership is checked again before every subsequent external action.
func (s *startupRun) navigate(ctx context.Context) (string, error) {
	t := s.task
	if ready := s.probe(); ready != "" {
		return ready, nil
	}
	if !s.owned() {
		return "", context.Canceled
	}
	state := t.snapshot()
	t.op.Lock()
	page := t.page
	t.op.Unlock()
	if str(state["task_type"]) == "HOMEWORK_DETAIL" && !s.detailTried {
		s.detailTried = true
		entry := str(obj(obj(state["target"])["portal"])["course_url"])
		if portalURL(entry) {
			response, err := page.Goto(entry, pw.PageGotoOptions{WaitUntil: pw.WaitUntilStateDomcontentloaded, Timeout: pw.Float(startupTimeout(ctx, 10*time.Second))})
			if !s.owned() {
				return "", context.Canceled
			}
			if err == nil && response != nil && response.Status() < 400 {
				if ready := s.probe(); ready != "" {
					return ready, nil
				}
			}
		}
	}
	s.phase = "portal_load"
	t.op.Lock()
	existing := len(portalLoginEntries(page)) == 1
	t.op.Unlock()
	if !existing {
		response, err := page.Goto(t.manager.cfg.StartURL, pw.PageGotoOptions{WaitUntil: pw.WaitUntilStateDomcontentloaded, Timeout: pw.Float(startupTimeout(ctx, 20*time.Second))})
		if !s.owned() {
			return "", context.Canceled
		}
		if err != nil {
			return "", err
		}
		if response != nil {
			s.httpStatus = response.Status()
		}
		if s.httpStatus >= 400 {
			return "", problem("PORTAL_HTTP_ERROR", "教学门户暂未返回正常页面")
		}
	}
	clicked := false
	for s.owned() {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		if ready := s.probe(); ready != "" {
			return ready, nil
		}
		if !clicked {
			t.op.Lock()
			entries := portalLoginEntries(page)
			t.op.Unlock()
			if len(entries) > 1 {
				return "", problem("LOGIN_ENTRY_AMBIGUOUS", "门户存在多个登录入口，请核对当前页面")
			}
			if len(entries) == 1 {
				if !s.owned() {
					return "", context.Canceled
				}
				s.phase = "login_redirect"
				if err := entries[0].Click(pw.LocatorClickOptions{Timeout: pw.Float(startupTimeout(ctx, 10*time.Second))}); err != nil {
					return "", err
				}
				clicked = true
			}
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(s.policy.poll):
		}
	}
	return "", context.Canceled
}

func (s *startupRun) succeeded(ready string, attempt int, started time.Time) {
	t := s.task
	t.op.Lock()
	defer t.op.Unlock()
	if !s.owned() {
		return
	}
	t.startupPending.Store(false)
	t.starting.Store(false)
	stage := "LOCATE_COURSE"
	if t.auth != nil {
		stage = "AUTHENTICATE"
	}
	_ = t.update(object{"startup_result": object{"attempts": attempt, "elapsed_seconds": time.Since(started).Seconds(), "auth_status": ready}, "startup_retry_kind": "", "startup_retry_at": nil, "auth_status": ready, "stage": stage, "error_code": nil, "last_error": nil})
	if t.auth == nil {
		if ready == "AUTHENTICATED" {
			t.human("LOGIN_COMPLETE", "官网登录已完成，请交还自动化继续")
		} else {
			t.human("LOGIN", "请在浏览器中完成登录，之后交还自动化")
		}
	}
}

func (t *Task) startup() {
	policy := defaultStartupPolicy()
	if t.manager.cfg.startup != nil {
		policy = *t.manager.cfg.startup
	}
	ctx, cancel := context.WithTimeout(t.ctx, policy.total)
	t.startupMu.Lock()
	if t.startupCancel != nil {
		t.startupCancel()
	}
	t.startupGeneration++
	generation := t.startupGeneration
	t.startupCancel = cancel
	t.starting.Store(true)
	t.startupMu.Unlock()
	defer func() {
		cancel()
		t.startupMu.Lock()
		defer t.startupMu.Unlock()
		if t.startupGeneration == generation {
			t.starting.Store(false)
			t.startupCancel = nil
		}
	}()
	s := &startupRun{task: t, ctx: ctx, epoch: num(t.snapshot()["control_epoch"]), policy: policy}
	started := time.Now()
	attempts := (len(policy.backoff) + 1) * (len(policy.cooldown) + 1)
	for index := 0; index < attempts && s.owned(); index++ {
		round, attempt := index/(len(policy.backoff)+1), index%(len(policy.backoff)+1)
		s.phase, s.httpStatus = "portal_load", 0
		_ = t.update(object{"startup_attempt": index + 1, "startup_round": round + 1, "startup_attempt_in_round": attempt + 1, "startup_max_attempts": attempts})
		attemptStarted := time.Now()
		attemptCtx, stop := context.WithTimeout(ctx, policy.attempt)
		ready, err := s.navigate(attemptCtx)
		stop()
		if !s.owned() {
			break
		}
		// A browser timeout can arrive after the login DOM has already appeared.
		if ready == "" {
			ready = s.probe()
		}
		if ready != "" {
			s.succeeded(ready, index+1, started)
			return
		}
		if err == nil {
			err = context.DeadlineExceeded
		}
		retryable := startupRetryable(err, s.httpStatus)
		codes := []any{}
		for _, match := range networkCode.FindAllStringSubmatch(err.Error(), -1) {
			codes = append(codes, match[1])
		}
		t.op.Lock()
		host, _ := url.Parse(t.manager.cfg.StartURL)
		if s.phase == "login_redirect" && t.page != nil {
			host, _ = url.Parse(t.page.URL())
		}
		exception := fmt.Sprintf("%T", err)
		if errors.Is(err, pw.ErrTimeout) || errors.Is(err, context.DeadlineExceeded) {
			exception = "TimeoutError"
		}
		t.failure(&issue{Code: "BROWSER_STARTUP_FAILED", Message: "官网登录页面暂未加载完成", Evidence: object{"phase": s.phase, "attempt": index + 1, "round": round + 1, "attempt_in_round": attempt + 1, "elapsed_ms": time.Since(attemptStarted).Milliseconds(), "host": host.Hostname(), "network_codes": codes, "http_status": s.httpStatus, "retryable": retryable, "exception_type": exception}}, "startup:"+s.phase)
		t.op.Unlock()
		if !retryable || index+1 == attempts {
			break
		}
		delay, kind := time.Duration(0), "backoff"
		if attempt == len(policy.backoff) {
			delay, kind = policy.cooldown[round], "cooldown"
		} else {
			delay = policy.backoff[attempt]
		}
		if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) <= delay {
			break
		}
		_ = t.update(object{"startup_retry_kind": kind, "startup_retry_at": now() + delay.Seconds()})
		t.event(fmt.Sprintf("官网连接暂未完成，等待 %.0f 秒后继续加载；可随时暂停或取消", delay.Seconds()))
		until := time.Now().Add(delay)
		for time.Now().Before(until) && s.owned() {
			if ready = s.probe(); ready != "" {
				s.succeeded(ready, index+1, started)
				return
			}
			if !s.sleep(min(time.Second, time.Until(until))) {
				break
			}
		}
		_ = t.update(object{"startup_retry_kind": "", "startup_retry_at": nil})
	}
	t.op.Lock()
	defer t.op.Unlock()
	// Deadline expiry ends the budget, while pause/cancel preserves its reason.
	state := t.snapshot()
	if t.ctx.Err() == nil && str(state["control_mode"]) == "AUTO" && num(state["control_epoch"]) == s.epoch && str(state["status"]) == "RUNNING" {
		t.human("BROWSER_STARTUP", "官网登录页加载失败，浏览器已保留，请重新加载登录页重试")
	}
}
