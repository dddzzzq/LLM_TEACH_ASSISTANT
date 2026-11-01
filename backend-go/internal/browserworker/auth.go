package browserworker

import (
	"context"
	"crypto/sha256"
	"errors"
	"net/url"
	"regexp"
	"strings"
	"time"

	pw "github.com/mxschmitt/playwright-go"
)

type authentication struct {
	task                     *Task
	credentials              object
	trust, filled, confirmed bool
	lastRequest, lastSubmit  time.Time
	submittedDigest          [32]byte
}

var codeRequest = regexp.MustCompile(`(?i)^(?:(?:重新)?(?:获取|发送)(?:短信)?验证码|重新发送|获取动态码|发送动态码|(?:get|send|request|resend)(?:\s+(?:a|the))?\s+(?:(?:sms|verification|security)\s+)?code|resend)$`)
var codeCountdown = regexp.MustCompile(`(?i)^(?:(?:重新获取|重新发送|获取验证码|发送验证码|resend(?:\s+code)?(?:\s+in)?)\s*)?[（(]?\s*\d+\s*(?:秒|s|seconds?)\s*[）)]?\s*(?:后(?:重新)?(?:获取|发送))?$`)
var trustTitle = regexp.MustCompile(`(?i)可信(?:浏览器|设备)|trust(?:ed)?\s+(?:(?:this|your)\s+)?(?:browser|device)`)
var codeInvalid = regexp.MustCompile(`(?i)验证码错误|验证码无效|验证码已过期|verification\s+code\s+(?:error|invalid|expired)|(?:invalid|incorrect|expired)\s+(?:(?:verification|security)\s+code|captcha)`)

func origin(value string) string {
	u, e := url.Parse(value)
	if e != nil || u.Scheme != "https" || u.Hostname() == "" || u.User != nil {
		return ""
	}
	host := strings.ToLower(u.Hostname())
	if u.Port() != "" && u.Port() != "443" {
		host += ":" + u.Port()
	}
	return "https://" + host
}
func loginOrigin(value string) bool {
	actual := origin(value)
	if actual == "" {
		return false
	}
	for _, value := range strings.Split(env("PORTAL_LOGIN_ORIGINS", "https://ids.xidian.edu.cn"), ",") {
		if origin(strings.TrimSpace(value)) == actual {
			return true
		}
	}
	return false
}
func namedControls(f pw.Frame, pattern *regexp.Regexp) []pw.Locator {
	out := []pw.Locator{}
	controls := f.Locator("button:visible,a:visible,input[type=button]:visible,input[type=submit]:visible,[role=button]:visible,[onclick]:visible")
	n, e := controls.Count()
	if e != nil {
		return out
	}
	for i := 0; i < n; i++ {
		l := controls.Nth(i)
		v, e := l.Evaluate(script("auth_control_label"), nil)
		if e == nil && pattern.MatchString(str(v)) {
			out = append(out, l)
		}
	}
	if len(out) == 0 {
		all := f.GetByText(pattern)
		for i := 0; i < count(all); i++ {
			if visible(all.Nth(i)) {
				out = append(out, all.Nth(i))
			}
		}
	}
	return out
}
func (a *authentication) find(kind string) pw.Frame {
	for _, p := range a.task.context.Pages() {
		if p.IsClosed() || !loginOrigin(p.URL()) {
			continue
		}
		for _, f := range p.Frames() {
			if !loginOrigin(f.URL()) {
				continue
			}
			switch kind {
			case "login":
				if count(f.Locator(constants["portal_auth.PASSWORD"])) > 0 {
					a.task.page = p
					return f
				}
			case "otp":
				if count(f.Locator(constants["portal_auth.PASSWORD"])) == 0 && count(f.Locator(constants["portal_auth.OTP"])) == 1 {
					a.task.page = p
					return f
				}
			case "trust":
				loc := f.GetByText(trustTitle)
				for i := 0; i < count(loc); i++ {
					if visible(loc.Nth(i)) {
						a.task.page = p
						return f
					}
				}
			}
		}
	}
	return nil
}
func (a *authentication) method(f pw.Frame) string {
	if f != nil {
		pattern := regexp.MustCompile(`(?i)dynamic\s*code\s*[（(]?\s*WeChat\s*Work|(?:企业微信|WeChat\s*Work).{0,32}(?:动态(?:验证码|码)?|dynamic\s*code)|(?:动态验证码|动态码).{0,20}(?:企业微信|WeChat\s*Work)`)
		loc := f.GetByText(pattern)
		for i := 0; i < count(loc); i++ {
			if visible(loc.Nth(i)) {
				return "wechat_work"
			}
		}
	}
	return ""
}
func (a *authentication) observation() (object, error) {
	login, otp, trust := a.find("login"), a.find("otp"), a.find("trust")
	entered := false
	requested := !a.lastRequest.IsZero()
	if login != nil && a.filled {
		wrong := login.GetByText(regexp.MustCompile(`(?i)密码错误|用户名或密码错误|账号或密码错误|账号被锁定|incorrect (?:username|password)|invalid (?:username|password)|account.{0,20}locked`))
		if count(wrong) > 0 && visible(wrong.First()) {
			return nil, problem("INVALID_CREDENTIALS", "官网提示账号或密码错误，请修改设置后重新连接")
		}
	}
	if login != nil {
		invalid := login.GetByText(codeInvalid)
		if count(invalid) > 0 && visible(invalid.First()) {
			return nil, problem("LOGIN_VERIFICATION_FAILED", "官网提示验证码校验失败，请检查当前登录验证")
		}
	}
	if otp != nil {
		value, e := otp.Locator(constants["portal_auth.OTP"]).InputValue()
		if e != nil {
			return nil, e
		}
		entered = value != ""
		invalid := otp.GetByText(codeInvalid)
		if count(invalid) > 0 && visible(invalid.First()) && (a.submittedDigest == [32]byte{} || !entered || a.submittedDigest == sha256.Sum256([]byte(value))) {
			return nil, problem("LOGIN_VERIFICATION_FAILED", "官网提示验证码无效或已过期，请核对验证码")
		}
		requested = requested || len(namedControls(otp, codeCountdown)) > 0
	}
	return object{"credentials_saved": true, "credentials_filled": a.filled, "login_form": login != nil, "otp_required": otp != nil, "otp_entered": entered, "code_method": a.method(otp), "code_requested": requested, "trust_prompt": trust != nil, "trust_device": a.trust, "authenticated": false}, nil
}
func (a *authentication) fill() error {
	f := a.find("login")
	if f == nil || a.filled {
		return problem("LOGIN_ALREADY_SUBMITTED", "当前不可重复填写账号密码，请根据页面继续认证")
	}
	user := f.Locator(env("PORTAL_USERNAME_SELECTOR", constants["portal_auth.USERNAME"]))
	password := f.Locator(constants["portal_auth.PASSWORD"])
	if count(user) != 1 || count(password) != 1 {
		return errors.New("未找到唯一的登录表单")
	}
	v, e := password.Evaluate(script("auth_form_origin"), nil)
	if e != nil {
		return e
	}
	if !loginOrigin(str(v)) {
		return problem("LOGIN_ORIGIN_REJECTED", "登录表单来源未获授权，已停止填写")
	}
	_ = a.task.update(object{"stage": "FILLING_CREDENTIALS", "message": "正在填写已保存的教务账号"})
	preflight, e := f.Evaluate("typeof window.checkNeedCaptcha === 'function'")
	if e != nil {
		return e
	}
	fill := func() error {
		if e := user.Fill(str(a.credentials["username"])); e != nil {
			return e
		}
		return password.Fill(str(a.credentials["password"]))
	}
	if flag(preflight) {
		res, e := f.Page().ExpectResponse(func(res pw.Response) bool {
			u, e := url.Parse(res.URL())
			return e == nil && origin(res.URL()) == origin(f.URL()) && u.Path == "/authserver/checkNeedCaptcha.htl"
		}, fill, pw.PageExpectResponseOptions{Timeout: pw.Float(5000)})
		if e != nil {
			return problem("LOGIN_CHECK_TIMEOUT", "官网登录验证检查超时，尚未提交账号密码")
		}
		var data object
		if e = res.JSON(&data); e != nil || !res.Ok() {
			return problem("LOGIN_CHECK_FAILED", "官网登录验证检查未正常完成")
		}
		expected, ok := data["isNeed"].(bool)
		if !ok {
			return problem("LOGIN_CHECK_FAILED", "官网登录验证检查返回无效结果")
		}
		deadline := time.Now().Add(3 * time.Second)
		for {
			v, e = f.Evaluate("expected=>typeof window.needCaptcha==='boolean'&&window.needCaptcha===expected", expected)
			if e == nil && flag(v) {
				break
			}
			if time.Now().After(deadline) {
				return problem("LOGIN_CHECK_TIMEOUT", "官网登录验证检查尚未完成")
			}
			time.Sleep(50 * time.Millisecond)
		}
	} else if e = fill(); e != nil {
		return e
	}
	a.filled = true
	a.confirmed = true
	if count(f.Locator("#sliderDiv:visible")) > 0 {
		return nil
	}
	submit := f.Locator(env("PORTAL_LOGIN_SUBMIT_SELECTOR", "#login_submit:visible,#login_submit_btn:visible,button[type=submit]:visible,input[type=submit]:visible"))
	if count(submit) == 1 {
		return submit.Click()
	}
	buttons := namedControls(f, regexp.MustCompile(`^登录$`))
	if len(buttons) == 1 {
		return buttons[0].Click()
	}
	return problem("LOGIN_SUBMIT_UNAVAILABLE", "未找到唯一的登录按钮，请核对认证页面")
}
func (a *authentication) requestCode() error {
	f := a.find("otp")
	if f == nil {
		return problem("LOGIN_CODE_PAGE_MISSING", "当前未找到官网验证码输入页")
	}
	if !a.lastRequest.IsZero() && time.Since(a.lastRequest) < time.Minute || len(namedControls(f, codeCountdown)) > 0 {
		return f.Locator(constants["portal_auth.OTP"]).Focus()
	}
	dynamic := f.Locator("#getDynamicCode:visible")
	_ = dynamic.WaitFor(pw.LocatorWaitForOptions{State: pw.WaitForSelectorStateVisible, Timeout: pw.Float(5000)})
	buttons := namedControls(f, codeRequest)
	if count(dynamic) == 1 {
		buttons = []pw.Locator{dynamic}
	}
	if len(buttons) != 1 {
		return problem("LOGIN_CODE_BUTTON_UNAVAILABLE", "未找到唯一可用的获取验证码按钮，请核对官网")
	}
	if !enabled(buttons[0]) {
		return problem("LOGIN_CODE_SEND_DISABLED", "官网获取验证码按钮暂不可用，请检查页面提示后继续")
	}
	a.lastRequest = time.Now()
	if e := buttons[0].Click(pw.LocatorClickOptions{Timeout: pw.Float(5000)}); e != nil {
		return problem("LOGIN_CODE_REQUEST_UNCONFIRMED", "获取验证码操作未确认，请核对官网是否已发送")
	}
	return f.Locator(constants["portal_auth.OTP"]).Focus()
}
func (a *authentication) trustDevice() error {
	f := a.find("trust")
	if f == nil {
		return errors.New("当前未出现可信设备弹窗")
	}
	pattern := regexp.MustCompile(`(?i)^(?:仅本次登录|暂不信任|不信任|(?:only\s+)?(?:this\s+time|log\s*in\s+once)|do\s+not\s+trust|don['’]t\s+trust(?:\s+this\s+(?:device|browser))?|not\s+now|skip)\s*[.!。]?$`)
	if a.trust {
		pattern = regexp.MustCompile(`(?i)^(?:信任此设备|信任此浏览器|trust\s+(?:this\s+)?(?:device|browser))\s*[?.！!。]?$`)
	}
	buttons := namedControls(f, pattern)
	if len(buttons) != 1 || !enabled(buttons[0]) {
		return problem("DEVICE_TRUST_UNAVAILABLE", "官网可信设备选项暂不可用，请核对弹窗")
	}
	_ = a.task.update(object{"stage": "TRUST_DECISION", "message": "正在应用教师的可信设备设置"})
	return buttons[0].Click(pw.LocatorClickOptions{Timeout: pw.Float(5000)})
}
func (a *authentication) identity() (bool, error) {
	selector := env("PORTAL_IDENTITY_SELECTOR", defaultIdentitySelector)
	for _, p := range a.task.context.Pages() {
		if p.IsClosed() || !resourceURL(p.URL()) {
			continue
		}
		for _, f := range p.Frames() {
			if !resourceURL(f.URL()) {
				continue
			}
			loc := f.Locator(selector)
			if count(loc) == 1 {
				v, e := loc.Evaluate(script("auth_identity"), nil)
				if e != nil {
					return false, e
				}
				if str(v) != "" {
					if str(v) != str(a.credentials["username"]) {
						return false, problem("IDENTITY_MISMATCH", "当前官网账号与绑定账号不一致，请断开后重新连接")
					}
					a.confirmed = true
					return true, nil
				}
			}
		}
	}
	return a.confirmed, nil
}
func (a *authentication) fillCode(value string, page pw.Page) error {
	if !regexp.MustCompile(`^[0-9]{4,12}$`).MatchString(value) {
		return problem("LOGIN_CODE_FORMAT", "请输入 4 至 12 位数字验证码")
	}
	f := a.find("otp")
	if f == nil || page != nil && f.Page() != page {
		return problem("LOGIN_CODE_PAGE_MISSING", "验证码页面已变化，请核对当前认证页面")
	}
	return f.Locator(constants["portal_auth.OTP"]).Fill(value)
}
func (a *authentication) submitCode() error {
	f := a.find("otp")
	if f == nil {
		return problem("LOGIN_CODE_PAGE_MISSING", "当前未找到官网验证码输入页")
	}
	value, e := f.Locator(constants["portal_auth.OTP"]).InputValue()
	if e != nil {
		return e
	}
	if !regexp.MustCompile(`^[0-9]{4,12}$`).MatchString(value) {
		return problem("LOGIN_CODE_FORMAT", "请输入 4 至 12 位数字验证码")
	}
	digest := sha256.Sum256([]byte(value))
	if digest == a.submittedDigest || time.Since(a.lastSubmit) < 3*time.Second {
		return problem("LOGIN_CODE_PENDING", "已提交验证码，请等待官网校验结果")
	}
	buttons := namedControls(f, regexp.MustCompile(`(?i)^(?:登录|验证并登录|log\s*in|sign\s*in|verify(?:\s+and\s+(?:log|sign)\s*in)?)$`))
	if len(buttons) != 1 {
		return problem("LOGIN_CODE_SUBMIT_UNAVAILABLE", "未找到唯一可见的验证码提交按钮")
	}
	a.submittedDigest = digest
	a.lastSubmit = time.Now()
	if err := buttons[0].Click(pw.LocatorClickOptions{Timeout: pw.Float(5000)}); err != nil {
		return err
	}
	// School handlers can consume the field asynchronously. Leave it intact,
	// observe the result, and never replay the same code after an uncertain click.
	for i := 0; i < 30; i++ {
		if a.find("trust") != nil || a.find("otp") == nil {
			return nil
		}
		invalid := f.GetByText(codeInvalid)
		if count(invalid) > 0 && visible(invalid.First()) {
			return problem("LOGIN_VERIFICATION_FAILED", "官网提示验证码无效或已过期，请重新核对验证码")
		}
		select {
		case <-a.task.ctx.Done():
			return context.Canceled
		case <-time.After(100 * time.Millisecond):
		}
		if !a.task.automatic() {
			return nil
		}
	}
	return nil
}
func (t *Task) waitCode(method string) {
	_ = t.update(object{"stage": "AUTHENTICATE", "login_code_method": method})
	message := "验证码已请求，请在任务浏览器输入本人收到的验证码，然后交还自动化"
	if method == "wechat_work" {
		message = "验证码已请求，请在企业微信查看本人动态验证码，在任务浏览器输入后交还自动化"
	}
	t.human("LOGIN_CODE", message)
}

func (t *Task) handoffLoginError(err error) bool {
	var e *issue
	if !errors.As(err, &e) {
		return false
	}
	reason := ""
	switch {
	case e.Code == "LOGIN_CODE_FORMAT", e.Code == "LOGIN_VERIFICATION_FAILED":
		reason = "LOGIN_CODE"
	case e.Code == "LOGIN_CODE_PENDING":
		reason = "LOGIN_CODE_RESULT"
	case strings.HasPrefix(e.Code, "LOGIN_CODE_"), strings.HasPrefix(e.Code, "DEVICE_TRUST_"):
		reason = "LOGIN_CODE_REQUEST"
	case e.Code == "TARGET_AMBIGUOUS", e.Code == "IDENTITY_MISMATCH", e.Code == "INVALID_CREDENTIALS":
		reason = e.Code
	}
	if reason == "" {
		return false
	}
	t.human(reason, e.Message)
	return true
}
