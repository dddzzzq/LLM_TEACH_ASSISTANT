package browserworker

import (
	"net/url"
	"regexp"
	"strings"

	pw "github.com/mxschmitt/playwright-go"
)

const defaultIdentitySelector = "[data-teacher-account],[data-login-account],#teacherAccount"

var portalLoginName = regexp.MustCompile(`(?i)^(?:登\s*录|log\s*in|sign\s*in)$`)
var portalLogoutName = regexp.MustCompile(`(?i)^(?:退出(?:登录)?|登出|log\s*out|sign\s*out)$`)

func learningPortal(value string) bool {
	u, err := url.Parse(value)
	return err == nil && allowedURL(value) && strings.EqualFold(u.Hostname(), "learning.xidian.edu.cn")
}

// A public portal contains course links and "个人空间" before login. These
// navigation items are never session evidence. Use its real login controls to
// open SSO, including layouts that no longer use the legacy .denglu class.
func portalLoginEntries(page pw.Page) []pw.Locator {
	entries := []pw.Locator{}
	if page == nil || page.IsClosed() || !allowedURL(page.URL()) {
		return entries
	}
	for _, frame := range page.Frames() {
		if !learningPortal(frame.URL()) {
			continue
		}
		legacy := frame.Locator(".denglu:visible")
		if n := count(legacy); n > 0 {
			for i := 0; i < n; i++ {
				entries = append(entries, legacy.Nth(i))
			}
			continue
		}
		for _, role := range []pw.AriaRole{"link", "button"} {
			controls := frame.GetByRole(role, pw.FrameGetByRoleOptions{Name: portalLoginName})
			for i := 0; i < count(controls); i++ {
				if visible(controls.Nth(i)) {
					entries = append(entries, controls.Nth(i))
				}
			}
		}
	}
	return entries
}

func (t *Task) loginChallengeRequired() bool {
	if t.context == nil {
		return true
	}
	for _, page := range t.context.Pages() {
		if page.IsClosed() {
			continue
		}
		for _, frame := range page.Frames() {
			if !allowedURL(frame.URL()) {
				continue
			}
			if count(frame.Locator("input[type=password]:visible,#sliderDiv:visible,input#captcha:visible")) > 0 {
				return true
			}
			if loginOrigin(frame.URL()) && count(frame.Locator(constants["portal_auth.OTP"])) > 0 {
				return true
			}
		}
	}
	return false
}

func (t *Task) loginRequired() bool {
	if t.loginChallengeRequired() {
		return true
	}
	// A separate, authenticated course tab may coexist with the old public
	// landing page. Its stale login button must not interrupt that session.
	if t.authenticatedPage() != nil {
		return false
	}
	for _, page := range t.context.Pages() {
		if len(portalLoginEntries(page)) > 0 {
			return true
		}
	}
	return false
}

func portalSessionEvidence(frame pw.Frame) bool {
	identity := frame.Locator(env("PORTAL_IDENTITY_SELECTOR", defaultIdentitySelector))
	if count(identity) == 1 {
		value, err := identity.Evaluate(script("auth_identity"), nil)
		if err == nil && str(value) != "" {
			return true
		}
	}
	for _, role := range []pw.AriaRole{"link", "button"} {
		controls := frame.GetByRole(role, pw.FrameGetByRoleOptions{Name: portalLogoutName})
		for i := 0; i < count(controls); i++ {
			if visible(controls.Nth(i)) {
				return true
			}
		}
	}
	return false
}

func (t *Task) authenticated() bool {
	page := t.authenticatedPage()
	if page == nil {
		return false
	}
	t.page = page
	return true
}

func (t *Task) authenticatedPage() pw.Page {
	if t.context == nil || t.loginChallengeRequired() {
		return nil
	}
	pages := append([]pw.Page{t.page}, t.context.Pages()...)
	for _, page := range pages {
		if page == nil || page.IsClosed() || !allowedURL(page.URL()) || len(portalLoginEntries(page)) > 0 {
			continue
		}
		for _, frame := range page.Frames() {
			if !allowedURL(frame.URL()) {
				continue
			}
			if learningPortal(frame.URL()) {
				if portalSessionEvidence(frame) {
					return page
				}
				continue
			}
			u, err := url.Parse(frame.URL())
			if err != nil {
				continue
			}
			host := strings.ToLower(u.Hostname())
			if host != "chaoxing.com" && !strings.HasSuffix(host, ".chaoxing.com") && host != "chaoxing.com.cn" && !strings.HasSuffix(host, ".chaoxing.com.cn") {
				continue
			}
			if count(frame.Locator("a[title='作业']:visible,[cname]:visible,.myde_course_item:visible,[data-courseid]:visible,[data-course-id]:visible,a[href*='courseid=']:visible")) > 0 {
				return page
			}
		}
	}
	return nil
}
