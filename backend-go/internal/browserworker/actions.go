package browserworker

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	pw "github.com/mxschmitt/playwright-go"
)

var semester = regexp.MustCompile(`^\d{4}\s*[-—–/]\s*\d{4}\s*第[一二三123]学期$`)
var forbidden = regexp.MustCompile(`(?i)删除|发布|评分|保存成绩|提交成绩|修改|编辑|新建|创建|退课|退出登录|logout|delete|publish`)

func regexpForEmpty(word string) *regexp.Regexp {
	return regexp.MustCompile(`(?:暂无|没有|尚无|未发布|还没有).{0,12}` + word + `|` + word + `.{0,8}(?:为空|暂无)`)
}
func (t *Task) execute(a object) error {
	name := str(a["tool"])
	s := t.snapshot()
	if str(s["stage"]) == "SOLVE_CAPTCHA" && name != "solve_slider_captcha" && name != "request_human" {
		return errors.New("滑块验证阶段仅允许验证工具或请求人工处理")
	}
	detail := str(s["task_type"]) == "HOMEWORK_DETAIL" || t.catalog != nil && t.catalog.detailActive()
	if detail && name != "switch_page" {
		t.detailSwitchStreak = 0
	}
	switch name {
	case "fill_portal_credentials", "request_login_code", "submit_login_code", "apply_device_trust":
		if t.auth == nil || !t.automatic() || str(s["stage"]) != "AUTHENTICATE" {
			return errors.New("当前任务没有可用的登录认证动作")
		}
		switch name {
		case "fill_portal_credentials":
			return t.auth.fill()
		case "request_login_code":
			if e := t.auth.requestCode(); e != nil {
				return e
			}
			t.waitCode(t.auth.method(t.auth.find("otp")))
			return nil
		case "submit_login_code":
			return t.auth.submitCode()
		default:
			return t.auth.trustDevice()
		}
	case "solve_slider_captcha":
		if str(s["stage"]) != "SOLVE_CAPTCHA" {
			return errors.New("当前不处于滑块验证阶段")
		}
		return t.solveCaptcha()
	case "read_course_list", "read_homework_list", "select_homework_detail", "return_to_course_list", "finish_catalog":
		if t.catalog == nil || t.auth != nil && t.verifiedSubject == "" {
			return errors.New("当前不处于已认证的目录采集阶段")
		}
		if detail {
			return problem("DETAIL_NAVIGATION_LOCKED", "请先读取当前作业详情")
		}
		switch name {
		case "read_course_list":
			return t.catalog.readCourses()
		case "read_homework_list":
			return t.catalog.readHomeworks(str(a["course_id"]))
		case "select_homework_detail":
			return t.catalog.selectDetail(str(a["homework_id"]))
		case "return_to_course_list":
			return t.catalog.returnToCourseList()
		default:
			return t.catalog.finish()
		}
	case "read_homework_detail":
		if !detail || t.auth != nil && t.verifiedSubject == "" {
			return errors.New("当前不处于已认证的详情读取阶段")
		}
		if t.detailPage != nil && t.page != t.detailPage {
			return problem("DETAIL_PAGE_NOT_ACTIVE", "请切换到已绑定的作业详情页")
		}
		if t.reading.Swap(true) {
			return errors.New("题目资源正在读取")
		}
		_ = t.update(object{"stage": "READING_DETAIL", "message": "正在读取题目正文、图片和附件"})
		t.manager.wg.Add(1)
		go func() {
			defer t.manager.wg.Done()
			defer t.reading.Store(false)
			t.op.Lock()
			defer t.op.Unlock()
			var detail object
			var e error
			for attempt := 0; attempt < 3; attempt++ {
				detail, e = t.readDetail()
				var problem *issue
				if e == nil || !errors.As(e, &problem) || !contains(problem.Code, "DETAIL_EMPTY", "DETAIL_CONTENT_MISSING", "DETAIL_TARGET_UNVERIFIED") || attempt == 2 || !t.automatic() {
					break
				}
				select {
				case <-t.ctx.Done():
					return
				case <-time.After(time.Second):
				}
			}
			if e == nil && t.catalog != nil && t.catalog.detailActive() {
				e = t.catalog.completeDetail(detail)
			} else if e == nil {
				e = t.update(object{"homework_detail": detail, "status": "SUCCEEDED", "stage": "DONE", "message": "作业详情已读取，正在解析文字、图片和附件"})
			}
			if e != nil && t.automatic() {
				t.failure(e, "read_homework_detail")
				t.human("DETAIL_READ", "作业详情读取未完成，请核对当前页面")
			}
		}()
		return nil
	case "request_human":
		reason, message := "AGENT_REQUEST", bounded(str(a["reason"]), 180)
		if str(s["stage"]) == "LOCATE_COURSE" && strings.Contains(message, "课程") && strings.Contains(message, "学期") {
			reason = "COURSE_OR_TERM"
		}
		t.human(reason, message)
		return nil
	case "request_target_confirmation":
		return t.proposeTarget(a)
	case "open_export_settings":
		if t.catalog != nil || detail || !t.courseConfirmed || !t.assignmentConfirmed {
			return errors.New("当前任务不可打开导出设置")
		}
		return t.prepareExport()
	case "switch_page":
		p, e := t.pageByID(str(a["page_id"]))
		if e != nil {
			return e
		}
		if detail && p == t.page {
			return problem("PAGE_ALREADY_ACTIVE", "该标签已是当前页，请读取详情或请求人工核对")
		}
		if detail && t.detailPage != nil && p != t.detailPage {
			return problem("DETAIL_NAVIGATION_LOCKED", "作业详情已打开，请继续使用已绑定详情页")
		}
		if detail {
			t.detailSwitchStreak++
			if t.detailSwitchStreak >= 4 {
				t.detailSwitchStreak = 0
				t.human("DETAIL_NAVIGATION", "详情任务连续切换标签页但没有读取进展，请核对已打开作业详情")
				return nil
			}
		}
		t.releasePointer("")
		t.page = p
		return nil
	case "scroll":
		delta := num(a["delta_y"])
		if delta > 1000 {
			delta = 1000
		}
		if delta < -1000 {
			delta = -1000
		}
		if t.catalog != nil && !detail {
			kind := "course"
			if t.catalog.pageCourses[t.page] != "" {
				kind = "homework"
			}
			_, e := t.catalog.viewport(kind, delta)
			return e
		}
		return t.page.Mouse().Wheel(0, delta)
	case "click", "fill":
		r, ok := t.refs[str(a["element_ref"])]
		if !ok {
			return errors.New("观察引用已失效")
		}
		item := r.item
		label := strings.TrimSpace(str(item["name"]))
		if flag(item["disabled"]) || !allowedURL(r.page.URL()) || forbidden.MatchString(label) || flag(item["submits_form"]) && !flag(item["search_submit"]) {
			return errors.New("该控件不属于获准的页面操作")
		}
		if name == "click" && exportEntryLabel.MatchString(label) {
			if t.catalog != nil || detail {
				return errors.New("当前任务不允许导出附件")
			}
			t.page = r.page
			return t.prepareExport()
		}
		if exportDownloadLabel.MatchString(label) || regexp.MustCompile(`(?i)确定|确认|confirm|提交|submit`).MatchString(label) && !regexp.MustCompile(`^(查看|浏览)(?:学生)?提交(?:记录|详情|列表|情况|附件)?$`).MatchString(label) {
			return errors.New("提交和下载必须由任务控制器核验")
		}
		if t.catalog != nil && regexp.MustCompile(`(?i)导出|下载|\bexport\b|\bdownload\b`).MatchString(label) {
			return errors.New("目录同步不执行导出或下载")
		}
		if detail && regexp.MustCompile(`(?i)导出|下载|查重|批阅|批改|\b(?:export|download|review|grade)\b`).MatchString(label) {
			return errors.New("作业详情只允许读取题目，不执行批阅或导出")
		}
		if str(item["semester_kind"]) != "" || semester.MatchString(label) || regexp.MustCompile(`^(选择|切换|搜索)?学期$`).MatchString(label) {
			t.human("COURSE_OR_TERM", "请核对课程及学期；如需切换学期，请手动操作后交还自动化")
			return nil
		}
		if name == "fill" {
			if !contains(strings.ToLower(label+str(item["context"])), "搜索", "search", "查找") {
				return errors.New("自动填写只允许搜索字段")
			}
			return r.loc.Fill(bounded(str(a["value"]), 120), pw.LocatorFillOptions{Timeout: pw.Float(8000)})
		}
		t.page = r.page
		if detail && t.detailPage != nil && label == str(obj(s["target"])["assignment_name"]) {
			t.page = t.detailPage
			return nil
		}
		if detail && t.detailNav != nil {
			return t.reconcileDetail()
		}
		oldCourse, oldAssignment := t.courseConfirmed, t.assignmentConfirmed
		courseRef := ""
		var e error
		if t.catalog != nil && !detail {
			courseRef, e = t.catalog.recordNavigation(r)
		} else {
			e = t.recordNavigation(r)
		}
		if e != nil {
			return e
		}
		opening := detail && label == str(obj(s["target"])["assignment_name"]) && t.assignmentConfirmed
		pages := t.context.Pages()
		known := map[pw.Page]bool{}
		for _, p := range pages {
			known[p] = true
		}
		if opening {
			frames := map[pw.Frame]string{}
			for _, f := range r.page.Frames() {
				frames[f] = f.URL()
			}
			t.detailNav = &navigation{r.page, r.page.URL(), pages, frames, time.Now()}
			if t.detailSource == nil {
				t.detailSource, t.detailSourceURL = r.page, r.page.URL()
			}
			t.detailListFrames = frames
		}
		if e = r.loc.Click(pw.LocatorClickOptions{Timeout: pw.Float(8000)}); e != nil {
			t.courseConfirmed, t.assignmentConfirmed = oldCourse, oldAssignment
			if opening {
				t.detailNav = nil
			}
			return e
		}
		deadline := time.Now().Add(200 * time.Millisecond)
		if opening || flag(item["opens_new_page"]) {
			deadline = time.Now().Add(2500 * time.Millisecond)
		}
		for time.Now().Before(deadline) {
			newPage := false
			for _, p := range t.context.Pages() {
				if known[p] || p.IsClosed() {
					continue
				}
				opener, _ := p.Opener()
				if opener == r.page {
					t.page = p
					newPage = true
				}
			}
			if newPage {
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
		if courseRef != "" && t.catalog != nil {
			t.catalog.pageCourses[t.page] = courseRef
		}
		if opening {
			return t.reconcileDetail()
		}
		return nil
	default:
		return errors.New("不支持的浏览器动作")
	}
}
func targetProof(r reference, name string) string {
	return hash([]any{r.page.URL(), r.item["frame_id"], r.item["record_id"], name, r.item["context"]})[:24]
}
func (t *Task) proposeTarget(a object) error {
	kind, name := str(a["kind"]), strings.TrimSpace(str(a["actual_name"]))
	if kind != "course" && kind != "assignment" || name == "" || len([]rune(name)) > 255 || name == "作业" || name == "课程" || name == "个人空间" {
		return errors.New("候选类型或名称无效")
	}
	s := t.snapshot()
	if flag(s["export_attempted"]) || t.assignmentConfirmed || kind == "course" && t.courseConfirmed || kind == "assignment" && !t.courseConfirmed {
		return errors.New("当前阶段不能更改目标")
	}
	r, ok := t.refs[str(a["element_ref"])]
	if !ok || flag(r.item["disabled"]) || !itemMatches(r.item, name) {
		return problem("TARGET_EVIDENCE_MISSING", "候选名称必须来自当前页面记录")
	}
	term := str(a["term"])
	if term != "" {
		found := false
		for _, item := range objects(t.view["elements"]) {
			found = found || itemMatches(item, term) || str(item["selected_option"]) == term
		}
		if !found {
			return problem("SEMESTER_EVIDENCE_MISSING", "学期必须来自当前页面证据")
		}
	}
	candidate := object{"id": targetProof(r, name), "name": name, "context": bounded(str(r.item["context"]), 220)}
	if hash(t.approved[kind]) == hash(candidate) {
		return nil
	}
	target := obj(s["target"])
	fields := object{"target_resolution": object{"id": uuid.NewString(), "kind": kind, "requested_name": target[kind+"_name"], "term": term, "previous_term": false, "semester_fallback": false, "candidates": []any{candidate}, "message": fmt.Sprintf("页面找到“%s”，请确认是否为本次目标", name), "model_reason": bounded(str(a["reason"]), 300)}}
	if s["requested_target"] == nil {
		fields["requested_target"] = clone(target)
	}
	if e := t.update(fields); e != nil {
		return e
	}
	t.human("TARGET_SUGGESTION", fmt.Sprintf("页面找到“%s”，请确认是否为本次目标", name))
	return nil
}
func (t *Task) resolveTarget(p object) error {
	s := t.snapshot()
	pending := obj(s["target_resolution"])
	if len(pending) == 0 || s["status"] != "WAITING_USER" || s["control_mode"] != "HUMAN" || flag(s["export_attempted"]) || p["resolution_id"] != pending["id"] {
		return errors.New("确认已过期，请刷新待处理事项")
	}
	if !t.authenticated() {
		return errors.New("请先完成官网登录")
	}
	target := obj(s["target"])
	kind := str(pending["kind"])
	switch str(p["choice"]) {
	case "edit":
		name := strings.TrimSpace(str(p["name"]))
		if name == "" || len([]rune(name)) > 255 {
			return errors.New("请填写有效名称")
		}
		target[kind+"_name"] = name
		delete(t.approved, kind)
	case "confirm":
		var chosen object
		for _, candidate := range objects(pending["candidates"]) {
			if candidate["id"] == p["candidate_id"] {
				chosen = candidate
			}
		}
		if chosen == nil {
			return errors.New("请选择已找到的候选")
		}
		view, e := t.inventory()
		if e != nil {
			return e
		}
		t.view = view
		match := false
		for _, r := range t.refs {
			if itemMatches(r.item, str(chosen["name"])) && targetProof(r, str(chosen["name"])) == str(chosen["id"]) {
				match = true
			}
		}
		if !match {
			return errors.New("页面候选已变化，请重新核对")
		}
		for _, el := range objects(view["elements"]) {
			term := str(el["selected_option"])
			if semester.MatchString(term) && str(pending["term"]) != "" && term != str(pending["term"]) {
				return errors.New("页面当前学期已变化，请重新核查")
			}
		}
		target[kind+"_name"] = chosen["name"]
		if kind == "course" && str(pending["term"]) != "" {
			target["term"] = pending["term"]
		}
		t.approved[kind] = chosen
		history := append(array(s["target_resolutions"]), object{"kind": kind, "requested_name": pending["requested_name"], "name": chosen["name"], "term": pending["term"], "semester_fallback": pending["semester_fallback"], "time": now()})
		if len(history) > 20 {
			history = history[len(history)-20:]
		}
		if err := t.update(object{"target_resolutions": history}); err != nil {
			return err
		}
	default:
		return errors.New("请选择确认候选或修改名称")
	}
	t.releasePointer("")
	t.observation = nil
	t.stagnant = 0
	t.marker = ""
	return t.update(object{"target": target, "target_resolution": nil, "status": "RUNNING", "control_mode": "AUTO", "waiting_reason": "", "control_epoch": num(s["control_epoch"]) + 1, "auto_actions": 0, "navigation_failures": 0, "decision_failures": 0})
}
