package browserworker

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	pw "github.com/mxschmitt/playwright-go"
)

var (
	exportEntryLabel       = regexp.MustCompile("(?i)" + constants["page_snapshot.EXPORT_LABEL"])
	exportMoreLabel        = regexp.MustCompile(`(?i)^(?:更多|more)(?:\s*[⋮…])?$`)
	exportDialogLabel      = regexp.MustCompile(`(?i)导出|\bexport\b`)
	exportConfirmLabel     = regexp.MustCompile(`(?i)^(?:确定|确认|confirm|ok|export|start export)$`)
	exportCancelLabel      = regexp.MustCompile(`(?i)^(?:取消|关闭|cancel|close|×|✕)$`)
	exportCenterLabel      = regexp.MustCompile(`(?i)^(?:下载中心|download\s+cent(?:er|re))$`)
	exportEmptyLabel       = regexp.MustCompile(`(?i)暂无|没有|暂时没有|\bno (?:downloads?|downloadable|files?|data)|\bnothing to download\b`)
	exportAttachmentLabel  = regexp.MustCompile("(?i)" + constants["portal_export.ATTACHMENT_LABEL"])
	exportOtherFormatLabel = regexp.MustCompile("(?i)" + constants["portal_export.OTHER_FORMAT_LABEL"])
	exportDownloadLabel    = regexp.MustCompile(`(?i)^(?:下载|重新下载|download|re-?download)$`)
	exportFailedLabel      = regexp.MustCompile(`(?i)失败|错误|已过期|\b(?:failed|failure|error|expired)(?:\s+(?:re-?download|download|re-?export))*\s*$`)
	exportReadyLabel       = regexp.MustCompile(`(?i)导出成功|已完成|\b(?:export(?:ed)?\s+success(?:ful(?:ly)?)?|exported|completed|ready)(?:\s+(?:re-?download|download|re-?export))*\s*$`)
)

// Reading navigation candidates never clicks a menu. Only prepareExport may
// disclose More and activate an export, after checking the assignment identity.
func (t *Task) exportCandidates(page pw.Page, hidden bool) []reference {
	if page == nil || page.IsClosed() {
		return nil
	}
	entries := []reference{}
	for _, frame := range page.Frames() {
		controls := frame.Locator("a,button,[onclick],[role=menuitem]")
		raw, err := controls.EvaluateAll(script("export_entry_candidates"), object{"pattern": constants["page_snapshot.EXPORT_LABEL"], "hidden": hidden})
		if err != nil {
			continue
		}
		for _, index := range array(raw) {
			entries = append(entries, reference{page: page, loc: controls.Nth(int(num(index)))})
		}
	}
	return entries
}

func (t *Task) exportEntry() *reference {
	if t.context == nil {
		return nil
	}
	for _, hidden := range []bool{false, true} {
		// A retained preview or a different course tab cannot make the current
		// review page ambiguous. Still reject sibling entries on the same page.
		entries := t.exportCandidates(t.page, hidden)
		if len(entries) == 1 {
			return &entries[0]
		}
		if len(entries) > 1 {
			return nil
		}
		for _, page := range t.context.Pages() {
			if page != t.page {
				entries = append(entries, t.exportCandidates(page, hidden)...)
			}
		}
		if len(entries) == 1 {
			return &entries[0]
		}
		if len(entries) > 1 {
			return nil
		}
	}
	return nil
}

func (t *Task) exportMoreControls(page pw.Page) []reference {
	if page == nil || page.IsClosed() {
		return nil
	}
	matches := []reference{}
	for _, frame := range page.Frames() {
		controls := frame.GetByText(exportMoreLabel)
		for i := 0; i < count(controls); i++ {
			if visible(controls.Nth(i)) {
				matches = append(matches, reference{page: page, loc: controls.Nth(i)})
			}
		}
	}
	return matches
}

func (t *Task) exportNavigationIssue(page pw.Page) error {
	evidence := object{
		"visible_entries": len(t.exportCandidates(page, false)), "legacy_entries": len(t.exportCandidates(page, true)), "more_controls": len(t.exportMoreControls(page)),
	}
	_ = t.update(object{"last_export_navigation": evidence})
	return &issue{Code: "EXPORT_SETTINGS_UNAVAILABLE", Message: "未找到唯一的作业附件导出入口", Evidence: evidence}
}

func exportPhaseError(err error, phase string) error {
	if err == nil {
		return nil
	}
	var known *issue
	if errors.As(err, &known) {
		evidence := clone(known.Evidence)
		if evidence == nil {
			evidence = object{}
		}
		if str(evidence["phase"]) == "" {
			evidence["phase"] = phase
		}
		return &issue{Code: known.Code, Message: known.Message, Evidence: evidence}
	}
	labels := map[string]string{
		"export_baseline": "读取下载中心记录", "find_export_entry": "定位导出菜单", "open_more_menu": "打开更多菜单",
		"select_target_class": "切换目标班级", "activate_export_entry": "触发导出设置", "wait_export_settings": "等待导出设置",
		"close_download_center": "关闭下载中心", "select_attachments": "选择提交附件", "verify_export_scope": "核验导出班级", "confirm_export": "提交导出确认",
	}
	label := labels[phase]
	if label == "" {
		label = "核验导出页面"
	}
	message := strings.ToLower(err.Error())
	// Keep actionable error categories, never raw Playwright call logs, HTML,
	// signed URLs, form values or tokens in the durable/public task state.
	return &issue{Code: "EXPORT_PAGE_OPERATION_FAILED", Message: "导出页面操作失败：" + label, Evidence: object{
		"phase": phase, "exception_type": fmt.Sprintf("%T", err),
		"timeout": strings.Contains(message, "timeout"), "detached": strings.Contains(message, "detached"),
		"obscured":    strings.Contains(message, "intercepts pointer events") || strings.Contains(message, "does not receive pointer events"),
		"strict_mode": strings.Contains(message, "strict mode"), "not_visible": strings.Contains(message, "not visible"),
	}}
}

func (t *Task) activateExportControl(entry reference, pattern string, legacy bool) error {
	if err := t.verifyExportPage(entry.page); err != nil {
		return err
	}
	value, err := entry.loc.Evaluate(script("export_control_activate"), object{"pattern": pattern, "legacy": legacy})
	if err != nil {
		return err
	}
	if !flag(value) {
		return problem("EXPORT_CONTROL_CHANGED", "官网导出控件已变化或不可用，尚未触发")
	}
	return nil
}

func (t *Task) verifyExportPage(page pw.Page) error {
	target := obj(obj(t.snapshot()["target"])["portal"])
	expectedCourse := strings.Split(str(target["course_id"]), ":")[0]
	expectedWork := str(target["homework_id"])
	seenCourse, seenWork, courseMatch, workMatch := false, false, false, false
	for _, frame := range page.Frames() {
		course := queryID(frame.URL(), "courseid")
		work := queryID(frame.URL(), "workid", "homeworkid", "taskid")
		if course != "" {
			seenCourse = true
			courseMatch = courseMatch || course == expectedCourse
		}
		if work != "" {
			seenWork = true
			workMatch = workMatch || work == expectedWork && (expectedCourse == "" || course == "" || course == expectedCourse)
		}
	}
	if expectedCourse != "" && seenCourse && !courseMatch || expectedWork != "" && seenWork && !workMatch {
		return problem("TARGET_ID_MISMATCH", "当前导出页面属于其他课程或作业，尚未提交导出")
	}
	return nil
}

func (t *Task) exportSettingsEntry() (*reference, error) {
	entry := t.exportEntry()
	page := t.page
	if entry != nil {
		page = entry.page
		if err := t.verifyExportPage(page); err != nil {
			return nil, err
		}
		if visible(entry.loc) {
			return entry, nil // More is already open; do not toggle it closed.
		}
	}
	if err := t.verifyExportPage(page); err != nil {
		return nil, err
	}
	if len(t.exportCandidates(page, false)) > 1 {
		return nil, t.exportNavigationIssue(page)
	}
	more := t.exportMoreControls(page)
	if len(more) == 1 {
		t.page = page
		if err := t.activateExportControl(more[0], strings.TrimPrefix(exportMoreLabel.String(), "(?i)"), false); err != nil {
			return nil, exportPhaseError(err, "open_more_menu")
		}
		deadline := time.Now().Add(3 * time.Second)
		for time.Now().Before(deadline) {
			entries := t.exportCandidates(page, false)
			if len(entries) == 1 {
				return &entries[0], nil
			}
			if len(entries) > 1 {
				return nil, t.exportNavigationIssue(page)
			}
			time.Sleep(100 * time.Millisecond)
		}
		if legacy := t.exportCandidates(page, true); len(legacy) == 1 {
			return &legacy[0], nil
		}
		return nil, t.exportNavigationIssue(page)
	}
	if len(more) == 0 && entry != nil {
		return entry, nil // Old portal: a site-owned hidden ul.morePop action.
	}
	return nil, t.exportNavigationIssue(page)
}

func exportStatus(text string) string {
	if exportFailedLabel.MatchString(text) {
		return "FAILED"
	}
	if exportReadyLabel.MatchString(text) {
		return "READY"
	}
	return "PENDING"
}
