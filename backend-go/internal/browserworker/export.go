package browserworker

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	pw "github.com/mxschmitt/playwright-go"
)

type exportLocator struct {
	page     pw.Page
	loc      pw.Locator
	platform string
}
type exportDialog struct {
	page  pw.Page
	frame pw.Frame
	loc   pw.Locator
}
type choice struct {
	loc  pw.Locator
	info object
}

var currentClassAlias = regexp.MustCompile("(?i)" + constants["portal_export.CURRENT_CLASS_LABEL"])

func (t *Task) configureDownloads() error {
	pageSession, e := t.context.NewCDPSession(t.page)
	if e != nil {
		return e
	}
	info, e := pageSession.Send("Target.getTargetInfo", nil)
	if e != nil {
		_ = pageSession.Detach()
		return e
	}
	session := pageSession
	if browser := t.context.Browser(); browser != nil {
		session, e = browser.NewBrowserCDPSession()
		if e != nil {
			_ = pageSession.Detach()
			return e
		}
		_ = pageSession.Detach()
	}
	session.On("Browser.downloadWillBegin", func(event object) {
		t.downloadMu.Lock()
		defer t.downloadMu.Unlock()
		guid := str(event["guid"])
		t.downloadURLs[str(event["url"])] = guid
		t.downloadBytes[guid] = 0
	})
	session.On("Browser.downloadProgress", func(event object) {
		t.downloadMu.Lock()
		t.downloadBytes[str(event["guid"])] = num(event["receivedBytes"])
		t.downloadMu.Unlock()
		_ = t.update(object{"download_progress": object{"received_bytes": event["receivedBytes"], "total_bytes": event["totalBytes"], "state": event["state"]}})
	})
	options := object{"behavior": "allowAndName", "downloadPath": filepath.Join(t.manager.cfg.Downloads, t.id, ".browser"), "eventsEnabled": true}
	if len(obj(t.snapshot()["portal_session"])) == 0 {
		if id := str(obj(obj(info)["targetInfo"])["browserContextId"]); id != "" {
			options["browserContextId"] = id
		}
	}
	_, e = session.Send("Browser.setDownloadBehavior", options)
	if e != nil {
		_ = session.Detach()
		return e
	}
	t.downloadCDP = session
	return nil
}

func (t *Task) exportArea() *exportDialog {
	if t.context == nil {
		return nil
	}
	for _, p := range t.context.Pages() {
		if p.IsClosed() {
			continue
		}
		for _, f := range p.Frames() {
			l := f.Locator(".popDiv.centerPop,[role=dialog]").Filter(pw.LocatorFilterOptions{HasText: exportDialogLabel})
			for i := 0; i < count(l); i++ {
				loc := l.Nth(i)
				if visible(loc) {
					return &exportDialog{p, f, loc}
				}
			}
		}
	}
	return nil
}
func (t *Task) choices(modal pw.Locator) ([]choice, error) {
	controls := modal.Locator(constants["portal_adapter.CHOICES"])
	value, e := controls.EvaluateAll(script("export_choices"), object{
		"attachments":   constants["portal_export.ATTACHMENT_LABEL"],
		"other_format":  constants["portal_export.OTHER_FORMAT_LABEL"],
		"current_class": constants["portal_export.CURRENT_CLASS_LABEL"],
		"all_classes":   constants["portal_export.ALL_CLASSES_LABEL"],
	})
	if e != nil {
		return nil, e
	}
	out := []choice{}
	for _, info := range objects(value) {
		out = append(out, choice{controls.Nth(int(num(info["i"]))), info})
	}
	return out, nil
}
func checked(l pw.Locator) bool {
	v, e := l.Evaluate(script("export_checked"), nil)
	return e == nil && flag(v)
}
func waitChecked(l pw.Locator, want bool) bool {
	for i := 0; i < 16; i++ {
		if checked(l) == want {
			return true
		}
		time.Sleep(100 * time.Millisecond)
	}
	return false
}
func (t *Task) classes(modal pw.Locator) ([]any, error) {
	choices, e := t.choices(modal)
	if e != nil {
		return nil, e
	}
	t.classLocators = map[string]pw.Locator{}
	out := []any{}
	for _, c := range choices {
		if str(c.info["group"]) != "scope" || str(c.info["name"]) == "" || len([]rune(str(c.info["name"]))) > 120 {
			continue
		}
		id := fmt.Sprintf("class_%d", len(out))
		r := object{"id": id, "name": c.info["name"], "all": c.info["all"]}
		v, e := c.loc.Evaluate(script("export_class_id"), nil)
		if e != nil {
			return nil, e
		}
		if str(v) != "" {
			r["platform_id"] = v
		}
		out = append(out, r)
		t.classLocators[id] = c.loc
	}
	return out, nil
}
func (t *Task) selectAttachments(modal pw.Locator) error {
	choices, e := t.choices(modal)
	if e != nil {
		return e
	}
	wanted, others := []pw.Locator{}, []pw.Locator{}
	formats := 0
	observed := []any{}
	for _, c := range choices {
		if str(c.info["group"]) != "format" {
			continue
		}
		formats++
		if len(observed) < 10 {
			observed = append(observed, object{"name": bounded(str(c.info["name"]), 180), "checked": checked(c.loc)})
		}
		if exportAttachmentLabel.MatchString(str(c.info["name"])) {
			wanted = append(wanted, c.loc)
		} else if exportOtherFormatLabel.MatchString(str(c.info["name"])) {
			others = append(others, c.loc)
		}
	}
	if len(wanted) > 1 {
		return &issue{Code: "ATTACHMENT_AMBIGUOUS", Message: "发现多个附件导出选项，请手动核对", Evidence: object{"formats": observed}}
	}
	if len(wanted) == 0 {
		if formats > 0 || count(modal.GetByText(exportOtherFormatLabel)) > 0 {
			return &issue{Code: "ATTACHMENT_MISSING", Message: "未识别到导出提交附件控件", Evidence: object{"formats": observed}}
		}
		return nil
	}
	if !checked(wanted[0]) {
		if e := wanted[0].Click(); e != nil {
			return e
		}
	}
	if !waitChecked(wanted[0], true) {
		return problem("ATTACHMENT_UNVERIFIED", "无法核验附件导出格式的选中状态")
	}
	for _, other := range others {
		if checked(other) {
			return problem("ATTACHMENT_MISMATCH", "其他导出格式仍处于选中状态，请核对")
		}
	}
	return nil
}
func (t *Task) closeCenters() error {
	for _, p := range t.context.Pages() {
		if p.IsClosed() {
			continue
		}
		for _, f := range p.Frames() {
			centers := f.Locator("#downloadcenter:visible,.downloadCenter:visible")
			for n := 0; n < count(centers); n++ {
				center := centers.Nth(n)
				controls := center.Locator("button,a,[role=button],[onclick],.close,.close_btn,.closeDown")
				matches := []pw.Locator{}
				for i := 0; i < count(controls); i++ {
					loc := controls.Nth(i)
					if !visible(loc) {
						continue
					}
					v, e := loc.Evaluate(script("export_center_close_control"), nil)
					if e != nil {
						return e
					}
					info := obj(v)
					if !flag(info["nested"]) && (regexp.MustCompile(`(?i)关闭|close|hideDownload`).MatchString(str(info["description"])) || str(info["text"]) == "关闭" || str(info["text"]) == "×" || str(info["text"]) == "✕") {
						matches = append(matches, loc)
					}
				}
				if len(matches) == 1 {
					if e := matches[0].Click(); e != nil {
						return e
					}
					if e := center.WaitFor(pw.LocatorWaitForOptions{State: pw.WaitForSelectorStateHidden, Timeout: pw.Float(5000)}); e != nil {
						return e
					}
				} else {
					v, e := center.Evaluate(script("export_center_overlay"), nil)
					if e != nil {
						return e
					}
					if flag(v) {
						return problem("EXPORT_CENTER_CLOSE_UNVERIFIED", "下载中心遮挡导出设置，请先关闭下载中心")
					}
				}
			}
		}
	}
	return nil
}
func (t *Task) exportBaseline() ([]any, bool, error) {
	rows, e := t.exportRows()
	if e != nil {
		return nil, false, e
	}
	present := len(rows) > 0
	for _, p := range t.context.Pages() {
		for _, f := range p.Frames() {
			present = present || count(f.Locator("#downloadcenter:visible,.downloadCenter:visible")) > 0
		}
	}
	if present {
		e = t.closeCenters()
		return rows, true, e
	}
	for _, p := range t.context.Pages() {
		if p.IsClosed() {
			continue
		}
		for _, f := range p.Frames() {
			entries := f.GetByText(exportCenterLabel)
			matches := []pw.Locator{}
			for i := 0; i < count(entries); i++ {
				if visible(entries.Nth(i)) {
					matches = append(matches, entries.Nth(i))
				}
			}
			if len(matches) != 1 {
				continue
			}
			v, e := matches[0].Evaluate(script("export_center_open"), nil)
			if e != nil || !flag(v) {
				continue
			}
			center := f.Locator("#downloadcenter:visible,.downloadCenter:visible")
			if e = center.First().WaitFor(pw.LocatorWaitForOptions{State: pw.WaitForSelectorStateVisible, Timeout: pw.Float(5000)}); e != nil {
				return nil, false, e
			}
			ready := center.First().Locator("tbody tr,.dataBody_td,.empty,.no-data,.noData,.no_data")
			deadline := time.Now().Add(5 * time.Second)
			for count(ready) == 0 && time.Now().Before(deadline) {
				text, _ := center.First().InnerText()
				if exportEmptyLabel.MatchString(text) {
					break
				}
				time.Sleep(100 * time.Millisecond)
			}
			rows, e = t.exportRows()
			if e != nil {
				return nil, false, e
			}
			e = t.closeCenters()
			return rows, true, e
		}
	}
	return []any{}, false, nil
}
func currentClass(page pw.Page) string {
	ids := map[string]bool{}
	for _, f := range page.Frames() {
		if id := queryID(f.URL(), "clazzid", "classid"); id != "" {
			ids[id] = true
		}
		nodes := f.Locator(constants["portal_catalog.CLASS_SELECTOR"])
		v, e := nodes.EvaluateAll(script("catalog_extract"))
		if e != nil {
			continue
		}
		for i, r := range objects(v) {
			selected, e := nodes.Nth(i).Evaluate("e=>e.matches('option:checked,.active,.selected,[aria-selected=true]')", nil)
			if e == nil && flag(selected) && str(r["class_id"]) != "" {
				return str(r["class_id"])
			}
		}
	}
	if len(ids) == 1 {
		for id := range ids {
			return id
		}
	}
	return ""
}
func (t *Task) selectClass(page pw.Page, id string) error {
	if currentClass(page) == id {
		return nil
	}
	matches := []pw.Locator{}
	available := false
	for _, f := range page.Frames() {
		nodes := f.Locator(constants["portal_catalog.CLASS_SELECTOR"])
		v, e := nodes.EvaluateAll(script("catalog_extract"))
		if e != nil {
			return e
		}
		for i, r := range objects(v) {
			key := str(r["class_id"])
			if key != "" && key != "0" && key != "-1" && key != "all" {
				available = true
			}
			if key == id {
				matches = append(matches, nodes.Nth(i))
			}
		}
	}
	if !available {
		return nil
	}
	if len(matches) != 1 {
		return problem("CLASS_CHANGED", "当前作业页没有唯一匹配的班级入口")
	}
	loc := matches[0]
	option, e := loc.Evaluate("e=>e.tagName==='OPTION'", nil)
	if e != nil {
		return e
	}
	if flag(option) {
		_, e = loc.Locator("..").SelectOption(pw.SelectOptionValues{Values: &[]string{id}})
	} else {
		_, e = loc.Evaluate("e=>e.click()", nil)
	}
	if e != nil {
		return e
	}
	_, e = loc.Evaluate(script("class_selection_ack"), nil)
	if e != nil || currentClass(page) != id {
		return problem("CLASS_CHANGED", "官网尚未确认所选班级")
	}
	return nil
}
func (t *Task) applyScope(modal pw.Locator, request object, options []any) (object, error) {
	request = reconcileScope(request, options)
	if request == nil {
		return nil, problem("SCOPE_CHANGED", "班级列表已变化，请重新选择")
	}
	selected := map[string]bool{}
	byID := map[string]object{}
	for _, o := range options {
		v := obj(o)
		byID[str(v["id"])] = v
	}
	mode := str(request["mode"])
	if mode == "current" {
		target := obj(obj(t.snapshot()["target"])["portal"])
		if len(options) > 0 || str(request["platform_id"]) == "" || currentClass(t.page) != str(target["class_id"]) {
			return nil, problem("CLASS_CHANGED", "当前批阅页班级与指定班级不一致")
		}
		if e := t.selectAttachments(modal); e != nil {
			return nil, e
		}
		return object{"mode": "selected", "classes": []any{object{"id": "current_page", "name": target["class_name"], "all": false, "platform_id": target["class_id"]}}}, nil
	}
	if mode == "all" {
		for id, v := range byID {
			if flag(v["all"]) {
				selected[id] = true
			}
		}
		if len(selected) > 1 {
			return nil, problem("SCOPE_AMBIGUOUS", "发现多个全部班级控件，请核对导出分组")
		}
		if len(selected) == 0 {
			for id, v := range byID {
				if contains(str(v["name"]), "本班级", "当前班级") {
					return nil, problem("ALL_CLASSES_UNAVAILABLE", "当前页面只支持当前班级，无法核验全部班级")
				}
				radio, e := t.classLocators[id].Evaluate(script("export_is_radio"), nil)
				if e != nil {
					return nil, e
				}
				if flag(radio) && len(options) > 1 {
					return nil, problem("ALL_CLASSES_UNAVAILABLE", "当前页面仅支持单选班级")
				}
				selected[id] = true
			}
		}
	} else if mode == "selected" {
		for _, value := range array(request["class_ids"]) {
			id := str(value)
			v := byID[id]
			if v == nil || flag(v["all"]) {
				return nil, problem("SCOPE_CHANGED", "班级列表已变化，请重新选择")
			}
			oldName := str(obj(request["option_names"])[id])
			if oldName != "" && oldName != str(v["name"]) {
				matches := []string{}
				for key, v := range byID {
					if !flag(v["all"]) && strings.Join(strings.Fields(str(v["name"])), " ") == strings.Join(strings.Fields(oldName), " ") {
						matches = append(matches, key)
					}
				}
				if len(matches) != 1 {
					return nil, problem("SCOPE_CHANGED", "班级列表已变化，请重新选择")
				}
				id = matches[0]
			}
			selected[id] = true
		}
	}
	if len(selected) == 0 {
		return nil, problem("SCOPE_MISSING", "当前页面没有可核验的班级范围")
	}
	master := false
	for id := range selected {
		master = master || flag(byID[id]["all"])
	}
	keys := []string{}
	for id := range byID {
		keys = append(keys, id)
	}
	sort.SliceStable(keys, func(i, j int) bool { return selected[keys[i]] && !selected[keys[j]] })
	for _, id := range keys {
		if master && !selected[id] {
			continue
		}
		want := selected[id]
		loc := t.classLocators[id]
		if checked(loc) != want {
			if e := loc.Click(); e != nil {
				return nil, e
			}
			if !waitChecked(loc, want) {
				return nil, problem("SCOPE_MISMATCH", "官网班级选中状态与已确认范围不一致")
			}
		}
	}
	actual := []any{}
	for _, o := range options {
		v := obj(o)
		id := str(v["id"])
		on := checked(t.classLocators[id])
		if !master && on != selected[id] || selected[id] && !on {
			return nil, problem("SCOPE_MISMATCH", "官网未确认完整班级范围")
		}
		if on {
			actual = append(actual, v)
		}
	}
	if e := t.selectAttachments(modal); e != nil {
		return nil, e
	}
	return object{"mode": mode, "classes": actual}, nil
}
func (t *Task) prepareExport() (resultErr error) {
	step := "verify_export_scope"
	defer func() {
		if resultErr != nil {
			resultErr = exportPhaseError(resultErr, step)
			_ = t.update(object{"last_export_step": obj(publicError(resultErr)["evidence"])["phase"]})
		}
	}()
	s := t.snapshot()
	if flag(s["export_attempted"]) {
		return nil
	}
	if !t.courseConfirmed || !t.assignmentConfirmed {
		return problem("TARGET_UNVERIFIED", "课程和作业归属尚未核验")
	}
	step = "export_baseline"
	before, observed, e := t.exportBaseline()
	if e != nil {
		return e
	}
	area := t.exportArea()
	target := obj(obj(s["target"])["portal"])
	if area == nil {
		step = "find_export_entry"
		entry, entryErr := t.exportSettingsEntry()
		if entryErr != nil {
			return entryErr
		}
		if str(target["class_id"]) != "" {
			step = "select_target_class"
			if e = t.selectClass(entry.page, str(target["class_id"])); e != nil {
				return e
			}
			t.page = entry.page
			step = "find_export_entry"
			entry, e = t.exportSettingsEntry()
			if e != nil {
				return e
			}
		}
		t.page = entry.page
		step = "activate_export_entry"
		e = t.activateExportControl(*entry, constants["page_snapshot.EXPORT_LABEL"], true)
		if e != nil {
			return e
		}
		step = "wait_export_settings"
		deadline := time.Now().Add(3 * time.Second)
		for area == nil && time.Now().Before(deadline) {
			time.Sleep(100 * time.Millisecond)
			area = t.exportArea()
		}
	}
	if area == nil {
		return problem("EXPORT_SETTINGS_UNAVAILABLE", "附件导出设置尚未打开")
	}
	if e = t.verifyExportPage(area.page); e != nil {
		return e
	}
	t.page = area.page
	step = "close_download_center"
	if e = t.closeCenters(); e != nil {
		return e
	}
	step = "select_attachments"
	if e = t.selectAttachments(area.loc); e != nil {
		e = exportPhaseError(e, step)
		_ = t.update(object{"last_export_step": step})
		t.failure(e, "select_attachments")
		t.human("EXPORT_FORMAT", str(publicError(e)["message"]))
		return nil
	}
	options, e := t.classes(area.loc)
	if e != nil {
		return e
	}
	if str(target["class_id"]) != "" && len(options) == 0 && currentClass(area.page) != str(target["class_id"]) {
		step = "select_target_class"
		if e = closeExportSettings(area.loc); e != nil {
			return e
		}
		if e = t.selectClass(area.page, str(target["class_id"])); e != nil {
			return e
		}
		step = "find_export_entry"
		entry, entryErr := t.exportSettingsEntry()
		if entryErr != nil {
			return entryErr
		}
		step = "activate_export_entry"
		e = t.activateExportControl(*entry, constants["page_snapshot.EXPORT_LABEL"], true)
		if e != nil {
			return e
		}
		step = "wait_export_settings"
		area = nil
		for i := 0; i < 20 && area == nil; i++ {
			time.Sleep(100 * time.Millisecond)
			area = t.exportArea()
		}
		if area == nil {
			return problem("EXPORT_SETTINGS_UNAVAILABLE", "切换班级后尚未打开附件导出设置")
		}
		t.page = area.page
		step = "select_attachments"
		if e = t.selectAttachments(area.loc); e != nil {
			return e
		}
		options, e = t.classes(area.loc)
		if e != nil {
			return e
		}
	}
	step = "verify_export_scope"
	request := obj(s["requested_scope"])
	names := object{}
	for _, o := range options {
		v := obj(o)
		names[str(v["id"])] = v["name"]
	}
	if str(target["class_id"]) != "" {
		matches := []string{}
		identified := false
		for _, o := range options {
			v := obj(o)
			identified = identified || !flag(v["all"]) && str(v["platform_id"]) != ""
		}
		for _, o := range options {
			v := obj(o)
			if flag(v["all"]) {
				continue
			}
			match := str(v["platform_id"]) == str(target["class_id"])
			if !identified {
				match = strings.TrimSpace(str(v["name"])) == strings.TrimSpace(str(target["class_name"]))
				if !match && currentClass(area.page) == str(target["class_id"]) {
					match = currentClassAlias.MatchString(str(v["name"]))
				}
			}
			if match {
				matches = append(matches, str(v["id"]))
			}
		}
		if len(options) == 0 && currentClass(area.page) == str(target["class_id"]) {
			request = object{"mode": "current", "platform_id": target["class_id"]}
		} else if len(matches) == 1 {
			request = object{"mode": "selected", "class_ids": []any{matches[0]}, "option_names": names}
		} else {
			return problem("CLASS_CHANGED", "导出班级与所选官网记录不一致")
		}
	} else if len(request) == 0 || len(target) > 0 {
		request = object{"mode": "all", "class_ids": []any{}, "option_names": names}
	}
	request = reconcileScope(request, options)
	_ = t.update(object{"stage": "SELECT_CLASSES", "classes": options, "requested_scope": request})
	scope, e := t.applyScope(area.loc, request, options)
	if e != nil {
		e = exportPhaseError(e, step)
		_ = t.update(object{"last_export_step": step})
		t.failure(e, "apply_scope")
		t.human("SCOPE_SELECTION", str(publicError(e)["message"]))
		return nil
	}
	if str(target["class_id"]) != "" {
		for _, c := range objects(scope["classes"]) {
			c["name"], c["platform_id"] = target["class_name"], target["class_id"]
		}
	}
	scope["baseline_observed"] = observed
	if str(scope["mode"]) == "all" {
		classes, e := readClasses(area.page, object{})
		if e != nil {
			return e
		}
		if len(classes) > 0 {
			scope["expected_classes"] = classes
		}
	}
	step = "confirm_export"
	confirm := area.loc.Locator("a.confirmDown").Filter(pw.LocatorFilterOptions{HasText: exportConfirmLabel})
	if count(confirm) == 0 {
		confirm = area.loc.GetByRole("button", pw.LocatorGetByRoleOptions{Name: exportConfirmLabel})
	}
	if count(confirm) != 1 || !visible(confirm) {
		return problem("EXPORT_CONFIRM_UNAVAILABLE", "未找到唯一可见的导出确认按钮")
	}
	refs := []any{}
	for _, row := range before {
		refs = append(refs, obj(row)["ref"])
	}
	if e = t.update(object{"class_scope": scope, "export_baseline": refs, "export_baseline_rows": before, "export_attempted": true, "export_started_at": now(), "stage": "EXPORT", "last_error": nil, "last_export_navigation": nil, "last_export_step": nil}); e != nil {
		return e
	}
	activated, e := confirm.Evaluate(script("export_confirm_activate"), nil)
	if e != nil {
		return e
	}
	if !flag(activated) {
		_ = t.update(object{"export_attempted": false})
		return problem("EXPORT_CONFIRM_UNAVAILABLE", "导出确认按钮已失效，尚未提交")
	}
	t.event("已提交导出，正在关联下载中心任务")
	return t.update(object{"stage": "WAIT_EXPORT"})
}

func reconcileScope(request object, options []any) object {
	if str(request["mode"]) == "current" {
		if str(request["platform_id"]) != "" && len(options) == 0 {
			return request
		}
		return nil
	}
	names := object{}
	for _, option := range objects(options) {
		names[str(option["id"])] = option["name"]
	}
	if str(request["mode"]) == "all" {
		if len(options) == 0 {
			return nil
		}
		return object{"mode": "all", "class_ids": []any{}, "option_names": names}
	}
	if str(request["mode"]) != "selected" || len(array(request["class_ids"])) == 0 {
		return nil
	}
	selected := []any{}
	for _, ref := range array(request["class_ids"]) {
		name := strings.Join(strings.Fields(str(obj(request["option_names"])[str(ref)])), " ")
		matches := []any{}
		for _, option := range objects(options) {
			if !flag(option["all"]) && name != "" && strings.Join(strings.Fields(str(option["name"])), " ") == name {
				matches = append(matches, option["id"])
			}
		}
		if len(matches) != 1 {
			return nil
		}
		selected = append(selected, matches[0])
	}
	return object{"mode": "selected", "class_ids": selected, "option_names": names}
}

func closeExportSettings(modal pw.Locator) error {
	controls := modal.Locator("button,a,[role=button],.close,.close_btn,.closePop")
	matches, cancels := []pw.Locator{}, []pw.Locator{}
	for i := 0; i < count(controls); i++ {
		control := controls.Nth(i)
		if !visible(control) {
			continue
		}
		text, _ := control.InnerText()
		label, _ := control.GetAttribute("aria-label")
		text = strings.TrimSpace(text)
		if exportCancelLabel.MatchString(text) || exportCancelLabel.MatchString(label) {
			matches = append(matches, control)
		}
		if text == "取消" || strings.EqualFold(text, "cancel") {
			cancels = append(cancels, control)
		}
	}
	if len(cancels) > 0 {
		matches = cancels
	}
	if len(matches) != 1 {
		return problem("EXPORT_CANCEL_UNAVAILABLE", "请先关闭导出设置，再选择目标班级；尚未提交导出")
	}
	if err := matches[0].Click(pw.LocatorClickOptions{Timeout: pw.Float(5000)}); err != nil {
		return err
	}
	return modal.WaitFor(pw.LocatorWaitForOptions{State: pw.WaitForSelectorStateHidden, Timeout: pw.Float(5000)})
}

func containsExact(value string, choices ...string) bool {
	for _, choice := range choices {
		if value == choice {
			return true
		}
	}
	return false
}

var exportTime = regexp.MustCompile(`(?:\d{4}[-/])?\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?`)
var exportState = regexp.MustCompile(`(?i)(?:重新下载|导出中|等待|导出成功|下载|失败|处理中|已完成|重新导出|已过期|排队中|\b(?:re-?download|exporting|waiting|export(?:ed)?\s+success(?:ful(?:ly)?)?|exported|download|failed|processing|completed|re-?export|expired|queued|ready)\b|\s)+$`)

func exportIdentity(text string) string {
	value := exportTime.ReplaceAllString(text, "")
	value = exportState.ReplaceAllString(value, "")
	return strings.Join(strings.Fields(value), "")
}
func exportRef(raw object) string {
	stable := str(raw["platform_id"])
	if stable == "" {
		stable = exportIdentity(str(raw["text"]))
	}
	return textHash(strings.TrimSpace(stable))[:24]
}
func (t *Task) exportRows() ([]any, error) {
	rows := []any{}
	t.exportLocators = map[string]exportLocator{}
	counts := map[string]int{}
	for _, p := range t.context.Pages() {
		if p.IsClosed() {
			continue
		}
		for _, f := range p.Frames() {
			center := f.Locator("#downloadcenter,.downloadCenter")
			if count(center) == 0 {
				continue
			}
			locs := center.Locator("tbody tr")
			if count(locs) == 0 {
				locs = center.Locator(".dataBody_td")
			}
			for i := 0; i < count(locs); i++ {
				loc := locs.Nth(i)
				if !visible(loc) {
					continue
				}
				value, e := loc.Evaluate(script("export_row"), nil)
				if e != nil {
					return nil, e
				}
				raw := obj(value)
				text := str(raw["text"])
				if text == "" {
					continue
				}
				ref := exportRef(raw)
				status := exportStatus(text)
				legacy := str(raw["platform_id"])
				if legacy == "" {
					legacy = regexp.MustCompile(`重新下载|导出中|等待|导出成功|下载|失败|处理中|已完成|重新导出|\s+`).ReplaceAllString(exportState.ReplaceAllString(text, ""), "")
				}
				rows = append(rows, object{"ref": ref, "legacy_ref": textHash(strings.TrimSpace(legacy))[:24], "text": text, "status": status})
				counts[ref]++
				t.exportLocators[ref] = exportLocator{p, loc, str(raw["platform_id"])}
			}
		}
	}
	for _, r := range rows {
		row := obj(r)
		ref := str(row["ref"])
		row["ambiguous"] = counts[ref] > 1
		if counts[ref] > 1 {
			delete(t.exportLocators, ref)
		}
	}
	return rows, nil
}
func rowMatchesAssignment(row object, title string) bool {
	text := str(row["text"])
	index := strings.Index(text, title)
	if index < 0 {
		return false
	}
	before := []rune(text[:index])
	after := []rune(text[index+len(title):])
	boundary := func(r rune) bool { return strings.ContainsRune(" \t\n_-/【[(（:：.】])）", r) }
	return (len(before) == 0 || boundary(before[len(before)-1])) && (len(after) == 0 || boundary(after[0]))
}
func exportStamp(text string, started float64) float64 {
	stamp := exportTime.FindString(text)
	if stamp == "" {
		return 0
	}
	stamp = strings.ReplaceAll(stamp, "/", "-")
	zone := time.FixedZone("Asia/Shanghai", 8*3600)
	if !regexp.MustCompile(`^\d{4}-`).MatchString(stamp) {
		stamp = fmt.Sprintf("%d-%s", time.Unix(int64(started), 0).In(zone).Year(), stamp)
	}
	for _, layout := range []string{"2006-1-2 15:04:05", "2006-1-2 15:04"} {
		v, e := time.ParseInLocation(layout, stamp, zone)
		if e == nil {
			return float64(v.Unix())
		}
	}
	return 0
}
func refreshedRows(rows, baseline []any, started float64) []any {
	previous := object{}
	for _, r := range baseline {
		row := obj(r)
		previous[str(row["ref"])] = row
	}
	for _, r := range rows {
		row := obj(r)
		prior := obj(previous[str(row["ref"])])
		a, b := exportStamp(str(row["text"]), started), exportStamp(str(prior["text"]), started)
		if str(prior["status"]) == "READY" && a > b && b > 0 && a >= float64(int64(started)/60*60) && a <= now()+300 && exportIdentity(str(row["text"])) == exportIdentity(str(prior["text"])) {
			row["generation_verified"] = true
		}
	}
	return rows
}
func freshRow(row object, baseline map[string]bool) bool {
	return !baseline[str(row["ref"])] || flag(row["generation_verified"])
}
func matchExports(rows []any, baseline map[string]bool, title string, scope object) []any {
	if !flag(scope["baseline_observed"]) {
		return nil
	}
	matches := []any{}
	for _, r := range rows {
		row := obj(r)
		if freshRow(row, baseline) && rowMatchesAssignment(row, title) {
			if flag(row["ambiguous"]) {
				return nil
			}
			matches = append(matches, row)
		}
	}
	classes := objects(scope["classes"])
	if str(scope["mode"]) == "all" {
		expected := objects(scope["expected_classes"])
		if len(expected) == 0 {
			for _, c := range classes {
				if !flag(c["all"]) {
					expected = append(expected, c)
				}
			}
		}
		refs := array(scope["expected_export_refs"])
		if len(refs) > 0 {
			fresh := map[string]bool{}
			for _, r := range matches {
				row := obj(r)
				if flag(row["generation_verified"]) {
					fresh[str(row["ref"])] = true
				}
			}
			if len(fresh) > 0 {
				for _, r := range refs {
					if !fresh[str(r)] {
						return nil
					}
				}
				return matches
			}
		}
		if len(matches) == 1 && (len(expected) <= 1 || regexp.MustCompile(`(?i)全部班级|所有班级|\ball classes\b`).MatchString(str(obj(matches[0])["text"]))) {
			return matches
		}
		if len(matches) == len(expected) && len(expected) > 0 {
			for _, c := range expected {
				name := str(c["name"])
				if name == "" {
					return nil
				}
				n := 0
				for _, r := range matches {
					if strings.Contains(str(obj(r)["text"]), name) {
						n++
					}
				}
				if n != 1 {
					return nil
				}
			}
			for _, r := range matches {
				n := 0
				for _, c := range expected {
					if strings.Contains(str(obj(r)["text"]), str(c["name"])) {
						n++
					}
				}
				if n != 1 {
					return nil
				}
			}
			return matches
		}
		return nil
	}
	selected := []any{}
	for _, r := range matches {
		row := obj(r)
		for _, c := range classes {
			if !flag(c["all"]) && str(c["name"]) != "" && strings.Contains(str(row["text"]), str(c["name"])) {
				selected = append(selected, row)
				break
			}
		}
	}
	if len(selected) != len(classes) || len(classes) == 0 {
		return nil
	}
	for _, c := range classes {
		n := 0
		for _, r := range selected {
			if strings.Contains(str(obj(r)["text"]), str(c["name"])) {
				n++
			}
		}
		if n != 1 {
			return nil
		}
	}
	return selected
}
func (t *Task) processExports() error {
	s := t.snapshot()
	if !t.automatic() || !flag(s["export_attempted"]) {
		return nil
	}
	if t.loginRequired() {
		t.human("LOGIN", "等待导出期间登录已过期，请重新登录")
		return nil
	}
	rows, e := t.exportRows()
	if e != nil {
		return e
	}
	rows = refreshedRows(rows, array(s["export_baseline_rows"]), num(s["export_started_at"]))
	baseline := map[string]bool{}
	for _, r := range array(s["export_baseline"]) {
		baseline[str(r)] = true
	}
	exports := array(s["exports"])
	if len(exports) == 0 {
		scope := clone(obj(s["class_scope"]))
		expected := []any{}
		for _, row := range objects(s["export_baseline_rows"]) {
			if rowMatchesAssignment(row, str(obj(s["target"])["assignment_name"])) && str(row["status"]) == "READY" {
				expected = append(expected, row["ref"])
			}
		}
		scope["expected_export_refs"] = expected
		exports = matchExports(rows, baseline, str(obj(s["target"])["assignment_name"]), scope)
		if len(exports) == 0 {
			candidates := []any{}
			for _, r := range rows {
				if freshRow(obj(r), baseline) {
					candidates = append(candidates, r)
				}
			}
			_ = t.update(object{"export_candidates": candidates})
			if now()-num(s["export_started_at"]) > 30 {
				t.human("EXPORT_ASSOCIATION", "请核对并绑定本次导出的全部记录；导出可能已提交，请勿重复导出")
			}
			if len(rows) == 0 {
				for _, p := range t.context.Pages() {
					if p.IsClosed() {
						continue
					}
					l := p.GetByText(exportCenterLabel)
					if count(l) > 0 && visible(l.First()) {
						_, _ = l.First().Evaluate(script("export_center_open"), nil)
						break
					}
				}
			}
			return nil
		}
	}
	claimed := map[string]bool{}
	resolved := []any{}
	for _, r := range exports {
		export := clone(obj(r))
		matches := []object{}
		for _, candidate := range objects(rows) {
			if flag(candidate["ambiguous"]) {
				continue
			}
			same := candidate["ref"] == export["ref"] || candidate["ref"] == export["row_ref"]
			if !same {
				same = !baseline[str(candidate["ref"])] && !baseline[str(candidate["legacy_ref"])] && exportIdentity(str(candidate["text"])) == exportIdentity(str(export["text"]))
			}
			if same {
				matches = append(matches, candidate)
			}
		}
		export["available"] = false
		if len(matches) == 1 && !claimed[str(matches[0]["ref"])] {
			original := export["ref"]
			for k, v := range matches[0] {
				export[k] = v
			}
			export["ref"], export["row_ref"], export["available"] = original, matches[0]["ref"], true
			claimed[str(matches[0]["ref"])] = true
		}
		resolved = append(resolved, export)
	}
	if e = t.update(object{"exports": resolved}); e != nil {
		return e
	}
	files := array(s["files"])
	done := map[string]bool{}
	verified := 0
	for _, f := range objects(files) {
		if str(f["status"]) == "VERIFIED" {
			verified++
			done[str(f["export_ref"])] = true
		}
		if str(f["status"]) == "FAILED" {
			done[str(f["export_ref"])] = true
		}
	}
	pending := false
	// Download every uniquely bound READY row before asking about another
	// missing row. Preserve completed files even if the remaining rows change.
	ordered := append([]any{}, resolved...)
	sort.SliceStable(ordered, func(i, j int) bool {
		a, b := obj(ordered[i]), obj(ordered[j])
		return flag(a["available"]) && str(a["status"]) == "READY" && !(flag(b["available"]) && str(b["status"]) == "READY")
	})
	for _, value := range ordered {
		export := obj(value)
		ref := str(export["ref"])
		if done[ref] {
			continue
		}
		if !flag(export["available"]) {
			t.human("EXPORT_ASSOCIATION", "已关联导出记录不可见或不唯一，请核对下载中心")
			return nil
		}
		switch str(export["status"]) {
		case "FAILED":
			files = append(files, object{"export_ref": ref, "status": "FAILED", "error": "平台导出失败或已过期"})
			if e = t.update(object{"files": files}); e != nil {
				return e
			}
		case "READY":
			entry, ok := t.exportLocators[str(export["row_ref"])]
			if !ok {
				return problem("DOWNLOAD_RECORD_MISSING", "已关联下载记录已变化")
			}
			d, e := entry.page.ExpectDownload(func() error { return activateDownload(entry, str(export["row_ref"])) }, pw.PageExpectDownloadOptions{Timeout: pw.Float(30000)})
			if e != nil {
				return e
			}
			t.page = entry.page
			t.downloadBusy = true
			_ = t.update(object{"stage": "DOWNLOAD", "download_progress": object{"received_bytes": 0, "total_bytes": 0, "state": "inProgress"}})
			t.manager.wg.Add(1)
			go t.saveDownload(d, export)
			return nil
		default:
			pending = true
		}
	}
	if pending {
		if now()-num(s["export_started_at"]) > 900 {
			t.human("EXPORT_TIMEOUT", "平台打包等待超过15分钟，请核对导出状态")
		}
		return nil
	}
	status := "FAILED"
	if verified == len(resolved) {
		status = "SUCCEEDED"
	} else if verified > 0 {
		status = "PARTIAL_SUCCESS"
	}
	return t.update(object{"status": status, "stage": "DONE", "files": files, "message": fmt.Sprintf("下载结束，已校验 %d/%d 个附件", verified, len(resolved))})
}
func activateDownload(entry exportLocator, ref string) error {
	row, e := entry.loc.ElementHandle()
	if e != nil || row == nil {
		return problem("DOWNLOAD_RECORD_MISSING", "已关联下载记录不再可见")
	}
	defer row.Dispose()
	buttons, e := row.QuerySelectorAll("a.download_ic,a[download]")
	if e != nil {
		return e
	}
	if len(buttons) == 0 {
		controls, e := row.QuerySelectorAll("a,button,[role=button],[role=link],[onclick]")
		if e != nil {
			return e
		}
		for _, c := range controls {
			text, _ := c.InnerText()
			if exportDownloadLabel.MatchString(strings.TrimSpace(text)) {
				buttons = append(buttons, c)
			}
		}
	}
	visibleButtons := []pw.ElementHandle{}
	for _, b := range buttons {
		v, _ := b.IsVisible()
		if v {
			visibleButtons = append(visibleButtons, b)
		}
	}
	if len(visibleButtons) != 1 {
		return problem("DOWNLOAD_CONTROL_AMBIGUOUS", "已关联记录内没有唯一下载按钮")
	}
	raw, e := row.Evaluate(script("export_row_identity"))
	if e != nil {
		return e
	}
	current := obj(raw)
	if exportRef(current) != ref || exportStatus(str(current["text"])) != "READY" {
		return problem("DOWNLOAD_RECORD_CHANGED", "导出记录或完成状态已变化")
	}
	v, e := visibleButtons[0].Evaluate(script("export_download_activate"), object{"row": row, "text": current["text"], "platform_id": current["platform_id"]})
	if e != nil {
		return e
	}
	if !flag(v) {
		return problem("DOWNLOAD_CONTROL_UNAVAILABLE", "下载按钮当前不可用")
	}
	return nil
}
func verifyZIP(path string) (object, error) {
	archive, e := zip.OpenReader(path)
	if e != nil {
		return nil, e
	}
	defer archive.Close()
	if len(archive.File) == 0 {
		return nil, errors.New("压缩包没有学生附件")
	}
	for _, entry := range archive.File {
		r, e := entry.Open()
		if e != nil {
			return nil, e
		}
		_, e = io.Copy(io.Discard, r)
		r.Close()
		if e != nil {
			return nil, errors.New("压缩包 CRC 校验失败")
		}
	}
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	h := sha256.New()
	size, e := io.Copy(h, f)
	if e != nil || size == 0 {
		return nil, errors.New("下载文件为空或不可读")
	}
	return object{"size": size, "sha256": hex.EncodeToString(h.Sum(nil)), "status": "VERIFIED"}, nil
}
func (t *Task) saveDownload(d pw.Download, export object) {
	defer t.manager.wg.Done()
	ref := str(export["ref"])
	name := safeName(d.SuggestedFilename())
	path := filepath.Join(t.manager.cfg.Downloads, t.id, textHash(ref)[:12]+"-"+name)
	partial := path + ".part"
	_ = os.MkdirAll(filepath.Dir(path), 0700)
	transfer := make(chan error, 1)
	go func() { transfer <- d.SaveAs(partial) }()
	deadline := time.NewTimer(30 * time.Minute)
	defer deadline.Stop()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	lastChange := time.Now()
	var size int64
	var err error
	for running := true; running; {
		select {
		case err = <-transfer:
			running = false
		case <-t.ctx.Done():
			err = t.ctx.Err()
			_ = d.Cancel()
			<-transfer
			running = false
		case <-deadline.C:
			err = errors.New("文件传输超过总时限")
			_ = d.Cancel()
			<-transfer
			running = false
		case <-ticker.C:
			t.downloadMu.Lock()
			received := int64(t.downloadBytes[t.downloadURLs[d.URL()]])
			t.downloadMu.Unlock()
			if received > size {
				size = received
				lastChange = time.Now()
			}
			if info, e := os.Stat(partial); e == nil && info.Size() > size {
				size = info.Size()
				lastChange = time.Now()
				_ = t.update(object{"download_progress": object{"received_bytes": size, "state": "inProgress"}})
			}
			if time.Since(lastChange) > 120*time.Second {
				err = errors.New("文件传输停滞")
				_ = d.Cancel()
				<-transfer
				running = false
			}
		}
	}
	result := object{"export_ref": ref, "status": "FAILED", "error": "下载或压缩包校验失败"}
	if err == nil {
		err = d.Failure()
	}
	if err == nil {
		_ = t.update(object{"stage": "VERIFY"})
		var checked object
		checked, err = verifyZIP(partial)
		if err == nil {
			err = os.Rename(partial, path)
			if err == nil {
				result = checked
				result["path"], result["filename"], result["export_ref"] = path, name, ref
				result["class_name"] = ""
				for _, c := range objects(obj(t.snapshot()["class_scope"])["classes"]) {
					if !flag(c["all"]) && str(c["name"]) != "" && strings.Contains(str(export["text"]), str(c["name"])) {
						result["class_name"] = c["name"]
						break
					}
				}
			}
		}
	}
	if err != nil {
		_ = os.Remove(partial)
	}
	t.op.Lock()
	defer t.op.Unlock()
	t.downloadBusy = false
	s := t.snapshot()
	files := array(s["files"])
	files = append(files, result)
	fields := object{"files": files}
	if !terminal(str(s["status"])) {
		fields["stage"] = "WAIT_EXPORT"
	}
	_ = t.update(fields)
}
