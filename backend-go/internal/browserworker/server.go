package browserworker

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
)

func (m *Manager) Handler() http.Handler { return http.HandlerFunc(m.serveHTTP) }
func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
func health() object {
	return object{"ok": true, "protocol_version": "browser.v1", "decision_driver": "go-eino", "execution_driver": "go-playwright", "captcha_driver": "python-opencv", "browser_stream": "browser.stream.v1", "target_resolution": "target.model-confirm.v1", "semester_selection": "manual", "download_activation": "download.bound-control.v1", "browser_startup": "bounded-cooldown.v2", "portal_auth_catalog": "same-browser.v1", "portal_automation": "teacher.portal.agent.v2", "portal_code_input": "teacher.code.browser.v1", "portal_wechat_code": "wechat.dynamic-code.v1", "catalog_reader": "homework.dom.v2", "catalog_navigation": "catalog.viewport.v1", "homework_detail": "question.resources.v1"}
}
func (m *Manager) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if !m.Authorized(r) {
		writeJSON(w, 401, object{"error": "Unauthorized"})
		return
	}
	if r.Method == "GET" && r.URL.Path == "/health" {
		writeJSON(w, 200, health())
		return
	}
	if r.Method == "GET" && r.URL.Path == "/capabilities" {
		raw, _ := assets.ReadFile("assets/actions.json")
		writeJSON(w, 200, object{"protocol_version": "browser.v1", "observation_version": "page.snapshot.v2", "execution_driver": "go-playwright", "action_schema": json.RawMessage(raw)})
		return
	}
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) == 3 && parts[0] == "portal" && parts[1] == "connections" && r.Method == "DELETE" {
		if e := m.deleteProfile(parts[2]); e != nil {
			writeJSON(w, 409, object{"error": str(publicError(e)["message"])})
			return
		}
		writeJSON(w, 200, object{"ok": true})
		return
	}
	if len(parts) < 2 || len(parts) > 3 || parts[0] != "tasks" || !validID(parts[1]) {
		writeJSON(w, 404, object{"error": "接口不存在"})
		return
	}
	id := parts[1]
	if len(parts) == 2 && r.Method == "POST" {
		payload, e := decodeObject(r)
		if e != nil {
			writeJSON(w, 400, object{"error": "请求参数无效"})
			return
		}
		t, e := m.Start(id, payload)
		if e != nil {
			writeJSON(w, 409, object{"error": bounded(e.Error(), 200)})
			return
		}
		writeJSON(w, 200, t.snapshot())
		return
	}
	t, e := m.task(id)
	if e != nil {
		writeJSON(w, 404, object{"error": "任务不存在"})
		return
	}
	if len(parts) == 2 && r.Method == "GET" {
		t.refreshLogin()
		writeJSON(w, 200, t.snapshot())
		return
	}
	if len(parts) != 3 {
		writeJSON(w, 404, object{"error": "接口不存在"})
		return
	}
	action := parts[2]
	if action == "stream" && r.Method == "GET" {
		m.serveStream(w, r, t)
		return
	}
	var result any
	var err error
	if r.Method == "GET" {
		switch action {
		case "observation":
			result, err = t.Observe()
		case "frame":
			result, err = t.Frame()
		case "portal-result":
			t.op.Lock()
			if str(t.snapshot()["task_type"]) == "" {
				err = errors.New("不是教务连接任务")
			} else {
				result = t.portalResult()
			}
			t.op.Unlock()
		case "export-diagnostics":
			t.op.Lock()
			frames := []any{}
			if t.context != nil {
				for _, p := range t.context.Pages() {
					if p.IsClosed() {
						continue
					}
					for _, f := range p.Frames() {
						v, e := f.Locator("select,option,li[onclick],[role=option],[class*='select' i],[class*='sel_' i]").EvaluateAll(script("export_diagnostics"))
						if e == nil && len(array(v)) > 0 {
							frames = append(frames, object{"controls": v})
						}
					}
				}
				rows, e := t.exportRows()
				err = e
				state := t.snapshot()
				result = object{"available": true, "frames": frames, "exports": rows, "last_error": state["last_error"], "last_export_step": state["last_export_step"], "export_navigation": object{"visible_entries": len(t.exportCandidates(t.page, false)), "legacy_entries": len(t.exportCandidates(t.page, true)), "more_controls": len(t.exportMoreControls(t.page))}}
			} else {
				result = object{"available": false}
			}
			t.op.Unlock()
		default:
			writeJSON(w, 404, object{"error": "未知操作"})
			return
		}
	} else if r.Method == "POST" {
		switch action {
		case "pause":
			err = t.Pause()
		case "resume":
			err = t.Resume()
		case "cancel":
			t.Stop("CANCELLED")
		default:
			var payload object
			payload, err = decodeObject(r)
			if err == nil {
				switch action {
				case "action":
					result, err = t.Apply(payload)
				case "decision-failed":
					result = t.DecisionFailed(payload)
				case "input":
					err = t.Input(payload, "http")
					result = object{"ok": err == nil}
				case "resolve-target", "class-scope", "bind-exports", "confirm-target":
					t.op.Lock()
					err = t.control(action, payload)
					t.op.Unlock()
				default:
					writeJSON(w, 404, object{"error": "未知操作"})
					return
				}
			}
		}
	} else {
		writeJSON(w, 405, object{"error": "不支持的请求方法"})
		return
	}
	if err != nil {
		message := "浏览器操作未完成，请刷新当前任务后重试"
		var issue *issue
		if errors.As(err, &issue) {
			message = issue.Message
		}
		writeJSON(w, 409, object{"error": message})
		return
	}
	if result == nil {
		result = t.snapshot()
	}
	writeJSON(w, 200, result)
}
func (t *Task) control(action string, p object) error {
	s := t.snapshot()
	human := str(s["control_mode"]) == "HUMAN" && !terminal(str(s["status"]))
	switch action {
	case "resolve-target":
		return t.resolveTarget(p)
	case "class-scope":
		if !human || flag(s["export_attempted"]) || !contains(str(s["waiting_reason"]), "CLASS_SCOPE", "SCOPE_SELECTION") {
			return errors.New("只有导出前的人工阶段可以确认范围")
		}
		mode := str(p["mode"])
		if mode != "all" && mode != "selected" {
			return errors.New("请选择班级范围")
		}
		options := objects(s["classes"])
		available, names := object{}, object{}
		hasAll := false
		for _, o := range options {
			id := str(o["id"])
			available[id] = o
			names[id] = o["name"]
			hasAll = hasAll || flag(o["all"])
		}
		if mode == "all" && !hasAll {
			return errors.New("当前导出窗口不支持全部班级")
		}
		ids := array(p["class_ids"])
		if mode == "selected" && len(ids) == 0 {
			return errors.New("请选择班级")
		}
		for _, v := range ids {
			o := obj(available[str(v)])
			if len(o) == 0 || flag(o["all"]) {
				return errors.New("班级列表已变化或选择无效")
			}
		}
		p["option_names"] = names
		return t.update(object{"requested_scope": p})
	case "bind-exports":
		if !human || !flag(s["export_attempted"]) || len(array(s["exports"])) > 0 {
			return errors.New("当前不能绑定导出记录")
		}
		rows, e := t.exportRows()
		if e != nil {
			return e
		}
		baseline := map[string]bool{}
		for _, v := range array(s["export_baseline"]) {
			baseline[str(v)] = true
		}
		candidates := object{}
		for _, row := range objects(rows) {
			if !baseline[str(row["ref"])] && !flag(row["ambiguous"]) {
				candidates[str(row["ref"])] = row
			}
		}
		selected := []any{}
		seen := map[string]bool{}
		for _, value := range array(p["refs"]) {
			if seen[str(value)] {
				continue
			}
			seen[str(value)] = true
			r, ok := candidates[str(value)]
			if !ok {
				return errors.New("导出记录已变化，请重新核对")
			}
			selected = append(selected, r)
		}
		if len(selected) == 0 {
			return errors.New("请选择导出记录")
		}
		return t.update(object{"exports": selected, "export_candidates": []any{}})
	case "confirm-target":
		if !human || flag(s["export_attempted"]) || str(s["waiting_reason"]) != "TARGET_CONFIRMATION" {
			return errors.New("当前不能重新确认目标")
		}
		target := obj(s["target"])
		if p["course_name"] != target["course_name"] || p["assignment_name"] != target["assignment_name"] || t.exportEntry() == nil {
			return errors.New("确认内容与任务目标或当前页面不一致")
		}
		t.courseConfirmed, t.assignmentConfirmed = true, true
		return nil
	}
	return errors.New("未知操作")
}
func (t *Task) refreshLogin() {
	if !t.op.TryLock() {
		return
	}
	defer t.op.Unlock()
	s := t.snapshot()
	if t.context == nil || terminal(str(s["status"])) || str(s["control_mode"]) != "HUMAN" || !contains(str(s["waiting_reason"]), "LOGIN") {
		return
	}
	authenticated := t.authenticated()
	status := "REQUIRED"
	if authenticated {
		status = "AUTHENTICATED"
	}
	if status != str(s["auth_status"]) {
		_ = t.update(object{"auth_status": status})
	}
}
func (t *Task) captureManualExport(area *exportDialog) error {
	options, e := t.classes(area.loc)
	if e != nil {
		return e
	}
	selected := []any{}
	mode := "manual"
	allSelected := false
	for _, o := range options {
		v := obj(o)
		if checked(t.classLocators[str(v["id"])]) {
			selected = append(selected, v)
			mode = "selected"
			if flag(v["all"]) {
				allSelected = true
			}
		}
	}
	if allSelected {
		mode = "all"
	}
	rows, e := t.exportRows()
	if e != nil {
		return e
	}
	refs := []any{}
	for _, r := range rows {
		refs = append(refs, obj(r)["ref"])
	}
	t.mu.Lock()
	t.manualPage = area.page
	t.mu.Unlock()
	return t.update(object{"manual_export_evidence": object{"baseline": refs, "rows": rows, "epoch": t.snapshot()["control_epoch"], "scope": object{"mode": mode, "classes": selected, "baseline_observed": true}}})
}

func (t *Task) checkpointManualConfirm(area *exportDialog, input object) error {
	if e := t.captureManualExport(area); e != nil {
		return e
	}
	x, y := num(input["x"]), num(input["y"])
	if area.frame.ParentFrame() != nil {
		element, e := area.frame.FrameElement()
		if e != nil {
			return e
		}
		box, e := element.BoundingBox()
		_ = element.Dispose()
		if e != nil || box == nil {
			return errors.New("导出设置页面位置已变化")
		}
		x -= box.X
		y -= box.Y
	}
	confirmed, e := area.loc.Evaluate(`(modal,{x,y,key})=>{const node=key?document.activeElement:document.elementFromPoint(x,y);const button=node?.closest('a.confirmDown,button');return !!button&&modal.contains(button)&&/确定|确认/.test(button.innerText||'')&&!button.disabled}`, object{"x": x, "y": y, "key": str(input["kind"]) == "key"})
	if e != nil {
		return e
	}
	if flag(confirmed) {
		_, e = t.manualExportSubmitted(area.page)
	}
	return e
}
func (t *Task) manualExportSubmitted(page any) (any, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := t.state
	proof := obj(s["manual_export_evidence"])
	if t.manualPage != page || str(s["control_mode"]) != "HUMAN" || terminal(str(s["status"])) || flag(s["export_attempted"]) || len(proof) == 0 || num(proof["epoch"]) != num(s["control_epoch"]) {
		return true, nil
	}
	s["export_attempted"], s["export_started_at"] = true, now()
	s["export_baseline"], s["export_baseline_rows"], s["class_scope"] = proof["baseline"], proof["rows"], proof["scope"]
	s["manual_export_evidence"] = nil
	s["stage"] = "WAIT_EXPORT"
	s["state_version"] = num(s["state_version"]) + 1
	t.storageErr = t.manager.persist(s)
	return true, t.storageErr
}
