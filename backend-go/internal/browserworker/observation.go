package browserworker

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

func (t *Task) inventory() (object, error) {
	pages, frames, elements := []any{}, []any{}, []any{}
	candidates := []reference{}
	target := obj(t.snapshot()["target"])
	names := []string{str(target["course_name"]), str(target["assignment_name"])}
	for pi, p := range t.context.Pages() {
		if p.IsClosed() {
			continue
		}
		title, e := p.Title()
		if e != nil {
			return nil, e
		}
		pages = append(pages, object{"page_id": fmt.Sprint(pi), "url": publicURL(p.URL()), "title": bounded(title, 120), "frame_count": len(p.Frames()), "active": p == t.page})
		if t.page != nil && p != t.page {
			continue
		}
		for fi, f := range p.Frames() {
			controls := f.Locator(constants["page_snapshot.CONTROLS"])
			raw, e := controls.EvaluateAll(script("page_snapshot"))
			if e != nil {
				return nil, e
			}
			data := obj(raw)
			headings := array(data["headings"])
			sort.SliceStable(headings, func(i, j int) bool {
				return containsName(obj(headings[i]), names) && !containsName(obj(headings[j]), names)
			})
			if len(headings) > 20 {
				headings = headings[:20]
			}
			data["headings"] = headings
			landmarks := array(data["landmarks"])
			if len(landmarks) > 10 {
				landmarks = landmarks[:10]
			}
			data["landmarks"] = landmarks
			frameID := fmt.Sprintf("p%df%d", pi, fi)
			items := objects(data["controls"])
			delete(data, "controls")
			data["frame_id"] = frameID
			data["page_id"] = fmt.Sprint(pi)
			data["name"] = bounded(f.Name(), 100)
			data["url"] = publicURL(f.URL())
			data["element_count"] = len(items)
			frames = append(frames, data)
			for _, item := range items {
				idx := int(num(item["i"]))
				delete(item, "i")
				if str(item["href"]) != "" {
					item["href"] = publicURL(str(item["href"]))
				}
				for _, pair := range [][2]string{{"context_id", "c"}, {"record_id", "r"}, {"semester_menu", "m"}} {
					if v, ok := item[pair[0]]; ok {
						if v != nil {
							item[pair[0]] = frameID + pair[1] + fmt.Sprint(v)
						} else {
							item[pair[0]] = ""
						}
					}
				}
				item["ref"] = fmt.Sprintf("%se%d", frameID, idx)
				item["page_id"] = fmt.Sprint(pi)
				item["frame_id"] = frameID
				candidates = append(candidates, reference{p, controls.Nth(idx), item})
			}
		}
	}
	priority := func(r reference) int {
		i := r.item
		if flag(i["in_viewport"]) {
			if str(i["semester_kind"]) == "option" {
				return 0
			}
			if str(i["semester_kind"]) == "trigger" {
				return 1
			}
		}
		if containsName(i, names) {
			return 2
		}
		if flag(i["in_viewport"]) {
			if contains(str(i["name"]), "个人空间", "我的课程", "课程", "作业", "批阅", "更多", "导出", "搜索") {
				return 3
			}
			return 4
		}
		return 5
	}
	sort.SliceStable(candidates, func(i, j int) bool { return priority(candidates[i]) < priority(candidates[j]) })
	t.refs = map[string]reference{}
	b, _ := json.Marshal(object{"pages": pages, "frames": frames})
	size := len(b)
	for _, r := range candidates {
		b, _ = json.Marshal(r.item)
		if size+len(b) > 60000 {
			continue
		}
		elements = append(elements, r.item)
		t.refs[str(r.item["ref"])] = r
		size += len(b)
		if len(elements) >= 400 {
			break
		}
	}
	targets := object{}
	for _, pair := range [][2]string{{"course", "course_name"}, {"assignment", "assignment_name"}} {
		groups := map[string]bool{}
		for _, r := range candidates {
			if itemMatches(r.item, str(target[pair[1]])) {
				group := str(r.item["record_id"])
				if group == "" {
					group = str(r.item["context_id"])
				}
				if group == "" {
					group = str(r.item["ref"])
				}
				groups[group] = str(target["term"]) != "" && strings.Contains(str(r.item["context"]), str(target["term"]))
			}
		}
		keys, matches := []any{}, []any{}
		ordered := []string{}
		for k := range groups {
			ordered = append(ordered, k)
		}
		sort.Strings(ordered)
		for _, k := range ordered {
			if len(keys) < 12 {
				keys = append(keys, k)
			}
			if groups[k] && len(matches) < 2 {
				matches = append(matches, k)
			}
		}
		targets[pair[0]] = object{"group_count": len(groups), "count_basis": "record", "record_ids": keys, "term_matches": matches}
	}
	return object{"pages": pages, "frames": frames, "elements": elements, "active_page_id": pageIndex(t.context, t.page), "target_candidates": targets, "coverage": object{"total_elements": len(candidates), "included_elements": len(elements), "truncated": len(elements) < len(candidates), "scope": "active_page", "snapshot_char_budget": 60000}}, nil
}
func itemMatches(item object, name string) bool {
	if name == str(item["name"]) {
		return true
	}
	for _, v := range array(item["context_labels"]) {
		if str(v) == name {
			return true
		}
	}
	return false
}
func containsName(item object, names []string) bool {
	for _, n := range names {
		if n == str(item["text"]) || itemMatches(item, n) {
			return true
		}
	}
	return false
}
func (t *Task) reconcile() {
	target := obj(t.snapshot()["target"])
	selected := obj(target["portal"])
	if len(selected) > 0 {
		expected := strings.Split(str(selected["course_id"]), ":")
		for _, p := range t.context.Pages() {
			if p.IsClosed() {
				continue
			}
			for _, f := range p.Frames() {
				course, clazz, work := queryID(f.URL(), "courseid"), queryID(f.URL(), "clazzid", "classid"), queryID(f.URL(), "workid", "homeworkid", "taskid")
				if course != "" && course != expected[0] {
					continue
				}
				if len(expected) > 1 && clazz != "" && clazz != expected[1] {
					continue
				}
				if course == expected[0] && (len(expected) == 1 || clazz == expected[1]) {
					t.courseConfirmed = true
				}
				if t.courseConfirmed && work == str(selected["homework_id"]) {
					t.assignmentConfirmed = true
				}
			}
		}
		return
	}
	for _, p := range t.context.Pages() {
		if p.IsClosed() {
			continue
		}
		for _, f := range p.Frames() {
			titles, _ := f.Locator("h1,h2,h3,h4,h5,h6,[role=heading],.courseName,.course-name,.coursename,[data-course-name]").EvaluateAll(script("verified_headings"))
			for _, title := range array(titles) {
				if str(title) == str(target["course_name"]) {
					t.courseConfirmed = true
				}
			}
			if t.courseConfirmed {
				titles, _ = f.Locator("h1,h2,h3,h4,h5,h6,[role=heading],.workName,.work-name,.work-title,.workTitle,.mark_title").EvaluateAll(script("verified_headings"))
				for _, title := range array(titles) {
					if str(title) == str(target["assignment_name"]) {
						if entry := t.exportEntry(); entry != nil && entry.page == p {
							t.assignmentConfirmed = true
						}
					}
				}
			}
		}
	}
}
func (t *Task) navigationView() (object, error) {
	if t.reading.Load() {
		return nil, nil
	}
	var auth object
	var err error
	if t.auth != nil {
		auth, err = t.auth.observation()
		if err != nil {
			return nil, err
		}
		if !flag(auth["login_form"]) && !flag(auth["otp_required"]) && !flag(auth["trust_prompt"]) && !t.loginRequired() && (t.verifiedSubject != "" || t.authenticated()) {
			ok, e := t.auth.identity()
			if e != nil {
				return nil, e
			}
			if !ok {
				t.human("LOGIN_IDENTITY", "请核对官网教师身份与绑定账号一致后交还自动化")
				return nil, nil
			}
			t.verifiedSubject = str(t.auth.credentials["username"])
			auth["authenticated"] = true
			if str(t.snapshot()["task_type"]) == "AUTH_BOOTSTRAP" && t.catalog == nil {
				t.catalog = newCollection(t)
			}
		}
		if flag(auth["otp_required"]) && flag(auth["code_requested"]) && !flag(auth["otp_entered"]) {
			t.waitCode(str(auth["code_method"]))
			return nil, nil
		}
	} else if t.loginRequired() && t.captchaFrame() == nil {
		t.human("LOGIN", "请在任务浏览器完成官网登录")
		return nil, nil
	}
	t.reconcile()
	stage := "LOCATE_COURSE"
	if t.courseConfirmed {
		stage = "LOCATE_ASSIGNMENT"
	}
	if t.assignmentConfirmed {
		stage = "EXPORT"
	}
	if t.catalog != nil {
		stage = "READING_CATALOG"
	}
	if str(t.snapshot()["task_type"]) == "HOMEWORK_DETAIL" || t.detailPage != nil || t.catalog != nil && t.catalog.detailActive() {
		stage = "READING_DETAIL"
	}
	if auth != nil && !flag(auth["authenticated"]) {
		stage = "AUTHENTICATE"
	}
	captcha := t.captchaFrame()
	if captcha != nil && !(flag(auth["login_form"]) && !flag(auth["credentials_filled"])) {
		stage = "SOLVE_CAPTCHA"
	}
	t.reconcile()
	if stage != str(t.snapshot()["stage"]) {
		_ = t.update(object{"stage": stage})
	}
	view, e := t.inventory()
	if e != nil {
		return nil, e
	}
	t.view = view
	view["task_type"] = t.snapshot()["task_type"]
	if view["task_type"] == "" {
		view["task_type"] = "DOWNLOAD"
	}
	view["target_evidence"] = object{"course_confirmed": t.courseConfirmed, "assignment_confirmed": t.assignmentConfirmed}
	if t.catalog == nil && str(t.snapshot()["task_type"]) != "HOMEWORK_DETAIL" {
		moreCount := len(t.exportMoreControls(t.page))
		view["export_navigation"] = object{"visible_entries": len(t.exportCandidates(t.page, false)), "more_controls": moreCount}
		view["can_open_export"] = t.courseConfirmed && t.assignmentConfirmed && (t.exportEntry() != nil || moreCount == 1)
	} else {
		view["can_open_export"] = false
	}
	if auth != nil {
		view["authentication"] = auth
	}
	if stage == "SOLVE_CAPTCHA" {
		view["captcha"] = object{"type": "slider", "visible": true, "attempts_remaining": 3 - num(t.snapshot()["captcha_attempts"])}
	}
	if t.catalog != nil && !t.catalog.detailActive() {
		c, e := t.catalog.observation()
		if e != nil {
			return nil, e
		}
		view["catalog"] = c
		if str(c["next_step"]) == "WAIT_FOR_LIST" {
			return nil, nil
		}
	}
	if str(t.snapshot()["task_type"]) == "HOMEWORK_DETAIL" || t.catalog != nil && t.catalog.detailActive() {
		detail, e := t.detailObservation()
		if e != nil {
			return nil, e
		}
		if flag(detail["pending"]) {
			return nil, nil
		}
		view["homework_detail"] = detail
		view["can_open_export"] = false
	}
	return view, nil
}
func (t *Task) recordNavigation(r reference) error {
	target := obj(t.snapshot()["target"])
	selected := obj(target["portal"])
	if len(selected) > 0 {
		raw, e := r.loc.Evaluate("e => ("+script("catalog_extract")+")([e])[0]", nil)
		if e != nil {
			return e
		}
		row := obj(raw)
		expected := strings.Split(str(selected["course_id"]), ":")
		course, clazz := str(row["course_id"]), str(row["class_id"])
		if course == "" {
			course = queryID(str(row["href"]), "courseid")
		}
		if clazz == "" {
			clazz = queryID(str(row["href"]), "clazzid", "classid")
		}
		if course != "" && course != expected[0] || len(expected) > 1 && clazz != "" && clazz != expected[1] {
			return problem("TARGET_ID_MISMATCH", "该页面入口属于其他官网课程或班级")
		}
		if itemMatches(r.item, str(target["course_name"])) {
			id := str(row["course_id"])
			if id == "" {
				id = queryID(str(row["href"]), "courseid")
			}
			expected := strings.Split(str(selected["course_id"]), ":")[0]
			if id != "" && id != expected {
				return problem("TARGET_ID_MISMATCH", "该课程与选定官网标识不一致")
			}
			if id == expected {
				t.courseConfirmed = true
			}
		}
		if t.courseConfirmed && itemMatches(r.item, str(target["assignment_name"])) {
			raw, e := r.loc.Evaluate("e => ("+script("homework_extract")+")([e])", nil)
			if e != nil {
				return e
			}
			rows := objects(raw)
			if len(rows) == 1 {
				id := str(rows[0]["work_id"])
				if flag(rows[0]["identity_conflict"]) || id != "" && id != str(selected["homework_id"]) {
					return problem("TARGET_ID_MISMATCH", "该作业与选定官网标识不一致")
				}
				if id == str(selected["homework_id"]) {
					t.assignmentConfirmed = true
				}
			}
		}
		return nil
	}
	for _, pair := range [][2]string{{"course", "course_name"}, {"assignment", "assignment_name"}} {
		if pair[0] == "assignment" && !t.courseConfirmed {
			continue
		}
		name := str(target[pair[1]])
		if !itemMatches(r.item, name) {
			continue
		}
		info := obj(obj(t.view["target_candidates"])[pair[0]])
		if num(info["group_count"]) > 1 && t.approved[pair[0]] == nil {
			matches := array(info["term_matches"])
			group := str(r.item["record_id"])
			if group == "" {
				group = str(r.item["context_id"])
			}
			if len(matches) != 1 || str(matches[0]) != group {
				return problem("TARGET_AMBIGUOUS", "页面存在同名目标，请先确认课程、作业及学期")
			}
		}
		if pair[0] == "course" {
			t.courseConfirmed = true
		} else {
			t.assignmentConfirmed = true
		}
	}
	return nil
}
