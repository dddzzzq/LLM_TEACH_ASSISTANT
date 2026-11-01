package browserworker

import (
	"errors"
	"fmt"
	"net/url"
	"sort"
	"strings"
	"time"

	pw "github.com/mxschmitt/playwright-go"
)

type collection struct {
	task                     *Task
	progress, original       object
	listPage                 pw.Page
	listURL                  string
	listSources              map[string]bool
	pageCourses              map[pw.Page]string
	detailCourse, detailWork string
}

func newCollection(t *Task) *collection {
	s := t.snapshot()
	progress := obj(s["catalog_progress"])
	if _, ok := progress["courses"].(map[string]any); !ok {
		progress = object{"courses": object{}, "courses_complete": false}
	}
	return &collection{task: t, original: clone(obj(s["target"])), progress: progress, pageCourses: map[pw.Page]string{}, listSources: map[string]bool{}}
}
func orderedKeys(values object) []string {
	keys := []string{}
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
func catalogSelector(kind string) string {
	if kind == "course" {
		return env("PORTAL_COURSE_SELECTOR", constants["portal_catalog.COURSE_SELECTOR"])
	}
	return env("PORTAL_HOMEWORK_SELECTOR", constants["portal_catalog.WORK_SELECTOR"])
}

func catalogSurface(value string) bool {
	return portalURL(strings.Replace(value, "http://", "https://", 1))
}

func catalogFrame(frame pw.Frame) bool {
	return frame.URL() == "about:blank" || frame.URL() == "about:srcdoc" || catalogSurface(frame.URL())
}
func sourceKey(raw object) string {
	parts := []string{}
	for _, k := range []string{"source_route", "source_path", "source_page", "title", "term"} {
		parts = append(parts, str(raw[k]))
	}
	return "pending:" + textHash(strings.Join(parts, "\x00"))[:24]
}
func cleanCatalogURL(value string) string {
	if !portalURL(value) {
		return ""
	}
	u, _ := url.Parse(value)
	allowed := map[string]bool{}
	for _, k := range []string{"courseid", "clazzid", "classid", "workid", "homeworkid", "taskid", "cpi", "ut", "mooc2", "enc", "t", "type", "page", "pageindex", "pagesize", "roleid"} {
		allowed[k] = true
	}
	for k := range u.Query() {
		if !allowed[strings.ToLower(k)] {
			return ""
		}
	}
	return u.String()
}
func normalizeRecord(raw object, kind, base string) object {
	dest := str(raw["href"])
	if dest != "" {
		u, e := url.Parse(base)
		r, er := url.Parse(dest)
		if e == nil && er == nil {
			dest = u.ResolveReference(r).String()
		}
	}
	id := str(raw["course_id"])
	if kind != "course" {
		id = str(raw["work_id"])
	}
	if id == "" && portalURL(strings.Replace(dest, "http://", "https://", 1)) {
		if kind == "course" {
			id = queryID(dest, "courseid")
		} else {
			id = queryID(dest, "workid", "homeworkid", "taskid")
		}
	}
	clazz := str(raw["class_id"])
	if clazz == "" && portalURL(strings.Replace(dest, "http://", "https://", 1)) {
		clazz = queryID(dest, "clazzid", "classid")
	}
	if id != "" && kind == "course" && clazz != "" {
		id += ":" + clazz
	}
	if str(raw["title"]) == "" {
		return nil
	}
	return object{"platform_id": id, "title": raw["title"], "url": cleanCatalogURL(dest), "term": raw["term"], "state": raw["state"], "deadline": raw["deadline"], "class_id": clazz, "class_name": raw["class_name"], "source_key": sourceKey(raw)}
}
func readRecords(page pw.Page, kind string) ([]object, error) {
	found := map[string]object{}
	if page == nil || page.IsClosed() || !catalogSurface(page.URL()) {
		return nil, errors.New("当前目录页面不可用")
	}
	for _, f := range page.Frames() {
		if !catalogFrame(f) {
			continue
		}
		js := script("catalog_extract")
		if kind == "homework" {
			js = script("homework_extract")
		}
		raw, e := f.Locator(catalogSelector(kind)).EvaluateAll(js)
		if e != nil {
			return nil, e
		}
		for _, r := range objects(raw) {
			if flag(r["identity_conflict"]) {
				return nil, problem("CATALOG_AMBIGUOUS", "同一作业记录包含冲突的官网标识")
			}
			row := normalizeRecord(r, kind, f.URL())
			if row == nil {
				continue
			}
			key := str(row["platform_id"])
			if key == "" {
				key = str(row["source_key"])
			}
			if prior := found[key]; prior != nil && prior["title"] != row["title"] {
				return nil, problem("CATALOG_AMBIGUOUS", "同一官网标识对应不同名称")
			}
			found[key] = row
		}
	}
	out := []object{}
	keys := []string{}
	for k := range found {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		out = append(out, found[k])
	}
	return out, nil
}
func emptyCatalog(page pw.Page, kind string) bool {
	word := "课程"
	if kind == "homework" {
		word = "作业"
	}
	for _, f := range page.Frames() {
		if !catalogFrame(f) {
			continue
		}
		loc := f.Locator("[data-empty],.empty,.no-data,.nodata")
		for i := 0; i < count(loc); i++ {
			if visible(loc.Nth(i)) {
				text, _ := loc.Nth(i).InnerText()
				if contains(text, "暂无", "没有", "无"+word) {
					return true
				}
			}
		}
		texts := f.GetByText(regexpForEmpty(word))
		for i := 0; i < count(texts); i++ {
			if visible(texts.Nth(i)) {
				return true
			}
		}
	}
	return false
}
func (c *collection) detailActive() bool { return c.detailWork != "" }
func (c *collection) courses() object    { return obj(c.progress["courses"]) }
func requirementsComplete(course object) bool {
	for _, work := range obj(course["homeworks"]) {
		if len(obj(obj(work)["detail"])) == 0 {
			return false
		}
	}
	return true
}
func (c *collection) viewport(kind string, delta float64) (object, error) {
	p := c.task.page
	if p == nil || p.IsClosed() {
		return nil, errors.New("当前列表页面不可用")
	}
	best := -1
	var frame pw.Frame
	var view object
	for _, f := range p.Frames() {
		if !catalogFrame(f) {
			continue
		}
		raw, e := f.Locator(catalogSelector(kind)).EvaluateAll(script("catalog_viewport"), object{"delta": 0})
		if e != nil {
			return nil, e
		}
		v := obj(raw)
		if int(num(v["record_count"])) > best {
			best = int(num(v["record_count"]))
			frame = f
			view = v
		}
	}
	if frame == nil {
		return nil, problem("CATALOG_VIEWPORT_UNAVAILABLE", "当前没有可读取的课程或作业列表")
	}
	if delta != 0 {
		before := view
		_, e := frame.Locator(catalogSelector(kind)).EvaluateAll(script("catalog_viewport"), object{"delta": delta})
		if e != nil {
			return nil, e
		}
		deadline, quiet := time.Now().Add(3*time.Second), time.Now()
		marker := ""
		for {
			time.Sleep(100 * time.Millisecond)
			raw, e := frame.Locator(catalogSelector(kind)).EvaluateAll(script("catalog_viewport"), object{"delta": 0})
			if e != nil {
				return nil, e
			}
			view = obj(raw)
			next := hash([]any{view["top"], view["height"], view["record_count"], view["loading"]})
			if marker != next || flag(view["loading"]) {
				marker = next
				quiet = time.Now()
			}
			if time.Now().After(deadline) || time.Since(quiet) >= 600*time.Millisecond {
				break
			}
		}
		if delta > 0 && flag(before["needs_scroll"]) && num(view["top"]) <= num(before["top"]) && num(view["record_count"]) <= num(before["record_count"]) {
			return nil, problem("CATALOG_SCROLL_STALLED", "目录列表未能向下滚动，请检查加载状态")
		}
	}
	view["kind"] = kind
	return view, nil
}
func (c *collection) readCourses() error {
	rows, e := readRecords(c.task.page, "course")
	if e != nil {
		return e
	}
	empty := len(rows) == 0 && emptyCatalog(c.task.page, "course")
	if len(rows) == 0 && !empty {
		return problem("CATALOG_CARDS_UNAVAILABLE", "未能读取当前课程卡片，请检查页面加载")
	}
	courses := c.courses()
	for _, r := range rows {
		key := str(r["platform_id"])
		if key == "" {
			key = str(r["source_key"])
		}
		var prior object
		for _, v := range courses {
			candidate := obj(v)
			if candidate["source_key"] == r["source_key"] {
				prior = candidate
				break
			}
		}
		if prior != nil && (str(r["platform_id"]) == "" || str(prior["platform_id"]) == "" || r["platform_id"] == prior["platform_id"]) {
			if str(prior["platform_id"]) == "" && str(r["platform_id"]) != "" {
				for _, k := range []string{"platform_id", "url", "class_id", "class_name"} {
					prior[k] = r[k]
				}
			}
			continue
		}
		if old := obj(courses[key]); len(old) > 0 {
			if old["title"] != r["title"] || old["term"] != r["term"] {
				return problem("CATALOG_AMBIGUOUS", "同一课程标识对应不同记录")
			}
			continue
		}
		r["classes"] = []any{}
		r["homeworks"] = object{}
		r["complete"] = false
		courses[key] = r
	}
	if len(courses) > 500 {
		return problem("CATALOG_LIMIT", "课程数超过读取上限")
	}
	view, e := c.viewport("course", 0)
	if e != nil {
		return e
	}
	c.progress["courses_complete"] = flag(c.progress["courses_complete"]) || empty || flag(view["at_end"])
	c.listPage, c.listURL = c.task.page, c.task.page.URL()
	c.listSources = map[string]bool{}
	for _, r := range rows {
		c.listSources[str(r["source_key"])] = true
	}
	delete(c.pageCourses, c.task.page)
	return c.task.update(object{"catalog_progress": c.progress})
}
func (c *collection) resolveCourse(key string, page pw.Page) bool {
	course := obj(c.courses()[key])
	if str(course["platform_id"]) != "" {
		return true
	}
	if c.pageCourses[page] != key {
		return false
	}
	ids, classes := map[string]bool{}, map[string]bool{}
	dest := ""
	for _, f := range page.Frames() {
		if !catalogFrame(f) {
			continue
		}
		id := queryID(f.URL(), "courseid")
		if id != "" {
			ids[id] = true
			if cleanCatalogURL(f.URL()) != "" {
				dest = cleanCatalogURL(f.URL())
			}
		}
		if clazz := queryID(f.URL(), "clazzid", "classid"); clazz != "" {
			classes[clazz] = true
		}
	}
	if len(ids) != 1 || len(classes) > 1 {
		return false
	}
	var id, clazz string
	for k := range ids {
		id = k
	}
	for k := range classes {
		clazz = k
	}
	if clazz != "" {
		id += ":" + clazz
	}
	course["platform_id"], course["class_id"], course["url"] = id, clazz, dest
	return true
}
func readClasses(page pw.Page, course object) ([]any, error) {
	found := object{}
	for _, f := range page.Frames() {
		if !catalogFrame(f) {
			continue
		}
		v, e := f.Locator(env("PORTAL_CLASS_SELECTOR", constants["portal_catalog.CLASS_SELECTOR"])).EvaluateAll(script("catalog_extract"))
		if e != nil {
			return nil, e
		}
		for _, r := range objects(v) {
			id := str(r["class_id"])
			title := str(r["class_name"])
			if title == "" {
				title = str(r["title"])
			}
			if id == "" || id == "0" || id == "-1" || id == "all" || contains(title, "全部班级", "所有班级", "请选择") {
				continue
			}
			found[id] = object{"platform_id": id, "name": title}
		}
	}
	if len(found) == 0 && str(course["class_id"]) != "" && str(course["class_name"]) != "" {
		found[str(course["class_id"])] = object{"platform_id": course["class_id"], "name": course["class_name"]}
	}
	out := []any{}
	for _, k := range orderedKeys(found) {
		out = append(out, found[k])
	}
	return out, nil
}
func (c *collection) readHomeworks(key string) error {
	course := obj(c.courses()[key])
	if len(course) == 0 {
		return errors.New("必须先读取官网课程记录")
	}
	p := c.task.page
	if !c.resolveCourse(key, p) {
		return problem("CATALOG_COURSE_ID_PENDING", "请先从课程卡片进入课程，核验唯一官网标识")
	}
	expected := strings.Split(str(course["platform_id"]), ":")
	matched := false
	for _, f := range p.Frames() {
		if !catalogFrame(f) {
			continue
		}
		id := queryID(f.URL(), "courseid")
		clazz := queryID(f.URL(), "clazzid", "classid")
		if id != "" && id != expected[0] || len(expected) > 1 && clazz != "" && clazz != expected[1] {
			return problem("CATALOG_COURSE_UNVERIFIED", "当前页面课程或班级与所选记录不一致")
		}
		matched = matched || id == expected[0]
	}
	if !matched && c.pageCourses[p] != key {
		return problem("CATALOG_COURSE_UNVERIFIED", "当前页面与要读取的课程不一致")
	}
	rows, e := readRecords(p, "homework")
	if e != nil {
		return e
	}
	empty := len(rows) == 0 && emptyCatalog(p, "homework")
	if len(rows) == 0 && !empty {
		return problem("HOMEWORK_IDS_UNAVAILABLE", "尚未提取到可核验的作业记录，请检查作业列表")
	}
	works := obj(course["homeworks"])
	for _, r := range rows {
		id := str(r["platform_id"])
		if id == "" {
			return problem("HOMEWORK_IDS_UNAVAILABLE", "作业缺少官网标识，请核对页面入口")
		}
		old := obj(works[id])
		if len(old) > 0 && old["title"] != r["title"] {
			return problem("CATALOG_AMBIGUOUS", "同一作业标识对应不同名称")
		}
		if old["detail"] != nil {
			r["detail"] = old["detail"]
		}
		works[id] = r
	}
	classes, e := readClasses(p, course)
	if e != nil {
		return e
	}
	merged := object{}
	for _, r := range append(array(course["classes"]), classes...) {
		v := obj(r)
		merged[str(v["platform_id"])] = v
	}
	all := []any{}
	for _, id := range orderedKeys(merged) {
		all = append(all, merged[id])
	}
	course["classes"] = all
	view, e := c.viewport("homework", 0)
	if e != nil {
		return e
	}
	course["complete"] = flag(course["complete"]) || empty || flag(view["at_end"])
	course["empty"] = empty && len(works) == 0
	course["homeworks"] = works
	c.pageCourses[p] = key
	if len(works) > 2000 || len(all) > 500 {
		return problem("CATALOG_LIMIT", "课程目录超过本次读取上限")
	}
	c.task.event(fmt.Sprintf("已读取课程“%s”：累计 %d 份作业", str(course["title"]), len(works)))
	return c.task.update(object{"catalog_progress": c.progress, "last_catalog_read": object{"course_id": key, "rows": len(rows), "total_homeworks": len(works), "complete": course["complete"], "empty": course["empty"]}})
}
func (c *collection) observation() (object, error) {
	courses := c.courses()
	current := c.pageCourses[c.task.page]
	course := obj(courses[current])
	completed := 0
	nextCourse := ""
	summaries := []any{}
	for _, key := range orderedKeys(courses) {
		v := obj(courses[key])
		works := obj(v["homeworks"])
		read := 0
		for _, w := range works {
			if len(obj(obj(w)["detail"])) > 0 {
				read++
			}
		}
		done := flag(v["complete"]) && requirementsComplete(v)
		if done {
			completed++
		} else if nextCourse == "" {
			nextCourse = key
		}
		summaries = append(summaries, object{"course_id": key, "title": v["title"], "term": v["term"], "homework_count": len(works), "class_count": len(array(v["classes"])), "classes": v["classes"], "complete": flag(v["complete"]), "requirements_read": read, "requirements_complete": requirementsComplete(v), "empty": flag(v["empty"]), "identity_pending": str(v["platform_id"]) == ""})
	}
	phase, kind := "COURSE_LIST", "course"
	if current != "" {
		phase, kind = "COLLECT_HOMEWORKS", "homework"
		if flag(course["complete"]) {
			phase = "READ_REQUIREMENTS"
		}
	}
	view, e := c.viewport(kind, 0)
	if e != nil {
		return nil, e
	}
	manifest, pending := []any{}, []any{}
	nextWork := ""
	works := obj(course["homeworks"])
	for _, key := range orderedKeys(works) {
		w := obj(works[key])
		status := "PENDING"
		if len(obj(w["detail"])) > 0 {
			status = "READ"
		} else if nextWork == "" {
			nextWork = key
		}
		manifest = append(manifest, object{"homework_id": key, "title": w["title"], "status": status})
	}
	step := "VISIT_NEXT_COURSE"
	listReady := false
	if current == "" {
		if flag(c.progress["courses_complete"]) && completed == len(courses) {
			step = "FINISH_CATALOG"
		}
	} else {
		rows, e := readRecords(c.task.page, "homework")
		if e != nil {
			return nil, e
		}
		listReady = len(rows) > 0 || emptyCatalog(c.task.page, "homework")
		unread := len(works) == 0
		for _, r := range rows {
			if works[str(r["platform_id"])] == nil {
				unread = true
			}
			if str(r["platform_id"]) == nextWork && flag(course["complete"]) {
				pending = append(pending, object{"homework_id": nextWork, "title": obj(works[nextWork])["title"]})
			}
		}
		switch {
		case flag(course["complete"]) && nextWork == "":
			step = "RETURN_TO_COURSE_LIST"
		case !listReady:
			step = "OPEN_HOMEWORK_LIST"
		case !flag(course["complete"]):
			if unread || flag(view["at_end"]) {
				step = "READ_HOMEWORK_LIST"
			} else if flag(view["loading"]) {
				step = "WAIT_FOR_LIST"
			} else if flag(view["has_next"]) {
				step = "NEXT_HOMEWORK_PAGE"
			} else {
				step = "SCROLL_HOMEWORK_LIST"
			}
		case len(pending) > 0:
			step = "SELECT_HOMEWORK_DETAIL"
		default:
			step = "LOCATE_HOMEWORK"
		}
	}
	if !flag(course["complete"]) {
		manifest = []any{}
	}
	return object{"courses_complete": flag(c.progress["courses_complete"]), "current_course_id": current, "phase": phase, "manifest_ready": flag(course["complete"]), "homework_manifest": manifest, "next_homework_id": nextWork, "next_course_id": nextCourse, "return_required": current != "" && flag(course["complete"]) && requirementsComplete(course), "total_courses": len(courses), "completed_courses": completed, "courses": summaries, "viewport": view, "pending_homeworks": pending, "homework_list_ready": listReady, "next_step": step}, nil
}
func (c *collection) recordNavigation(r reference) (string, error) {
	raw, e := r.loc.Evaluate("e=>("+script("catalog_extract")+")([e])[0]", nil)
	if e != nil {
		return "", e
	}
	value := obj(raw)
	row := normalizeRecord(value, "course", r.page.URL())
	for _, key := range orderedKeys(c.courses()) {
		course := obj(c.courses()[key])
		if row != nil && str(row["platform_id"]) != "" && row["platform_id"] == course["platform_id"] || course["source_key"] == sourceKey(value) {
			return key, nil
		}
	}
	return c.pageCourses[r.page], nil
}
func (c *collection) selectDetail(id string) error {
	if c.detailActive() {
		return problem("DETAIL_SELECTION_INVALID", "请先读取当前已打开作业详情")
	}
	view, e := c.observation()
	if e != nil {
		return e
	}
	found := false
	for _, r := range objects(view["pending_homeworks"]) {
		found = found || str(r["homework_id"]) == id
	}
	if !found {
		return problem("DETAIL_SELECTION_INVALID", "请从已核验作业清单选择当前可见的待办作业")
	}
	current := c.pageCourses[c.task.page]
	course := obj(c.courses()[current])
	work := obj(obj(course["homeworks"])[id])
	c.detailCourse, c.detailWork = current, id
	t := c.task
	t.detailSource, t.detailSourceURL = t.page, t.page.URL()
	t.detailPage = nil
	t.detailURL = ""
	t.detailNav = nil
	t.courseConfirmed, t.assignmentConfirmed = true, false
	return t.update(object{"target": object{"course_name": course["title"], "assignment_name": work["title"], "term": course["term"], "portal": object{"course_id": course["platform_id"], "homework_id": id}}, "stage": "READING_DETAIL"})
}
func (c *collection) completeDetail(detail object) error {
	course := obj(c.courses()[c.detailCourse])
	work := obj(obj(course["homeworks"])[c.detailWork])
	if detail["course_id"] != course["platform_id"] || detail["homework_id"] != c.detailWork || detail["title"] != work["title"] {
		return problem("DETAIL_TARGET_UNVERIFIED", "读取结果与当前作业不一致")
	}
	work["detail"] = detail
	t := c.task
	if e := t.returnToHomeworkList(); e != nil {
		return e
	}
	c.detailCourse, c.detailWork = "", ""
	t.detailPage = nil
	t.detailURL = ""
	t.detailNav = nil
	t.courseConfirmed, t.assignmentConfirmed = false, false
	return t.update(object{"target": clone(c.original), "catalog_progress": c.progress, "stage": "READING_CATALOG"})
}
func (c *collection) returnToCourseList() error {
	t := c.task
	if c.detailActive() {
		return problem("DETAIL_NAVIGATION_LOCKED", "请先读取当前作业详情")
	}
	if current := c.pageCourses[t.page]; current != "" {
		course := obj(c.courses()[current])
		if !flag(course["complete"]) || !requirementsComplete(course) {
			t.event("当前课程尚未完成作业清单核验或要求读取，继续当前课程")
			return nil
		}
	}
	if c.listPage == nil || c.listPage.IsClosed() {
		return problem("CATALOG_LIST_UNAVAILABLE", "原课程列表页已关闭，请重新同步")
	}
	current := t.page
	if current != c.listPage {
		if e := current.Close(); e != nil {
			return e
		}
	}
	t.page = c.listPage
	if !c.isCourseList() {
		if !allowedURL(c.listURL) {
			return errors.New("原课程列表来源无效")
		}
		if _, e := t.page.Goto(c.listURL, pw.PageGotoOptions{WaitUntil: pw.WaitUntilStateDomcontentloaded}); e != nil {
			return e
		}
	}
	for attempt := 0; attempt < 20; attempt++ {
		if c.isCourseList() {
			delete(c.pageCourses, t.page)
			return nil
		}
		if !t.automatic() {
			return errors.New("目录读取已暂停或取消")
		}
		select {
		case <-t.ctx.Done():
			return t.ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
	return problem("CATALOG_LIST_UNAVAILABLE", "尚未恢复已核验的原课程列表")
}

func (c *collection) isCourseList() bool {
	t := c.task
	if t.page == nil || t.page.IsClosed() || t.page.URL() != c.listURL {
		return false
	}
	rows, e := readRecords(t.page, "course")
	if e != nil {
		return false
	}
	match := len(c.listSources) == 0 && emptyCatalog(t.page, "course")
	for _, r := range rows {
		match = match || c.listSources[str(r["source_key"])]
	}
	return match
}
func (c *collection) finish() error {
	if e := c.returnToCourseList(); e != nil {
		return e
	}
	if current := c.pageCourses[c.task.page]; current != "" {
		return nil
	}
	if e := c.readCourses(); e != nil {
		return e
	}
	if !flag(c.progress["courses_complete"]) {
		c.task.event("已返回课程列表，尚需滚动或翻页核验课程清单末尾")
		return nil
	}
	out := []any{}
	ids := map[string]bool{}
	total := 0
	for _, key := range orderedKeys(c.courses()) {
		course := obj(c.courses()[key])
		id := str(course["platform_id"])
		if id == "" || !flag(course["complete"]) || !requirementsComplete(course) || ids[id] {
			return problem("CATALOG_INCOMPLETE", "课程标识、作业清单或题目详情尚未完整核验")
		}
		ids[id] = true
		works := []any{}
		for _, k := range orderedKeys(obj(course["homeworks"])) {
			w := clone(obj(obj(course["homeworks"])[k]))
			delete(w, "source_key")
			works = append(works, w)
		}
		total += len(works)
		out = append(out, object{"platform_id": id, "title": course["title"], "term": course["term"], "url": course["url"], "classes": course["classes"], "homeworks": works})
	}
	if total > 10000 {
		return problem("CATALOG_LIMIT", "作业数量超过读取上限")
	}
	catalog := object{"complete": true, "details_included": true, "courses": out}
	b, _ := jsonMarshal(catalog)
	if len(b) > 6*1024*1024 {
		return problem("CATALOG_LIMIT", "目录超过本次读取上限")
	}
	return c.task.update(object{"portal_catalog": catalog, "catalog_progress": nil, "status": "SUCCEEDED", "stage": "DONE", "message": "课程、作业及作业详情读取完成"})
}
