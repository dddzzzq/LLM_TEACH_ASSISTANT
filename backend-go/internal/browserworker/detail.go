package browserworker

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	pw "github.com/mxschmitt/playwright-go"
)

func (t *Task) reconcileDetail() error {
	nav := t.detailNav
	if nav == nil {
		return nil
	}
	known := map[pw.Page]bool{}
	for _, p := range nav.pages {
		known[p] = true
	}
	candidates := []pw.Page{}
	for _, p := range t.context.Pages() {
		if known[p] || p.IsClosed() {
			continue
		}
		opener, _ := p.Opener()
		if opener == nav.page {
			candidates = append(candidates, p)
		}
	}
	if len(candidates) > 1 {
		t.human("DETAIL_NAVIGATION", "入口打开了多个页面，请核对唯一目标详情")
		return nil
	}
	destination := nav.page
	if len(candidates) == 1 {
		destination = candidates[0]
	}
	if destination.IsClosed() || destination.URL() == "about:blank" {
		return nil
	}
	changed := destination != nav.page || destination.URL() != nav.url
	for f, u := range nav.frames {
		changed = changed || !f.IsDetached() && f.URL() != u
	}
	old := t.page
	t.page = destination
	t.detailPage, t.detailURL = destination, destination.URL()
	if !changed {
		f, _ := t.detailFrame()
		if f == nil || count(f.Locator(constants["homework_detail.QUESTION_ROOTS"]+",[data-homework-detail]")) == 0 {
			t.detailPage = nil
			t.detailURL = ""
			t.page = old
			return nil
		}
	}
	t.detailNav = nil
	return nil
}
func resourceURL(value string) bool {
	u, e := url.Parse(value)
	if e != nil || u.Scheme != "https" || u.User != nil || u.Port() != "" && u.Port() != "443" {
		return false
	}
	h := u.Hostname()
	return h == "learning.xidian.edu.cn" || h == "chaoxing.com" || strings.HasSuffix(h, ".chaoxing.com") || strings.HasSuffix(h, ".chaoxing.com.cn")
}
func (t *Task) detailFrame() (pw.Frame, string) {
	target := obj(t.snapshot()["target"])
	selected := obj(target["portal"])
	if str(selected["homework_id"]) == "" || !t.courseConfirmed {
		return nil, "COURSE_UNVERIFIED"
	}
	p := t.page
	if p == nil || p.IsClosed() {
		return nil, "DETAIL_PAGE_UNAVAILABLE"
	}
	reason := "DETAIL_PAGE_NOT_FOUND"
	for _, f := range p.Frames() {
		if !resourceURL(f.URL()) {
			continue
		}
		id := queryID(f.URL(), "courseid")
		if id != "" && id != strings.Split(str(selected["course_id"]), ":")[0] {
			reason = "COURSE_ID_MISMATCH"
			continue
		}
		ids := map[string]bool{}
		if id := queryID(f.URL(), "workid", "homeworkid", "taskid"); id != "" {
			ids[id] = true
		}
		raw, e := f.Locator("input[type=hidden][name='workid' i],input[type=hidden][id='workid' i],[data-homework-detail][data-workid]").EvaluateAll("els=>els.map(e=>e.value||e.getAttribute('data-workid')).filter(Boolean)")
		if e != nil {
			continue
		}
		for _, v := range array(raw) {
			ids[str(v)] = true
		}
		linked := p == t.detailPage && p.URL() == t.detailURL
		u, _ := url.Parse(f.URL())
		preview := strings.HasSuffix(strings.TrimSuffix(u.Path, "/"), "/work/library/view-work")
		identity := len(ids) == 1 && ids[str(selected["homework_id"])]
		if len(ids) > 0 && !identity && !(linked && preview) {
			reason = "HOMEWORK_ID_MISMATCH"
			continue
		}
		titles, e := f.Locator("h1,h2,h3,.mark_title,.work-title,.workTitle").EvaluateAll(script("question_titles"))
		if e != nil {
			continue
		}
		titleOK := false
		for _, v := range array(titles) {
			titleOK = titleOK || str(v) == str(target["assignment_name"])
		}
		if !titleOK {
			reason = "DETAIL_TITLE_MISSING"
			continue
		}
		if identity || linked {
			return f, ""
		}
		reason = "DETAIL_ENTRY_UNVERIFIED"
	}
	return nil, reason
}
func (t *Task) detailObservation() (object, error) {
	if e := t.reconcileDetail(); e != nil {
		return nil, e
	}
	f, reason := t.detailFrame()
	opened := t.detailPage != nil && !t.detailPage.IsClosed()
	active := opened && t.page == t.detailPage
	v := object{"ready": f != nil, "read_only": true, "reason": reason, "open": opened, "active": active, "page_id": nil}
	if opened {
		v["page_id"] = pageIndex(t.context, t.detailPage)
	}
	if t.detailNav != nil && !opened {
		v["ready"], v["pending"], v["reason"] = false, true, "DETAIL_NAVIGATION_PENDING"
		if time.Since(t.detailNav.started) > 15*time.Second {
			t.human("DETAIL_NAVIGATION", "详情入口尚未打开，请核对页面加载；本次入口已保留")
		}
	}
	if opened && !active {
		v["ready"], v["reason"] = false, "DETAIL_PAGE_NOT_ACTIVE"
	}
	return v, nil
}
func (t *Task) fetchResource(value string) ([]byte, string, error) {
	for i := 0; i < 4; i++ {
		if !resourceURL(value) {
			return nil, "", errors.New("资源地址不属于支持的平台")
		}
		res, e := t.context.Request().Get(value, pw.APIRequestContextGetOptions{Timeout: pw.Float(15000), MaxRedirects: pw.Int(0)})
		if e != nil {
			return nil, "", e
		}
		status := res.Status()
		headers := res.Headers()
		if status == 301 || status == 302 || status == 303 || status == 307 || status == 308 {
			u, _ := url.Parse(value)
			r, e := url.Parse(headers["location"])
			_ = res.Dispose()
			if e != nil {
				return nil, "", e
			}
			value = u.ResolveReference(r).String()
			continue
		}
		mime := strings.ToLower(strings.Split(headers["content-type"], ";")[0])
		if status != 200 || mime == "text/html" || mime == "application/xhtml+xml" {
			_ = res.Dispose()
			return nil, "", errors.New("资源返回了无效文件或登录网页")
		}
		if n := parseSize(headers["content-length"]); n > 20*1024*1024 {
			_ = res.Dispose()
			return nil, "", errors.New("资源超过 20MiB")
		}
		data, e := res.Body()
		_ = res.Dispose()
		if e != nil {
			return nil, "", e
		}
		if len(data) > 20*1024*1024 {
			return nil, "", errors.New("资源超过 20MiB")
		}
		return data, mime, nil
	}
	return nil, "", errors.New("资源跳转次数过多")
}
func (t *Task) readDetail() (object, error) {
	f, reason := t.detailFrame()
	if f == nil {
		return nil, problem("DETAIL_TARGET_UNVERIFIED", "当前详情不可读取："+reason)
	}
	roots := f.Locator(constants["homework_detail.QUESTION_ROOTS"])
	if count(roots) == 0 {
		roots = f.Locator("[data-homework-detail],main,[role=main]")
	}
	if count(roots) == 0 {
		return nil, problem("DETAIL_CONTENT_MISSING", "尚未定位到题目正文区域")
	}
	value, e := roots.EvaluateAll(script("question_extract"))
	if e != nil {
		return nil, e
	}
	raw := obj(value)
	target := obj(t.snapshot()["target"])
	sources, warnings := []any{}, []any{}
	if flag(raw["truncated"]) {
		warnings = append(warnings, "题目内容超过读取上限，未完整提取")
	}
	dir := filepath.Join(t.manager.cfg.Downloads, t.id, "requirements")
	if e = os.MkdirAll(dir, 0700); e != nil {
		return nil, e
	}
	total := 0
	for i, s := range objects(raw["sources"]) {
		if i >= 40 {
			break
		}
		if !t.automatic() {
			return nil, problem("DETAIL_STOPPED", "详情读取已暂停或取消")
		}
		kind := str(s["kind"])
		if kind == "text" {
			lines := []string{}
			for _, line := range strings.Split(str(s["text"]), "\n") {
				trim := strings.TrimSpace(line)
				if trim == str(target["assignment_name"]) || trim == "作业详情" || trim == "显示答案" || trim == "答案" || trim == "加载中" || trim == "加载中..." || trim == "正在加载..." {
					continue
				}
				lines = append(lines, line)
			}
			text := strings.TrimSpace(strings.Join(lines, "\n"))
			if text != "" {
				sources = append(sources, object{"kind": "text", "text": bounded(text, 100000)})
			}
			continue
		}
		entry := object{"kind": kind, "name": bounded(str(s["name"]), 255)}
		data, mime, err := t.readSource(f, s, entry)
		if err == nil {
			total += len(data)
			if len(data) == 0 || len(data) > 20*1024*1024 || total > 60*1024*1024 {
				err = errors.New("题目资源为空或超过读取上限")
			}
		}
		ext := ""
		if err == nil {
			ext = resourceExtension(mime, str(s["url"]), str(entry["name"]))
			if ext == "" {
				err = errors.New("题目附件格式暂不支持")
			}
		}
		if err == nil {
			digest := textDigest(data)
			path := filepath.Join(dir, fmt.Sprintf("%02d-%s%s", i, digest[:16], ext))
			err = os.WriteFile(path, data, 0600)
			if err == nil {
				entry["path"], entry["sha256"], entry["size"] = path, digest, len(data)
			}
		}
		if err != nil {
			t.lastErr = err
			entry["error"] = "题目资源读取失败，请核对原始图片或附件"
			warnings = append(warnings, str(entry["name"])+"："+str(entry["error"]))
		}
		sources = append(sources, entry)
	}
	if len(sources) == 0 {
		return nil, problem("DETAIL_EMPTY", "题目正文尚未加载或没有可读取内容")
	}
	if !t.automatic() {
		return nil, problem("DETAIL_STOPPED", "作业详情读取已暂停或取消")
	}
	selected := obj(target["portal"])
	return object{"course_id": selected["course_id"], "homework_id": selected["homework_id"], "title": target["assignment_name"], "sources": sources, "warnings": warnings, "complete": len(warnings) == 0}, nil
}
func (t *Task) readSource(f pw.Frame, s, entry object) ([]byte, string, error) {
	raw := str(s["url"])
	u, _ := url.Parse(f.URL())
	r, e := url.Parse(raw)
	if e != nil {
		return nil, "", e
	}
	value := u.ResolveReference(r).String()
	control := f.Locator(fmt.Sprintf("[data-rpa-question-resource='%d']", int(num(s["resource_index"]))))
	if str(s["kind"]) == "attachment" && (raw == "" || strings.HasPrefix(raw, "javascript:") || strings.HasPrefix(raw, "#") || strings.HasPrefix(raw, "blob:")) {
		d, e := t.page.ExpectDownload(func() error {
			activated, err := control.Evaluate(`e=>{if(!e.isConnected||!e.getClientRects().length||e.disabled||e.closest('[hidden],[aria-hidden=true],[aria-disabled=true],[inert]'))return false;e.click();return true}`, nil)
			if err != nil {
				return err
			}
			if !flag(activated) {
				return errors.New("题目附件控件已变化")
			}
			return nil
		}, pw.PageExpectDownloadOptions{Timeout: pw.Float(10000)})
		if e != nil {
			return nil, "", e
		}
		path, e := t.detailDownloadPath(d, 20*time.Second)
		if e != nil {
			return nil, "", e
		}
		info, e := os.Stat(path)
		if e != nil || info.Size() > 20*1024*1024 {
			_ = d.Cancel()
			return nil, "", errors.New("附件过大或下载失败")
		}
		entry["name"] = safeName(d.SuggestedFilename())
		data, e := os.ReadFile(path)
		return data, "application/octet-stream", e
	}
	if strings.HasPrefix(raw, "data:image/") {
		parts := strings.SplitN(raw, ",", 2)
		if len(parts) != 2 || !strings.Contains(parts[0], ";base64") || len(parts[1]) > 40*1024*1024 {
			return nil, "", errors.New("图片编码不受支持")
		}
		data, e := base64.StdEncoding.Strict().DecodeString(parts[1])
		return data, strings.Split(strings.TrimPrefix(parts[0], "data:"), ";")[0], e
	}
	if raw != "" && resourceURL(value) {
		data, mime, e := t.fetchResource(value)
		if e == nil || str(s["kind"]) != "image" {
			return data, mime, e
		}
	}
	if str(s["kind"]) == "image" {
		data, e := control.Screenshot(pw.LocatorScreenshotOptions{Timeout: pw.Float(8000)})
		return data, "image/png", e
	}
	return nil, "", errors.New("附件地址不属于支持的平台")
}

func (t *Task) detailDownloadPath(d pw.Download, timeout time.Duration) (string, error) {
	type result struct {
		path string
		err  error
	}
	finished := make(chan result, 1)
	go func() { path, err := d.Path(); finished <- result{path, err} }()
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case r := <-finished:
			return r.path, r.err
		case <-t.ctx.Done():
			_ = d.Cancel()
			return "", t.ctx.Err()
		case <-deadline.C:
			_ = d.Cancel()
			return "", problem("DETAIL_DOWNLOAD_TIMEOUT", "题目附件下载超时")
		case <-ticker.C:
			if !t.automatic() {
				_ = d.Cancel()
				return "", problem("DETAIL_STOPPED", "作业详情读取已暂停或取消")
			}
		}
	}
}

func resourceExtension(mime, value, name string) string {
	extensions := map[string]string{"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif", "application/pdf": ".pdf", "application/msword": ".doc", "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx", "text/plain": ".txt", "text/markdown": ".md"}
	ext := extensions[mime]
	if ext == "" {
		u, _ := url.Parse(value)
		ext = strings.ToLower(filepath.Ext(u.Path))
	}
	allowed := regexp.MustCompile(`^\.(pdf|docx?|txt|md|png|jpe?g|webp|gif)$`)
	if allowed.MatchString(ext) {
		return ext
	}
	matches := regexp.MustCompile(`(?i)\.(pdf|docx?|txt|md)(?:\b|$)`).FindStringSubmatch(name)
	if len(matches) > 1 {
		return "." + strings.ToLower(matches[1])
	}
	return ""
}
func (t *Task) returnToHomeworkList() error {
	source := t.detailSource
	if source == nil || source.IsClosed() {
		return problem("DETAIL_LIST_UNAVAILABLE", "原作业列表页已关闭，请重新同步")
	}
	if t.page != source {
		if e := t.page.Close(); e != nil {
			return e
		}
	}
	t.page = source
	if source.URL() != t.detailSourceURL {
		if !allowedURL(t.detailSourceURL) {
			return errors.New("原作业列表来源无效")
		}
		if _, e := source.Goto(t.detailSourceURL, pw.PageGotoOptions{WaitUntil: pw.WaitUntilStateDomcontentloaded}); e != nil {
			return e
		}
	} else {
		for f, u := range t.detailListFrames {
			if !f.IsDetached() && f.URL() != u && allowedURL(u) {
				if _, e := f.Goto(u, pw.FrameGotoOptions{WaitUntil: pw.WaitUntilStateDomcontentloaded}); e != nil {
					return e
				}
			}
		}
	}
	id := str(obj(obj(t.snapshot()["target"])["portal"])["homework_id"])
	title := str(obj(t.snapshot()["target"])["assignment_name"])
	for attempt := 0; attempt < 20; attempt++ {
		if !t.automatic() {
			return problem("DETAIL_STOPPED", "作业详情读取已暂停或取消")
		}
		rows, e := readRecords(source, "homework")
		if e != nil {
			return e
		}
		for _, r := range rows {
			if str(r["platform_id"]) == id && str(r["title"]) == title {
				return nil
			}
		}
		select {
		case <-t.ctx.Done():
			return t.ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
	return problem("DETAIL_LIST_UNAVAILABLE", "尚未恢复包含原作业的列表页")
}
