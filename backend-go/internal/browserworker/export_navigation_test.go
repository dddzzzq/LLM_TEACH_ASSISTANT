package browserworker

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

func TestExportFailureReportsPhaseWithoutRawPageData(t *testing.T) {
	err := exportPhaseError(errors.New("playwright: timeout: element is not visible; body intercepts pointer events https://example.test/export?token=SECRET_CANARY"), "activate_export_entry")
	report := publicError(err)
	evidence := obj(report["evidence"])
	encoded, _ := json.Marshal(report)
	if report["code"] != "EXPORT_PAGE_OPERATION_FAILED" || evidence["phase"] != "activate_export_entry" || !flag(evidence["timeout"]) || !flag(evidence["obscured"]) || !flag(evidence["not_visible"]) || strings.Contains(string(encoded), "SECRET_CANARY") {
		t.Fatal("missing or unsafe export diagnostics", report)
	}
	known := &issue{Code: "CLASS_CHANGED", Message: "班级未确认", Evidence: object{"class_id": "22"}}
	wrapped := publicError(exportPhaseError(known, "select_target_class"))
	if wrapped["code"] != "CLASS_CHANGED" || obj(wrapped["evidence"])["phase"] != "select_target_class" || known.Evidence["phase"] != nil {
		t.Fatal("changed issue identity or mutated prior evidence", wrapped)
	}
}

func TestExportLabelsAndEnglishRecordIdentity(t *testing.T) {
	for _, label := range []string{"导出提交附件", "导出学生提交的附件", "Export commit attachments", "Export submission attachments", "Export submitted attachments", "Export attachments"} {
		if !exportAttachmentLabel.MatchString(label) {
			t.Fatal("missing attachment format", label)
		}
	}
	for _, label := range []string{"Export the complete answer record The exported content contains students submission of attachments", "Export only document marking annotations Export only attachments submitted by students", "Export the class work attachment", "Export all class work attachments", "导出完整答题记录 学生提交附件", "仅导出文档留痕批注"} {
		if exportAttachmentLabel.MatchString(label) {
			t.Fatal("unrelated option accepted as attachment format", label)
		}
	}
	for _, label := range []string{"导出本班级作业附件", "当前班级", "Current class", "Export the class work attachment"} {
		if !currentClassAlias.MatchString(label) {
			t.Fatal("missing current class alias", label)
		}
	}
	for _, label := range []string{"导出作业下所有班级作业附件", "All classes", "Export all class work attachments"} {
		if currentClassAlias.MatchString(label) {
			t.Fatal("all classes accepted as current class", label)
		}
	}
	for _, label := range []string{"导出作业附件", "导出附件", "Export homework", "Export assignment attachments", "export submission files", "Export attachments"} {
		if !exportEntryLabel.MatchString(label) {
			t.Fatal("missing export label", label)
		}
	}
	for _, label := range []string{"Export grades", "Import the score", "Download", "Export homework and grades"} {
		if exportEntryLabel.MatchString(label) {
			t.Fatal("unrelated action accepted", label)
		}
	}
	if exportIdentity("第一次作业 A班 Exporting") != exportIdentity("第一次作业 A班 Export successful Download") {
		t.Fatal("English completion changed the stable export identity")
	}
	for label, want := range map[string]string{"第一次作业 A班 Export successful Download": "READY", "第一次作业 A班 Exported Download": "READY", "第一次作业 A班 Failed Download": "FAILED", "第一次作业 A班 Expired": "FAILED", "第一次作业 A班 Exporting": "PENDING", "Ready application A班 Exporting": "PENDING", "Failure analysis A班 Exporting": "PENDING"} {
		if got := exportStatus(label); got != want {
			t.Fatal(label, got, want)
		}
	}
}

func TestEnglishExportSwitchesClassAndDoesNotToggleOpenMenu(t *testing.T) {
	for _, variant := range []struct {
		name, selected string
		open           bool
		more           int
		scope          bool
	}{
		{"closed-menu-wrong-class", "23", false, 2, false},
		{"open-menu-wrong-class", "23", true, 1, false},
		{"open-menu-correct-class", "22", true, 0, false},
		{"all-scope-must-be-replaced", "23", false, 2, true},
	} {
		t.Run(variant.name, func(t *testing.T) {
			_, task := controlFixture(t)
			task.op.Lock()
			defer task.op.Unlock()
			body := `<meta charset=utf-8>` + fixturePortalSession + `
<select name=clazzid onchange="document.querySelector('#menu').hidden=true;document.querySelector('#menu').innerHTML=''">
  <option value=22 SELECT_A>班级124CS512601</option><option value=23 SELECT_B>班级124CS512602</option>
</select>
<button id=more onclick="window.moreClicks=(window.moreClicks||0)+1;openMenu()">More</button>
<ul id=menu class=morePop hidden></ul>
<div role=dialog id=settings hidden><h3>Export homework</h3>
  <fieldset><legend>Export format</legend>
    <label><input type=radio name=format checked>Export the complete answer record<p>The exported content contains students answers, teachers corrections, and students submission of attachments</p></label>
    <label><input type=radio name=format>Export only document marking annotations<p>Export only the teachers document marking annotation attachments in zip format</p></label>
    <label><input id=attachment type=radio name=format>Export commit attachments<p>Export only attachments submitted by students in compressed package format</p></label>
  </fieldset>
  <button onclick="this.parentElement.hidden=true">Cancel</button>
  <button onclick="if(!document.querySelector('#attachment').checked||document.querySelector('#all-scope')?.checked)return;window.exportedClass=document.querySelector('select').value;window.exports=(window.exports||0)+1;this.parentElement.hidden=true">Confirm</button>
</div><div id=downloadcenter>No downloads</div>
<script>function openMenu(){const menu=document.querySelector('#menu');menu.hidden=false;menu.innerHTML='<li role=menuitem><a onclick="document.querySelector(\'#settings\').hidden=false">Export homework</a></li>'}</script>`
			body = strings.ReplaceAll(body, "SELECT_A", map[bool]string{true: "selected", false: ""}[variant.selected == "22"])
			body = strings.ReplaceAll(body, "SELECT_B", map[bool]string{true: "selected", false: ""}[variant.selected == "23"])
			if variant.scope {
				body = strings.Replace(body, "<fieldset><legend>Export format", `<fieldset><legend>Export range</legend><label><input type=radio name=scope>Export the class work attachment</label><label><input type=radio id=all-scope name=scope checked>Export all class work attachments</label></fieldset><fieldset><legend>Export format`, 1)
			}
			if err := task.page.SetContent(body); err != nil {
				t.Fatal(err)
			}
			if variant.open {
				if _, err := task.page.Evaluate("openMenu()"); err != nil {
					t.Fatal(err)
				}
			}
			_ = task.update(object{"target": object{"course_name": "课程", "assignment_name": "作业", "portal": object{"class_id": "22", "class_name": "班级124CS512601"}}})
			task.courseConfirmed, task.assignmentConfirmed = true, true
			if err := task.prepareExport(); err != nil {
				t.Fatal(err)
			}
			values, err := task.page.Evaluate("[window.moreClicks||0,window.exports,window.exportedClass]")
			if err != nil || hash(values) != hash([]any{float64(variant.more), float64(1), "22"}) || currentClass(task.page) != "22" {
				t.Fatal("wrong class or menu toggled", values, err)
			}
			if variant.scope {
				classes := objects(obj(task.snapshot()["class_scope"])["classes"])
				if len(classes) != 1 || flag(classes[0]["all"]) || classes[0]["platform_id"] != "22" {
					t.Fatal("single-class task expanded to all classes", classes)
				}
			}
			if err := task.prepareExport(); err != nil {
				t.Fatal(err)
			}
			n, _ := task.page.Evaluate("window.exports")
			if num(n) != 1 {
				t.Fatal("export submitted twice", n)
			}
		})
	}
}

func TestEnglishExportRejectsAmbiguityAndUnknownFormats(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	task.courseConfirmed, task.assignmentConfirmed = true, true
	if err := task.page.SetContent(fixturePortalSession + `<button onclick="window.moreClicks=1">More</button><a>Export homework</a><a>Export homework</a>`); err != nil {
		t.Fatal(err)
	}
	err := task.prepareExport()
	requireIssue(t, err, "EXPORT_SETTINGS_UNAVAILABLE")
	if num(obj(task.snapshot()["last_export_navigation"])["visible_entries"]) != 2 {
		t.Fatal("missing ambiguity evidence", task.snapshot())
	}
	clicked, _ := task.page.Evaluate("window.moreClicks||0")
	if num(clicked) != 0 {
		t.Fatal("ambiguous export toggled More")
	}
	if err := task.page.SetContent(fixturePortalSession + `<div role=dialog>Export homework<fieldset><legend>Export format</legend><label><input type=radio checked>Export an unknown format</label></fieldset><button onclick="window.exported=1">Confirm</button></div>`); err != nil {
		t.Fatal(err)
	}
	modal := task.page.Locator("[role=dialog]")
	requireIssue(t, task.selectAttachments(modal), "ATTACHMENT_MISSING")
	if flag(task.snapshot()["export_attempted"]) {
		t.Fatal("unknown format was exported")
	}
}

// Mirror the portal's custom grade_check controls, including descriptions
// that mention attachments under the wrong (default) export format.
func TestPortalAttachmentFormatsAndSingleClassTranslations(t *testing.T) {
	for _, variant := range []string{"chinese", "english", "english-without-section-classes"} {
		t.Run(variant, func(t *testing.T) {
			_, task := controlFixture(t)
			task.op.Lock()
			defer task.op.Unlock()
			fixture, err := os.ReadFile("testdata/portal-export.html")
			if err != nil {
				t.Fatal(err)
			}
			body := fixturePortalSession + `<select name=clazzid><option value=22 selected>班级124CS512601</option><option value=23>班级124CS512602</option></select>` + string(fixture)
			body = strings.Replace(body, "role=dialog hidden", "role=dialog", 1)
			body = strings.Replace(body, "!document.querySelector('.export-range.all .grade_checked')", "document.querySelector('.export-range.all .grade_checked')", 1)
			body = strings.Replace(body, "fetch('/export')", "window.exports=(window.exports||0)+1", 1)
			if variant != "chinese" {
				body = strings.NewReplacer(
					"选择格式", "Choose Type",
					"导出本班级作业附件", "Export the class work attachment",
					"导出作业下所有班级作业附件", "Export all class work attachments",
					"若班级内没有学生提交，将自动跳过导出", "If no student in the class submits, the export is automatically skipped",
					"导出范围", "Export range",
					"导出格式", "Export format",
					"导出完整答题记录", "Export the complete answer record",
					"导出内容包含学生作答情况、教师批改情况、学生提交附件", "The exported content contains students answers, teachers corrections, and students submission of attachments",
					"仅导出文档留痕批注", "Export only document marking annotations",
					"导出提交附件", "Export commit attachments",
					"以压缩包格式导出学生提交的附件", "Export only attachments submitted by students in compressed package format",
				).Replace(body)
			}
			if variant == "english-without-section-classes" {
				body = strings.NewReplacer("class=export-range", "data-scope", `class="export-range all"`, "data-scope class=all", `class="out export-flexBox"`, "data-format", ".export-range", "[data-scope]", ".out", "[data-format]").Replace(body)
			}
			if err = task.page.SetContent(body); err != nil {
				t.Fatal(err)
			}
			task.courseConfirmed, task.assignmentConfirmed = true, true
			_ = task.update(object{"target": object{"course_name": "操作系统", "assignment_name": "第一次作业", "portal": object{"class_id": "22", "class_name": "班级124CS512601"}}})
			if err = task.prepareExport(); err != nil {
				t.Fatal(err)
			}
			values, err := task.page.Evaluate(`[window.exports||0,!!document.querySelector('#attachment-option .grade_checked'),document.querySelectorAll('.out .grade_checked,[data-format] .grade_checked').length,window.namingClicks||0]`)
			if err != nil || hash(values) != hash([]any{float64(1), true, float64(1), float64(0)}) {
				t.Fatal("attachment format not uniquely selected or naming changed", values, err, task.snapshot())
			}
			classes := objects(obj(task.snapshot()["class_scope"])["classes"])
			if len(classes) != 1 || flag(classes[0]["all"]) || classes[0]["platform_id"] != "22" {
				t.Fatal("wrong single-class scope", classes)
			}
		})
	}
}

func TestExportRejectsOtherAssignmentBeforeOpeningMore(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	if _, err := task.page.Goto("https://learning.xidian.edu.cn/work?courseid=11&workid=99"); err != nil {
		t.Fatal(err)
	}
	if err := task.page.SetContent(fixturePortalSession + `<button onclick="window.moreClicks=1">More</button>`); err != nil {
		t.Fatal(err)
	}
	_ = task.update(object{"target": object{"course_name": "课程", "assignment_name": "作业", "portal": object{"course_id": "11", "homework_id": "33"}}})
	task.courseConfirmed, task.assignmentConfirmed = true, true
	requireIssue(t, task.prepareExport(), "TARGET_ID_MISMATCH")
	clicked, _ := task.page.Evaluate("window.moreClicks||0")
	if num(clicked) != 0 {
		t.Fatal("opened another assignment's More menu")
	}
}

func TestExportMenuThatHidesWhenPointerLeavesMore(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	task.page.SetDefaultTimeout(700)
	body := fixturePortalSession + `<meta charset=utf-8>
<style>#more{position:absolute;left:100px;top:60px}#menu{position:absolute;left:100px;top:100px}#menu a{display:block;padding:20px}</style>
<button id=more onclick="document.querySelector('#menu').hidden=false" onmouseleave="document.querySelector('#menu').hidden=true">More</button>
<ul id=menu class=morePop hidden><a onclick="document.querySelector('#settings').hidden=false">Export homework</a></ul>
<div id=settings role=dialog hidden><h3>Export homework</h3>
<fieldset><legend>Export scope</legend><label><input type=checkbox checked>All classes</label></fieldset>
<fieldset><legend>Export format</legend><label><input type=radio checked>Export submission attachments</label></fieldset>
<button onclick="window.exports=(window.exports||0)+1;this.parentElement.hidden=true">Confirm</button></div>
<div id=downloadcenter>No downloads</div>`
	if err := task.page.SetContent(body); err != nil {
		t.Fatal(err)
	}
	task.courseConfirmed, task.assignmentConfirmed = true, true
	if err := task.prepareExport(); err != nil {
		t.Fatal(err)
	}
	count, _ := task.page.Evaluate("window.exports")
	if num(count) != 1 {
		t.Fatal("hover-dependent menu did not export exactly once", count)
	}
}

func TestNativeExportRejectsChangedAndInactiveControls(t *testing.T) {
	_, task := controlFixture(t)
	task.op.Lock()
	defer task.op.Unlock()
	for _, body := range []string{
		`<a onclick="window.clicked=1">Export grades</a>`,
		`<div hidden><ul class=morePop><a onclick="window.clicked=1">Export homework</a></ul></div>`,
		`<a aria-disabled=true onclick="window.clicked=1">Export homework</a>`,
	} {
		if err := task.page.SetContent(body); err != nil {
			t.Fatal(err)
		}
		entry := reference{page: task.page, loc: task.page.Locator("a")}
		requireIssue(t, task.activateExportControl(entry, constants["page_snapshot.EXPORT_LABEL"], true), "EXPORT_CONTROL_CHANGED")
		value, _ := task.page.Evaluate("window.clicked||0")
		if num(value) != 0 {
			t.Fatal("activated an unverified control")
		}
	}
}
