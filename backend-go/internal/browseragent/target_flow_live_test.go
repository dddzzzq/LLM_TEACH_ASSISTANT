package browseragent

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"grading-gateway/internal/agent/catalog"
	"grading-gateway/internal/agent/modelclient"
)

// Explicit opt-in only. All page content is synthetic; no real browser task is created.
func TestLiveModelTargetFlow(t *testing.T) {
	if os.Getenv("RPA_LIVE_MODEL_TEST") != "1" {
		t.Skip("opt-in synthetic model decision check")
	}
	python := os.Getenv("RPA_BROWSER_PYTHON")
	if python == "" {
		t.Fatal("RPA_BROWSER_PYTHON required")
	}
	cmd := exec.Command(python, "-c", "from app.rpa.tools import tool_instructions; print(tool_instructions())")
	cmd.Dir = "../../../ai_engine_python"
	contract, err := cmd.Output()
	if err != nil {
		t.Fatal("cannot read browser contract")
	}
	skills, err := catalog.Default("portal-browser")
	if err != nil {
		t.Fatal(err)
	}
	client := modelclient.NewDeepSeekClient("")
	cases := []struct{ name, input, tool, actual, term string }{
		{"portal-personal-space", `{"target":{"course_name":"操作系统","assignment_name":"作业1"},"stage":"LOCATE_COURSE","observation":{"pages":[{"page_id":"0","title":"教学门户","url":"https://learning.xidian.edu.cn/portal"}],"frames":[{"frame_id":"p0f0","headings":[{"text":"教学新闻"}]}],"elements":[{"ref":"p0f0e0","name":"个人空间","role":"link","context_labels":["教学新闻"],"context":"教学新闻 个人空间"}],"target_evidence":{"course_confirmed":false,"assignment_confirmed":false},"coverage":{"truncated":false}},"recent_actions":[]}`, "click", "", ""},
		{"assignment-near-name", `{"target":{"course_name":"操作系统","assignment_name":"作业1"},"stage":"LOCATE_ASSIGNMENT","observation":{"pages":[{"page_id":"0","title":"操作系统 作业列表"}],"elements":[{"ref":"p0f0e0","name":"批阅","role":"button","record_id":"r1","context_labels":["作业一"],"context":"作业一 36已交 批阅"},{"ref":"p0f0e1","name":"批阅","role":"button","record_id":"r2","context_labels":["作业二"],"context":"作业二 40已交 批阅"}],"target_evidence":{"course_confirmed":true,"assignment_confirmed":false},"coverage":{"truncated":false}},"recent_actions":[]}`, "request_target_confirmation", "作业一", ""},
		{"current-course-missing", `{"target":{"course_name":"操作系统","assignment_name":"作业1"},"stage":"LOCATE_COURSE","observation":{"pages":[{"page_id":"0","title":"个人空间 我的课程"}],"frames":[{"frame_id":"p0f0","headings":[{"text":"我的课程"}],"scroll":{"x":0,"y":0,"height":850},"viewport":{"width":1440,"height":1000}}],"elements":[{"ref":"p0f0e0","tag":"select","role":"combobox","name":"学期","selected_option":"2026-2027第一学期","options":["2026-2027第一学期","2025-2026第二学期","2025-2026第三学期"]},{"ref":"p0f0e1","name":"高等数学","role":"link","context_labels":["高等数学"]}],"target_evidence":{"course_confirmed":false,"assignment_confirmed":false},"coverage":{"truncated":false}},"recent_actions":[]}`, "request_human", "", ""},
		{"open-semester-menu-needs-teacher", `{"target":{"course_name":"分布式计算","assignment_name":"第1次作业"},"stage":"LOCATE_COURSE","observation":{"active_page_id":"6","pages":[{"page_id":"0","title":"教学门户","active":false},{"page_id":"1","title":"个人空间","active":false},{"page_id":"2","title":"个人空间","active":false},{"page_id":"3","title":"个人空间","active":false},{"page_id":"4","title":"个人空间","active":false},{"page_id":"5","title":"个人空间","active":false},{"page_id":"6","title":"个人空间 我的课程","active":true}],"frames":[{"frame_id":"p6f0","page_id":"6","headings":[{"text":"我的课程"}],"scroll":{"x":0,"y":0,"height":850},"viewport":{"width":1440,"height":1000}}],"elements":[{"ref":"p6f0e0","page_id":"6","tag":"input","name":"学期","selected_option":"2026-2027第一学期","semester_kind":"trigger"},{"ref":"p6f0e1","page_id":"6","tag":"div","name":"2026-2027第一学期","semester_kind":"option","semester_menu":"p6f0m0","semester_index":1,"semester_option_count":3,"in_viewport":true},{"ref":"p6f0e2","page_id":"6","tag":"div","name":"2025-2026第二学期","semester_kind":"option","semester_menu":"p6f0m0","semester_index":2,"semester_option_count":3,"in_viewport":true},{"ref":"p6f0e3","page_id":"6","tag":"div","name":"2025-2026第三学期","semester_kind":"option","semester_menu":"p6f0m0","semester_index":3,"semester_option_count":3,"in_viewport":true},{"ref":"p6f0e4","page_id":"6","name":"移动互联网导论","role":"link","context_labels":["移动互联网导论"]}],"target_evidence":{"course_confirmed":false,"assignment_confirmed":false},"coverage":{"scope":"active_page","truncated":false}},"recent_actions":[]}`, "request_human", "", ""},
		{"manual-term-exact-course", `{"target":{"course_name":"操作系统","assignment_name":"作业1"},"stage":"LOCATE_COURSE","observation":{"pages":[{"page_id":"0","title":"个人空间 我的课程"}],"elements":[{"ref":"p0f0e0","tag":"select","name":"学期","selected_option":"2025-2026第二学期","options":["2026-2027第一学期","2025-2026第二学期"]},{"ref":"p0f0e1","name":"操作系统","role":"link","record_id":"r1","context_labels":["操作系统"],"context":"操作系统 教师甲"}],"target_evidence":{"course_confirmed":false,"assignment_confirmed":false},"coverage":{"truncated":false}},"recent_actions":[]}`, "click", "", ""},
		{"current-course-missing-other-term", `{"target":{"course_name":"操作系统","assignment_name":"作业1"},"stage":"LOCATE_COURSE","observation":{"pages":[{"title":"个人空间 我的课程"}],"elements":[{"ref":"p0f0e0","name":"学期","tag":"select","selected_option":"2025-2026第二学期","options":["2026-2027第一学期","2025-2026第二学期"]},{"ref":"p0f0e1","name":"大学英语","role":"link","context_labels":["大学英语"]}],"target_evidence":{"course_confirmed":false,"assignment_confirmed":false},"coverage":{"truncated":false}},"recent_actions":[]}`, "request_human", "", ""},
		{"specified-term-needs-teacher", `{"target":{"course_name":"操作系统","assignment_name":"作业1","term":"2026-2027第一学期"},"stage":"LOCATE_COURSE","observation":{"pages":[{"page_id":"0","title":"个人空间 我的课程"}],"elements":[{"ref":"p0f0e0","tag":"select","name":"学期","selected_option":"2025-2026第二学期","options":["2026-2027第一学期","2025-2026第二学期"]},{"ref":"p0f0e1","name":"操作系统","role":"link","record_id":"r1","context_labels":["操作系统"],"context":"操作系统 教师甲"}],"target_evidence":{"course_confirmed":false,"assignment_confirmed":false},"coverage":{"truncated":false}},"recent_actions":[]}`, "request_human", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
			defer cancel()
			var input struct {
				Observation map[string]any `json:"observation"`
			}
			if err := json.Unmarshal([]byte(tc.input), &input); err != nil {
				t.Fatal(err)
			}
			// Include the same page-focus evidence emitted by the worker.
			var payload map[string]any
			_ = json.Unmarshal([]byte(tc.input), &payload)
			if _, ok := input.Observation["active_page_id"]; !ok {
				input.Observation["active_page_id"] = "0"
			}
			if _, ok := input.Observation["frames"]; !ok {
				input.Observation["frames"] = []any{map[string]any{
					"frame_id": "p0f0", "page_id": "0",
					"scroll":   map[string]any{"x": 0, "y": 0, "height": 850},
					"viewport": map[string]any{"width": 1440, "height": 1000},
				}}
			}
			payload["observation"] = input.Observation
			encoded, err := json.Marshal(payload)
			if err != nil {
				t.Fatal(err)
			}
			action, err := Decide(ctx, encoded, skills["fetch-homework"], contract, client.CallMessages)
			if err != nil {
				t.Fatalf("model decision failed (%T)", err)
			}
			var got map[string]string
			if json.Unmarshal(action, &got) != nil || got["tool"] != tc.tool || (tc.actual != "" && got["actual_name"] != tc.actual) || (tc.term != "" && got["term"] != tc.term) {
				t.Fatalf("unexpected synthetic decision: %s", action)
			}
			if tc.tool == "request_human" && (!strings.Contains(got["reason"], "课程") || !strings.Contains(got["reason"], "学期")) {
				t.Fatal("missing course-name/semester reminder")
			}
			t.Logf("verified %s", got["tool"])
		})
	}
}
