package agent

import (
	"context"
	"strings"
	"testing"
)

type fullFlowTool struct{ runtimeTool }

func (t *fullFlowTool) RequiredSkill(string) string { return "download-homework" }
func (t *fullFlowTool) AdditionalSkills(args string) []string {
	return (&FetchAndGradeHomeworkTool{}).AdditionalSkills(args)
}

func TestFullDownloadFlowRequiresBothSkillsAndMaterialsRemainSeparate(t *testing.T) {
	tool := &fullFlowTool{runtimeTool: runtimeTool{name: "fetch_homework"}}
	r := NewToolRegistry()
	r.Register(tool)
	g := newGateway("teacher", "teacher", r)
	g.visible = map[string]string{"download-homework": "v2"}
	output, err := g.execute(context.Background(), "fetch_homework", `{"course_name":"C","assignment_name":"A","after_download":"grade"}`)
	if err != nil || tool.calls != 0 || !strings.Contains(output, "grade-homework") {
		t.Fatal("full flow bypassed grading skill")
	}
	g.visible["grade-homework"] = "v2"
	if _, err := g.execute(context.Background(), "fetch_homework", `{"course_name":"C","assignment_name":"A","after_download":"grade"}`); err != nil || tool.calls != 1 {
		t.Fatal("both skills did not authorize explicit flow")
	}
	prepare := &MaterialTool{Operation: "prepare"}
	if prepare.RequiredSkill(`{"source_job_id":"download-id"}`) != "download-homework" || prepare.RequiredSkill(`{"source_job_id":"download-id","auto_grade":true}`) != "grade-homework" {
		t.Fatal("material tool mixes extraction and grading policy")
	}
	for _, name := range []string{"extract_homework_files", "get_homework_materials", "cancel_homework_materials"} {
		if _, ok := NewBuiltinToolRegistry(7, "teacher").GetTool(name); !ok {
			t.Fatal("missing tool", name)
		}
	}
}
