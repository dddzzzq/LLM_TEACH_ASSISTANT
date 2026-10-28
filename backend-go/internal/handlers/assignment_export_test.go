package handlers

import (
	"strings"
	"testing"
	"unicode/utf8"

	"grading-gateway/internal/models"
)

func TestAssignmentWorkbookSeparatesIdenticallyNamedClassesAndKeepsZeroReview(t *testing.T) {
	a := models.Assignment{ClassName: "全部班级", PortalHomeworkID: "portal-homework"}
	classes := []AssignmentClass{{ID: "class-a", Name: "同名/班级", Source: "portal"}, {ID: "class-b", Name: "同名/班级", Source: "portal"}, {ID: unassignedClassID, Name: "未分班", Source: "unassigned"}}
	batches := []models.MaterialBatch{{ID: "a", PortalClassID: "class-a", ClassName: "同名/班级"}, {ID: "b", PortalClassID: "class-b", ClassName: "同名/班级"}}
	// Same student ID and report pair in different classes must remain separate.
	report := `[{"similar_to":"peer","content_type":"code","initial_score":0.9,"llm_analysis":{"is_plagiarism":true,"reasoning":"需要复核"}}]`
	submissions := []models.Submission{
		{ID: 1, MaterialBatchID: "a", StudentID: "same-id", StudentName: "A班学生", Score: 90, IsHumanReviewed: true, HumanScore: 0, PlagiarismReport: report},
		{ID: 2, MaterialBatchID: "b", StudentID: "same-id", StudentName: "B班学生", Score: 70, PlagiarismReport: report},
		{ID: 3, MaterialBatchID: "missing", StudentID: "legacy", Score: 80},
	}
	f, err := assignmentWorkbook(a, classes, batches, submissions, "")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sheets := f.GetSheetList()
	if len(sheets) != 6 {
		t.Fatalf("got %v; expected a pair of sheets per class", sheets)
	}
	for i, want := range []string{"same-id-A班学生", "same-id-B班学生", "legacy-"} {
		rows, err := f.GetRows(sheets[i*2])
		if err != nil || len(rows) != 2 || rows[1][0] != want {
			t.Fatalf("class sheet %q mixed or lost students: %v, %v", sheets[i*2], rows, err)
		}
		if i == 0 && (rows[1][3] != "是" || rows[1][4] != "0") {
			t.Fatalf("reviewed zero score was lost: %v", rows[1])
		}
		if i < 2 {
			reports, err := f.GetRows(sheets[i*2+1])
			if err != nil || len(reports) != 2 {
				t.Fatalf("class report deduplication crossed class boundaries: %v, %v", reports, err)
			}
		}
	}
}

func TestAssignmentWorkbookSingleClassKeepsReportNamesAndHandlesMalformedReports(t *testing.T) {
	f, err := assignmentWorkbook(models.Assignment{}, nil, nil, []models.Submission{{StudentID: "s1", Score: 60, AIGCReport: `{"ai_probability":"bad"}`, CodeDocMatchReport: `not-json`}}, "class-a")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sheets := f.GetSheetList()
	if len(sheets) != 2 || sheets[0] != "成绩单" || sheets[1] != "抄袭检测报告" {
		t.Fatalf("single class export contract changed: %v", sheets)
	}
	rows, err := f.GetRows("成绩单")
	if err != nil || rows[1][6] != "N/A" || rows[1][7] != "N/A" {
		t.Fatalf("malformed report should not break score export: %v, %v", rows, err)
	}
}

func TestUniqueSheetNameIsValidAndCollisionSafe(t *testing.T) {
	used := map[string]bool{}
	name := strings.Repeat("班级", 30) + "[]:*?/\\"
	first, second := uniqueSheetName(name, used), uniqueSheetName(name, used)
	if first == second || utf8.RuneCountInString(first) > 31 || utf8.RuneCountInString(second) > 31 || strings.ContainsAny(first, "[]:*?/\\") {
		t.Fatalf("invalid sheet names: %q %q", first, second)
	}
	if uniqueSheetName("A", used) == uniqueSheetName("a", used) {
		t.Fatal("sheet collisions must be case insensitive")
	}
}

func TestLegacyClassIdentityDoesNotInventPortalOwnership(t *testing.T) {
	local := legacyAssignmentClass(models.Assignment{ClassName: "软件工程一班"})
	if !strings.HasPrefix(local.ID, "local:") || local.Name != "软件工程一班" || local.CanGrade || local.CanImport {
		t.Fatalf("invalid local class: %+v", local)
	}
	portal := legacyAssignmentClass(models.Assignment{ClassName: "全部班级", PortalHomeworkID: "homework"})
	if portal.ID != unassignedClassID {
		t.Fatalf("legacy portal grades must remain explicitly unassigned: %+v", portal)
	}
}
