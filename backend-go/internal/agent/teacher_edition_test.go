package agent

import (
	"strings"
	"testing"
)

func TestTeacherEditionDoesNotAdvertiseExamGrading(t *testing.T) {
	catalog, err := DefaultGradingSkills()
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := catalog["grade-exam"]; exists {
		t.Fatal("removed exam Skill remains available")
	}
	if _, exists := catalog["grade-homework"]; !exists {
		t.Fatal("homework Skill removed")
	}
	for _, operation := range []string{"inspect", "submit"} {
		schema := (&GradingTool{Operation: operation}).Schema()
		if strings.Contains(schema, `"exam"`) || strings.Contains(schema, `"image_paths"`) {
			t.Fatal("exam input remains in a grading tool")
		}
	}
}
