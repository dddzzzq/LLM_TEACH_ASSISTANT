package agent

import (
	"context"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
	"grading-gateway/internal/testutil"
	"strings"
	"testing"
)

func TestIntegrationTeacherMetadataPreservesDisabledToolsAndDoesNotGrantRoles(t *testing.T) {
	db, _ := testutil.ExecutionDB(t)
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old })
	if err := db.AutoMigrate(&models.ToolDefinition{}); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, before, after string }{
		{"start_grading_job", `["student","teacher"]`, `["teacher"]`},
		{"query_student_score", `["student"]`, `[]`},
	} {
		var prior models.ToolDefinition
		existed := db.Where("name = ?", tc.name).First(&prior).Error == nil
		if existed {
			db.Delete(&prior)
		}
		row := models.ToolDefinition{Name: tc.name, ImplKey: tc.name, Description: "查询或批改试卷", SchemaJSON: `{"kind":"exam"}`, AllowedRoles: tc.before}
		if err := db.Create(&row).Error; err != nil {
			t.Fatal(err)
		}
		if err := db.Model(&row).Update("enabled", false).Error; err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			db.Delete(&models.ToolDefinition{}, "name = ?", tc.name)
			if existed {
				db.Create(&prior)
			}
		})
	}
	EnsureTeacherEditionTools(context.Background())
	for _, tc := range []struct{ name, roles string }{{"start_grading_job", `["teacher"]`}, {"query_student_score", `[]`}} {
		var row models.ToolDefinition
		if err := db.Where("name = ?", tc.name).First(&row).Error; err != nil {
			t.Fatal(err)
		}
		if row.Enabled || row.AllowedRoles != tc.roles || strings.Contains(row.SchemaJSON, `"exam"`) || strings.Contains(row.Description, "试卷") {
			t.Fatal("teacher metadata changed availability, granted access, or retained exam capabilities")
		}
	}
}
