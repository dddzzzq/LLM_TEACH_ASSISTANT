package agent

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
)

func TestToolExposureUsesRegisteredContractAndConfiguredPolicy(t *testing.T) {
	r := NewBuiltinToolRegistry(42)
	rows := []toolConfigDTO{
		{Name: "fetch_homework", Enabled: true, AllowedRoles: `["teacher"]`,
			Description: "custom description", SchemaJSON: `{"properties":{"password":{}}}`, ImplKey: "arbitrary_executor"},
		{Name: "query_student_score", Enabled: false, AllowedRoles: `["teacher"]`},
		{Name: "trigger_async_pipeline", Enabled: true, AllowedRoles: `["admin"]`},
		{Name: "unregistered_from_database", Enabled: true, AllowedRoles: `["teacher"]`},
		{Name: "get_fetch_job", Enabled: true, AllowedRoles: `invalid`},
	}
	definitions, err := buildToolsForRole(rows, "teacher", r)
	if err != nil || len(definitions) != 1 {
		t.Fatalf("unexpected tools: %v, %v", definitions, err)
	}
	function := definitions[0]["function"].(map[string]interface{})
	tool, _ := r.GetTool("fetch_homework")
	var actual map[string]interface{}
	if err := json.Unmarshal([]byte(tool.Schema()), &actual); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(function["parameters"], actual) {
		t.Fatal("database schema changed the executable contract")
	}
	if function["description"] != "custom description" || function["name"] != tool.Name() {
		t.Fatal("description override or registered tool name was lost")
	}
	for _, role := range []string{"student", "", "unknown"} {
		got, err := buildToolsForRole(rows, role, r)
		if err != nil || len(got) != 0 {
			t.Fatalf("unexpected tools for %q: %v, %v", role, got, err)
		}
	}
}

func TestMissingExecutorCatalogFailsClosed(t *testing.T) {
	if _, err := BuildToolsForRole(context.Background(), "admin", nil); err == nil {
		t.Fatal("missing registry must fail before loading database policy")
	}
}

func TestLegacyToolNamesRetainServerIdentity(t *testing.T) {
	r := NewBuiltinToolRegistry(42)
	for _, name := range []string{"fetch_homework", "fetch_and_grade_homework"} {
		tool, ok := r.GetTool(name)
		if !ok || tool.(*FetchAndGradeHomeworkTool).UserID != 42 {
			t.Fatalf("server identity missing from %s", name)
		}
	}
	for _, op := range []string{"get", "pause", "resume", "cancel"} {
		tool, ok := r.GetTool(op + "_fetch_job")
		if !ok || tool.(*FetchJobControlTool).UserID != 42 {
			t.Fatal("control lost identity")
		}
	}
}
