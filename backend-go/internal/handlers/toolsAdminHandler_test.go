package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm/schema"
	"grading-gateway/internal/agent"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
)

func TestToolAdminReportsCodeSchemaAndPreservesStoredPolicy(t *testing.T) {
	row := models.ToolDefinition{Name: "fetch_homework", SchemaJSON: `{"password":"old"}`,
		ImplKey: "FetchAndGradeHomeworkSkill", Enabled: false, AllowedRoles: `["admin"]`}
	r := agent.NewBuiltinToolRegistry(0)
	view := toolView(row, r)
	tool, _ := r.GetTool(row.Name)
	if !view.Registered || view.SchemaJSON != tool.Schema() || view.ImplKey != tool.Name() {
		t.Fatal("admin view does not show the runtime contract")
	}
	if view.Enabled || view.AllowedRoles != row.AllowedRoles || row.SchemaJSON == view.SchemaJSON {
		t.Fatal("projection changed stored policy or metadata")
	}
	if toolView(models.ToolDefinition{Name: "unregistered"}, r).Registered {
		t.Fatal("unknown tool registered")
	}
	// Renaming the Go model must not create an empty replacement table during AutoMigrate.
	for _, model := range []any{&models.ToolDefinition{}, &models.SkillDefinition{}} {
		parsed, err := schema.Parse(model, &sync.Map{}, schema.NamingStrategy{})
		if err != nil || parsed.Table != "skill_definitions" {
			t.Fatalf("table changed: %v, %v", parsed, err)
		}
	}
}

func TestToolAdminRejectsContractEditsButAcceptsLegacyRoundTrip(t *testing.T) {
	tool := &agent.FetchAndGradeHomeworkTool{}
	row := models.ToolDefinition{ImplKey: "FetchAndGradeHomeworkSkill"}
	for _, tc := range []struct {
		name, schema, impl string
		wantError          bool
	}{
		{"unchanged legacy client", "\n" + tool.Schema(), row.ImplKey, false},
		{"canonical client", tool.Schema(), tool.Name(), false},
		{"schema drift", `{"type":"object","properties":{"password":{}}}`, tool.Name(), true},
		{"invalid schema", "broken", tool.Name(), true},
		{"executor replacement", tool.Schema(), "another_executor", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateToolContract(ToolAdminUpdateRequest{SchemaJSON: &tc.schema, ImplKey: &tc.impl}, tool, row)
			if (err != nil) != tc.wantError {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestToolAdminAliasesShareAuthentication(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	group := router.Group("/api/admin", middleware.AuthMiddleware(), middleware.RBACMiddleware("teacher", "admin"))
	RegisterToolAdminRoutes(group)
	for _, prefix := range []string{"/tools", "/skills"} {
		for _, route := range []struct{ method, suffix string }{{"GET", ""}, {"PUT", "/fetch_homework"}, {"POST", "/cache/refresh"}} {
			req := httptest.NewRequest(route.method, "/api/admin"+prefix+route.suffix, strings.NewReader(`{}`))
			response := httptest.NewRecorder()
			router.ServeHTTP(response, req)
			if response.Code != http.StatusUnauthorized {
				t.Fatalf("%s %s got %d", route.method, req.URL, response.Code)
			}
		}
	}
}
