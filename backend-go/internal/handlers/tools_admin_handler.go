package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"

	"grading-gateway/internal/agent"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"

	"github.com/gin-gonic/gin"
)

type ToolAdminUpdateRequest struct {
	Enabled      *bool     `json:"enabled,omitempty"`
	Description  *string   `json:"description,omitempty"`
	SchemaJSON   *string   `json:"schema_json,omitempty"`
	AllowedRoles *[]string `json:"allowed_roles,omitempty"`
	ImplKey      *string   `json:"impl_key,omitempty"`
}

// RegisterToolAdminRoutes mounts canonical routes and legacy aliases with identical middleware.
// /skills continues to mean tool configuration for existing clients; it is not a Skill catalog.
func RegisterToolAdminRoutes(group *gin.RouterGroup) {
	for _, path := range []string{"/tools", "/skills"} {
		group.GET(path, ListToolsAdmin)
		group.PUT(path+"/:name", UpdateToolAdmin)
		group.POST(path+"/cache/refresh", RefreshToolsCacheAdmin)
	}
}

type toolAdminView struct {
	models.ToolDefinition
	Registered bool `json:"registered"`
}

func toolView(row models.ToolDefinition, registry *agent.ToolRegistry) toolAdminView {
	tool, registered := registry.GetTool(row.Name)
	if registered {
		row.SchemaJSON = tool.Schema()
		row.ImplKey = tool.Name()
		if row.Description == "" {
			row.Description = tool.Description()
		}
	}
	return toolAdminView{ToolDefinition: row, Registered: registered}
}

// ListToolsAdmin lists policy configuration with the effective, code-owned schema.
func ListToolsAdmin(c *gin.Context) {
	var rows []models.ToolDefinition
	if err := database.DB.WithContext(c.Request.Context()).Order("id ASC").Find(&rows).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("查询工具失败: %v", err)})
		return
	}
	registry := agent.NewBuiltinToolRegistry(0)
	views := make([]toolAdminView, 0, len(rows))
	for _, row := range rows {
		views = append(views, toolView(row, registry))
	}
	c.JSON(http.StatusOK, views)
}

// validateToolContract permits unchanged legacy fields, but never edits executable contracts.
func validateToolContract(req ToolAdminUpdateRequest, tool agent.Tool, stored models.ToolDefinition) error {
	if req.SchemaJSON != nil {
		var supplied, actual any
		if json.Unmarshal([]byte(*req.SchemaJSON), &supplied) != nil {
			return fmt.Errorf("schema_json 不是合法 JSON")
		}
		if json.Unmarshal([]byte(tool.Schema()), &actual) != nil || !reflect.DeepEqual(supplied, actual) {
			return fmt.Errorf("工具参数 Schema 由执行器代码定义，不能通过配置修改；请重新加载工具配置")
		}
	}
	if req.ImplKey != nil && *req.ImplKey != tool.Name() && *req.ImplKey != stored.ImplKey {
		return fmt.Errorf("工具执行器由代码注册，不能通过 impl_key 修改")
	}
	return nil
}

// UpdateToolAdmin updates availability, role policy and description overrides only.
func UpdateToolAdmin(c *gin.Context) {
	name := strings.TrimSpace(c.Param("name"))
	if name == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "name 不能为空"})
		return
	}

	var req ToolAdminUpdateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求格式错误"})
		return
	}

	registry := agent.NewBuiltinToolRegistry(0)
	tool, registered := registry.GetTool(name)
	if !registered {
		c.JSON(http.StatusNotFound, gin.H{"error": "未注册该工具的执行器"})
		return
	}
	var config models.ToolDefinition
	if err := database.DB.WithContext(c.Request.Context()).Where("name = ?", name).First(&config).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "未找到该工具的配置"})
		return
	}
	if err := validateToolContract(req, tool, config); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if req.AllowedRoles != nil {
		// normalize/validate roles
		allowed := make([]string, 0, len(*req.AllowedRoles))
		for _, r := range *req.AllowedRoles {
			r = strings.TrimSpace(r)
			if r == "" {
				continue
			}
			switch r {
			case "student", "teacher", "admin":
				allowed = append(allowed, r)
			default:
				c.JSON(http.StatusBadRequest, gin.H{"error": "allowed_roles 仅支持 student/teacher/admin"})
				return
			}
		}
		b, _ := json.Marshal(allowed)
		config.AllowedRoles = string(b)
	}

	if req.Enabled != nil {
		config.Enabled = *req.Enabled
	}
	if req.Description != nil {
		config.Description = strings.TrimSpace(*req.Description)
	}

	if err := database.DB.WithContext(c.Request.Context()).Save(&config).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("保存失败: %v", err)})
		return
	}

	// A committed policy change must invalidate the cache even if the client disconnects.
	agent.InvalidateToolsCache(context.Background())
	c.JSON(http.StatusOK, toolView(config, registry))
}

// RefreshToolsCacheAdmin 手动清缓存
func RefreshToolsCacheAdmin(c *gin.Context) {
	agent.InvalidateToolsCache(c.Request.Context())
	c.JSON(http.StatusOK, gin.H{"ok": true})
}
