package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"grading-gateway/internal/agent"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"

	"github.com/gin-gonic/gin"
)

type SkillAdminUpdateRequest struct {
	Enabled      *bool     `json:"enabled,omitempty"`
	Description  *string   `json:"description,omitempty"`
	SchemaJSON   *string   `json:"schema_json,omitempty"`
	AllowedRoles *[]string `json:"allowed_roles,omitempty"`
	ImplKey      *string   `json:"impl_key,omitempty"`
}

// ListSkillsAdmin 获取所有技能定义（含禁用的）
func ListSkillsAdmin(c *gin.Context) {
	var skills []models.SkillDefinition
	if err := database.DB.Order("id ASC").Find(&skills).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("查询技能失败: %v", err)})
		return
	}
	c.JSON(http.StatusOK, skills)
}

// UpdateSkillAdmin 更新指定技能定义（enabled/roles/desc/schema/impl_key）
func UpdateSkillAdmin(c *gin.Context) {
	name := strings.TrimSpace(c.Param("name"))
	if name == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "name 不能为空"})
		return
	}

	var req SkillAdminUpdateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求格式错误"})
		return
	}

	var skill models.SkillDefinition
	if err := database.DB.Where("name = ?", name).First(&skill).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "未找到该 skill"})
		return
	}

	// validate schema json if provided
	if req.SchemaJSON != nil {
		var tmp any
		if err := json.Unmarshal([]byte(*req.SchemaJSON), &tmp); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "schema_json 不是合法 JSON"})
			return
		}
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
		skill.AllowedRoles = string(b)
	}

	if req.Enabled != nil {
		skill.Enabled = *req.Enabled
	}
	if req.Description != nil {
		skill.Description = strings.TrimSpace(*req.Description)
	}
	if req.SchemaJSON != nil {
		skill.SchemaJSON = strings.TrimSpace(*req.SchemaJSON)
	}
	if req.ImplKey != nil {
		skill.ImplKey = strings.TrimSpace(*req.ImplKey)
	}

	if err := database.DB.Save(&skill).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("保存失败: %v", err)})
		return
	}

	agent.InvalidateSkillsCache(context.Background())
	c.JSON(http.StatusOK, skill)
}

// RefreshSkillsCacheAdmin 手动清缓存
func RefreshSkillsCacheAdmin(c *gin.Context) {
	agent.InvalidateSkillsCache(context.Background())
	c.JSON(http.StatusOK, gin.H{"ok": true})
}
