package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"gorm.io/gorm"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

const (
	toolsCacheKey   = "skills:enabled:v1"
	toolsCacheTTL   = 5 * time.Hour
	toolsSeedMarker = "skills:seeded:v1"
)

type toolConfigDTO struct {
	Name         string `json:"name"`
	Description  string `json:"description"`
	SchemaJSON   string `json:"schema_json"`
	Enabled      bool   `json:"enabled"`
	AllowedRoles string `json:"allowed_roles"`
	ImplKey      string `json:"impl_key"`
}

func roleAllowed(allowedRolesJSON string, role string) bool {
	if role == "" {
		return false
	}
	var roles []string
	if err := json.Unmarshal([]byte(allowedRolesJSON), &roles); err != nil {
		return false
	}
	for _, r := range roles {
		if r == role {
			return true
		}
	}
	return false
}

// EnsureDefaultToolsSeeded 会在工具配置表为空时写入默认的内置工具配置。
// 该函数是幂等的：若工具配置表已存在数据，则不会重复写入。
func EnsureDefaultToolsSeeded(ctx context.Context) {
	if database.DB == nil {
		log.Printf("EnsureDefaultToolsSeeded: database.DB is nil, skip seeding")
		return
	}

	var count int64
	if err := database.DB.Model(&models.ToolDefinition{}).Count(&count).Error; err != nil {
		log.Printf("EnsureDefaultToolsSeeded: count failed: %v", err)
		return
	}
	if count > 0 {
		return
	}

	defaultAllowedAll, _ := json.Marshal([]string{"student", "teacher", "admin"})
	defaultAllowedTeacherAdmin, _ := json.Marshal([]string{"teacher", "admin"})

	seed := []models.ToolDefinition{
		{
			Name:         "query_student_score",
			ImplKey:      "QueryStudentScoreTool",
			Description:  (&QueryStudentScoreTool{}).Description(),
			SchemaJSON:   (&QueryStudentScoreTool{}).Schema(),
			Enabled:      true,
			AllowedRoles: string(defaultAllowedAll),
		},
		{
			Name:         "trigger_async_pipeline",
			ImplKey:      "TriggerPipelineTool",
			Description:  (&TriggerPipelineTool{}).Description(),
			SchemaJSON:   (&TriggerPipelineTool{}).Schema(),
			Enabled:      true,
			AllowedRoles: string(defaultAllowedTeacherAdmin),
		},
		{
			Name:         "fetch_and_grade_homework",
			ImplKey:      "FetchAndGradeHomeworkTool",
			Description:  (&FetchAndGradeHomeworkTool{}).Description(),
			SchemaJSON:   (&FetchAndGradeHomeworkTool{}).Schema(),
			Enabled:      true,
			AllowedRoles: string(defaultAllowedTeacherAdmin),
		},
	}

	if err := database.DB.Create(&seed).Error; err != nil {
		log.Printf("EnsureDefaultToolsSeeded: seed insert failed: %v", err)
		return
	}
	log.Printf("EnsureDefaultToolsSeeded: seeded %d default tools", len(seed))

	// Best-effort: set a marker in Redis to avoid repeated reads in multi-instance startup storms.
	rc := database.GetRedisClient()
	if rc != nil {
		_ = rc.Set(ctx, toolsSeedMarker, "1", 10*time.Minute).Err()
	}
}

func loadEnabledToolsFromDB(ctx context.Context) ([]toolConfigDTO, error) {
	if database.DB == nil {
		return nil, fmt.Errorf("database not initialized")
	}
	var rows []models.ToolDefinition
	if err := database.DB.WithContext(ctx).Where("enabled = ?", true).Order("name ASC").Find(&rows).Error; err != nil {
		return nil, err
	}

	out := make([]toolConfigDTO, 0, len(rows))
	for _, r := range rows {
		out = append(out, toolConfigDTO{
			Name:         r.Name,
			Description:  r.Description,
			SchemaJSON:   r.SchemaJSON,
			Enabled:      r.Enabled,
			AllowedRoles: r.AllowedRoles,
			ImplKey:      r.ImplKey,
		})
	}
	return out, nil
}

func loadEnabledToolsCached(ctx context.Context) ([]toolConfigDTO, error) {
	rc := database.GetRedisClient()
	if rc == nil {
		return loadEnabledToolsFromDB(ctx)
	}

	if raw, err := rc.Get(ctx, toolsCacheKey).Result(); err == nil && raw != "" {
		var cached []toolConfigDTO
		if err := json.Unmarshal([]byte(raw), &cached); err == nil {
			return cached, nil
		}
		// fallthrough: cache parse error -> DB
	}

	rows, err := loadEnabledToolsFromDB(ctx)
	if err != nil {
		return nil, err
	}

	if b, err := json.Marshal(rows); err == nil {
		_ = rc.Set(ctx, toolsCacheKey, string(b), toolsCacheTTL).Err()
	}
	return rows, nil
}

// BuildToolsForRole 从 MySQL/Redis 获取启用的工具定义，按角色过滤，并生成 DeepSeek/OpenAI tools 格式。
// registry 用于确保只下发“代码里确实注册了执行器”的工具，避免 DB 配置与实现不一致导致 LLM 调用未知工具。
func BuildToolsForRole(ctx context.Context, role string, registry *ToolRegistry) ([]map[string]interface{}, error) {
	if registry == nil {
		return nil, fmt.Errorf("tool registry is required")
	}
	rows, err := loadEnabledToolsCached(ctx)
	if err != nil {
		return nil, err
	}

	return buildToolsForRole(rows, role, registry)
}

func buildToolsForRole(rows []toolConfigDTO, role string, registry *ToolRegistry) ([]map[string]interface{}, error) {
	if registry == nil {
		return nil, fmt.Errorf("tool registry is required")
	}
	tools := make([]map[string]interface{}, 0)
	for _, r := range rows {
		if !r.Enabled {
			continue
		}
		if !roleAllowed(r.AllowedRoles, role) {
			continue
		}
		executor, ok := registry.GetTool(r.Name)
		if !ok {
			continue
		}
		// Database rows configure availability and descriptions, never executable contracts.
		tool, err := ToolFunction(executor, r.Description)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

// InvalidateToolsCache clears the shared legacy policy cache for old and new callers.
func InvalidateToolsCache(ctx context.Context) {
	rc := database.GetRedisClient()
	if rc == nil {
		return
	}
	_ = rc.Del(ctx, toolsCacheKey).Err()
}

// EnsureRPATools migrates the credential-based built-in schema and adds task controls.
// Preserve existing enable switches and customized descriptions unless migrating the legacy schema.
func EnsureRPATools(ctx context.Context) {
	if database.DB == nil {
		return
	}
	roles := `["teacher","admin"]`
	tools := []Tool{&FetchAndGradeHomeworkTool{}, &FetchAndGradeHomeworkTool{UseFetchAlias: true}}
	for _, op := range []string{"get", "pause", "resume", "cancel"} {
		tools = append(tools, &FetchJobControlTool{Operation: op})
	}
	for _, tool := range tools {
		var row models.ToolDefinition
		result := database.DB.Where("name = ?", tool.Name()).First(&row)
		if result.Error != nil {
			if result.Error == gorm.ErrRecordNotFound {
				database.DB.Create(&models.ToolDefinition{Name: tool.Name(), ImplKey: tool.Name(), Description: tool.Description(), SchemaJSON: tool.Schema(), Enabled: true, AllowedRoles: roles})
			}
			continue
		}
		var schema map[string]any
		_ = json.Unmarshal([]byte(row.SchemaJSON), &schema)
		properties, _ := schema["properties"].(map[string]any)
		if _, legacy := properties["password"]; legacy {
			database.DB.Model(&row).Updates(map[string]any{"schema_json": tool.Schema(), "description": tool.Description(), "allowed_roles": roles})
		}
	}
	InvalidateToolsCache(ctx)
}

// EnsureGradingTools adds new capabilities without resetting existing administrator policy.
func EnsureGradingTools(ctx context.Context) {
	if database.DB == nil {
		return
	}
	for _, op := range []string{"inspect", "submit", "status"} {
		tool := &GradingTool{Operation: op}
		row := models.ToolDefinition{Name: tool.Name(), ImplKey: tool.Name(), Description: tool.Description(), SchemaJSON: tool.Schema(), Enabled: true, AllowedRoles: `["teacher","admin"]`}
		if err := database.DB.WithContext(ctx).Where("name = ?", tool.Name()).FirstOrCreate(&row).Error; err != nil {
			log.Printf("初始化批改工具 %s 失败: %v", tool.Name(), err)
		}
	}
	InvalidateToolsCache(ctx)
}
