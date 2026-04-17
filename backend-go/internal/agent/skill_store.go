package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

const (
	skillsCacheKey   = "skills:enabled:v1"
	skillsCacheTTL   = 5 * time.Hour
	skillsSeedMarker = "skills:seeded:v1"
)

type skillToolDTO struct {
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

// EnsureDefaultSkillsSeeded 会在 skills 表为空时写入默认的内置技能定义。
// 该函数是幂等的：若 skills 已存在数据，则不会重复写入。
func EnsureDefaultSkillsSeeded(ctx context.Context) {
	if database.DB == nil {
		log.Printf("EnsureDefaultSkillsSeeded: database.DB is nil, skip seeding")
		return
	}

	var count int64
	if err := database.DB.Model(&models.SkillDefinition{}).Count(&count).Error; err != nil {
		log.Printf("EnsureDefaultSkillsSeeded: count failed: %v", err)
		return
	}
	if count > 0 {
		return
	}

	defaultAllowedAll, _ := json.Marshal([]string{"student", "teacher", "admin"})
	defaultAllowedTeacherAdmin, _ := json.Marshal([]string{"teacher", "admin"})

	seed := []models.SkillDefinition{
		{
			Name:         "query_student_score",
			ImplKey:      "QueryStudentScoreSkill",
			Description:  (&QueryStudentScoreSkill{}).Description(),
			SchemaJSON:   (&QueryStudentScoreSkill{}).Schema(),
			Enabled:      true,
			AllowedRoles: string(defaultAllowedAll),
		},
		{
			Name:         "trigger_async_pipeline",
			ImplKey:      "TriggerPipelineSkill",
			Description:  (&TriggerPipelineSkill{}).Description(),
			SchemaJSON:   (&TriggerPipelineSkill{}).Schema(),
			Enabled:      true,
			AllowedRoles: string(defaultAllowedTeacherAdmin),
		},
		{
			Name:         "fetch_and_grade_homework",
			ImplKey:      "FetchAndGradeHomeworkSkill",
			Description:  (&FetchAndGradeHomeworkSkill{}).Description(),
			SchemaJSON:   (&FetchAndGradeHomeworkSkill{}).Schema(),
			Enabled:      true,
			AllowedRoles: string(defaultAllowedAll),
		},
	}

	if err := database.DB.Create(&seed).Error; err != nil {
		log.Printf("EnsureDefaultSkillsSeeded: seed insert failed: %v", err)
		return
	}
	log.Printf("EnsureDefaultSkillsSeeded: seeded %d default skills", len(seed))

	// Best-effort: set a marker in Redis to avoid repeated reads in multi-instance startup storms.
	rc := database.GetRedisClient()
	if rc != nil {
		_ = rc.Set(ctx, skillsSeedMarker, "1", 10*time.Minute).Err()
	}
}

func loadEnabledSkillToolsFromDB(ctx context.Context) ([]skillToolDTO, error) {
	if database.DB == nil {
		return nil, fmt.Errorf("database not initialized")
	}
	var rows []models.SkillDefinition
	if err := database.DB.Where("enabled = ?", true).Find(&rows).Error; err != nil {
		return nil, err
	}

	out := make([]skillToolDTO, 0, len(rows))
	for _, r := range rows {
		out = append(out, skillToolDTO{
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

func loadEnabledSkillToolsCached(ctx context.Context) ([]skillToolDTO, error) {
	rc := database.GetRedisClient()
	if rc == nil {
		return loadEnabledSkillToolsFromDB(ctx)
	}

	if raw, err := rc.Get(ctx, skillsCacheKey).Result(); err == nil && raw != "" {
		var cached []skillToolDTO
		if err := json.Unmarshal([]byte(raw), &cached); err == nil {
			return cached, nil
		}
		// fallthrough: cache parse error -> DB
	}

	rows, err := loadEnabledSkillToolsFromDB(ctx)
	if err != nil {
		return nil, err
	}

	if b, err := json.Marshal(rows); err == nil {
		_ = rc.Set(ctx, skillsCacheKey, string(b), skillsCacheTTL).Err()
	}
	return rows, nil
}

// BuildToolsForRole 从 MySQL/Redis 获取启用的技能定义，按角色过滤，并生成 DeepSeek/OpenAI tools 格式。
// registry 用于确保只下发“代码里确实注册了执行器”的工具，避免 DB 配置与实现不一致导致 LLM 调用未知工具。
func BuildToolsForRole(ctx context.Context, role string, registry *SkillRegistry) ([]map[string]interface{}, error) {
	rows, err := loadEnabledSkillToolsCached(ctx)
	if err != nil {
		return nil, err
	}

	var tools []map[string]interface{}
	for _, r := range rows {
		if !r.Enabled {
			continue
		}
		if !roleAllowed(r.AllowedRoles, role) {
			continue
		}
		if registry != nil {
			if _, ok := registry.GetSkill(r.Name); !ok {
				continue
			}
		}

		var schemaMap map[string]interface{}
		if err := json.Unmarshal([]byte(r.SchemaJSON), &schemaMap); err != nil {
			continue
		}

		tool := map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        r.Name,
				"description": r.Description,
				"parameters":  schemaMap,
			},
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

// InvalidateSkillsCache 主动清理 skills tools 的 Redis 缓存。
func InvalidateSkillsCache(ctx context.Context) {
	rc := database.GetRedisClient()
	if rc == nil {
		return
	}
	_ = rc.Del(ctx, skillsCacheKey).Err()
}
