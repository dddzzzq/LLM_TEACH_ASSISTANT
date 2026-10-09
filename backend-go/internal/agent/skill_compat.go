package agent

import "context"

// Deprecated compatibility names. New code uses Tool; Skill knowledge lives in skills/.
type Skill = Tool
type SkillRegistry = ToolRegistry
type QueryStudentScoreSkill = QueryStudentScoreTool
type TriggerPipelineSkill = TriggerPipelineTool
type FetchAndGradeHomeworkSkill = FetchAndGradeHomeworkTool
type FetchJobControlSkill = FetchJobControlTool

func NewSkillRegistry() *ToolRegistry                     { return NewToolRegistry() }
func (r *ToolRegistry) GetSkill(name string) (Tool, bool) { return r.GetTool(name) }
func (r *ToolRegistry) ListSkills() []string              { return r.ListTools() }
func (r *ToolRegistry) RemoveSkill(name string)           { r.RemoveTool(name) }
func EnsureDefaultSkillsSeeded(ctx context.Context)       { EnsureDefaultToolsSeeded(ctx) }
func EnsureRPASkills(ctx context.Context)                 { EnsureRPATools(ctx) }
func InvalidateSkillsCache(ctx context.Context)           { InvalidateToolsCache(ctx) }
