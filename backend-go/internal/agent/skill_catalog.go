package agent

import "grading-gateway/internal/agent/catalog"

type SkillDocument = catalog.Document
type SkillCatalog = catalog.Catalog

func LoadGradingSkills(root string) (SkillCatalog, error) {
	return catalog.Load(root, "teaching-assistant")
}
func DefaultGradingSkills() (SkillCatalog, error) { return catalog.Default("teaching-assistant") }
