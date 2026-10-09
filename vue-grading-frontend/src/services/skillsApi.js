// Deprecated import: this API manages Tools, not Markdown Skills.
import toolsApi from './toolsApi'
export default {
  listSkills: toolsApi.listTools,
  updateSkill: toolsApi.updateTool,
  refreshSkillsCache: toolsApi.refreshToolsCache
}
