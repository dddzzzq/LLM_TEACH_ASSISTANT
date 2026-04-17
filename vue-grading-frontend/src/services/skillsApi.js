import authApi from './authApi'

const apiClient = authApi.getClient()

export default {
  listSkills() {
    return apiClient.get('/api/admin/skills')
  },

  updateSkill(name, payload) {
    return apiClient.put(`/api/admin/skills/${encodeURIComponent(name)}`, payload)
  },

  refreshSkillsCache() {
    return apiClient.post('/api/admin/skills/cache/refresh')
  }
}

