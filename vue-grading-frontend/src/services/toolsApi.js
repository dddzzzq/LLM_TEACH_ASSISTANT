import authApi from './authApi'

const apiClient = authApi.getClient()

// Permit frontend/backend rolling upgrades. Only a missing new route uses the alias;
// an existing tool's validation/permission error must never be retried elsewhere.
function isMissingRoute(error) {
  return error?.response?.status === 404 &&
    typeof error.response.data === 'string' &&
    error.response.data.trim() === '404 page not found'
}

async function request(method, suffix = '', payload) {
  try {
    return await apiClient.request({ method, url: `/api/admin/tools${suffix}`, data: payload })
  } catch (error) {
    if (!isMissingRoute(error)) throw error
    return apiClient.request({ method, url: `/api/admin/skills${suffix}`, data: payload })
  }
}

export default {
  listTools: () => request('get'),
  updateTool: (name, payload) => request('put', `/${encodeURIComponent(name)}`, payload),
  refreshToolsCache: () => request('post', '/cache/refresh')
}
