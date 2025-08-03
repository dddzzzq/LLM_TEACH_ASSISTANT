import axios from 'axios'

const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:8000',
});

export default {
  // 任务端点
  createAssignment(data) {
    return apiClient.post('/assignments/', data, { headers: { 'Content-Type': 'application/json' } });
  },
  getAssignments() {
    return apiClient.get('/assignments/');
  },
  getAssignment(id) {
    return apiClient.get(`/assignments/${id}`);
  },
  deleteAssignment(id) {
    return apiClient.delete(`/assignments/${id}`);
  },

  // 提交端点
  getResultsForAssignment(id) {
    return apiClient.get(`/assignments/${id}/results`);
  },
  submitBatch(assignmentId, formData) {
    return apiClient.post(`/assignments/${assignmentId}/submit`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  deleteAllSubmissions(assignmentId) {
    return apiClient.delete(`/assignments/${assignmentId}/results`);
  },
  // Submission endpoints (核心修改点)
  deleteSubmission(submissionId) {
    // 路径从 /assignments/submissions/... 简化为 /submissions/...
    return apiClient.delete(`/submissions/${submissionId}`);
  },
  updateSubmission(submissionId, data) {
    // 路径从 /assignments/submissions/... 简化为 /submissions/...
    return apiClient.put(`/submissions/${submissionId}`, data, { headers: { 'Content-Type': 'application/json' } });
  }
}