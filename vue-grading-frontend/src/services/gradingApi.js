import axios from 'axios'

const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:8000',
});

export default {
  // Assignment endpoints
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

  // Submission endpoints
  getResultsForAssignment(id) {
    return apiClient.get(`/assignments/${id}/results`);
  },
  submitBatch(assignmentId, formData) {
    return apiClient.post(`/assignments/${assignmentId}/submit`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  deleteSubmission(submissionId) {
    return apiClient.delete(`/submissions/${submissionId}`);
  },
  deleteAllSubmissions(assignmentId) {
    return apiClient.delete(`/assignments/${assignmentId}/results`);
  }
}