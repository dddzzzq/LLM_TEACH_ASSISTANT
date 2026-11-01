// 2. 引入你配置好的 authApi (请根据实际目录结构调整相对路径，例如 '../api/authApi')
import authApi from './authApi';

// 3. 获取带有 Token 拦截器的 axios 实例
const apiClient = authApi.getClient();

export default {
  // --- 现有作业 (Assignments) API ---

  gradeHomework(formData) {
    return apiClient.post('/homework/grade', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  getAssignments() {
    return apiClient.get('/assignments/');
  },

  getAssignment(id) {
    return apiClient.get(`/assignments/${id}`);
  },

  getAssignmentClasses(id) {
    return apiClient.get(`/assignments/${id}/classes`);
  },

  createAssignment(assignmentData) {
    return apiClient.post('/assignments/', assignmentData);
  },

  deleteAssignment(id) {
    return apiClient.delete(`/assignments/${id}`);
  },

  submitBatch(assignmentId, formData) {
    return apiClient.post(`/assignments/${assignmentId}/submit`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  getResultsForAssignment(assignmentId, batchId = '', classId = '') {
    return apiClient.get(`/assignments/${assignmentId}/results`, {
      params: {
        ...(batchId ? { batch_id: batchId } : {}),
        ...(classId ? { class_id: classId } : {}),
      },
    });
  },

  deleteAllSubmissions(assignmentId) {
    return apiClient.delete(`/assignments/${assignmentId}/results`);
  },

  deleteClassSubmissions(assignmentId, classId) {
    if (!classId) throw new Error('删除本班作业必须指定班级');
    return apiClient.delete(`/assignments/${assignmentId}/results`, {
      params: { class_id: classId },
    });
  },

  deleteSubmission(submissionId) {
    return apiClient.delete(`/submissions/${submissionId}`);
  },

  updateSubmission(submissionId, updateData) {
    return apiClient.put(`/submissions/${submissionId}`, updateData);
  },

  exportAssignment(assignmentId, batchId = '', classId = '') {
    return apiClient.get(`/assignments/${assignmentId}/export`, {
      responseType: 'blob', // 重要：告诉axios期望一个blob
      params: {
        ...(batchId ? { batch_id: batchId } : {}),
        ...(classId ? { class_id: classId } : {}),
      },
    });
  },
};
