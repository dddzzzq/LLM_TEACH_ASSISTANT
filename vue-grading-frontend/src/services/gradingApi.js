import axios from 'axios';

// 设置后端的 API 基础 URL
const API_URL = 'http://127.0.0.1:8000';   // 8000端口
// const API_URL = 'http://450992b4.r7.cpolar.cn';

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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

  getResultsForAssignment(assignmentId) {
    return apiClient.get(`/assignments/${assignmentId}/results`);
  },
  
  deleteAllSubmissions(assignmentId) {
    return apiClient.delete(`/assignments/${assignmentId}/results`);
  },

  deleteSubmission(submissionId) {
    return apiClient.delete(`/submissions/${submissionId}`);
  },

  updateSubmission(submissionId, updateData) {
    return apiClient.put(`/submissions/${submissionId}`, updateData);
  },

  exportAssignment(assignmentId) {
    return apiClient.get(`/assignments/${assignmentId}/export`, {
      responseType: 'blob', // 重要：告诉axios期望一个blob
    });
  },

  // --- 新增试卷 (Exams) API ---

  /**
   * 获取所有试卷列表
   */
  getExams() {
    return apiClient.get('/exams/');
  },

  /**
   * 创建一个新的试卷
   * @param {{ name: string }} examData - e.g. { name: "2022期末考" }
   */
  createExam(examData) {
    return apiClient.post('/exams/', examData);
  },

  /**
   * 获取单个试卷的详细信息，包括所有题目
   * @param {string|number} examId - 试卷ID
   */
  getExamDetails(examId) {
    return apiClient.get(`/exams/${examId}`);
  },

  /**
   * 删除一个试卷
   * @param {string|number} examId - 试卷ID
   */
  deleteExam(examId) {
    return apiClient.delete(`/exams/${examId}`);
  },

  /**
   * 为试卷添加一道题目
   * @param {string|number} examId - 试卷ID
   * @param {object} questionData - e.g. { question_number, question_text, standard_answer, rubric }
   */
  addExamQuestion(examId, questionData) {
    return apiClient.post(`/exams/${examId}/questions`, questionData);
  },

  /**
   * 提交一个学生的试卷图片（可多张）进行评分
   * @param {string|number} examId - 试卷ID
   * @param {FormData} formData - 包含 student_id 和 images
   */
  gradeStudentExam(examId, formData) {
    return apiClient.post(`/exams/${examId}/grade_submission`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  /**
   * 获取某次试卷的所有学生成绩总结
   * @param {string|number} examId - 试卷ID
   */
  getExamResultsSummary(examId) {
    return apiClient.get(`/exams/${examId}/results`);
  },

  /**
   * 获取单个学生的详细报告（包括每题得分）
   * @param {string|number} examId - 试卷ID
   * @param {string|number} studentExamId - 学生试卷提交ID (注意：不是学生学号)
   */
  getStudentDetailedReport(examId, studentExamId) {
    return apiClient.get(`/exams/${examId}/results/${studentExamId}`);
  },
  /**
   * 删除特定学生的评分结果
   * @param {string|number} examId 
   * @param {string|number} studentExamId 
   */
  deleteStudentExamResult(examId, studentExamId) {
    return apiClient.delete(`/exams/${examId}/results/${studentExamId}`);
  },

};

