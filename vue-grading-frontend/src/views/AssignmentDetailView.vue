<template>
  <div class="p-4 md:p-8">
    <div v-if="isLoadingAssignment" class="py-10 text-center">
      <Loader />
    </div>

    <div v-else-if="error" class="p-4 text-red-500 bg-red-100 rounded-lg">
      {{ error }}
    </div>

    <div v-else-if="assignment" class="space-y-8">
      <div class="p-6 bg-white rounded-lg shadow-lg">
        <h1 class="text-3xl font-bold text-gray-800">{{ assignment.task_name }}</h1>
        <p class="mt-4 text-gray-600 whitespace-pre-wrap">{{ assignment.question }}</p>
        <div class="mt-4">
          <h3 class="font-semibold text-gray-700">评分标准:</h3>
          <pre
            class="p-4 mt-2 overflow-x-auto text-sm text-gray-700 bg-gray-50 border rounded-md whitespace-pre-wrap"
            >{{ JSON.stringify(assignment.rubric, null, 2) }}</pre
          >
        </div>
      </div>

      <div class="p-6 bg-white rounded-lg shadow-lg">
        <h2 class="mb-4 text-2xl font-bold text-gray-800">提交学生作业</h2>
        <form @submit.prevent="submitBatchFile">
          <label for="file-upload" class="block text-sm font-medium text-gray-700">
            上传包含所有学生作业的ZIP压缩包
          </label>
          <input
            id="file-upload"
            type="file"
            accept=".zip,.rar"
            class="block w-full mt-2 text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 rounded-full hover:file:bg-indigo-100"
            required
            @change="handleFileChange"
          />
          <p v-if="fileName" class="mt-2 text-sm text-green-600">
            已选择文件: {{ fileName }}
          </p>
          <button
            type="submit"
            :disabled="isSubmitting"
            class="flex justify-center px-6 py-2 mt-4 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md shadow-sm w-full sm:w-auto hover:bg-indigo-700 disabled:bg-indigo-400"
          >
            <span v-if="!isSubmitting">提交并开始后台评分</span>
            <span v-else>提交中...</span>
          </button>
        </form>
        <div
          v-if="submissionMessage"
          class="p-3 mt-4 rounded-md"
          :class="
            submissionError ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
          "
        >
          {{ submissionMessage }}
        </div>
      </div>

      <div class="p-6 bg-white rounded-lg shadow-lg">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-2xl font-bold text-gray-800">评分结果</h2>
          <div class="flex items-center gap-4">
            <button
              @click="deleteAllResults"
              :disabled="isLoadingResults || results.length === 0"
              class="text-sm font-semibold text-red-600 hover:text-red-800 disabled:text-gray-400 disabled:cursor-not-allowed"
            >
              一键清空
            </button>
            <button
              @click="fetchResults"
              :disabled="isLoadingResults"
              class="text-sm text-indigo-600 hover:text-indigo-800 disabled:text-gray-400"
            >
              刷新结果
            </button>
          </div>
        </div>

        <div v-if="isLoadingResults" class="py-5 text-center">
          <Loader />
        </div>
        <div v-else-if="results.length > 0" class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th
                  class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase"
                >
                  学生ID
                </th>
                <th
                  class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase"
                >
                  分数
                </th>
                <th
                  class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase"
                >
                  查重报告
                </th>
                <th
                  class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase"
                >
                  AIGC检测
                </th>
                <th
                  class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase"
                >
                  评语
                </th>
                <th
                  class="px-6 py-3 text-xs font-medium text-right text-gray-500 uppercase"
                >
                  操作
                </th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr v-for="(result, index) in results" :key="result.id">
                <td class="px-6 py-4 text-sm font-medium text-gray-900 whitespace-nowrap">
                  <div class="flex items-center">
                    <span>{{ result.student_id }}</span>
                    <svg
                      v-if="result.is_human_reviewed"
                      xmlns="http://www.w3.org/2000/svg"
                      class="h-5 w-5 ml-2 text-green-500"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                      title="已由教师复查"
                    >
                      <path
                        fill-rule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                        clip-rule="evenodd"
                      />
                    </svg>
                  </div>
                </td>
                <td
                  class="px-6 py-4 text-sm font-bold whitespace-nowrap"
                  :class="getScoreColor(result.score)"
                >
                  {{ result.score.toFixed(1) }}
                </td>
                <td class="px-6 py-4 text-sm whitespace-nowrap">
                  <button
                    @click="showReport(result, 'plagiarism')"
                    class="font-semibold hover:underline disabled:text-gray-400 disabled:cursor-not-allowed"
                    :class="getPlagiarismSummaryClass(result.plagiarism_reports)"
                    :disabled="!hasHighRiskPlagiarism(result.plagiarism_reports)"
                  >
                    {{ formatPlagiarismSummary(result.plagiarism_reports) }}
                  </button>
                </td>
                <td
                  class="px-6 py-4 whitespace-nowrap text-sm font-semibold"
                  :class="getAIGCColor(result.aigc_report)"
                >
                  {{ formatAIGC(result.aigc_report) }}
                </td>
                <td
                  class="px-6 py-4 text-sm text-gray-500 whitespace-normal truncate max-w-xs"
                  :title="result.human_feedback || result.feedback"
                >
                  {{ result.human_feedback || result.feedback }}
                </td>
                <td class="px-6 py-4 text-sm font-medium text-right whitespace-nowrap">
                  <button
                    @click="openReviewModal(result)"
                    title="教师复查"
                    class="p-1 text-gray-400 rounded-full hover:text-indigo-600"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      class="h-5 w-5"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z"
                      />
                      <path
                        fill-rule="evenodd"
                        d="M2 6a2 2 0 012-2h4a1 1 0 010 2H4v10h10v-4a1 1 0 112 0v4a2 2 0 01-2 2H4a2 2 0 01-2-2V6z"
                        clip-rule="evenodd"
                      />
                    </svg>
                  </button>
                  <button
                    @click="showReport(result, 'feedback')"
                    title="查看完整评语"
                    class="p-1 text-gray-400 rounded-full hover:text-blue-600"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      class="h-5 w-5"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                      <path
                        fill-rule="evenodd"
                        d="M.458 10C3.732 4.943 9.522 1 10 1s6.268 3.943 9.542 9c-3.274 5.057-9.03 9-9.542 9S3.732 15.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                        clip-rule="evenodd"
                      />
                    </svg>
                  </button>
                  <button
                    @click="deleteSingleResult(result.id, index)"
                    title="删除此条记录"
                    class="p-1 text-gray-400 rounded-full hover:text-red-600"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      class="w-5 h-5"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fill-rule="evenodd"
                        d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z"
                        clip-rule="evenodd"
                      />
                    </svg>
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="py-5 text-center text-gray-500">暂无提交结果。</div>
      </div>
    </div>
  </div>

  <ReportModal
    v-if="isModalVisible"
    :submission="selectedSubmission"
    :reportType="reportType"
    @close="isModalVisible = false"
  />
  <ReviewModal
    v-if="isReviewModalVisible"
    :submission="submissionToReview"
    @close="isReviewModalVisible = false"
    @save="handleSaveReview"
  />
</template>

<script setup lang="ts">
import { ref, onMounted } from "vue";
import { useRoute } from "vue-router";
import gradingApi from "../services/gradingApi";
import Loader from "../components/Loader.vue";
import ReportModal from "../components/ReportModal.vue";
import ReviewModal from "../components/ReviewModal.vue";

// --- Interfaces ---
interface AIGCReport {
  predicted_label: string;
  confidence: number;
  ai_probability: number;
  detection_source?: string;
}
interface LLMPlagiarismAnalysis {
  similarity_score: number;
  reasoning: string;
  suspicious_parts: any[];
}
interface PlagiarismReport {
  similar_to: string;
  initial_score: number;
  content_type: string;
  llm_analysis?: LLMPlagiarismAnalysis;
}
interface SubmissionResult {
  id: number;
  student_id: string;
  score: number;
  feedback: string;
  plagiarism_reports: PlagiarismReport[];
  aigc_report?: AIGCReport;
  is_human_reviewed: boolean;
  human_feedback?: string;
}
interface Assignment {
  task_name: string;
  question: string;
  rubric: object;
  submissions: SubmissionResult[];
}

const props = defineProps<{ id: string }>();

// --- Component State ---
const assignment = ref<Assignment | null>(null);
const results = ref<SubmissionResult[]>([]);
const selectedSubmission = ref<SubmissionResult | null>(null);
const file = ref<File | null>(null);
const fileName = ref("");
const isLoadingAssignment = ref(true);
const isLoadingResults = ref(false);
const isSubmitting = ref(false);
const isModalVisible = ref(false);
const error = ref<string | null>(null);
const submissionMessage = ref("");
const submissionError = ref(false);
const reportType = ref("feedback");
const isReviewModalVisible = ref(false);
const submissionToReview = ref<SubmissionResult | null>(null);

// --- API Methods ---
const fetchAssignmentDetails = async () => {
  isLoadingAssignment.value = true;
  try {
    const response = await gradingApi.getAssignment(props.id);
    assignment.value = response.data;
    results.value = response.data.submissions || [];
  } catch (e) {
    error.value = "无法加载作业详情。";
    console.error(e);
  } finally {
    isLoadingAssignment.value = false;
  }
};
const fetchResults = async () => {
  isLoadingResults.value = true;
  try {
    const response = await gradingApi.getResultsForAssignment(props.id);
    results.value = response.data;
  } catch (e) {
    console.error("无法加载结果:", e);
  } finally {
    isLoadingResults.value = false;
  }
};
const submitBatchFile = async () => {
  if (!file.value) return;
  isSubmitting.value = true;
  submissionMessage.value = "";
  submissionError.value = false;
  const formData = new FormData();
  formData.append("batch_file", file.value);
  try {
    const response = await gradingApi.submitBatch(props.id, formData);
    submissionMessage.value = response.data.message;
    fileName.value = "";
    file.value = null;
    (document.getElementById("file-upload") as HTMLInputElement).value = "";
  } catch (e: any) {
    submissionError.value = true;
    submissionMessage.value = e.response?.data?.detail || "提交失败。";
  } finally {
    isSubmitting.value = false;
  }
};
const deleteSingleResult = async (submissionId: number, index: number) => {
  if (confirm(`确定要删除学生 ${results.value[index].student_id} 的评分记录吗？`)) {
    try {
      await gradingApi.deleteSubmission(submissionId);
      results.value.splice(index, 1);
    } catch (e) {
      alert("删除失败，请重试。");
    }
  }
};
const deleteAllResults = async () => {
  if (results.value.length === 0) return;
  if (confirm(`确定要清空此作业下的所有 ${results.value.length} 条评分记录吗？`)) {
    try {
      await gradingApi.deleteAllSubmissions(props.id);
      results.value = [];
    } catch (e) {
      alert("清空失败，请重试。");
    }
  }
};

// --- UI Logic ---
const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement;
  const selectedFile = target.files?.[0];
  if (selectedFile) {
    file.value = selectedFile;
    fileName.value = selectedFile.name;
  }
};

const showReport = (submission: SubmissionResult, type: string) => {
  if (type === "plagiarism") {
    const submissionCopy = JSON.parse(JSON.stringify(submission));
    submissionCopy.plagiarism_reports = submissionCopy.plagiarism_reports.filter(
      (report: PlagiarismReport) => (report.llm_analysis?.similarity_score || 0) >= 80
    );
    selectedSubmission.value = submissionCopy;
  } else {
    selectedSubmission.value = submission;
  }
  reportType.value = type;
  isModalVisible.value = true;
};

const openReviewModal = (submission: SubmissionResult) => {
  submissionToReview.value = submission;
  isReviewModalVisible.value = true;
};

const handleSaveReview = async (updatedData: {
  id: number;
  score: number;
  human_feedback: string;
}) => {
  try {
    const response = await gradingApi.updateSubmission(updatedData.id, {
      score: updatedData.score,
      human_feedback: updatedData.human_feedback,
    });
    const index = results.value.findIndex((r) => r.id === updatedData.id);
    if (index !== -1) {
      results.value[index] = response.data;
    }
    isReviewModalVisible.value = false;
  } catch (e) {
    alert("保存失败，请重试。");
    console.error(e);
  }
};

const getScoreColor = (score: number) => {
  if (score >= 90) return "text-green-600";
  if (score >= 60) return "text-yellow-600";
  return "text-red-600";
};

// --- Helper Functions ---
const getMaxLlmScore = (reports?: PlagiarismReport[]): number => {
  if (!reports || reports.length === 0) return 0;
  return Math.max(0, ...reports.map((r) => r.llm_analysis?.similarity_score || 0));
};

const hasHighRiskPlagiarism = (reports?: PlagiarismReport[]): boolean => {
  return getMaxLlmScore(reports) >= 90;
};

const formatPlagiarismSummary = (reports?: PlagiarismReport[]): string => {
  const maxScore = getMaxLlmScore(reports);
  if (maxScore >= 90) {
    return `高度疑似 (${maxScore}分)`;
  }
  return "不存在抄袭风险";
};

const getPlagiarismSummaryClass = (reports?: PlagiarismReport[]): string => {
  const maxScore = getMaxLlmScore(reports);
  if (maxScore >= 90) {
    return "text-red-600";
  }
  return "text-green-600";
};

const formatAIGC = (report?: AIGCReport): string => {
  if (!report) return "未检测";
  const probability = (report.ai_probability * 100).toFixed(1);
  const source = report.detection_source ? ` (${report.detection_source})` : "";
  return `${probability}% AI生成${source}`;
};

const getAIGCColor = (report?: AIGCReport): string => {
  if (!report) return "text-gray-400";
  if (report.ai_probability > 0.8) return "text-red-600";
  if (report.ai_probability > 0.5) return "text-yellow-600";
  return "text-green-600";
};

onMounted(fetchAssignmentDetails);
</script>
