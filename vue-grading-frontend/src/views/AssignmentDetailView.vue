<template>
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
        >{{ JSON.stringify(assignment.rubric, null, 2) }}</pre>
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
        >
        <p v-if="fileName" class="mt-2 text-sm text-green-600">已选择文件: {{ fileName }}</p>
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
        :class="submissionError ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'"
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
              <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">学生ID</th>
              <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">分数</th>
              <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">查重报告</th>
              <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">AIGC检测</th>
              <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">评语</th>
              <th class="px-6 py-3 text-xs font-medium text-right text-gray-500 uppercase">操作</th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr v-for="(result, index) in results" :key="result.id">
              <td class="px-6 py-4 text-sm font-medium text-gray-900 whitespace-nowrap">{{ result.student_id }}</td>
              <td
                class="px-6 py-4 text-sm font-bold whitespace-nowrap"
                :class="result.score >= 60 ? 'text-green-600' : 'text-red-600'"
              >
                {{ result.score.toFixed(1) }}
              </td>
              <td class="px-6 py-4 text-sm whitespace-nowrap">
                <button
                  v-if="result.plagiarism_report?.highest_similarity"
                  @click="showReportDetail(result.plagiarism_report)"
                  class="flex items-center font-bold hover:underline"
                  :class="getSimilarityClass(result.plagiarism_report)"
                >
                  <svg v-if="result.plagiarism_report?.llm_analysis?.is_plagiarized" xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 mr-1" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.21 3.03-1.742 3.03H4.42c-1.532 0-2.492-1.696-1.742-3.03l5.58-9.92zM10 13a1 1 0 110-2 1 1 0 010 2zm-1-4a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1z" clip-rule="evenodd" /></svg>
                  {{ (result.plagiarism_report.highest_similarity.score * 100).toFixed(1) }}%
                </button>
                <span v-else class="text-gray-400">-</span>
              </td>
              <!-- 新增AIGC检测结果列 -->
              <td class="px-6 py-4 whitespace-nowrap text-sm font-semibold" :class="getAIGCColor(result.aigc_report)">
                {{ formatAIGC(result.aigc_report) }}
              </td>
              <td class="px-6 py-4 text-sm text-gray-500 whitespace-normal">{{ result.feedback }}</td>
              <td class="px-6 py-4 text-sm font-medium text-right whitespace-nowrap">
                <button @click="deleteSingleResult(result.id, index)" title="删除此条记录" class="p-1 text-gray-400 rounded-full hover:text-red-600">
                  <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z" clip-rule="evenodd" /></svg>
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else class="py-5 text-center text-gray-500">暂无提交结果。</div>
    </div>
  </div>

  <ReportModal v-if="isModalVisible && selectedReport" :report="selectedReport" @close="isModalVisible = false" />
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import gradingApi from '../services/gradingApi';
import Loader from '../components/Loader.vue';
import ReportModal from '../components/ReportModal.vue';

// --- Interfaces ---
interface AIGCReport {
  predicted_label: string;
  confidence: number;
  ai_probability: number;
}

interface SimilarityMatch {
  score: number;
  similar_to: string;
}

interface LLMAnalysis {
  is_plagiarized: boolean;
  reasoning: string;
  suspicious_parts?: string[];
}

interface PlagiarismReport {
  highest_similarity?: SimilarityMatch;
  llm_analysis?: LLMAnalysis;
}

interface SubmissionResult {
  id: number;
  student_id: string;
  score: number;
  feedback: string;
  plagiarism_report?: PlagiarismReport;
  aigc_report?: AIGCReport; // 添加AIGC报告字段
}

interface Assignment {
  task_name: string;
  question: string;
  rubric: object; 
  submissions: SubmissionResult[];
}

const props = defineProps<{
  id: string;
}>();

// --- Component State ---
const assignment = ref<Assignment | null>(null);
const results = ref<SubmissionResult[]>([]);
const selectedReport = ref<PlagiarismReport | null>(null);
const file = ref<File | null>(null);
const fileName = ref('');
const isLoadingAssignment = ref(true);
const isLoadingResults = ref(false);
const isSubmitting = ref(false);
const isModalVisible = ref(false);
const error = ref<string | null>(null);
const submissionMessage = ref('');
const submissionError = ref(false);

// --- API Methods ---
const fetchAssignmentDetails = async () => {
  isLoadingAssignment.value = true;
  try {
    const response = await gradingApi.getAssignment(props.id);
    assignment.value = response.data;
    results.value = response.data.submissions || [];
  } catch (e) {
    error.value = '无法加载作业详情。';
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
    console.error('无法加载结果:', e);
  } finally {
    isLoadingResults.value = false;
  }
};

const submitBatchFile = async () => {
  if (!file.value) return;
  isSubmitting.value = true;
  submissionMessage.value = '';
  submissionError.value = false;

  const formData = new FormData();
  formData.append('batch_file', file.value);

  try {
    const response = await gradingApi.submitBatch(props.id, formData);
    submissionMessage.value = response.data.message;
    fileName.value = '';
    file.value = null;
    (document.getElementById('file-upload') as HTMLInputElement).value = '';
  } catch (e: any) {
    submissionError.value = true;
    submissionMessage.value = e.response?.data?.detail || '提交失败。';
    console.error(e);
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
      alert('删除失败，请重试。');
      console.error(e);
    }
  }
};

const deleteAllResults = async () => {
  if (results.value.length === 0) return;
  if (confirm(`确定要清空此作业下的所有 ${results.value.length} 条评分记录吗？此操作不可撤销。`)) {
    try {
      await gradingApi.deleteAllSubmissions(props.id);
      results.value = [];
    } catch (e) {
      alert('清空失败，请重试。');
      console.error(e);
    }
  }
};

// 
const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement;
  const selectedFile = target.files?.[0];
  if (selectedFile) {
    file.value = selectedFile;
    fileName.value = selectedFile.name;
  }
};

const getSimilarityClass = (report?: PlagiarismReport): string => {
  if (report?.llm_analysis?.is_plagiarized) {
    return 'text-red-600';
  }
  if (report?.highest_similarity && report.highest_similarity.score >= 0.7) {
    return 'text-yellow-600';
  }
  return 'text-gray-500';
};

const showReportDetail = (report: PlagiarismReport) => {
  selectedReport.value = report;
  isModalVisible.value = true;
};

// 新增aigc得分展示模块
const formatAIGC = (report?: AIGCReport): string => {
  if (!report) return '未检测';
  const probability = (report.ai_probability * 100).toFixed(1);
  return `${probability}% AI生成`;
};

const getAIGCColor = (report?: AIGCReport): string => {
  if (!report) return 'text-gray-400';
  if (report.ai_probability > 0.8) return 'text-red-600';
  if (report.ai_probability > 0.5) return 'text-yellow-600';
  return 'text-green-600';
};

onMounted(fetchAssignmentDetails);
</script>
