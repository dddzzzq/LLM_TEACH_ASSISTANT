<template>
  <div class="detail-page">
    <div v-if="isLoadingAssignment" class="py-10 text-center">
      <Loader />
    </div>

    <div v-else-if="error" class="p-4 text-red-500 bg-red-100 rounded-lg">
      {{ error }}
    </div>

    <div v-else-if="assignment" class="assignment-detail-workspace">
      <router-link to="/assignments" class="back-link">← 返回作业批改</router-link>
      <header class="detail-heading">
        <div
          ><p class="detail-course"
            >{{ assignment.course_name }} · {{ assignment.class_name || '全部班级' }}</p
          ><h1 class="page-title">{{ assignment.task_name }}</h1></div
        >
        <button
          v-if="activeTab === 'content'"
          class="workspace-button primary"
          @click="selectTab('materials')"
          >{{ assignment.portal_homework_id ? '按班级批改 →' : '导入学生作业 →' }}</button
        >
      </header>
      <nav class="detail-tabs" aria-label="作业详情导航">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          :class="{ active: activeTab === tab.key }"
          :aria-current="activeTab === tab.key ? 'page' : undefined"
          @click="selectTab(tab.key)"
          >{{ tab.label }}<span v-if="tab.key === 'results'">{{ results.length }}</span></button
        >
      </nav>
      <section v-if="activeTab === 'content'" class="assignment-content-grid">
        <div class="surface content-section"
          ><div class="content-section-heading"><h2>题目要求</h2><span>作业内容</span></div
          ><p
            v-if="assignment.portal_homework_id && assignment.requirements_status !== 'READY'"
            role="status"
            >官网要求正在同步或尚未完整读取，完成后才能按班级批改。</p
          ><RequirementContent
            v-if="assignment.portal_homework_id && assignment.question"
            class="full-question"
            :text="assignment.question"
          /><p v-else class="full-question">{{
            assignment.question ||
            (assignment.portal_homework_id ? '等待自动读取官网要求…' : '尚未填写题目要求。')
          }}</p></div
        >
        <aside class="surface content-section"
          ><div class="content-section-heading"><h2>评分标准</h2></div
          ><RubricSummary :rubric="assignment.rubric" /><p class="grading-scope"
            >批改范围：{{ assignment.class_name || '全部班级' }}</p
          ><button
            v-if="assignment.portal_homework_id"
            class="workspace-button"
            @click="selectTab('materials')"
            >编辑评分标准 / 选择班级</button
          ></aside
        >
      </section>
      <section v-show="activeTab === 'materials'" class="p-5 md:p-7 surface">
        <AssignmentPortalPanel
          v-if="assignment.portal_homework_id"
          :key="props.id"
          :assignment-id="props.id"
          :rubric="assignment.rubric"
          @context="updatePortalContext"
          @rubric="assignment.rubric = $event"
          @results="showBatchResults('')"
        />
        <h2 class="mb-4 text-2xl font-bold text-gray-800">导入学生作业</h2>
        <label v-if="portalClasses.length" class="result-class-filter"
          >本地材料所属班级 / 筛选材料
          <select v-model="resultsClass" aria-label="材料班级"
            ><option value="">全部班级</option
            ><option v-for="item in portalClasses" :key="item.id" :value="item.id">{{
              item.name
            }}</option></select
          >
        </label>
        <form @submit.prevent="submitBatchFile">
          <label for="file-upload" class="block text-sm font-medium text-gray-700">
            上传学生作业 ZIP / RAR / 7z 压缩包
          </label>
          <input
            id="file-upload"
            type="file"
            accept=".zip,.rar,.7z"
            class="block w-full mt-2 text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 rounded-full hover:file:bg-indigo-100"
            required
            @change="handleFileChange"
          />
          <p v-if="fileName" class="mt-2 text-sm text-green-600"> 已选择文件: {{ fileName }} </p>
          <label class="upload-autograde"
            ><input v-model="autoGrade" type="checkbox" />材料整理完成后自动批改</label
          >
          <button
            type="submit"
            :disabled="isSubmitting"
            class="flex justify-center px-6 py-2 mt-4 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md shadow-sm w-full sm:w-auto hover:bg-indigo-700 disabled:bg-indigo-400"
          >
            <span v-if="!isSubmitting">{{ autoGrade ? '上传并自动批改' : '上传并整理材料' }}</span>
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
        <MaterialBatchPanel
          v-if="activeTab === 'materials'"
          :assignment-id="props.id"
          :class-id="resultsClass"
          :refresh-key="materialsVersion"
          @results="showBatchResults"
        />
      </section>

      <section v-show="activeTab === 'results'" class="p-5 md:p-7 surface results-section">
        <div class="flex items-center justify-between mb-4">
          <div
            ><h2 class="text-2xl font-bold text-gray-800">评分结果</h2
            ><p v-if="resultsBatchId" class="text-xs text-slate-500 mt-2"
              >仅显示批次 {{ resultsBatchId }}
              <button class="text-blue-700 ml-2" @click="showBatchResults('')"
                >查看全部批次</button
              ></p
            ></div
          >
          <div class="flex items-center gap-4">
            <button
              @click="exportResults"
              :disabled="isExporting || results.length === 0"
              class="text-sm font-semibold text-green-600 hover:text-green-800 disabled:text-gray-400 disabled:cursor-not-allowed"
            >
              <span v-if="!isExporting">导出为Excel</span>
              <span v-else>正在导出...</span>
            </button>
            <button
              @click="deleteAllResults"
              v-if="!resultsBatchId && !resultsClass"
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

        <p v-if="resultsError" class="workspace-error" role="alert">{{ resultsError }}</p>
        <label v-if="portalClasses.length" class="result-class-filter"
          >筛选班级
          <select v-model="resultsClass" aria-label="结果班级" @change="fetchResults">
            <option value="">全部班级</option
            ><option v-for="item in portalClasses" :key="item.id" :value="item.id">{{
              item.name
            }}</option>
          </select>
        </label>
        <div v-if="isLoadingResults" class="py-5 text-center">
          <Loader />
        </div>
        <div v-else-if="results.length > 0" class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">
                  学号 & 姓名
                </th>
                <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">
                  分数
                </th>
                <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">
                  查重报告
                </th>
                <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">
                  AIGC检测
                </th>
                <!-- +++ 新增列 +++ -->
                <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">
                  代码/文档匹配度
                </th>
                <th class="px-6 py-3 text-xs font-medium text-left text-gray-500 uppercase">
                  评语
                </th>
                <th class="px-6 py-3 text-xs font-medium text-right text-gray-500 uppercase">
                  操作
                </th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr v-for="result in sortedResults" :key="result.id">
                <td class="px-6 py-4 text-sm font-medium text-gray-900 whitespace-nowrap">
                  <div class="flex items-center">
                    <span
                      >{{ result.student_id
                      }}{{ result.student_name ? ' - ' + result.student_name : '' }}</span
                    >
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
                  :class="
                    getScoreColor(
                      result.is_human_reviewed && typeof result.human_score === 'number'
                        ? result.human_score
                        : result.score
                    )
                  "
                >
                  {{
                    (result.is_human_reviewed && typeof result.human_score === 'number'
                      ? result.human_score
                      : result.score
                    ).toFixed(1)
                  }}
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
                <!-- +++ 新增列的渲染逻辑 +++ -->
                <td
                  class="px-6 py-4 whitespace-nowrap text-sm font-semibold"
                  :class="getCodeDocMatchColor(result.code_doc_match_report)"
                  :title="result.code_doc_match_report?.reasoning"
                >
                  {{ formatCodeDocMatch(result.code_doc_match_report) }}
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
                    @click="deleteSingleResult(result.id)"
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
        <div v-else class="py-5 text-center text-gray-500"
          >暂无评分结果。请在“材料与批改”中导入学生作业并开始批改。</div
        >
      </section>
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
import { ref, onMounted, computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import gradingApi from '../services/gradingApi';
import Loader from '../components/Loader.vue';
import ReportModal from '../components/ReportModal.vue';
import ReviewModal from '../components/ReviewModal.vue';
import MaterialBatchPanel from '../components/MaterialBatchPanel.vue';
import RubricSummary from '../components/RubricSummary.vue';
import AssignmentPortalPanel from '../components/AssignmentPortalPanel.vue';
import RequirementContent from '../components/RequirementContent.vue';
const route = useRoute(),
  router = useRouter();
const tabs = [
  { key: 'content', label: '作业内容' },
  { key: 'materials', label: '材料与批改' },
  { key: 'results', label: '评分结果' },
];
const activeTab = computed(() =>
  tabs.some((tab) => tab.key === route.query.tab) ? route.query.tab : 'content'
);
const selectTab = (tab: string) => router.replace({ query: { ...route.query, tab } });
const autoGrade = ref(false);
const materialsVersion = ref(0);
const resultsBatchId = ref('');
const resultsClass = ref('');
const portalClasses = ref<Array<{ id: string; name: string }>>([]);
function updatePortalContext(data: any) {
  portalClasses.value = data.classes || [];
  if (assignment.value && data.available) {
    assignment.value.question = data.question || '';
    assignment.value.requirements_status = data.requirements_status;
  }
}

// --- 接口 ---
interface CodeDocMatchReport {
  score: number;
  reasoning: string;
}
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
  student_name?: string;
  score: number;
  feedback: string;
  plagiarism_reports: PlagiarismReport[];
  aigc_report?: AIGCReport;
  code_doc_match_report?: CodeDocMatchReport;
  is_human_reviewed: boolean;
  human_feedback?: string;
  human_score?: number;
}
interface Assignment {
  portal_homework_id?: string;
  requirements_status?: string;
  course_name: string;
  class_name: string;
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
const fileName = ref('');
const isLoadingAssignment = ref(true);
const isLoadingResults = ref(false);
const resultsError = ref('');
const isSubmitting = ref(false);
const isModalVisible = ref(false);
const error = ref<string | null>(null);
const submissionMessage = ref('');
const submissionError = ref(false);
const reportType = ref('feedback');
const isReviewModalVisible = ref(false);
const submissionToReview = ref<SubmissionResult | null>(null);
const isExporting = ref(false);

const exportResults = async () => {
  if (results.value.length === 0) {
    alert('没有可导出的评分结果');
    return;
  }
  isExporting.value = true;
  try {
    const response = await gradingApi.exportAssignment(
      props.id,
      resultsBatchId.value,
      resultsClass.value
    );
    const blob = new Blob([response.data], { type: response.headers['content-type'] });

    const contentDisposition = response.headers['content-disposition'];
    let filename = `${assignment.value?.task_name || 'assignment'}_results.xlsx`;
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
      if (filenameMatch && filenameMatch.length > 1) {
        filename = decodeURIComponent(filenameMatch[1].replace(/['"]/g, ''));
      }
    }

    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(link.href);
  } catch (e: any) {
    if (e.response && e.response.status === 404) {
      alert('导出失败: 该作业没有任何评分结果。');
    } else {
      alert('导出失败，请检查网络或联系管理员。');
    }
    console.error('导出时发生错误:', e);
  } finally {
    isExporting.value = false;
  }
};

const sortedResults = computed(() => {
  return [...results.value].sort((a, b) => {
    const scoreA = typeof a.human_score === 'number' ? a.human_score : a.score;
    const scoreB = typeof b.human_score === 'number' ? b.human_score : b.score;
    return scoreB - scoreA;
  });
});

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
  resultsError.value = '';
  try {
    const response = await gradingApi.getResultsForAssignment(
      props.id,
      resultsBatchId.value,
      resultsClass.value
    );
    // 添加 || []，防止 Go 后端返回 null 导致前端崩溃
    results.value = response.data || [];
  } catch (e) {
    resultsError.value = '评分结果暂时无法读取，请点击刷新结果重试。';
  } finally {
    isLoadingResults.value = false;
  }
};

const showBatchResults = async (batchId: string) => {
  resultsBatchId.value = batchId;
  selectTab('results');
  await fetchResults();
};

const submitBatchFile = async () => {
  if (!file.value) return;
  isSubmitting.value = true;
  submissionMessage.value = '';
  submissionError.value = false;
  const formData = new FormData();
  formData.append('batch_file', file.value);
  formData.append('auto_grade', String(autoGrade.value));
  if (resultsClass.value) formData.append('class_id', resultsClass.value);
  try {
    const response = await gradingApi.submitBatch(props.id, formData);
    submissionMessage.value = response.data.message;
    materialsVersion.value++;
    fileName.value = '';
    file.value = null;
    (document.getElementById('file-upload') as HTMLInputElement).value = '';
  } catch (e: any) {
    submissionError.value = true;
    submissionMessage.value = e.response?.data?.detail || '提交失败。';
  } finally {
    isSubmitting.value = false;
  }
};

const deleteSingleResult = async (submissionId: number) => {
  const resultToDelete = results.value.find((r) => r.id === submissionId);
  if (!resultToDelete) return;
  if (confirm(`确定要删除学生 ${resultToDelete.student_id} 的评分记录吗？`)) {
    try {
      await gradingApi.deleteSubmission(submissionId);
      results.value = results.value.filter((r) => r.id !== submissionId);
    } catch (e) {
      alert('删除失败，请重试。');
      console.error(e);
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
      alert('清空失败，请重试。');
    }
  }
};

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement;
  const selectedFile = target.files?.[0];
  if (selectedFile) {
    file.value = selectedFile;
    fileName.value = selectedFile.name;
  }
};

const showReport = (submission: SubmissionResult, type: string) => {
  selectedSubmission.value = submission;
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
      human_score: updatedData.score,
      human_feedback: updatedData.human_feedback,
    });
    const index = results.value.findIndex((r) => r.id === updatedData.id);
    if (index !== -1) {
      results.value[index] = response.data;
    }
    isReviewModalVisible.value = false;
  } catch (e) {
    alert('保存失败，请重试。');
    console.error(e);
  }
};

const getScoreColor = (score: number) => {
  if (score >= 85) return 'text-green-600';
  if (score >= 60) return 'text-yellow-600';
  return 'text-red-600';
};

const getMaxLlmScore = (reports?: PlagiarismReport[]): number => {
  if (!reports || reports.length === 0) return 0;
  return Math.max(0, ...reports.map((r) => r.llm_analysis?.similarity_score || 0));
};

const hasHighRiskPlagiarism = (reports?: PlagiarismReport[]): boolean => {
  if (!reports) return false;
  return reports.some((r) => (r.llm_analysis?.similarity_score || 0) >= 85);
};

const formatPlagiarismSummary = (reports?: PlagiarismReport[]): string => {
  const maxScore = getMaxLlmScore(reports);
  if (maxScore > 85) return `高度疑似 (${maxScore}分)`;
  if (maxScore > 70) return `中度疑似 (${maxScore}分)`;
  return '无风险';
};

const getPlagiarismSummaryClass = (reports?: PlagiarismReport[]): string => {
  const maxScore = getMaxLlmScore(reports);
  if (maxScore >= 85) return 'text-red-600';
  if (maxScore >= 70) return 'text-yellow-600';
  return 'text-green-600';
};

const formatAIGC = (report?: AIGCReport): string => {
  if (!report) return '未检测';
  const probability = (report.ai_probability * 100).toFixed(1);
  const source = report.detection_source ? ` (${report.detection_source})` : '';
  return `${probability}% AI生成${source}`;
};

const getAIGCColor = (report?: AIGCReport): string => {
  if (!report) return 'text-gray-400';
  const probPercent = report.ai_probability * 100;
  if (probPercent > 80) return 'text-red-600';
  if (probPercent > 50) return 'text-yellow-600';
  return 'text-green-600';
};

const formatCodeDocMatch = (report?: CodeDocMatchReport): string => {
  if (!report) return '未检测';
  return `${report.score}分`;
};

const getCodeDocMatchColor = (report?: CodeDocMatchReport): string => {
  if (!report) return 'text-gray-400';
  if (report.score < 50) return 'text-red-600';
  if (report.score < 70) return 'text-yellow-600';
  return 'text-green-600';
};

onMounted(fetchAssignmentDetails);
</script>

<style scoped>
.detail-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 20px;
  margin-bottom: 28px;
}
.detail-course {
  font-size: 12px;
  color: var(--color-muted);
  margin-bottom: 10px;
}
.detail-tabs {
  display: flex;
  gap: 28px;
  border-bottom: 1px solid var(--color-line);
  margin-bottom: 26px;
}
.detail-tabs button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 0 14px;
  color: var(--color-muted);
  font-size: 13px;
  border-bottom: 2px solid transparent;
}
.detail-tabs button.active {
  color: var(--color-accent);
  border-bottom-color: var(--color-accent);
  font-weight: 600;
}
.detail-tabs span {
  font-size: 10px;
  border-radius: 5px;
  background: #edf0f6;
  padding: 1px 6px;
}
.assignment-content-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 24px;
  align-items: start;
}
.content-section {
  padding: 26px 28px;
}
.content-section-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--color-line);
  margin-bottom: 20px;
}
.content-section h2 {
  font-size: 14px;
  font-weight: 600;
}
.content-section-heading > span {
  font-size: 10px;
  color: var(--color-muted);
}
.full-question {
  font-size: 14px;
  line-height: 2;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  min-height: 220px;
  color: #4b5670;
}
.full-question.requirement-content {
  white-space: normal;
}
.grading-scope {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid var(--color-line);
  font-size: 11px;
  color: var(--color-muted);
}
.upload-autograde {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #667086;
  margin-top: 20px;
}
.results-section {
  min-width: 0;
}
.results-section > div:first-child {
  gap: 16px;
  flex-wrap: wrap;
}
.results-section > div:first-child > div:last-child {
  flex-wrap: wrap;
}
.detail-page h2.text-2xl {
  font-size: 16px;
}
@media (max-width: 1100px) {
  .assignment-content-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 760px) {
  .detail-heading {
    align-items: flex-start;
    flex-direction: column;
  }
  .detail-tabs {
    gap: 22px;
  }
  .detail-tabs button {
    font-size: 12px;
  }
  .content-section {
    padding: 20px;
  }
  .assignment-content-grid {
    gap: 16px;
  }
  .full-question {
    min-height: 150px;
    font-size: 13px;
  }
}
</style>
