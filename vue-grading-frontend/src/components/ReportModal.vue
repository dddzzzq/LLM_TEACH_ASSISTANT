<template>
  <component
    :is="embedded ? 'section' : AppModal"
    :title="reportType === 'feedback' ? '完整评语' : '查重报告'"
    @close="close"
  >
    <div v-if="reportType === 'feedback'">
      <h2 class="text-2xl font-bold text-gray-800 mb-4">评分评语</h2>
      <h3 class="font-semibold text-gray-700">来自: {{ submission.student_id }}</h3>
      <p class="mt-4 p-4 bg-gray-50 rounded-md whitespace-pre-wrap">
        {{
          submission.is_human_reviewed
            ? (submission.human_feedback ?? submission.feedback)
            : submission.feedback
        }}
      </p>
    </div>

    <div v-if="reportType === 'plagiarism'">
      <h2 class="text-2xl font-bold text-gray-800 mb-4">抄袭检测详细报告</h2>
      <div v-if="!submission.plagiarism_reports || submission.plagiarism_reports.length === 0">
        <p class="text-gray-600">未发现与该学生相关的可疑抄袭记录。</p>
      </div>
      <div v-else class="space-y-6">
        <div
          v-for="(report, index) in sortedPlagiarismReports"
          :key="index"
          class="p-4 border rounded-lg"
        >
          <h3 class="text-lg font-bold mb-2" :class="getReportHeaderClass(report)">
            与 {{ report.similar_to }} 的
            {{ report.content_type === 'code' ? '代码' : '文本' }} 相似度分析
          </h3>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div class="p-3 bg-gray-50 rounded">
              <p class="text-sm text-gray-500">初步语义相似度</p>
              <p class="text-xl font-bold"> {{ (report.initial_score * 100).toFixed(1) }}% </p>
            </div>
            <div class="p-3 bg-gray-50 rounded">
              <p class="text-sm text-gray-500">LLM 综合评估分数</p>
              <p class="text-xl font-bold"> {{ report.llm_analysis?.similarity_score }} / 100 </p>
            </div>
            <!-- +++ 新增代码/文档匹配度得分展示 +++ -->
            <div class="p-3 bg-gray-50 rounded" v-if="submission.code_doc_match_report">
              <p class="text-sm text-gray-500">代码/文档匹配度</p>
              <p
                class="text-xl font-bold"
                :class="getCodeDocMatchColor(submission.code_doc_match_report)"
              >
                {{ submission.code_doc_match_report.score }} / 100
              </p>
            </div>
          </div>

          <div class="mt-4">
            <h4 class="font-semibold text-gray-700">AI 分析理由:</h4>
            <p class="mt-1 p-3 bg-blue-50 text-gray-700 rounded-md whitespace-pre-wrap">
              {{ report.llm_analysis?.reasoning }}
            </p>
          </div>
          <!-- +++ 新增代码/文档匹配度理由展示 +++ -->
          <div class="mt-4" v-if="submission.code_doc_match_report">
            <h4 class="font-semibold text-gray-700">代码/文档匹配度分析理由:</h4>
            <p class="mt-1 p-3 bg-yellow-50 text-gray-700 rounded-md whitespace-pre-wrap">
              {{ submission.code_doc_match_report.reasoning }}
            </p>
          </div>
          <div
            class="mt-4"
            v-if="
              report.llm_analysis?.suspicious_parts &&
              report.llm_analysis?.suspicious_parts.length > 0
            "
          >
            <h4 class="font-semibold text-gray-700">可疑片段对比:</h4>
            <div
              v-for="(part, pIndex) in report.llm_analysis?.suspicious_parts"
              :key="pIndex"
              class="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2"
            >
              <div class="p-2 border rounded">
                <p class="text-xs font-mono text-gray-500 mb-1">
                  学生: {{ submission.student_id }}
                </p>
                <pre
                  class="text-xs bg-gray-100 p-2 rounded overflow-x-auto"
                ><code>{{ part.student_A_content }}</code></pre>
              </div>
              <div class="p-2 border rounded">
                <p class="text-xs font-mono text-gray-500 mb-1"> 学生: {{ report.similar_to }} </p>
                <pre
                  class="text-xs bg-gray-100 p-2 rounded overflow-x-auto"
                ><code>{{ part.student_B_content }}</code></pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </component>
</template>

<script setup>
import { computed } from 'vue';
import AppModal from './AppModal.vue';

const props = defineProps({ submission: Object, reportType: String, embedded: Boolean });
const emit = defineEmits(['close']);

const sortedPlagiarismReports = computed(() => {
  if (!props.submission || !props.submission.plagiarism_reports) {
    return [];
  }
  return [...props.submission.plagiarism_reports].sort((a, b) => {
    const scoreA = a.llm_analysis?.similarity_score || 0;
    const scoreB = b.llm_analysis?.similarity_score || 0;
    return scoreB - scoreA;
  });
});

const close = () => emit('close');
const getReportHeaderClass = (report) => {
  if (report.llm_analysis && report.llm_analysis?.similarity_score >= 85) return 'text-red-600';
  if (report.llm_analysis && report.llm_analysis?.similarity_score >= 70) return 'text-yellow-600';
  return 'text-gray-800';
};

// +++ 新增辅助函数 +++
const getCodeDocMatchColor = (report) => {
  if (!report) return 'text-gray-400';
  if (report.score < 50) return 'text-red-600';
  if (report.score < 70) return 'text-yellow-600';
  return 'text-green-600';
};
</script>
