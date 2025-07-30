<template>
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60"
    @click.self="close"
  >
    <div
      class="relative w-full max-w-4xl max-h-[90vh] p-6 mx-4 bg-white rounded-lg shadow-xl overflow-y-auto"
    >
      <button
        @click="close"
        class="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          class="h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </button>

      <div v-if="reportType === 'feedback'">
        <h2 class="text-2xl font-bold text-gray-800 mb-4">AI 综合评语</h2>
        <h3 class="font-semibold text-gray-700">来自: {{ submission.student_id }}</h3>
        <p class="mt-4 p-4 bg-gray-50 rounded-md whitespace-pre-wrap">
          {{ submission.feedback }}
        </p>
      </div>

      <!-- 修改抄袭检测报告的形式  -->
      <div v-if="reportType === 'plagiarism'">
        <h2 class="text-2xl font-bold text-gray-800 mb-4">抄袭检测详细报告</h2>
        <div
          v-if="
            !submission.plagiarism_reports || submission.plagiarism_reports.length === 0
          "
        >
          <p class="text-gray-600">未发现与该学生相关的可疑抄袭记录。</p>
        </div>
        <div v-else class="space-y-6">
          <div
            v-for="(report, index) in submission.plagiarism_reports"
            :key="index"
            class="p-4 border rounded-lg"
          >
            <h3 class="text-lg font-bold mb-2" :class="getReportHeaderClass(report)">
              与 {{ report.similar_to }} 的
              {{ report.content_type === "code" ? "代码" : "文本" }} 相似度分析
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div class="p-3 bg-gray-50 rounded">
                <p class="text-sm text-gray-500">初步语义相似度</p>
                <p class="text-xl font-bold">
                  {{ (report.initial_score * 100).toFixed(1) }}%
                </p>
              </div>
              <div class="p-3 bg-gray-50 rounded">
                <p class="text-sm text-gray-500">LLM 综合评估分数</p>
                <p class="text-xl font-bold">
                  {{ report.llm_analysis.similarity_score }} / 100
                </p>
              </div>
            </div>
            <div class="mt-4">
              <h4 class="font-semibold text-gray-700">AI 分析理由:</h4>
              <p class="mt-1 p-3 bg-blue-50 text-gray-700 rounded-md whitespace-pre-wrap">
                {{ report.llm_analysis.reasoning }}
              </p>
            </div>
            <div
              class="mt-4"
              v-if="
                report.llm_analysis.suspicious_parts &&
                report.llm_analysis.suspicious_parts.length > 0
              "
            >
              <h4 class="font-semibold text-gray-700">可疑片段对比:</h4>
              <div
                v-for="(part, pIndex) in report.llm_analysis.suspicious_parts"
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
                  <p class="text-xs font-mono text-gray-500 mb-1">
                    学生: {{ report.similar_to }}
                  </p>
                  <pre
                    class="text-xs bg-gray-100 p-2 rounded overflow-x-auto"
                  ><code>{{ part.student_B_content }}</code></pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({ submission: Object, reportType: String });
const emit = defineEmits(["close"]);
const close = () => emit("close");
const getReportHeaderClass = (report) => {
  if (report.llm_analysis && report.llm_analysis.similarity_score > 85)
    return "text-red-600";
  if (report.llm_analysis && report.llm_analysis.similarity_score > 70)
    return "text-yellow-600";
  return "text-gray-800";
};
</script>
