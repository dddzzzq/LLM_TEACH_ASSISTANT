<template>
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black bg-opacity-60"
    @click.self="$emit('close')"
  >
    <div
      class="flex w-full max-w-2xl max-h-[90vh] flex-col bg-white rounded-lg shadow-2xl"
      @click.stop
    >
      <header class="flex items-center justify-between p-4 border-b">
        <h2 class="text-xl font-bold text-gray-800">详细查重报告</h2>
        <button @click="$emit('close')" class="text-gray-400 hover:text-gray-700">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            class="w-6 h-6"
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
      </header>

      <main class="p-6 space-y-6 overflow-y-auto">
        <div v-if="report.highest_similarity">
          <h3 class="font-semibold text-gray-700">初步检测 (TF-IDF)</h3>
          <p class="mt-1 text-sm text-gray-600">
            最高相似度为
            <strong class="text-lg">{{ (report.highest_similarity.score * 100).toFixed(1) }}%</strong>，
            与学生 <strong class="font-mono">{{ report.highest_similarity.similar_to }}</strong> 的作业最相似。
          </p>
        </div>

        <div v-if="report.llm_analysis">
          <h3 class="font-semibold text-gray-700">大模型深度分析</h3>
          <div class="p-4 mt-2 space-y-3 bg-gray-50 border rounded-lg">
            <div>
              <p class="text-sm font-medium">分析结论:</p>
              <p
                class="text-lg font-bold"
                :class="report.llm_analysis.is_plagiarized ? 'text-red-600' : 'text-green-600'"
              >
                {{ report.llm_analysis.is_plagiarized ? '高度疑似抄袭' : '不像抄袭' }}
              </p>
            </div>
            <div>
              <p class="text-sm font-medium">分析理由:</p>
              <p class="text-sm text-gray-800 whitespace-pre-wrap">{{ report.llm_analysis.reasoning }}</p>
            </div>
            <div v-if="report.llm_analysis.suspicious_parts?.length">
              <p class="text-sm font-medium">可疑片段摘录:</p>
              <ul class="mt-1 space-y-2 list-disc list-inside">
                <li
                  v-for="(part, index) in report.llm_analysis.suspicious_parts"
                  :key="index"
                  class="p-2 text-sm text-gray-600 bg-gray-100 rounded"
                >
                  {{ part }}
                </li>
              </ul>
            </div>
          </div>
        </div>
        <div v-else class="mt-4">
          <h3 class="font-semibold text-gray-700">大模型深度分析</h3>
          <p class="mt-1 text-sm text-gray-500">
            未进行。(原因: 初步检测的相似度低于启动深度分析的阈值)
          </p>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">

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

defineProps<{
  report: PlagiarismReport;
}>();

defineEmits<{
  (e: 'close'): void;
}>();
</script>