<template>
  <div class="p-4 md:p-8 max-w-5xl mx-auto">
    <div v-if="isLoading" class="py-10 text-center">
      <Loader />
    </div>

    <div v-else-if="error" class="p-4 text-red-500 bg-red-100 rounded-lg">
      {{ error }}
    </div>

    <div v-else-if="reportData" class="space-y-8">
      <!-- 返回按钮 -->
      <router-link
        :to="`/exams/${reportData.exam_id}`"
        class="inline-flex items-center text-sm font-medium text-indigo-600 hover:text-indigo-800"
      >
        <svg
          class="h-5 w-5 mr-1"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fill-rule="evenodd"
            d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z"
            clip-rule="evenodd"
          />
        </svg>
        返回试卷详情
      </router-link>

      <!-- 总结报告 (移到顶部) -->
      <div class="p-6 bg-white rounded-lg shadow-xl">
        <h1 class="text-3xl font-bold text-gray-800">
          学生 {{ reportData.student_id }} 的试卷报告
        </h1>
        <div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="p-4 bg-gray-50 rounded-lg">
            <h2 class="text-sm font-medium text-gray-500 uppercase">试卷名称</h2>
            <p class="text-lg font-semibold text-gray-800">{{ examName }}</p>
          </div>
          <div class="p-4 bg-gray-50 rounded-lg">
            <h2 class="text-sm font-medium text-gray-500 uppercase">总得分</h2>
            <p
              class="text-lg font-bold"
              :class="
                getScoreColor(reportData.report.total_score.toFixed(1), examTotalScore)
              "
            >
              {{ reportData.report.total_score.toFixed(1) }}
            </p>
          </div>
        </div>
        <div class="mt-6">
          <h2 class="text-xl font-semibold text-gray-800 mb-2">AI 总结与建议</h2>
          <p
            class="text-gray-700 whitespace-pre-wrap leading-relaxed p-4 bg-blue-50 rounded-md border border-blue-100"
          >
            {{ reportData.report.summary_report }}
          </p>
        </div>
      </div>

      <!-- 未归类/公共图片区域 (如果有) -->
      <div
        v-if="unassignedImages.length > 0"
        class="p-6 bg-white rounded-lg shadow-xl border-l-4 border-gray-300"
      >
        <h2 class="text-lg font-bold text-gray-600 mb-4">未归类/其他图片</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div
            v-for="(img, idx) in unassignedImages"
            :key="img.id"
            class="relative group"
          >
            <a :href="getImageUrl(img.image_path)" target="_blank">
              <img
                :src="getImageUrl(img.image_path)"
                class="w-full h-32 object-cover rounded border hover:opacity-90 transition cursor-zoom-in"
              />
            </a>
          </div>
        </div>
      </div>

      <!-- 题目详情 (每道题下列出关联的图片) -->
      <div class="p-6 bg-white rounded-lg shadow-xl">
        <h2 class="text-2xl font-bold text-gray-800 mb-6">每题详情</h2>
        <div class="space-y-8">
          <div
            v-for="answer in sortedAnswers"
            :key="answer.id"
            class="border-b border-gray-200 pb-8 last:border-0"
          >
            <div class="flex justify-between items-start">
              <h3 class="text-lg font-semibold text-gray-900">
                题号 {{ answer.question.question_number }}
              </h3>
              <span
                class="text-xl font-bold"
                :class="getScoreColor(answer.score, answer.question.max_score)"
              >
                {{ answer.score.toFixed(1) }} /
                {{
                  answer.question.max_score !== null
                    ? answer.question.max_score.toFixed(1)
                    : "?"
                }}
              </span>
            </div>

            <!-- 题目内容 -->
            <details class="mt-2 text-sm">
              <summary class="cursor-pointer text-gray-600">
                查看题目、答案和评分标准
              </summary>
              <div class="mt-2 p-3 bg-gray-50 rounded-md space-y-2">
                <p><strong>题目:</strong> {{ answer.question.question_text }}</p>
                <p><strong>标准答案:</strong> {{ answer.question.standard_answer }}</p>
                <p class="whitespace-pre-wrap">
                  <strong>评分标准:</strong> {{ answer.question.rubric }}
                </p>
              </div>
            </details>

            <!-- 核心区域：图片与OCR左右对照布局 -->
            <div class="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
              <!-- 左侧：答题图片区域 -->
              <div class="flex flex-col">
                <h4 class="font-semibold text-gray-700 mb-3 flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    class="h-5 w-5 mr-2 text-indigo-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  答题图片
                </h4>

                <div
                  class="bg-gray-50 rounded-lg border border-gray-200 p-4 flex-grow flex flex-col justify-center"
                >
                  <div
                    v-if="getImagesForQuestion(answer.question.id).length > 0"
                    class="grid gap-3"
                    :class="
                      getImagesForQuestion(answer.question.id).length === 1
                        ? 'grid-cols-1'
                        : 'grid-cols-2'
                    "
                  >
                    <div
                      v-for="img in getImagesForQuestion(answer.question.id)"
                      :key="img.id"
                      class="relative group w-full"
                    >
                      <a
                        :href="getImageUrl(img.image_path)"
                        target="_blank"
                        class="block border rounded-md overflow-hidden shadow-sm hover:shadow-md transition bg-gray-200 flex items-center justify-center"
                        :class="
                          getImagesForQuestion(answer.question.id).length === 1
                            ? ''
                            : 'h-48'
                        "
                      >
                        <img
                          :src="getImageUrl(img.image_path)"
                          class="w-full object-contain transition-transform duration-300 group-hover:scale-105"
                          :class="
                            getImagesForQuestion(answer.question.id).length === 1
                              ? 'max-h-[600px]'
                              : 'h-full'
                          "
                          alt="题目图片"
                        />
                        <div
                          class="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition flex items-center justify-center pointer-events-none"
                        >
                          <span
                            class="text-white font-bold opacity-0 group-hover:opacity-100 transition bg-black bg-opacity-50 px-3 py-1 rounded-full text-sm"
                            >点击预览</span
                          >
                        </div>
                      </a>
                    </div>
                  </div>
                  <div
                    v-else
                    class="h-full flex flex-col items-center justify-center text-gray-400 text-sm min-h-[120px]"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      class="h-8 w-8 mb-2 opacity-50"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                      />
                    </svg>
                    (此题未检测到关联图片)
                  </div>
                </div>
              </div>

              <!-- 右侧：OCR 文本区域 -->
              <div class="flex flex-col">
                <h4 class="font-semibold text-gray-700 mb-3 flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    class="h-5 w-5 mr-2 text-indigo-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  学生答案 (OCR提取)
                </h4>
                <div
                  class="bg-white rounded-lg border border-gray-200 p-4 shadow-sm flex-grow"
                >
                  <p
                    class="text-gray-700 whitespace-pre-wrap text-sm leading-relaxed font-mono"
                  >
                    {{ answer.student_answer_text || "未提取到文本内容..." }}
                  </p>
                </div>
              </div>
            </div>

            <!-- AI评语 (保持在下方，作为对整体的评价) -->
            <div class="mt-6">
              <h4 class="font-semibold text-gray-700 mb-2 flex items-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  class="h-5 w-5 mr-2 text-yellow-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
                AI 评分依据
              </h4>
              <p
                class="p-4 bg-yellow-50 text-gray-800 rounded-lg border border-yellow-100 text-sm leading-relaxed"
              >
                {{ answer.feedback }}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, onMounted, computed } from "vue";
import gradingApi from "../services/gradingApi";
import Loader from "../components/Loader.vue";

const props = defineProps({
  id: String, // exam_id
  studentExamId: String,
});

const reportData = ref(null);
const examName = ref("");
const examTotalScore = ref(0);
const isLoading = ref(true);
const error = ref(null);
const API_BASE_URL = "https://53ee3fcb.r25.cpolar.top";  // 8000端口网址

const sortedAnswers = computed(() => {
  if (!reportData.value || !reportData.value.answers) return [];
  return [...reportData.value.answers].sort(
    (a, b) => a.question.question_number - b.question.question_number
  );
});

// 计算属性：获取未分配给任何题目的图片
const unassignedImages = computed(() => {
  if (!reportData.value || !reportData.value.images) return [];
  return reportData.value.images.filter((img) => !img.exam_question_id);
});

// 方法：获取特定题目的图片
const getImagesForQuestion = (questionId) => {
  if (!reportData.value || !reportData.value.images) return [];
  return reportData.value.images.filter((img) => img.exam_question_id === questionId);
};

const getImageUrl = (relativePath) => {
  if (!relativePath) return "";
  return `${API_BASE_URL}${relativePath}`;
};

const fetchReport = async () => {
  isLoading.value = true;
  error.value = null;
  try {
    const reportResponse = await gradingApi.getStudentDetailedReport(
      props.id,
      props.studentExamId
    );
    reportData.value = reportResponse.data;
    const examResponse = await gradingApi.getExamDetails(props.id);
    examName.value = examResponse.data.name;
    examTotalScore.value = examResponse.data.total_score || 100;
  } catch (e) {
    console.error("无法加载报告:", e);
    error.value = "无法加载学生报告详情。";
  } finally {
    isLoading.value = false;
  }
};

const getScoreColor = (score, max_score) => {
  if (max_score === null || max_score === undefined || max_score === 0)
    return "text-gray-600";
  const ratio = score / max_score;
  if (ratio >= 0.85) return "text-green-600";
  if (ratio >= 0.6) return "text-yellow-600";
  return "text-red-600";
};

onMounted(fetchReport);
</script>
