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

      <!-- 总结报告 -->
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
              :class="getScoreColor(reportData.report.total_score)"
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

      <!-- 题目详情 -->
      <div class="p-6 bg-white rounded-lg shadow-xl">
        <h2 class="text-2xl font-bold text-gray-800 mb-6">每题详情</h2>
        <div class="space-y-6">
          <div
            v-for="answer in sortedAnswers"
            :key="answer.id"
            class="border-b border-gray-200 pb-6"
          >
            <div class="flex justify-between items-start">
              <h3 class="text-lg font-semibold text-gray-900">
                题号 {{ answer.question.question_number }}
              </h3>
              <span class="text-xl font-bold" :class="getScoreColor(answer.score)">
                {{ answer.score.toFixed(1) }} / {{ answer.question.max_score.toFixed(1) }}
              </span>
            </div>

            <!-- 题目内容 -->
            <details class="mt-2 text-sm">
              <summary class="cursor-pointer text-gray-600">查看题目与标准答案</summary>
              <div class="mt-2 p-3 bg-gray-50 rounded-md space-y-2">
                <p><strong>题目:</strong> {{ answer.question.question_text }}</p>
                <p><strong>标准答案:</strong> {{ answer.question.standard_answer }}</p>
              </div>
            </details>

            <!-- 学生答案 -->
            <div class="mt-4">
              <h4 class="font-medium text-gray-700">学生答案 (OCR提取):</h4>
              <p
                class="mt-1 p-3 bg-gray-100 text-gray-600 rounded-md whitespace-pre-wrap text-sm"
              >
                {{ answer.student_answer_text || "未提取到答案" }}
              </p>
            </div>

            <!-- AI评语 -->
            <div class="mt-4">
              <h4 class="font-medium text-gray-700">AI 评判依据:</h4>
              <p
                class="mt-1 p-3 bg-yellow-50 text-yellow-800 rounded-md whitespace-pre-wrap text-sm"
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
import { useRoute, RouterLink } from "vue-router";
import gradingApi from "../services/gradingApi";
import Loader from "../components/Loader.vue";

const props = defineProps({
  id: String, // exam_id
  studentExamId: String,
});

const route = useRoute();
const reportData = ref(null);
const examName = ref("");
const isLoading = ref(true);
const error = ref(null);

const sortedAnswers = computed(() => {
  if (!reportData.value || !reportData.value.answers) return [];
  return [...reportData.value.answers].sort(
    (a, b) => a.question.question_number - b.question.question_number
  );
});

const fetchReport = async () => {
  isLoading.value = true;
  error.value = null;
  try {
    // 1. 获取学生详细报告
    const reportResponse = await gradingApi.getStudentDetailedReport(
      props.id,
      props.studentExamId
    );
    reportData.value = reportResponse.data;

    // 2. (可选) 获取试卷名称
    const examResponse = await gradingApi.getExamDetails(props.id);
    examName.value = examResponse.data.name;
  } catch (e) {
    console.error("无法加载报告:", e);
    error.value = "无法加载学生报告详情。";
  } finally {
    isLoading.value = false;
  }
};

const getScoreColor = (score) => {
  // 假设总分不为0，计算得分率
  // 注意：我们在这里没有总分，暂时只根据分数判断
  if (score > 8) return "text-green-600"; // 假设10分制
  if (score > 5) return "text-yellow-600";
  return "text-red-600";
};

onMounted(fetchReport);
</script>
