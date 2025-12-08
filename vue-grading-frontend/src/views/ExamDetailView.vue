<template>
  <div class="p-4 md:p-8">
    <div v-if="isLoadingExam" class="py-10 text-center">
      <Loader />
    </div>

    <div v-else-if="error" class="p-4 text-red-500 bg-red-100 rounded-lg">
      {{ error }}
    </div>

    <div v-else-if="exam" class="space-y-8">
      <div class="p-6 bg-white rounded-lg shadow-lg">
        <div class="flex justify-between items-start">
          <div>
            <h1 class="text-3xl font-bold text-gray-800">{{ exam.name }}</h1>
            <p class="mt-2 text-gray-600">
              共 {{ exam.question_count }} 道题目，
              <span class="font-semibold text-indigo-600"
                >满分 {{ exam.total_score }} 分</span
              >
            </p>
          </div>
        </div>

        <details class="mt-4">
          <summary class="cursor-pointer font-semibold text-indigo-600">
            查看题目详情
          </summary>
          <ul class="mt-2 space-y-2 text-sm text-gray-700">
            <li
              v-for="q in exam.questions"
              :key="q.id"
              class="p-2 bg-gray-50 rounded flex justify-between"
            >
              <span
                ><strong class="font-bold">{{ q.question_number }}.</strong>
                {{ q.question_text }}</span
              >
              <span class="font-mono text-gray-500">({{ q.max_score }}分)</span>
            </li>
          </ul>
        </details>
      </div>

      <div class="p-6 bg-white rounded-lg shadow-lg">
        <h2 class="mb-4 text-2xl font-bold text-gray-800">提交学生试卷评分</h2>
        <form @submit.prevent="submitStudentExam">
          <div class="space-y-4">
            <div>
              <label for="student_id" class="block text-sm font-medium text-gray-700">
                学生学号
              </label>
              <input
                id="student_id"
                v-model="studentId"
                type="text"
                class="block w-full sm:w-1/2 px-3 py-2 mt-1 border border-gray-300 rounded-md shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                required
                placeholder="请输入学生学号"
              />
            </div>
            <div>
              <label for="file-upload" class="block text-sm font-medium text-gray-700">
                上传试卷图片 (可多选)
              </label>
              <input
                id="file-upload"
                type="file"
                accept="image/png, image/jpeg"
                class="block w-full mt-2 text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 rounded-full hover:file:bg-indigo-100"
                required
                multiple
                @change="handleFileChange"
              />
              <p v-if="fileNames.length" class="mt-2 text-sm text-green-600">
                已选择 {{ fileNames.length }} 个文件: {{ fileNames.join(", ") }}
              </p>
            </div>
          </div>

          <button
            type="submit"
            :disabled="isSubmitting"
            class="flex justify-center px-6 py-2 mt-6 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md shadow-sm w-full sm:w-auto hover:bg-indigo-700 disabled:bg-indigo-400"
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
          <button
            @click="fetchResults"
            :disabled="isLoadingResults"
            class="text-sm text-indigo-600 hover:text-indigo-800 disabled:text-gray-400"
          >
            刷新结果
          </button>
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
                  得分 / 总分
                </th>
                <th
                  class="px-6 py-3 text-xs font-medium text-right text-gray-500 uppercase"
                >
                  操作
                </th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr v-for="result in results" :key="result.student_exam_id">
                <td class="px-6 py-4 text-sm font-medium text-gray-900 whitespace-nowrap">
                  {{ result.student_id }}
                </td>
                <td
                  class="px-6 py-4 text-sm font-bold whitespace-nowrap"
                  :class="getScoreColor(result.total_score)"
                >
                  {{ result.total_score.toFixed(1) }}
                  <span class="text-gray-400 font-normal mx-1">/</span>
                  <span class="text-gray-500 font-medium">{{
                    exam.total_score.toFixed(1)
                  }}</span>
                </td>
                <td
                  class="px-6 py-4 text-sm font-medium text-right whitespace-nowrap space-x-4"
                >
                  <router-link
                    :to="`/exams/${exam.id}/student/${result.student_exam_id}`"
                    class="text-indigo-600 hover:text-indigo-900"
                  >
                    查看详情
                  </router-link>

                  <button
                    @click="deleteResult(result.student_exam_id)"
                    class="text-red-600 hover:text-red-900 disabled:opacity-50 disabled:cursor-not-allowed"
                    :disabled="isDeleting === result.student_exam_id"
                  >
                    <span v-if="isDeleting === result.student_exam_id">删除中...</span>
                    <span v-else>删除</span>
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="py-5 text-center text-gray-500">暂无评分结果。</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue"; // 修改：移除了 computed
import { useRoute, RouterLink } from "vue-router";
import gradingApi from "../services/gradingApi";
import Loader from "../components/Loader.vue";

const props = defineProps({ id: String });
const route = useRoute();

const exam = ref(null);
const results = ref([]);
const studentId = ref("");
const files = ref([]);
const fileNames = ref([]);

const isLoadingExam = ref(true);
const isLoadingResults = ref(false);
const isSubmitting = ref(false);
const isDeleting = ref(null);
const error = ref(null);
const submissionMessage = ref("");
const submissionError = ref(false);

// 修改：删除了 examTotalScore 的 computed 属性，因为现在后端直接返回 total_score

const fetchExamDetails = async () => {
  isLoadingExam.value = true;
  error.value = null;
  try {
    const response = await gradingApi.getExamDetails(props.id);
    exam.value = response.data;
  } catch (e) {
    error.value = "无法加载试卷详情。";
    console.error(e);
  } finally {
    isLoadingExam.value = false;
  }
};

const fetchResults = async () => {
  isLoadingResults.value = true;
  try {
    const response = await gradingApi.getExamResultsSummary(props.id);
    results.value = response.data;
  } catch (e) {
    console.error("无法加载结果:", e);
    error.value = "无法加载评分结果。";
  } finally {
    isLoadingResults.value = false;
  }
};

const handleFileChange = (event) => {
  files.value = Array.from(event.target.files);
  fileNames.value = files.value.map((f) => f.name);
};

const submitStudentExam = async () => {
  if (!files.value.length || !studentId.value) {
    submissionError.value = true;
    submissionMessage.value = "请填写学生学号并选择图片文件。";
    return;
  }

  isSubmitting.value = true;
  submissionMessage.value = "";
  submissionError.value = false;

  const formData = new FormData();
  formData.append("student_id", studentId.value);
  files.value.forEach((file) => {
    formData.append("images", file);
  });

  try {
    const response = await gradingApi.gradeStudentExam(props.id, formData);
    submissionMessage.value = response.data.message;

    // 重置表单
    studentId.value = "";
    files.value = [];
    fileNames.value = [];
    const fileInput = document.getElementById("file-upload");
    if (fileInput) fileInput.value = "";

    // 提交后立即刷新一次结果
    fetchResults();
  } catch (e) {
    submissionError.value = true;
    submissionMessage.value = e.response?.data?.detail || "提交失败。";
    console.error(e);
  } finally {
    isSubmitting.value = false;
  }
};

const deleteResult = async (studentExamId) => {
  if (!window.confirm("确定要删除该学生的评分结果吗？此操作不可恢复。")) {
    return;
  }

  isDeleting.value = studentExamId;
  try {
    await gradingApi.deleteStudentExamResult(props.id, studentExamId);
    results.value = results.value.filter((r) => r.student_exam_id !== studentExamId);
  } catch (e) {
    console.error(e);
    alert("删除失败，请重试。");
  } finally {
    isDeleting.value = null;
  }
};

const getScoreColor = (score) => {
  // 修改：直接使用 exam.value.total_score
  const max = exam.value ? exam.value.total_score : 0;
  if (!max) return "text-gray-800";

  const ratio = score / max;

  if (ratio >= 0.85) return "text-green-600";
  if (ratio >= 0.6) return "text-yellow-600";
  return "text-red-600";
};

onMounted(async () => {
  await fetchExamDetails();
  await fetchResults();
});
</script>
