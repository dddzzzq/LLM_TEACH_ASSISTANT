<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-3xl font-bold text-gray-800">试卷列表</h1>
      <router-link
        to="/exams/new"
        class="px-4 py-2 font-semibold text-white bg-indigo-600 rounded-lg shadow-md transition-colors hover:bg-indigo-700"
      >
        + 新建试卷
      </router-link>
    </div>

    <div v-if="isLoading" class="py-10 text-center">
      <Loader />
    </div>

    <div v-else-if="error" class="p-4 text-red-500 bg-red-100 rounded-lg">
      {{ error }}
    </div>

    <div v-else-if="exams.length === 0" class="py-10 text-center text-gray-500">
      暂无试卷，请点击右上角“新建试卷”开始。
    </div>

    <div v-else class="overflow-hidden bg-white rounded-lg shadow-xl">
      <ul class="divide-y divide-gray-200">
        <li
          v-for="(exam, index) in exams"
          :key="exam.id"
          class="p-4 transition-colors hover:bg-gray-50 flex items-center justify-between"
        >
          <router-link
            :to="`/exams/${exam.id}`"
            class="flex-grow flex items-center justify-between"
          >
            <div>
              <p class="text-lg font-semibold text-indigo-700">{{ exam.name }}</p>
              <p class="mt-1 text-sm text-gray-600">
                题目数量: {{ exam.question_count }}
                <span class="mx-2 text-gray-300">|</span>
                总分: {{ exam.total_score }}
              </p>
            </div>
            <svg
              class="h-5 w-5 text-gray-400"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fill-rule="evenodd"
                d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                clip-rule="evenodd"
              />
            </svg>
          </router-link>

          <button
            @click.prevent="deleteExam(exam.id, index)"
            :disabled="isDeleting[exam.id]"
            title="删除试卷"
            class="ml-4 p-2 text-gray-400 rounded-full hover:bg-red-100 hover:text-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg
              v-if="!isDeleting[exam.id]"
              xmlns="http://www.w3.org/2000/svg"
              class="h-5 w-5"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fill-rule="evenodd"
                d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z"
                clip-rule="evenodd"
              />
            </svg>
            <svg
              v-else
              class="animate-spin h-5 w-5 text-red-600"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                class="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                stroke-width="4"
              ></circle>
              <path
                class="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { RouterLink } from "vue-router";
import gradingApi from "../services/gradingApi";
import Loader from "../components/Loader.vue";

const exams = ref([]);
const isLoading = ref(true);
const error = ref(null);
const isDeleting = ref({});

const fetchExams = async () => {
  isLoading.value = true;
  error.value = null;
  try {
    const response = await gradingApi.getExams();
    exams.value = response.data;
  } catch (e) {
    console.error(e);
    error.value = "无法加载试卷列表。";
  } finally {
    isLoading.value = false;
  }
};

const deleteExam = async (id, index) => {
  if (!window.confirm("确定要删除这个试卷吗？")) return;

  isDeleting.value[id] = true;
  error.value = null;
  try {
    await gradingApi.deleteExam(id);
    exams.value.splice(index, 1);
  } catch (e) {
    console.error(e);
    const examName = exams.value[index] ? exams.value[index].name : id;
    error.value = `删除试卷 "${examName}" 失败。`;
  } finally {
    isDeleting.value[id] = false;
  }
};

onMounted(fetchExams);
</script>
