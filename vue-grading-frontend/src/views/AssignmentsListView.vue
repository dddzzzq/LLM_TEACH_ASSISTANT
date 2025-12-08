<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-3xl font-bold text-gray-800">所有作业任务</h1>
      <router-link
        to="/assignments/new"
        class="px-4 py-2 font-semibold text-white bg-indigo-600 rounded-lg shadow-md transition-colors hover:bg-indigo-700 flex items-center"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          class="h-5 w-5 mr-1"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fill-rule="evenodd"
            d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z"
            clip-rule="evenodd"
          />
        </svg>
        新建作业
      </router-link>
    </div>

    <div v-if="isLoading" class="py-20 text-center">
      <Loader />
      <p class="text-gray-500 mt-4">正在加载作业列表...</p>
    </div>

    <div
      v-else-if="error"
      class="p-6 text-red-700 bg-red-100 rounded-lg border border-red-200 flex items-start"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        class="h-6 w-6 mr-2 flex-shrink-0"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
      <div>
        <h3 class="font-bold text-lg">加载失败</h3>
        <p>{{ error }}</p>
        <button
          @click="fetchAssignments"
          class="mt-2 text-sm font-semibold underline hover:text-red-900"
        >
          重试
        </button>
      </div>
    </div>

    <div
      v-else-if="assignments.length === 0"
      class="py-20 text-center bg-white rounded-lg shadow-sm border border-gray-100"
    >
      <div
        class="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-100 mb-4"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          class="h-8 w-8 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"
          />
        </svg>
      </div>
      <h3 class="text-lg font-medium text-gray-900">暂无作业任务</h3>
      <p class="mt-1 text-gray-500">点击右上角的“新建作业”按钮创建一个新的作业任务。</p>
    </div>

    <div v-else class="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      <div
        v-for="(assignment, index) in assignments"
        :key="assignment.id"
        class="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300 border border-gray-100 overflow-hidden flex flex-col"
      >
        <div class="p-6 flex-grow">
          <div class="flex justify-between items-start mb-4">
            <h3
              class="text-xl font-bold text-gray-800 line-clamp-2"
              :title="assignment.task_name"
            >
              {{ assignment.task_name }}
            </h3>
            <span
              class="px-2 py-1 text-xs font-semibold text-indigo-600 bg-indigo-50 rounded-full"
              >ID: {{ assignment.id }}</span
            >
          </div>
          <p class="text-gray-600 text-sm line-clamp-3 mb-4 h-12">
            {{ assignment.question || "无题目描述" }}
          </p>
        </div>

        <div
          class="px-6 py-4 bg-gray-50 border-t border-gray-100 flex justify-between items-center mt-auto"
        >
          <router-link
            :to="`/assignments/${assignment.id}`"
            class="text-indigo-600 hover:text-indigo-800 font-medium text-sm flex items-center"
          >
            进入评分
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="h-4 w-4 ml-1"
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
            @click.prevent="deleteAssignment(assignment.id, index)"
            title="删除此作业"
            class="text-gray-400 hover:text-red-600 transition-colors p-1 rounded-full hover:bg-red-50"
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
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from "vue";
import { RouterLink } from "vue-router";
import gradingApi from "../services/gradingApi";
import Loader from "../components/Loader.vue";

interface Assignment {
  id: number;
  task_name: string;
  question: string;
}

const assignments = ref<Assignment[]>([]);
const isLoading = ref(true);
const error = ref<string | null>(null);

const fetchAssignments = async () => {
  isLoading.value = true;
  error.value = null;
  try {
    console.log("开始获取作业列表..."); // 调试日志
    const response = await gradingApi.getAssignments();
    console.log("API 响应数据:", response.data); // 调试日志
    assignments.value = response.data;
  } catch (e: any) {
    console.error("获取作业列表失败:", e); // 调试日志
    error.value = e.message || "无法加载作业列表，请检查后端服务是否启动。";
    // 如果是网络错误，通常 e.message 会是 "Network Error"
  } finally {
    isLoading.value = false;
  }
};

const deleteAssignment = async (id: number, index: number) => {
  if (confirm("确定要删除这个作业及其所有评分记录吗？\n此操作不可撤销。")) {
    try {
      await gradingApi.deleteAssignment(id);
      assignments.value.splice(index, 1);
    } catch (e) {
      alert("删除失败，请重试。");
      console.error(e);
    }
  }
};

onMounted(fetchAssignments);
</script>
