<template>
  <div class="max-w-3xl mx-auto">
    <h1 class="mb-6 text-3xl font-bold text-gray-800">新建作业任务</h1>

    <form @submit.prevent="createAssignment" class="p-8 space-y-6 bg-white rounded-lg shadow-xl">
      <div>
        <label for="task_name" class="block text-sm font-medium text-gray-700">任务名称</label>
        <input
          id="task_name"
          v-model="taskName"
          type="text"
          required
          class="block w-full px-3 py-2 mt-1 border border-gray-300 rounded-md shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
          placeholder="例如：第一次分布式作业"
        >
      </div>

      <div>
        <label for="question" class="block text-sm font-medium text-gray-700">题目要求</label>
        <textarea
          id="question"
          v-model="question"
          rows="4"
          required
          class="block w-full px-3 py-2 mt-1 border border-gray-300 rounded-md shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
          placeholder="请详细描述本次作业的具体要求..."
        />
      </div>

      <div>
        <label for="rubric" class="block text-sm font-medium text-gray-700">评分标准 (JSON格式)</label>
        <textarea
          id="rubric"
          v-model="rubric"
          rows="8"
          required
          class="block w-full px-3 py-2 mt-1 font-mono border border-gray-300 rounded-md shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
        />
      </div>

      <div v-if="error" class="text-sm text-red-500">
        {{ error }}
      </div>

      <button
        type="submit"
        :disabled="isLoading"
        class="flex justify-center w-full px-4 py-3 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-400"
      >
        <span v-if="!isLoading">创建作业</span>
        <span v-else>创建中...</span>
      </button>
    </form>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import gradingApi from '../services/gradingApi';

const router = useRouter();

// 数据形式
const taskName = ref('');
const question = ref('');
const rubric = ref(
  JSON.stringify(
    {
      "功能实现": { description: "核心功能完整无误", score: 60 },
      "代码质量": { description: "代码规范，注释清晰", score: 20 },
      "文档报告": { description: "报告内容详实，格式正确", score: 20 },
    },
    null,
    2,
  ),
);
const isLoading = ref(false);
const error = ref<string | null>(null);

// 处理逻辑
const createAssignment = async () => {
  isLoading.value = true;
  error.value = null;

  try {
    const rubricData = JSON.parse(rubric.value);
    await gradingApi.createAssignment({
      task_name: taskName.value,
      question: question.value,
      rubric: rubricData,
    });
    
    router.push({ name: 'assignments-list' });
  } catch (e: unknown) {
    if (e instanceof SyntaxError) {
      error.value = '评分标准不是一个有效的JSON格式。';
    } else {
      error.value = '创建作业失败，请重试。';
      console.error(e);
    }
  } finally {
    isLoading.value = false;
  }
};
</script>