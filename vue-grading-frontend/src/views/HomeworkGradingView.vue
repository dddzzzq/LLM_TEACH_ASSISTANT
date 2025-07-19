<template>
  <div>
    <header class="text-center mb-8">
      <h1 class="text-3xl md:text-4xl font-bold text-gray-800">智能作业批改</h1>
      <p class="text-gray-500 mt-2">上传学生作业压缩包，AI将进行综合评估</p>
    </header>

    <main class="max-w-3xl mx-auto bg-white rounded-lg shadow-xl p-6 md:p-8">
      <form @submit.prevent="submitGradingTask">
        <div class="space-y-6">
          <!-- 任务名称 -->
          <div>
            <label for="task_name" class="block text-sm font-medium text-gray-700">任务名称</label>
            <input type="text" v-model="taskName" id="task_name" required
                  class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="例如：第一次分布式作业">
          </div>

          <!-- 题目要求 -->
          <div>
            <label for="question" class="block text-sm font-medium text-gray-700">题目要求</label>
            <textarea v-model="question" id="question" rows="4" required
                      class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                      placeholder="请详细描述本次作业的具体要求..."></textarea>
          </div>

          <!-- 评分标准 -->
          <div>
            <label for="rubric" class="block text-sm font-medium text-gray-700">评分标准 (JSON格式)</label>
            <textarea v-model="rubric" id="rubric" rows="6" required
                      class="mt-1 block w-full px-3 py-2 font-mono text-sm bg-gray-50 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      placeholder='例如：\n{\n  "功能实现": {"description": "核心功能完整无误", "score": 60},\n  "代码质量": {"description": "代码规范，注释清晰", "score": 20},\n  "文档报告": {"description": "报告内容详实，格式正确", "score": 20}\n}'></textarea>
          </div>

          <!-- 文件上传 -->
          <div>
            <label for="file-upload" class="block text-sm font-medium text-gray-700">上传作业压缩包</label>
            <div class="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
              <div class="space-y-1 text-center">
                <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                  <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
                <div class="flex text-sm text-gray-600">
                  <label for="file-upload" class="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                    <span>选择文件</span>
                    <input id="file-upload" name="file-upload" type="file" class="sr-only" @change="handleFileChange" accept=".zip" required>
                  </label>
                  <p class="pl-1">或拖拽到此处</p>
                </div>
                <p class="text-xs text-gray-500">仅支持 .zip 格式</p>
                <p v-if="fileName" class="text-sm text-green-600 pt-2 font-semibold">{{ fileName }}</p>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 提交按钮 -->
        <div class="mt-8">
          <button type="submit" :disabled="isLoading" class="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-300 disabled:cursor-not-allowed">
            <span v-if="!isLoading">开始AI批改</span>
            <span v-else>批改中，请稍候...</span>
          </button>
        </div>
      </form>
      
      <!-- 加载、错误和结果展示 -->
      <Loader v-if="isLoading" class="mt-8" />
      <div v-if="error" class="mt-8 p-4 bg-red-100 border border-red-400 text-red-700 rounded-md">
        <h3 class="font-bold">发生错误</h3>
        <p>{{ error }}</p>
      </div>
      <GradingResult v-if="result" :result-data="result" class="mt-8" />
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import gradingApi from '../services/gradingApi';
import GradingResult from '../components/GradingResult.vue';
import Loader from '../components/Loader.vue';

// 响应式状态定义
const taskName = ref('第一次Web开发大作业');
const question = ref('利用HTML, CSS, JavaScript技术实现一个个人作品集网站。');
const rubric = ref(JSON.stringify({
  "页面设计": { "description": "界面美观，布局合理", "score": 30 },
  "功能实现": { "description": "交互功能完整，无明显bug", "score": 50 },
  "代码质量": { "description": "代码规范，有适当注释", "score": 20 }
}, null, 2));
const file = ref(null);
const fileName = ref('');
const isLoading = ref(false);
const error = ref(null);
const result = ref(null);

const handleFileChange = (event) => {
  const selectedFile = event.target.files[0];
  if (selectedFile) {
    file.value = selectedFile;
    fileName.value = selectedFile.name;
  }
};

const submitGradingTask = async () => {
  if (!file.value) {
    error.value = "请选择要上传的ZIP文件。";
    return;
  }
  
  isLoading.value = true;
  error.value = null;
  result.value = null;

  const formData = new FormData();
  formData.append('task_name', taskName.value);
  formData.append('question', question.value);
  formData.append('rubric', rubric.value);
  formData.append('file', file.value);

  try {
    const response = await gradingApi.gradeHomework(formData);
    result.value = response.data;
  } catch (e) {
    console.error('提交失败:', e);
    if (e.response && e.response.data && e.response.data.detail) {
        error.value = e.response.data.detail;
    } else {
        error.value = '请求失败，请检查网络连接或后端服务是否正常。';
    }
  } finally {
    isLoading.value = false;
  }
};
</script>