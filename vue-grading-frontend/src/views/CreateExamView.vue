<template>
  <div class="max-w-4xl mx-auto">
    <!-- 步骤一：创建试卷 -->
    <div v-if="step === 1" class="p-8 bg-white rounded-lg shadow-xl">
      <h1 class="mb-6 text-3xl font-bold text-gray-800">步骤 1: 新建试卷</h1>
      <form @submit.prevent="handleCreateExam">
        <div>
          <label for="exam_name" class="block text-sm font-medium text-gray-700"
            >试卷名称</label
          >
          <input
            id="exam_name"
            v-model="examName"
            type="text"
            required
            class="block w-full px-3 py-2 mt-1 border border-gray-300 rounded-md shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            placeholder="例如：2022年期末试卷"
          />
        </div>
        <div v-if="error" class="mt-4 text-sm text-red-500">
          {{ error }}
        </div>
        <button
          type="submit"
          :disabled="isLoading"
          class="flex justify-center w-full px-4 py-3 mt-6 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-400"
        >
          <span v-if="!isLoading">创建试卷并添加题目</span>
          <span v-else>创建中...</span>
        </button>
      </form>
    </div>

    <!-- 步骤二：添加题目 -->
    <div v-if="step === 2" class="space-y-8">
      <h1 class="text-3xl font-bold text-gray-800">
        步骤 2: 为 "{{ examName }}" 添加题目
      </h1>

      <!-- 题目列表 -->
      <div v-if="questions.length > 0" class="p-8 bg-white rounded-lg shadow-xl">
        <h2 class="text-xl font-semibold text-gray-700 mb-4">
          已添加题目 ({{ questions.length }} 道)
        </h2>
        <ul class="divide-y divide-gray-200">
          <li v-for="q in questions" :key="q.id" class="py-2">
            <span class="font-bold">{{ q.question_number }}.</span>
            {{ q.question_text.substring(0, 100) }}...
          </li>
        </ul>
        <button
          @click="finishCreation"
          class="w-full px-4 py-3 mt-6 text-sm font-medium text-white bg-green-600 border border-transparent rounded-md shadow-sm hover:bg-green-700"
        >
          完成创建（共 {{ questions.length }} 道题）
        </button>
      </div>

      <!-- 添加题目表单 -->
      <form
        @submit.prevent="handleAddQuestion"
        class="p-8 space-y-6 bg-white rounded-lg shadow-xl"
      >
        <h2 class="text-xl font-semibold text-gray-700">添加新题目</h2>

        <div class="grid grid-cols-3 gap-4">
          <div class="col-span-2">
            <label for="q_number" class="block text-sm font-medium text-gray-700"
              >题号</label
            >
            <input
              id="q_number"
              v-model.number="newQuestion.question_number"
              type="number"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
              placeholder="例如：1"
            />
          </div>
          <div>
            <label for="q_max_score" class="block text-sm font-medium text-gray-700"
              >题目总分</label
            >
            <input
              id="q_max_score"
              v-model.number="newQuestion.max_score"
              type="number"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
              placeholder="例如：10"
            />
          </div>
        </div>

        <div>
          <label for="q_text" class="block text-sm font-medium text-gray-700"
            >题目内容</label
          >
          <textarea
            id="q_text"
            v-model="newQuestion.question_text"
            rows="3"
            required
            class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
            placeholder="请输入完整的题目内容..."
          ></textarea>
        </div>
        <div>
          <label for="q_answer" class="block text-sm font-medium text-gray-700"
            >标准答案</label
          >
          <textarea
            id="q_answer"
            v-model="newQuestion.standard_answer"
            rows="3"
            required
            class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
            placeholder="请输入该题的标准答案..."
          ></textarea>
        </div>
        <div>
          <label for="q_rubric" class="block text-sm font-medium text-gray-700"
            >评分标准 (纯文本)</label
          >
          <textarea
            id="q_rubric"
            v-model="newQuestion.rubric"
            rows="5"
            required
            class="mt-1 block w-full px-3 py-2 font-mono border border-gray-300 rounded-md shadow-sm"
            placeholder="请输入自然语言评分标准，例如：&#10;1. 答对A点得5分。&#10;2. 答对B点得5分。"
          ></textarea>
        </div>
        <div v-if="error" class="text-sm text-red-500">
          {{ error }}
        </div>
        <button
          type="submit"
          :disabled="isLoading"
          class="flex justify-center w-full px-4 py-3 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md shadow-sm hover:bg-indigo-700 disabled:bg-indigo-400"
        >
          <span v-if="!isLoading">添加这道题</span>
          <span v-else>添加中...</span>
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
import gradingApi from "../services/gradingApi";

const router = useRouter();

const step = ref(1);
const examName = ref("");
const createdExamId = ref(null);
const questions = ref([]); // 已添加的题目列表

const newQuestion = ref({
  question_number: 1,
  question_text: "",
  standard_answer: "",
  rubric: "", // <--- 修改：默认为空字符串
  max_score: 10, // <--- 新增
});

const isLoading = ref(false);
const error = ref(null);

// 步骤一：创建试卷
const handleCreateExam = async () => {
  isLoading.value = true;
  error.value = null;
  try {
    const response = await gradingApi.createExam({ name: examName.value });
    createdExamId.value = response.data.id;
    newQuestion.value.question_number = (response.data.question_count || 0) + 1;
    step.value = 2; // 进入步骤二
  } catch (e) {
    console.error(e);
    error.value = "创建试卷失败，请重试。";
  } finally {
    isLoading.value = false;
  }
};

// 步骤二：添加题目
const handleAddQuestion = async () => {
  isLoading.value = true;
  error.value = null;

  // <--- 删除：移除 JSON.parse 验证 ---

  try {
    const questionData = {
      question_number: newQuestion.value.question_number,
      question_text: newQuestion.value.question_text,
      standard_answer: newQuestion.value.standard_answer,
      rubric: newQuestion.value.rubric, // <--- 修改：直接发送字符串
      max_score: newQuestion.value.max_score, // <--- 新增
    };

    const response = await gradingApi.addExamQuestion(createdExamId.value, questionData);

    // 添加成功
    questions.value.push(response.data);

    // 重置表单
    newQuestion.value.question_number += 1;
    newQuestion.value.question_text = "";
    newQuestion.value.standard_answer = "";
    newQuestion.value.rubric = ""; // <--- 修改：重置为空字符串
    newQuestion.value.max_score = 10; // <--- 新增：重置
  } catch (e) {
    console.error(e);
    error.value = "添加题目失败，请重试。";
  } finally {
    isLoading.value = false;
  }
};

// 步骤二：完成创建
const finishCreation = () => {
  // 跳转到新创建的试卷详情页
  router.push({ name: "exam-detail", params: { id: createdExamId.value } });
};
</script>
