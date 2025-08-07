<template>
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60"
    @click.self="close"
  >
    <div class="relative w-full max-w-2xl p-6 mx-4 bg-white rounded-lg shadow-xl">
      <h2 class="text-2xl font-bold text-gray-800 mb-4">教师复查与评分修改</h2>
      <p class="text-sm text-gray-600 mb-6">
        您正在为学生
        <span class="font-bold">{{ submission.student_id }}</span> 进行人工复核。
      </p>

      <form @submit.prevent="saveReview">
        <div class="space-y-4">
          <div>
            <label for="score" class="block text-sm font-medium text-gray-700"
              >最终分数</label
            >
            <input
              type="number"
              id="human_score"
              v-model.number="editableScore"
              step="1"
              min="0"
              max="100"
              required
              class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            />
            <p class="text-xs text-gray-500 mt-1">
              AI 原始评分: {{ submission.score.toFixed(1) }}
            </p>
          </div>
          <div>
            <label for="human_feedback" class="block text-sm font-medium text-gray-700"
              >最终评语</label
            >
            <textarea
              id="human_feedback"
              v-model="editableFeedback"
              rows="6"
              required
              class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            ></textarea>
            <p class="text-xs text-gray-500 mt-1">
              AI 原始评语: {{ submission.feedback }}
            </p>
          </div>
        </div>
        <div class="mt-6 flex justify-end space-x-4">
          <button
            type="button"
            @click="close"
            class="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
          >
            取消
          </button>
          <button
            type="submit"
            class="px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700"
          >
            保存修改
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, defineProps, defineEmits } from "vue";

const props = defineProps({
  submission: {
    type: Object,
    required: true,
  },
});

const emit = defineEmits(["close", "save"]);

const editableScore = ref(props.submission.human_score || props.submission.sc);
const editableFeedback = ref(
  props.submission.human_feedback || props.submission.feedback
);

const close = () => {
  emit("close");
};

const saveReview = () => {
  emit("save", {
    id: props.submission.id,
    score: editableScore.value,
    human_feedback: editableFeedback.value,
  });
};
</script>
