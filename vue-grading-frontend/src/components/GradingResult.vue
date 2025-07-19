<template>
  <div class="p-6 bg-gray-50 rounded-lg border">
    <h2 class="text-2xl font-bold text-gray-800 mb-4">
      批改结果
    </h2>
    
    <div class="space-y-4">
      <div>
        <h3 class="font-semibold text-gray-700">任务名称:</h3>
        <p class="text-gray-600">{{ resultData.task_name }}</p>
      </div>
      
      <div>
        <h3 class="font-semibold text-gray-700">分数:</h3>
        <p class="text-3xl font-bold text-indigo-600">
          {{ resultData.score.toFixed(1) }}
        </p>
      </div>
      
      <div>
        <h3 class="font-semibold text-gray-700">AI评语:</h3>
        <p class="text-gray-600 whitespace-pre-wrap">
          {{ resultData.feedback }}
        </p>
      </div>
      
      <div>
        <h3 class="font-semibold text-gray-700">学生提交内容预览 (合并后):</h3>
        <pre
          class="mt-2 p-4 bg-white border rounded-md text-xs text-gray-700 whitespace-pre-wrap overflow-x-auto max-h-96"
        >{{ resultData.merged_content }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// 为 resultData 定义一个更具体的类型
interface ResultData {
  task_name: string;
  score: number;
  feedback: string;
  merged_content: string;
}

defineProps<{
  resultData: ResultData;
}>();
</script>