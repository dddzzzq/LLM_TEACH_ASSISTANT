<template>
  <div class="ai-assistant-view">
    <!-- 页面标题 -->
    <header class="text-center mb-8">
      <h1 class="text-3xl md:text-4xl font-bold text-gray-800">AI 教学助手</h1>
      <p class="text-gray-500 mt-2">智能问答、成绩查询与批改助手</p>
    </header>

    <main class="max-w-5xl mx-auto">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <!-- 左侧：聊天组件 -->
        <div class="lg:col-span-2">
          <AgentChatWidget />
        </div>

        <!-- 右侧：功能说明和示例 -->
        <div class="lg:col-span-1">
          <div class="bg-white rounded-lg shadow-xl p-6 border border-gray-200">
            <h2 class="text-xl font-bold text-gray-800 mb-4">📋 功能说明</h2>
            
            <div class="space-y-4">
              <div class="p-3 bg-blue-50 rounded-lg border border-blue-100">
                <h3 class="font-medium text-blue-800 mb-1">🔍 学生成绩查询</h3>
                <p class="text-sm text-gray-600">查询学生的历史作业和试卷成绩、评语等详细信息。</p>
                <div class="mt-2 text-xs text-gray-500">
                  <strong>示例问题：</strong>
                  <p class="mt-1">"查询学生 23009200042 的成绩"</p>
                  <p>"查看学生 23009200045 的作业和试卷记录"</p>
                </div>
              </div>

              <div class="p-3 bg-green-50 rounded-lg border border-green-100">
                <h3 class="font-medium text-green-800 mb-1">⚡ 批改流水线触发</h3>
                <p class="text-sm text-gray-600">启动后台批改任务，自动处理学生作业。</p>
                <div class="mt-2 text-xs text-gray-500">
                  <strong>示例问题：</strong>
                  <p class="mt-1">"开始批改作业 ID 1，文件路径 /path/to/submissions.zip"</p>
                  <p>"触发批改流水线，作业ID 2，文件路径 /data/homework.zip"</p>
                </div>
              </div>

              <div class="p-3 bg-purple-50 rounded-lg border border-purple-100">
                <h3 class="font-medium text-purple-800 mb-1">💡 教学相关问题</h3>
                <p class="text-sm text-gray-600">解答教学管理、评分标准、教学建议等相关问题。</p>
                <div class="mt-2 text-xs text-gray-500">
                  <strong>示例问题：</strong>
                  <p class="mt-1">"如何设计有效的评分标准？"</p>
                  <p>"教学过程中如何预防学生抄袭？"</p>
                </div>
              </div>
            </div>

            <!-- 使用提示 -->
            <div class="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <h3 class="font-medium text-yellow-800 mb-2">💡 使用提示</h3>
              <ul class="text-sm text-gray-700 space-y-1">
                <li>• 尽量提供明确的学生ID或作业ID</li>
                <li>• 文件路径需要使用绝对路径</li>
                <li>• 系统会自动识别您的意图调用相应工具</li>
                <li>• 回复支持 Markdown 格式，可显示表格、列表等</li>
              </ul>
            </div>

            <!-- 快速操作 -->
            <div class="mt-6">
              <h3 class="font-medium text-gray-800 mb-3">🚀 快速操作</h3>
              <div class="grid grid-cols-1 gap-2">
                <button 
                  @click="insertExample('score')"
                  class="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors text-sm"
                >
                  <span class="font-medium text-gray-700">查询成绩示例</span>
                  <p class="text-gray-500 mt-1">插入学生成绩查询示例问题</p>
                </button>
                <button 
                  @click="insertExample('pipeline')"
                  class="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors text-sm"
                >
                  <span class="font-medium text-gray-700">批改触发示例</span>
                  <p class="text-gray-500 mt-1">插入批改流水线触发示例问题</p>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import AgentChatWidget from '../components/AgentChatWidget.vue'

// 示例问题
const exampleQuestions = {
  score: '查询学生 23009200042 的成绩',
  pipeline: '开始批改作业 ID 1，文件路径 /path/to/submissions.zip'
}

// 插入示例问题到聊天输入框
const insertExample = (type) => {
  const question = exampleQuestions[type]
  if (question) {
    // 这里需要通过事件总线或直接操作子组件来插入文本
    // 由于组件结构，我们暂时先显示提示
    alert(`请手动输入：${question}`)
  }
}
</script>

<style scoped>
.ai-assistant-view {
  min-height: calc(100vh - 8rem);
}
</style>