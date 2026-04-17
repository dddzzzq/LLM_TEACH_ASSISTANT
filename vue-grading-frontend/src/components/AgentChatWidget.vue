<template>
  <div class="agent-chat-widget">
    <!-- 聊天标题 -->
    <div class="chat-header bg-indigo-600 text-white p-4 rounded-t-lg flex justify-between items-center">
      <div>
        <h2 class="text-xl font-bold">AI 教学助手</h2>
        <p class="text-sm opacity-80">智能问答、成绩查询与批改助手</p>
      </div>
      <div v-if="currentSessionId" class="text-xs bg-indigo-500 px-3 py-1 rounded-full">
        会话: {{ currentSessionId.substring(0, 8) }}...
      </div>
    </div>

    <!-- 会话管理工具栏 -->
    <div v-if="isAuthenticated" class="toolbar bg-gray-100 border-b border-gray-200 px-4 py-2 flex justify-between items-center">
      <div class="flex space-x-2">
        <button
          @click="createNewSession"
          class="px-3 py-1 text-xs bg-white border border-gray-300 rounded hover:bg-gray-50 transition-colors"
          :disabled="loading"
        >
          🆕 新会话
        </button>
        <button
          @click="loadSessions"
          class="px-3 py-1 text-xs bg-white border border-gray-300 rounded hover:bg-gray-50 transition-colors"
          :disabled="loading"
        >
          📋 历史会话
        </button>
      </div>
      <div class="text-xs text-gray-600">
        用户: {{ currentUser?.name || currentUser?.username || '未登录' }}
      </div>
    </div>

    <!-- 会话选择下拉菜单 -->
    <div v-if="showSessionDropdown && sessions.length > 0" class="session-dropdown bg-white border border-gray-200 mx-4 mt-2 rounded shadow-lg max-h-48 overflow-y-auto">
      <div class="p-2 text-xs text-gray-500 border-b">选择历史会话</div>
      <div
        v-for="session in sessions"
        :key="session.id"
        @click="selectSession(session)"
        class="px-3 py-2 hover:bg-gray-100 cursor-pointer flex justify-between items-center"
      >
        <div>
          <div class="font-medium">{{ session.title }}</div>
          <div class="text-xs text-gray-500">{{ formatDate(session.updated_at) }}</div>
        </div>
        <div class="text-xs text-gray-400">
          {{ session.id.substring(0, 6) }}...
        </div>
      </div>
    </div>

    <!-- 聊天消息区域 -->
    <div
      ref="messagesContainer"
      class="messages-container p-4 bg-gray-50 h-96 overflow-y-auto"
    >
      <!-- 认证提示 -->
      <div v-if="!isAuthenticated" class="auth-prompt text-center py-8">
        <p class="text-gray-600 mb-4">请先登录以使用 AI 教学助手</p>
        <button
          @click="redirectToLogin"
          class="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors"
        >
          前往登录
        </button>
      </div>

      <!-- 欢迎消息 -->
      <div v-else-if="messages.length === 0" class="welcome-message text-center py-8">
        <p class="text-gray-500">👋 你好{{ currentUser?.name ? ` ${currentUser.name}` : '' }}！我是您的 AI 教学助手，可以帮您查询学生成绩或触发批改流水线。</p>
        <p class="text-sm text-gray-400 mt-2">当前会话 ID: {{ currentSessionId.substring(0, 12) }}...</p>
      </div>

      <!-- 消息列表 -->
      <div
        v-for="(msg, index) in messages"
        :key="index"
        :class="['message-item flex mb-4', msg.role === 'user' ? 'justify-end' : 'justify-start']"
      >
        <div
          :class="['message-content max-w-3/4 rounded-lg p-3', msg.role === 'user' ? 'bg-indigo-500 text-white' : 'bg-white text-gray-800 border border-gray-200']"
        >
          <!-- 用户消息：纯文本 -->
          <div v-if="msg.role === 'user'" class="user-message">
            {{ msg.content }}
          </div>
          
          <!-- Agent 消息：支持 Markdown 渲染 -->
          <div v-else class="agent-message markdown-content" v-html="renderMarkdown(msg.content)"></div>
          
          <!-- 消息时间 -->
          <div
            :class="['message-time text-xs mt-1', msg.role === 'user' ? 'text-indigo-200' : 'text-gray-400']"
          >
            {{ formatTime(msg.timestamp) }}
          </div>
        </div>
      </div>

      <!-- 加载状态 -->
      <div v-if="loading" class="loading-indicator flex justify-center my-4">
        <div class="typing-indicator flex space-x-1">
          <div class="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
          <div class="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-150"></div>
          <div class="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-300"></div>
        </div>
      </div>
    </div>

    <!-- 快捷指令区域 -->
    <div v-if="isAuthenticated" class="quick-commands bg-gray-50 border-t border-gray-200 px-4 py-3">
      <div class="text-xs text-gray-500 mb-2">💡 快捷指令：</div>
      <div class="flex flex-wrap gap-2">
        <button
          @click="useQuickCommand('fetch_homework')"
          class="quick-cmd-btn flex items-center px-3 py-1.5 text-xs bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-full hover:from-purple-600 hover:to-indigo-600 transition-all shadow-sm"
          :disabled="loading"
        >
          <span class="mr-1">📥</span>
          从教务系统下载作业
        </button>
        <button
          @click="useQuickCommand('query_score')"
          class="quick-cmd-btn flex items-center px-3 py-1.5 text-xs bg-gradient-to-r from-green-500 to-teal-500 text-white rounded-full hover:from-green-600 hover:to-teal-600 transition-all shadow-sm"
          :disabled="loading"
        >
          <span class="mr-1">📊</span>
          查询学生成绩
        </button>
        <button
          @click="useQuickCommand('trigger_pipeline')"
          class="quick-cmd-btn flex items-center px-3 py-1.5 text-xs bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-full hover:from-orange-600 hover:to-red-600 transition-all shadow-sm"
          :disabled="loading"
        >
          <span class="mr-1">🚀</span>
          触发批改流水线
        </button>
      </div>
    </div>

    <!-- 输入区域 -->
    <div v-if="isAuthenticated" class="input-area p-4 bg-white border-t border-gray-200 rounded-b-lg">
      <form @submit.prevent="sendMessage" class="flex space-x-2">
        <input
          ref="inputRef"
          v-model="inputMessage"
          type="text"
          placeholder="输入您的问题，或点击上方快捷指令"
          class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          :disabled="loading"
        />
        <button
          type="submit"
          :disabled="!inputMessage.trim() || loading"
          class="px-6 py-2 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          发送
        </button>
      </form>
      <div v-if="currentSessionId" class="text-xs text-gray-500 mt-2 flex justify-between">
        <span>会话 ID: {{ currentSessionId.substring(0, 16) }}...</span>
        <button
          @click="copySessionId"
          class="text-indigo-500 hover:text-indigo-700"
        >
          复制
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import authApi from '../services/authApi'

// 消息数据
const messages = ref([])
const inputMessage = ref('')
const loading = ref(false)
const messagesContainer = ref(null)
const currentSessionId = ref('')
const sessions = ref([])
const showSessionDropdown = ref(false)
const inputRef = ref(null)

// 快捷指令配置
const quickCommands = {
  fetch_homework: {
    template: '请帮我从教务系统下载作业，需要提供以下信息：\n- 用户名：\n- 密码：\n- 课程名称：\n- 作业名称：',
    placeholder: '请输入教务系统用户名、密码、课程名称和作业名称'
  },
  query_score: {
    template: '请帮我查询学生 ',
    placeholder: '请输入学号，例如：23009200042'
  },
  trigger_pipeline: {
    template: '请帮我触发批改流水线，作业ID为 ',
    placeholder: '请输入作业ID和文件路径'
  }
}

// 计算属性
const isAuthenticated = computed(() => authApi.isAuthenticated())
const currentUser = computed(() => authApi.getCurrentUser())

// 初始化
onMounted(() => {
  if (isAuthenticated.value) {
    // 检查是否有保存的会话ID
    const savedSessionId = localStorage.getItem('current_session_id')
    if (savedSessionId) {
      currentSessionId.value = savedSessionId
      loadSessionHistory(savedSessionId)
    } else {
      createNewSession()
    }
  }
})

// 格式化时间
const formatTime = (date) => {
  return new Date(date).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 格式化日期
const formatDate = (dateString) => {
  const date = new Date(dateString)
  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// Markdown 渲染方法
const renderMarkdown = (content) => {
  if (!content) return ''
  
  // 使用 marked 将 Markdown 转换为 HTML
  const rawHtml = marked.parse(content, {
    breaks: true, // 允许换行符
    gfm: true, // GitHub Flavored Markdown
    headerIds: false // 禁用自动生成的 header IDs
  })
  
  // 使用 DOMPurify 进行安全过滤
  return DOMPurify.sanitize(rawHtml, {
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'em', 'b', 'i', 'u', 's', 'code', 'pre',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li', 'blockquote',
      'table', 'thead', 'tbody', 'tr', 'th', 'td',
      'a', 'img', 'div', 'span',
      'hr', 'sup', 'sub'
    ],
    ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'class', 'id', 'target']
  })
}

// 创建新会话
const createNewSession = async () => {
  try {
    loading.value = true
    
    // 【修改点 1】：使用普通的 client，手动添加 /api 前缀
    const client = authApi.getClient()
    const response = await client.post('/api/sessions', {
      title: `会话 ${new Date().toLocaleDateString('zh-CN')}`
    })
    
    if (response.data && response.data.id) {
      currentSessionId.value = response.data.id
      localStorage.setItem('current_session_id', response.data.id)
      messages.value = []
      
      // 添加欢迎消息
      messages.value.push({
        role: 'assistant',
        content: '您好！我是您的智能教学助手，可以帮您：\n\n1. **查询学生成绩** - 例如："查询学生 23009200042 的成绩"\n2. **触发批改流水线** - 例如："开始批改作业 ID 1，文件路径 /path/to/submissions.zip"\n3. **回答教学相关问题**\n\n请问有什么可以帮您的？',
        timestamp: new Date()
      })
    } else {
      throw new Error('创建会话失败，未返回会话ID')
    }
  } catch (error) {
    console.error('创建会话失败:', error)
    // 显示错误提示
    alert('创建会话失败，请检查网络连接或稍后重试。')
    // 清除本地存储的会话ID
    localStorage.removeItem('current_session_id')
    currentSessionId.value = ''
    messages.value = []
    
    // 添加错误提示消息
    messages.value.push({
      role: 'assistant',
      content: '抱歉，系统暂时无法创建新会话，请检查网络连接或稍后重试。',
      timestamp: new Date()
    })
  } finally {
    loading.value = false
    showSessionDropdown.value = false
  }
}

// 加载用户的所有会话
const loadSessions = async () => {
  try {
    loading.value = true
    // 【修改点 2】：使用普通的 client，手动添加 /api 前缀
    const client = authApi.getClient()
    const response = await client.get('/api/sessions')
    
    if (response.data) {
      sessions.value = response.data
      showSessionDropdown.value = !showSessionDropdown.value
      if (sessions.value.length === 0) {
        alert('暂无历史会话记录，请创建新会话')
      }
    } else {
      throw new Error('未获取到会话数据')
    }
  } catch (error) {
    console.error('加载会话失败:', error)
    showSessionDropdown.value = false
    alert('加载历史会话失败，请检查网络连接或重新登录')
  } finally {
    loading.value = false
  }
}

// 选择会话
const selectSession = async (session) => {
  currentSessionId.value = session.id
  localStorage.setItem('current_session_id', session.id)
  showSessionDropdown.value = false
  
  // 加载会话历史
  await loadSessionHistory(session.id)
}

// 加载会话历史
const loadSessionHistory = async (sessionId) => {
  try {
    loading.value = true
    // 【修改点 3】：使用普通的 client，手动添加 /api 前缀
    const client = authApi.getClient()
    const response = await client.get(`/api/sessions/${sessionId}/history`)
    
    if (response.data && response.data.messages) {
      messages.value = response.data.messages.map(msg => ({
        role: msg.role,
        content: msg.content,
        timestamp: new Date(msg.timestamp || msg.created_at)
      }))
    } else {
      // 如果没有历史消息，添加欢迎消息
      messages.value = [{
        role: 'assistant',
        content: `欢迎回来！继续之前的对话。`,
        timestamp: new Date()
      }]
    }
  } catch (error) {
    console.error('加载会话历史失败:', error)
    messages.value = [{
      role: 'assistant',
      content: `欢迎回来！开始新的对话。`,
      timestamp: new Date()
    }]
  } finally {
    loading.value = false
    scrollToBottom()
  }
}

// 发送消息
const sendMessage = async () => {
  const message = inputMessage.value.trim()
  if (!message) return

  // 检查是否已认证
  if (!isAuthenticated.value) {
    redirectToLogin()
    return
  }

  // 检查是否有会话ID
  if (!currentSessionId.value) {
    await createNewSession()
  }

  // 添加用户消息
  messages.value.push({
    role: 'user',
    content: message,
    timestamp: new Date()
  })

  // 清空输入框
  inputMessage.value = ''
  
  // 设置加载状态
  loading.value = true

  // 滚动到底部
  scrollToBottom()

  try {
    // 【修改点 4】：使用普通的 client，手动添加 /api 前缀解决 404 报错
    const client = authApi.getClient()
    const response = await client.post('/api/agent/chat', {
      message: message,
      session_id: currentSessionId.value
    })

    const data = response.data

    // 添加 Agent 回复
    messages.value.push({
      role: 'assistant',
      content: data.reply || '抱歉，暂时无法回答您的问题。',
      timestamp: new Date()
    })

  } catch (error) {
    console.error('发送消息失败:', error)
    
    // 检查是否为认证错误
    if (error.response?.status === 401) {
      // Token 可能过期，尝试刷新
      try {
        await authApi.refreshToken(authApi.getRefreshToken())
        // 重试发送消息
        await sendMessage()
        return
      } catch (refreshError) {
        // 刷新失败，重定向到登录
        redirectToLogin()
        return
      }
    }
    
    // 添加错误消息
    messages.value.push({
      role: 'assistant',
      content: '抱歉，网络请求失败，请检查网络连接或稍后重试。',
      timestamp: new Date()
    })
  } finally {
    loading.value = false
    // 滚动到底部
    scrollToBottom()
  }
}

// 复制会话ID
const copySessionId = () => {
  navigator.clipboard.writeText(currentSessionId.value).then(() => {
    alert('会话ID已复制到剪贴板')
  }).catch(err => {
    console.error('复制失败:', err)
  })
}

// 重定向到登录页
const redirectToLogin = () => {
  // 这里可以根据实际路由配置调整
  window.location.href = '/login'
}

// 使用快捷指令
const useQuickCommand = (commandType) => {
  const command = quickCommands[commandType]
  if (command) {
    inputMessage.value = command.template
    // 聚焦到输入框
    nextTick(() => {
      if (inputRef.value) {
        inputRef.value.focus()
        // 将光标移到文本末尾
        inputRef.value.setSelectionRange(command.template.length, command.template.length)
      }
    })
  }
}

// 滚动到底部
const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

// 监听消息变化，自动滚动
watch(messages, () => {
  scrollToBottom()
}, { deep: true })

// 监听认证状态变化
watch(isAuthenticated, (newVal) => {
  if (newVal) {
    // 重新加载会话
    const savedSessionId = localStorage.getItem('current_session_id')
    if (savedSessionId) {
      currentSessionId.value = savedSessionId
      loadSessionHistory(savedSessionId)
    } else {
      createNewSession()
    }
  } else {
    // 清除会话数据
    messages.value = []
    currentSessionId.value = ''
    localStorage.removeItem('current_session_id')
  }
})
</script>

<style scoped>
.agent-chat-widget {
  border-radius: 0.5rem;
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  overflow: hidden;
}

.messages-container {
  min-height: 384px; /* h-96 */
}

.message-content {
  word-wrap: break-word;
  overflow-wrap: break-word;
}

.session-dropdown {
  z-index: 10;
  position: relative;
}

/* 打字指示器动画 */
@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.animate-pulse {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

.delay-150 {
  animation-delay: 150ms;
}

.delay-300 {
  animation-delay: 300ms;
}
</style>

<style>
/* Markdown 样式重置和美化 */
.markdown-content {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  line-height: 1.6;
}

.markdown-content p {
  margin-bottom: 1rem;
}

.markdown-content strong,
.markdown-content b {
  font-weight: 600;
  color: #1f2937;
}

.markdown-content em,
.markdown-content i {
  font-style: italic;
}

.markdown-content h1,
.markdown-content h2,
.markdown-content h3,
.markdown-content h4,
.markdown-content h5,
.markdown-content h6 {
  font-weight: 600;
  margin-top: 1.5rem;
  margin-bottom: 1rem;
  color: #111827;
}

.markdown-content h1 {
  font-size: 1.875rem;
  border-bottom: 2px solid #e5e7eb;
  padding-bottom: 0.5rem;
}

.markdown-content h2 {
  font-size: 1.5rem;
}

.markdown-content h3 {
  font-size: 1.25rem;
}

.markdown-content ul,
.markdown-content ol {
  padding-left: 1.5rem;
  margin-bottom: 1rem;
}

.markdown-content li {
  margin-bottom: 0.5rem;
}

.markdown-content ul {
  list-style-type: disc;
}

.markdown-content ol {
  list-style-type: decimal;
}

.markdown-content blockquote {
  border-left: 4px solid #e5e7eb;
  padding-left: 1rem;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1rem;
  color: #6b7280;
  font-style: italic;
}

.markdown-content code {
  background-color: #f3f4f6;
  padding: 0.2rem 0.4rem;
  border-radius: 0.25rem;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.875rem;
}

.markdown-content pre {
  background-color: #1f2937;
  color: #f3f4f6;
  padding: 1rem;
  border-radius: 0.5rem;
  overflow-x: auto;
  margin-bottom: 1rem;
}

.markdown-content pre code {
  background-color: transparent;
  padding: 0;
  color: inherit;
}

.markdown-content table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1rem;
  font-size: 0.875rem;
}

.markdown-content th,
.markdown-content td {
  border: 1px solid #e5e7eb;
  padding: 0.75rem;
  text-align: left;
}

.markdown-content th {
  background-color: #f9fafb;
  font-weight: 600;
  color: #374151;
}

.markdown-content tr:nth-child(even) {
  background-color: #f9fafb;
}

.markdown-content a {
  color: #4f46e5;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.markdown-content a:hover {
  color: #3730a3;
}

.markdown-content hr {
  border: 0;
  height: 1px;
  background-color: #e5e7eb;
  margin: 1.5rem 0;
}
</style>