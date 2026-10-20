<template>
  <div class="agent-chat-widget">
    <!-- 聊天标题 -->
    <div class="chat-header">
      <div class="chat-identity"><div class="chat-symbol"><AppIcon name="sparkles" /></div><div>
        <h2>教学对话</h2><p>{{ canManageHomework ? '拉取与解压、作业批改、成绩查询' : '学习问答与个人成绩查询' }}</p>
      </div></div>
      <div v-if="currentSessionId" class="session-tag">
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
          <AppIcon name="plus" /> 新会话
        </button>
        <button
          @click="loadSessions"
          class="px-3 py-1 text-xs bg-white border border-gray-300 rounded hover:bg-gray-50 transition-colors"
          :disabled="loading"
        >
          <AppIcon name="history" /> 历史会话
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
      :class="{'messages-idle': !hasUserMessages}"
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
        <p class="text-gray-500">{{ welcomeText }}</p>
      </div>

      <!-- 消息列表 -->
      <div
        v-for="(msg, index) in messages"
        :key="index"
        :class="['message-item flex mb-4', msg.role === 'user' ? 'justify-end' : 'justify-start']"
      >
        <div v-if="msg.role !== 'user'" class="message-avatar"><AppIcon name="sparkles" /></div>
        <div
          :class="['message-content max-w-3/4 rounded-lg p-3', msg.role === 'user' ? 'bg-indigo-500 text-white' : 'bg-white text-gray-800 border border-gray-200']"
        >
          <!-- 用户消息：纯文本 -->
          <div v-if="msg.role === 'user'" class="user-message">
            {{ msg.content }}
          </div>
          
          <!-- Agent 消息：支持 Markdown 渲染 -->
          <div v-else class="agent-message markdown-content" v-html="renderMarkdown(msg.content)"></div>
          
          <div v-if="msg.jobId && ['HOMEWORK', 'EXAM','homework_materials'].includes(msg.jobType)" class="mt-2 border-t pt-2 text-xs text-gray-600">
            <p>{{ msg.jobType === 'homework_materials' ? '材料整理任务' : '批改任务' }}：{{ msg.jobId }}</p>
            <router-link v-if="msg.resultUrl" :to="msg.resultUrl" class="text-indigo-600 underline">{{ msg.jobType === 'homework_materials' ? '查看材料与后续批改' : '查看批改详情' }}</router-link>
          </div>
          <button v-if="msg.jobId && (!msg.jobType || msg.jobType === 'rpa_fetch_homework')" type="button"
            class="mt-2 text-sm text-indigo-600 underline" @click="emit('fetch-task', msg.jobId)">
            打开作业控制台
          </button>
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
    <div v-if="isAuthenticated" class="quick-commands border-t border-gray-200" aria-label="教学助手快捷入口">
      <div class="command-heading"><span>快捷入口</span><span>选择后补充信息，再发送</span></div>
      <div class="command-options" :class="{'single-option': !canManageHomework}">
        <button v-for="command in visibleCommands" :key="command.key" type="button"
          class="quick-cmd-btn" :class="{'command-selected': selectedEntry === command.key}"
          :aria-label="command.label" :aria-pressed="selectedEntry === command.key"
          :disabled="loading" @click="useQuickCommand(command.key)">
          <span class="command-icon"><AppIcon :name="command.icon" /></span>
          <span class="command-copy"><strong>{{ command.label }}</strong><span>{{ command.description }}</span></span>
        </button>
      </div>
      <div v-if="selectedCommand" class="command-guidance" role="status">
        <p>{{ selectedCommand.guidance }}</p>
        <button v-if="selectedEntry === 'grade_homework'" type="button" :disabled="loading"
          @click="useQuickCommand(selectedCommandKey === 'fetch_and_grade' ? 'grade_homework' : 'fetch_and_grade')">
          {{ selectedCommandKey === 'fetch_and_grade' ? '改用已有作业目录' : '改为拉取、解压并批改' }} <AppIcon name="arrow" />
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
          aria-label="消息内容"
          :placeholder="selectedCommand?.placeholder || '输入需求，或选择上方快捷入口'"
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
import AppIcon from './AppIcon.vue'
const emit = defineEmits(['fetch-task'])

// 消息数据
const messages = ref([])
const inputMessage = ref('')
const loading = ref(false)
const messagesContainer = ref(null)
const currentSessionId = ref('')
const sessions = ref([])
const showSessionDropdown = ref(false)
const inputRef = ref(null)
const selectedCommandKey = ref('')

// 快捷指令配置
const quickCommands = {
  fetch_homework: {
    key: 'fetch_homework', label: '拉取作业并解压', icon: 'download',
    description: '从教务系统获取，整理后暂不批改',
    template: '请从教务系统拉取“课程名称”的“作业名称”并解压，暂不批改。',
    selection: '课程名称',
    placeholder: '填写课程、作业名称；可补充解压保存目录',
    guidance: '提供课程与作业名称，可补充保存位置。登录在任务浏览器完成，整理结束后由你决定何时批改。'
  },
  grade_homework: {
    key: 'grade_homework', label: '批改作业', icon: 'pen',
    description: '使用已有材料，也可拉取后直接批改',
    template: '请批改“课程名称”的“作业名称”，作业目录是“目录路径”。',
    selection: '课程名称',
    placeholder: '填写所属作业与目录，也可说明使用刚才整理的材料',
    guidance: '提供所属作业和本机目录，或说明使用已整理的材料；按该作业配置的题目与评分标准批改。'
  },
  fetch_and_grade: {
    template: '请从教务系统拉取“课程名称”的“作业名称”，解压后直接批改。',
    selection: '课程名称',
    placeholder: '填写课程与作业名称，完成拉取、解压和批改全过程',
    guidance: '本次会在拉取、解压完成后继续批改；缺少题目或评分标准时会提醒补充。'
  },
  query_score: {
    key: 'query_score', label: '查询成绩', icon: 'chart',
    description: '按学号查看作业与试卷成绩',
    template: '请查询学号“学生学号”的作业和试卷成绩。',
    selection: '学生学号',
    placeholder: '填写学生学号，也可补充课程或作业名称',
    guidance: '提供学生学号，可补充课程、作业或试卷名称，缩小查询范围。'
  }
}

// 计算属性
const isAuthenticated = computed(() => authApi.isAuthenticated())
const currentUser = computed(() => authApi.getCurrentUser())
const hasUserMessages = computed(() => messages.value.some(message => message.role === 'user'))
const canManageHomework = computed(() => ['teacher','admin'].includes(currentUser.value?.role))
const visibleCommands = computed(() => canManageHomework.value
  ? [quickCommands.fetch_homework, quickCommands.grade_homework, quickCommands.query_score]
  : [{...quickCommands.query_score, label:'查询我的成绩', description:'查看自己的作业与试卷成绩'}])
const selectedEntry = computed(() => selectedCommandKey.value === 'fetch_and_grade' ? 'grade_homework' : selectedCommandKey.value)
const selectedCommand = computed(() => {
  if (!selectedCommandKey.value) return null
  if (!canManageHomework.value && selectedCommandKey.value === 'query_score') return {
    template:'请查询我的作业和试卷成绩。', placeholder:'可补充课程、作业或试卷名称', guidance:'只查询你本人的成绩，可补充课程或作业名称。'
  }
  return quickCommands[selectedCommandKey.value]
})
const welcomeText = computed(() => canManageHomework.value
  ? '您好！可以帮您拉取并解压作业、批改已有材料，或查询学生成绩。也可以一次完成拉取、解压和批改。'
  : '您好！可以帮您解答学习问题，或查询您本人的作业与试卷成绩。')

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
        content: welcomeText.value,
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
  if (!message || loading.value) return

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
    }, { headers: { 'Idempotency-Key': globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}` } })

    const data = response.data
    if (data.job_id && (!data.job_type || data.job_type === 'rpa_fetch_homework')) emit('fetch-task', data.job_id)

    // 添加 Agent 回复
    messages.value.push({
      role: 'assistant',
      content: data.reply || '抱歉，暂时无法回答您的问题。',
      jobId: data.job_id,
      jobType: data.job_type,
      resultUrl: /^(\/materials|\/(assignments|exams)\/\d+)$/.test(data.result_url || '') ? data.result_url : '',
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
  if (loading.value || (!canManageHomework.value && commandType !== 'query_score')) return
  selectedCommandKey.value = commandType
  const command = selectedCommand.value
  if (command) {
    inputMessage.value = command.template
    // 聚焦到输入框
    nextTick(() => {
      if (inputRef.value) {
        inputRef.value.focus()
        const start = command.selection ? command.template.indexOf(command.selection) : -1
        inputRef.value.setSelectionRange(start < 0 ? command.template.length : start, start < 0 ? command.template.length : start + command.selection.length)
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
.agent-chat-widget { border: 1px solid #e0e7ee; border-radius: 16px; background: #fff; overflow: hidden; box-shadow: 0 4px 20px rgb(20 45 64 / .035); }
.chat-header { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 21px 25px; background: #fff; color: #20384c; }
.chat-identity { display: flex; align-items: center; gap: 12px; }
.chat-symbol { display: grid; place-items: center; width: 40px; height: 40px; background: #edf7f5; border-radius: 12px; color: #197b70; flex-shrink: 0; }
.chat-symbol svg { width: 23px; height: 23px; }
.chat-header h2 { font-size: 15px; font-weight: 600; }
.chat-header p { color: #66788a; font-size: 11px; margin-top: 5px; }
.session-tag { font-size: 10px; padding: 5px 9px; border: 1px solid #e0e7ee; border-radius: 5px; color: #66788a; white-space: nowrap; }
.toolbar { padding: 11px 25px; background: #fff; border-top: 1px solid #eff3f7; gap: 12px; flex-wrap: wrap; }
.toolbar button { display: inline-flex; align-items: center; gap: 6px; min-height: 32px; padding: 6px 10px; border-color: #e0e7ee; font-size: 11px; border-radius: 6px; }
.toolbar button svg { width: 14px; height: 14px; color: #66788a; }
.toolbar > div:last-child { font-size: 10px; }
.messages-container { height: clamp(300px, 43vh, 520px); min-height: 300px; padding: 26px; background: #f7f9fb; }
.messages-container.messages-idle { height:200px; min-height:180px; }
.message-item { gap: 10px; align-items: flex-start; margin-bottom: 22px; }
.message-avatar { display: grid; place-items: center; width: 30px; height: 30px; flex-shrink: 0; border: 1px solid #e0e7ee; border-radius: 9px; color: #197b70; background: #fff; }
.message-avatar svg { width: 17px; height: 17px; }
.message-content { min-width: 0; max-width: 85%; padding: 15px 18px; border-radius: 0 12px 12px; font-size: 13px; line-height: 1.8; word-wrap: break-word; overflow-wrap: anywhere; }
.message-content.bg-indigo-500 { background: #197b70; border-radius: 12px 0 12px 12px; }
.message-content.bg-indigo-500 a, .message-content.bg-indigo-500 button { color: #d9eeea; }
.message-time { font-size: 9px; margin-top: 8px; }
.quick-commands { padding: 16px 25px; background: #fff; }
.command-heading { display:flex; justify-content:space-between; gap:12px; margin-bottom:11px; color:#66788a; font-size:10px; }
.command-options { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
.command-options.single-option { grid-template-columns:1fr; }
.quick-cmd-btn { display:flex; align-items:flex-start; gap:10px; min-width:0; border:1px solid #dce7e5; background:#fbfdfc; color:#14655d; padding:13px 12px; border-radius:9px; text-align:left; box-shadow:none; transition:background .15s,border-color .15s; }
.quick-cmd-btn:hover { background: #edf7f5; border-color: #83c4b9; }
.quick-cmd-btn:disabled { opacity: .5; }
.quick-cmd-btn:focus-visible { outline:2px solid #197b70; outline-offset:3px; }
.quick-cmd-btn.command-selected { background:#edf7f4; border-color:#5aab9f; }
.command-icon { display:grid; place-items:center; width:28px; height:28px; flex-shrink:0; background:#edf7f4; border-radius:7px; }
.quick-cmd-btn svg { width:16px; height:16px; }
.command-copy { min-width:0; }
.command-copy strong { display:block; font-size:12px; font-weight:600; line-height:1.6; }
.command-copy>span { display:block; margin-top:5px; color:#718499; font-size:10px; line-height:1.7; }
.command-guidance { display:flex; flex-wrap:wrap; align-items:center; gap:8px 16px; margin-top:12px; font-size:11px; line-height:1.8; color:#66788a; }
.command-guidance p { flex:1 1 300px; }
.command-guidance button { display:inline-flex; align-items:center; gap:5px; color:#197b70; white-space:nowrap; }
.command-guidance button svg { width:14px; height:14px; }
.command-guidance button:disabled { opacity:.5; }
.input-area { padding: 19px 25px; border-top-color: #eff3f7; }
.input-area input { min-width: 0; min-height: 46px; font-size: 12px; background: #f7f9fb; }
.input-area button[type="submit"] { padding: 10px 21px; font-size: 12px; }
.input-area > div:last-child { margin-top: 12px; font-size: 9px; }
.session-dropdown { z-index: 10; position: relative; border-radius: 8px; }
.welcome-message p { font-size: 13px; line-height: 1.9; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
.animate-pulse { animation: pulse 2s cubic-bezier(.4, 0, .6, 1) infinite; }
.delay-150 { animation-delay: 150ms; }
.delay-300 { animation-delay: 300ms; }
@media (max-width: 760px) {
  .chat-header, .toolbar, .quick-commands, .input-area { padding-left: 16px; padding-right: 16px; }
  .chat-header { flex-wrap: wrap; padding-top: 17px; padding-bottom: 17px; }
  .chat-header p { font-size: 10px; }
  .session-tag { margin-left: 52px; }
  .messages-container { padding: 20px 14px; height: 360px; }
  .messages-container.messages-idle { height:180px; min-height:180px; }
  .message-item { gap: 7px; }
  .message-content { max-width: calc(100% - 37px); padding: 12px; font-size: 12px; }
  .message-content.bg-indigo-500 { max-width: 90%; }
  .input-area button[type="submit"] { padding: 10px 15px; }
}
@media (max-width:600px) {
  .chat-header .session-tag { display:none; }
  .command-options { gap:7px; }
  .quick-cmd-btn { flex-direction:column; align-items:center; gap:8px; padding:12px 5px; text-align:center; }
  .command-copy strong { font-size:11px; }
  .command-copy>span { display:none; }
  .single-option .quick-cmd-btn { flex-direction:row; text-align:left; padding:12px; }
  .single-option .command-copy>span { display:block; }
  .command-guidance p { flex-basis:100%; }
}
</style>

<style>
/* Markdown 样式重置和美化 */
.markdown-content {
  min-width: 0;
  font-family: inherit;
  line-height: 1.8;
}

.markdown-content p {
  margin-bottom: 1rem;
}

.markdown-content strong,
.markdown-content b {
  font-weight: 600;
  color: #20384c;
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
  color: #142d40;
}

.markdown-content h1 {
  font-size: 1.875rem;
  border-bottom: 2px solid #e0e7ee;
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
  border-left: 4px solid #e0e7ee;
  padding-left: 1rem;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1rem;
  color: #66788a;
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
  background-color: #20384c;
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
  display: block;
  overflow-x: auto;
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1rem;
  font-size: 0.875rem;
}

.markdown-content th,
.markdown-content td {
  border: 1px solid #e0e7ee;
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

.markdown-content p:last-child { margin-bottom: 0; }

.markdown-content a {
  color: #197b70;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.markdown-content a:hover {
  color: #14655d;
}

.markdown-content hr {
  border: 0;
  height: 1px;
  background-color: #e0e7ee;
  margin: 1.5rem 0;
}
</style>
