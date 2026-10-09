<template>
  <div class="login-view">
    <section class="login-story" aria-labelledby="login-story-title">
      <div class="login-brand"><span><AppIcon name="book" /></span><div>智能教学助理<small>教学有方，批改有助</small></div></div>
      <div class="login-story-copy">
        <div class="story-emblem" aria-hidden="true"><AppIcon name="book" /><span class="emblem-spark"><AppIcon name="sparkles" /></span></div>
        <h1 id="login-story-title">把时间留给<br />更有价值的教学。</h1>
        <p>从一份作业到一次考试，<br />为每一次认真教学，提供恰到好处的帮助。</p>
        <ul class="login-capabilities">
          <li><AppIcon name="check" /><span>作业查重与智能评分</span></li>
          <li><AppIcon name="pen" /><span>答卷识别与逐题反馈</span></li>
          <li><AppIcon name="chat" /><span>教务作业下载与成绩查询</span></li>
        </ul>
      </div>
      <p class="login-story-footer">大语言模型辅助教学 · 教师把关每一份评价</p>
    </section>
    <div class="login-main">
      <div class="login-heading">
        <span class="login-heading-icon"><AppIcon name="book" /></span>
        <h2>{{ showRegister ? '创建您的账户' : '欢迎回来' }}</h2>
        <p>{{ showRegister ? '填写账户信息，开启教学工作台。' : '登录智能作业批改系统，开始今天的工作。' }}</p>
      </div>
      <div class="login-card">
        <!-- 登录表单 -->
        <div v-if="!showRegister">
          <form class="space-y-6" @submit.prevent="handleLogin">
            <div>
              <label for="username" class="block text-sm font-medium text-gray-700">
                用户名 / 学号
              </label>
              <div class="mt-1">
                <input
                  id="username"
                  v-model="loginForm.username"
                  name="username"
                  type="text"
                  required
                  class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="请输入用户名"
                />
              </div>
            </div>

            <div>
              <label for="password" class="block text-sm font-medium text-gray-700">
                密码
              </label>
              <div class="mt-1">
                <input
                  id="password"
                  v-model="loginForm.password"
                  name="password"
                  type="password"
                  required
                  class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="请输入密码"
                />
              </div>
            </div>

            <div class="flex items-center justify-between">
              <div class="flex items-center">
                <input
                  id="remember-me"
                  v-model="rememberMe"
                  name="remember-me"
                  type="checkbox"
                  class="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                />
                <label for="remember-me" class="ml-2 block text-sm text-gray-900">
                  记住我
                </label>
              </div>
            </div>

            <div>
              <button
                type="submit"
                :disabled="loading"
                class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {{ loading ? '登录中...' : '登录' }}
              </button>
            </div>
          </form>

          <div class="mt-6">
            <div class="relative">
              <div class="absolute inset-0 flex items-center">
                <div class="w-full border-t border-gray-300"></div>
              </div>
              <div class="relative flex justify-center text-sm">
                <span class="px-2 bg-white text-gray-500">
                  或
                </span>
              </div>
            </div>

            <div class="mt-6">
              <button
                @click="showRegister = true"
                class="w-full flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                注册新账户
              </button>
            </div>
          </div>
        </div>

        <!-- 注册表单 -->
        <div v-else>
          <form class="space-y-6" @submit.prevent="handleRegister">
            <div>
              <label for="reg-username" class="block text-sm font-medium text-gray-700">
                用户名 / 学号
              </label>
              <div class="mt-1">
                <input
                  id="reg-username"
                  v-model="registerForm.username"
                  name="username"
                  type="text"
                  required
                  class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="请输入用户名"
                />
              </div>
            </div>

            <div>
              <label for="reg-password" class="block text-sm font-medium text-gray-700">
                密码
              </label>
              <div class="mt-1">
                <input
                  id="reg-password"
                  v-model="registerForm.password"
                  name="password"
                  type="password"
                  required
                  class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="请输入密码"
                />
              </div>
            </div>

            <div>
              <label for="reg-name" class="block text-sm font-medium text-gray-700">
                姓名
              </label>
              <div class="mt-1">
                <input
                  id="reg-name"
                  v-model="registerForm.name"
                  name="name"
                  type="text"
                  required
                  class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="请输入真实姓名"
                />
              </div>
            </div>

            <div>
              <label for="reg-role" class="block text-sm font-medium text-gray-700">
                角色
              </label>
              <div class="mt-1">
                <select
                  id="reg-role"
                  v-model="registerForm.role"
                  name="role"
                  required
                  class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                >
                  <option value="">请选择角色</option>
                  <option value="student">学生</option>
                  <option value="teacher">教师</option>
                  <option value="admin">管理员</option>
                </select>
              </div>
            </div>

            <div class="flex space-x-4">
              <button
                type="button"
                @click="showRegister = false"
                class="flex-1 py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                返回登录
              </button>
              <button
                type="submit"
                :disabled="loading"
                class="flex-1 flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {{ loading ? '注册中...' : '注册' }}
              </button>
            </div>
          </form>
        </div>

        <!-- 错误提示 -->
        <div v-if="errorMessage" class="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <p class="text-sm text-red-700">{{ errorMessage }}</p>
            </div>
          </div>
        </div>

        <!-- 成功提示 -->
        <div v-if="successMessage" class="mt-4 p-3 bg-green-50 border border-green-200 rounded-md">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <p class="text-sm text-green-700">{{ successMessage }}</p>
            </div>
          </div>
        </div>
      </div>

      <div class="test-accounts">
        <p>测试账户：</p>
        <p class="mt-1 text-xs">
          学生：student / student123<br>
          教师：teacher / teacher123<br>
          管理员：admin / admin123
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import authApi from '../services/authApi'
import AppIcon from '../components/AppIcon.vue'

const router = useRouter()

// 表单数据
const loginForm = reactive({
  username: '',
  password: ''
})

const registerForm = reactive({
  username: '',
  password: '',
  name: '',
  role: 'student'
})

// 状态
const loading = ref(false)
const rememberMe = ref(true)
const showRegister = ref(false)
const errorMessage = ref('')
const successMessage = ref('')

// 检查是否已登录
onMounted(() => {
  if (authApi.isAuthenticated()) {
    // 如果已登录，重定向到首页
    router.push('/')
  }
})

// 处理登录
const handleLogin = async () => {
  errorMessage.value = ''
  successMessage.value = ''
  loading.value = true

  try {
    const response = await authApi.login(loginForm)
    
    if (response.data) {
      const { access_token, refresh_token, user_id, role, name } = response.data
      
      // 存储用户信息和 Token
      authApi.setUserInfo({
        user_id,
        role,
        name,
        username: loginForm.username,
        access_token,
        refresh_token
      })

      // 初始化认证
      authApi.init()

      // 显示成功消息
      successMessage.value = `欢迎回来，${name}！`

      // 根据角色进行路由分流
      let redirectPath = '/';
      if (role === 'student') {
        redirectPath = '/student/exams';
      } else if (role === 'teacher') {
        redirectPath = '/teacher/dashboard';
      } else if (role === 'admin') {
        redirectPath = '/admin/dashboard';
      }
      
      // 延迟跳转
      setTimeout(() => {
        router.push(redirectPath);
      }, 1000)
    }
  } catch (error) {
    console.error('登录失败:', error)
    errorMessage.value = error.response?.data?.error || '登录失败，请检查用户名和密码'
  } finally {
    loading.value = false
  }
}

// 处理注册
const handleRegister = async () => {
  errorMessage.value = ''
  successMessage.value = ''
  loading.value = true

  try {
    const response = await authApi.register(registerForm)
    
    if (response.data) {
      successMessage.value = '注册成功！已自动登录。'
      
      // 自动登录
      const { access_token, refresh_token, user_id, role, name } = response.data
      
      authApi.setUserInfo({
        user_id,
        role,
        name,
        username: registerForm.username,
        access_token,
        refresh_token
      })

      authApi.init()

      // 延迟跳转
      setTimeout(() => {
        router.push('/')
      }, 1500)
    }
  } catch (error) {
    console.error('注册失败:', error)
    errorMessage.value = error.response?.data?.error || '注册失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

// 快速填充测试账户
const fillTestAccount = (role) => {
  const testAccounts = {
    student: { username: 'student', password: 'student123', name: '测试学生', role: 'student' },
    teacher: { username: 'teacher', password: 'teacher123', name: '测试教师', role: 'teacher' },
    admin: { username: 'admin', password: 'admin123', name: '测试管理员', role: 'admin' }
  }

  if (showRegister.value) {
    Object.assign(registerForm, testAccounts[role])
  } else {
    Object.assign(loginForm, { username: testAccounts[role].username, password: testAccounts[role].password })
  }
}
</script>

<style scoped>
.login-view { display: grid; grid-template-columns: 1fr 1fr; min-height: 100vh; min-height: 100dvh; background: #fff; }
.login-story { position: relative; display: flex; flex-direction: column; padding: 42px 54px 30px; background: #153f3e; color: #fff; overflow: hidden; }
.login-story::after { content: ''; position: absolute; width: 550px; height: 550px; right: -310px; bottom: -280px; border: 1px solid #2e5957; border-radius: 50%; box-shadow: 0 0 0 70px rgb(255 255 255 / .015), 0 0 0 140px rgb(255 255 255 / .015); pointer-events: none; }
.login-brand { display: flex; align-items: center; gap: 12px; font-size: 17px; font-weight: 600; }
.login-brand > span { display: grid; place-items: center; width: 42px; height: 42px; border: 1px solid #547673; border-radius: 11px; }
.login-brand svg { width: 24px; height: 24px; }
.login-brand small { display: block; color: #b8cfcb; font-size: 10px; font-weight: 400; margin-top: 5px; }
.login-story-copy { position: relative; z-index: 1; margin: auto 0; padding: 55px 0; }
.story-emblem { position: relative; display: grid; place-items: center; width: 82px; height: 82px; border: 1px solid #547673; border-radius: 22px; background: #204a48; margin-bottom: 32px; }
.story-emblem > svg { width: 44px; height: 44px; color: #d4e7df; }
.emblem-spark { position: absolute; right: -12px; top: -10px; display: grid; place-items: center; width: 34px; height: 34px; background: #d8e9da; border: 4px solid #153f3e; border-radius: 11px; color: #197b70; }
.emblem-spark svg { width: 17px; height: 17px; }
.login-story h1 { font-size: clamp(32px, 3.2vw, 48px); letter-spacing: -1.5px; font-weight: 600; line-height: 1.5; }
.login-story-copy > p { color: #c0d4cf; font-size: 13px; line-height: 1.9; margin-top: 23px; }
.login-capabilities { display: flex; flex-direction: column; gap: 18px; margin-top: 40px; }
.login-capabilities li { display: flex; align-items: center; gap: 11px; font-size: 12px; color: #d4e5e0; }
.login-capabilities svg { width: 18px; height: 18px; color: #95c5b8; }
.login-story-footer { position: relative; z-index: 1; color: #b8cfcb; font-size: 10px; }
.login-main { display: flex; flex-direction: column; justify-content: center; width: 100%; max-width: 490px; padding: 54px; margin: 0 auto; }
.login-heading h2 { color: #142d40; font-size: 29px; font-weight: 700; letter-spacing: -.7px; }
.login-heading > p { color: #66788a; font-size: 12px; line-height: 1.8; margin-top: 11px; }
.login-heading-icon { display: none; }
.login-card { margin-top: 32px; }
.login-card input:not([type="checkbox"]), .login-card select { min-height: 46px; padding: 11px 13px; font-size: 13px; box-shadow: none; }
.login-card label { font-size: 12px; }
.login-card button { min-height: 43px; box-shadow: none; font-size: 12px; }
.login-card input[type="checkbox"] { width: 14px; height: 14px; }
.test-accounts { border-top: 1px solid #e0e7ee; margin-top: 27px; padding-top: 18px; color: #66788a; font-size: 11px; line-height: 1.8; }
.test-accounts p + p { font-size: 10px; line-height: 1.9; }
@media (min-width: 1600px) { .login-story { padding-left: max(54px, calc((100vw - 1400px) / 2)); } }
@media (max-width: 1000px) { .login-story { padding: 32px; } .login-main { padding: 38px; } .login-story h1 { font-size: 34px; } }
@media (max-width: 760px) {
  .login-view { grid-template-columns: 1fr; }
  .login-story { padding: 22px 24px 27px; }
  .login-brand { font-size: 15px; }
  .login-brand > span { width: 35px; height: 35px; }
  .login-story-copy { margin: 0; padding: 24px 0 0; }
  .story-emblem, .login-capabilities, .login-story-footer { display: none; }
  .login-story h1 { font-size: 27px; letter-spacing: -.6px; line-height: 1.45; }
  .login-story h1 br { display: none; }
  .login-story-copy > p { font-size: 11px; margin-top: 12px; }
  .login-story-copy > p br { display: none; }
  .login-main { max-width: 470px; padding: 30px 26px; }
  .login-heading h2 { font-size: 25px; }
  .login-card { margin-top: 25px; }
}
</style>
