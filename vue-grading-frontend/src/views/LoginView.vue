<template>
  <div class="login-view min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
    <div class="sm:mx-auto sm:w-full sm:max-w-md">
      <div class="flex justify-center">
        <div class="w-16 h-16 bg-indigo-600 rounded-full flex items-center justify-center">
          <span class="text-white text-2xl font-bold">AI</span>
        </div>
      </div>
      <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900">
        智能作业批改系统
      </h2>
      <p class="mt-2 text-center text-sm text-gray-600">
        基于 JWT + Redis 的安全认证系统
      </p>
    </div>

    <div class="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
      <div class="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
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

      <div class="mt-6 text-center text-sm text-gray-600">
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
.login-view {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.bg-white {
  background-color: rgba(255, 255, 255, 0.95);
}

input, select {
  transition: all 0.2s;
}

input:focus, select:focus {
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}
</style>