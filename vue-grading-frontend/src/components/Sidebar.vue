<template>
  <div class="flex flex-col w-64 bg-gray-800 text-gray-100 h-screen shadow-xl">
    <!-- 主标题和用户信息 -->
    <div class="h-16 flex items-center justify-between px-4 shadow-md">
      <h1 class="text-lg font-bold text-white truncate">智能评分系统</h1>
      <div v-if="currentUser" class="flex items-center space-x-2">
        <div class="w-8 h-8 bg-indigo-500 rounded-full flex items-center justify-center text-xs font-bold">
          {{ currentUser.name ? currentUser.name.charAt(0).toUpperCase() : 'U' }}
        </div>
      </div>
    </div>

    <!-- 用户信息面板 -->
    <div v-if="currentUser" class="px-4 py-3 border-t border-gray-700">
      <div class="flex items-center justify-between">
        <div>
          <p class="text-sm font-medium truncate">{{ currentUser.name || currentUser.username }}</p>
          <p class="text-xs text-gray-400 capitalize">{{ getRoleName(currentUser.role) }}</p>
        </div>
      </div>
    </div>

    <!-- 导航菜单 -->
    <nav class="flex-1 mt-2 px-2 space-y-1">
      <router-link
        v-for="item in filteredMenuItems"
        :key="item.name"
        :to="item.path"
        class="flex items-center px-3 py-3 text-sm font-medium rounded-md transition-colors duration-150"
        :class="{
          'bg-gray-900 text-white': $route.path === item.path,
          'hover:bg-gray-700 hover:text-white': $route.path !== item.path
        }"
      >
        <component :is="item.icon" class="h-5 w-5 mr-3 flex-shrink-0" aria-hidden="true" />
        <span class="truncate">{{ item.name }}</span>
      </router-link>
    </nav>

    <!-- 退出登录按钮 -->
    <div class="mt-auto p-4 border-t border-gray-700">
      <button
        @click="handleLogout"
        class="w-full flex items-center justify-center px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-md transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-800"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
        </svg>
        退出登录
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from "vue";
import { useRouter, useRoute } from "vue-router";
import authApi from "../services/authApi";

const router = useRouter();
const route = useRoute();

// 获取当前用户信息
const currentUser = computed(() => authApi.getCurrentUser());

// 获取角色名称
const getRoleName = (role) => {
  const roleMap = {
    'student': '学生',
    'teacher': '教师',
    'admin': '管理员'
  }
  return roleMap[role] || '用户'
}

// 图标组件
const HomeIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M2.25 12l8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25" /></svg>`,
};

const MagnifyingGlassIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-5.4-5.4m2.7-4.1A7.5 7.5 0 1111.1 3.6a7.5 7.5 0 017.2 7.5z" /></svg>`,
};

const CheckBadgeIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12c0 1.268-.63 2.4-1.683 3.036M15.75 9c0 2.485-2.015 4.5-4.5 4.5S6.75 11.485 6.75 9c0-2.485 2.015-4.5 4.5-4.5S15.75 6.515 15.75 9c0 .825-.223 1.58-.62 2.22m-1.121-3.662a9.046 9.046 0 015.13 5.13m0 0a8.997 8.997 0 01-5.13 8.28m-5.13 0a8.997 8.997 0 01-8.28-5.13m0 0c-.114-.378-.23-.75-.353-1.122m1.121 3.662a8.997 8.997 0 01-5.13-8.28m5.13 0a8.997 8.997 0 018.28-5.13M3.75 9c0 2.485 2.015 4.5 4.5 4.5S12.75 11.485 12.75 9c0-2.485-2.015-4.5-4.5-4.5S3.75 6.515 3.75 9z" /></svg>`,
};

const PencilSquareIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" /></svg>`,
};

const ChatBubbleLeftRightIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193-.34.027-.68.052-1.02.072v3.091l-3-3c-1.354 0-2.694-.055-4.02-.163a2.115 2.115 0 01-.825-.242m9.345-8.334a2.126 2.126 0 00-.476-.095 48.64 48.64 0 00-8.048 0c-1.131.094-1.976 1.057-1.976 2.192v4.286c0 .837.46 1.58 1.155 1.951m9.345-8.334V6.637c0-1.621-1.422-2.87-3.14-2.902a48.78 48.78 0 00-8.66 0C3.422 3.767 2 5.016 2 6.637v4.333c0 1.621 1.422 2.87 3.14 2.902.835.022 1.67.045 2.505.072v-.592c0-.906.672-1.654 1.562-1.654h3.386c.89 0 1.562.748 1.562 1.654v.592c.835-.027 1.67-.05 2.505-.072 1.718-.032 3.14-1.281 3.14-2.902V8.511z" /></svg>`,
};

const WrenchScrewdriverIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M11.42 2.25a.75.75 0 01.75.75v1.068a6.72 6.72 0 012.037.842l.756-.756a.75.75 0 011.06 0l1.768 1.768a.75.75 0 010 1.06l-.756.756c.37.646.65 1.36.842 2.037H21a.75.75 0 01.75.75v2.5A.75.75 0 0121 13.75h-1.068a6.72 6.72 0 01-.842 2.037l.756.756a.75.75 0 010 1.06l-1.768 1.768a.75.75 0 01-1.06 0l-.756-.756a6.72 6.72 0 01-2.037.842V21a.75.75 0 01-.75.75h-2.5A.75.75 0 019.25 21v-1.068a6.72 6.72 0 01-2.037-.842l-.756.756a.75.75 0 01-1.06 0L3.63 17.834a.75.75 0 010-1.06l.756-.756a6.72 6.72 0 01-.842-2.037H2.5a.75.75 0 01-.75-.75v-2.5A.75.75 0 012.5 9.75h1.068c.192-.677.472-1.391.842-2.037l-.756-.756a.75.75 0 010-1.06l1.768-1.768a.75.75 0 011.06 0l.756.756a6.72 6.72 0 012.037-.842V3a.75.75 0 01.75-.75h2.5zM10.5 12a2.25 2.25 0 104.5 0 2.25 2.25 0 00-4.5 0z" /></svg>`,
}

// 完整菜单项
const allMenuItems = [
  { name: "主页", path: "/home", icon: HomeIcon },
  { name: "作业自动查重", path: "/assignments", icon: MagnifyingGlassIcon },
  { name: "作业自动评分", path: "/assignments", icon: CheckBadgeIcon },
  { name: "主观题自动评分", path: "/exams", icon: PencilSquareIcon },
  { name: "AI教学助手", path: "/ai-assistant", icon: ChatBubbleLeftRightIcon },
  { name: "Skills 管理", path: "/skills-admin", icon: WrenchScrewdriverIcon },
]

// 根据角色过滤菜单项
const filteredMenuItems = computed(() => {
  const userRole = currentUser.value?.role || 'student'
  
  // 学生只能看到AI教学助手
  if (userRole === 'student') {
    return allMenuItems.filter(item => item.path === '/ai-assistant')
  }
  
  // 教师/管理员可见；其中 Skills 管理建议仅管理员可见
  if (userRole === 'teacher') {
    return allMenuItems.filter(item => item.path !== '/skills-admin')
  }
  return allMenuItems
})

// 处理退出登录
const handleLogout = async () => {
  try {
    authApi.logout();
    router.push('/login');
    alert('已成功退出登录');
  } catch (error) {
    console.error('退出登录失败:', error);
    alert('退出登录失败，请重试');
  }
};
</script>

<style scoped>
.router-link-exact-active {
  background-color: #1f2937;
  color: #ffffff;
}
</style>