<template>
  <!-- 登录页面不使用侧边栏布局 -->
  <div v-if="isLoginPage" class="h-screen">
    <router-view />
  </div>
  
  <!-- 其他页面使用带侧边栏的布局 -->
  <div v-else class="flex h-screen bg-gray-100">
    <!-- 侧边栏 -->
    <Sidebar />

    <!-- 主内容区 -->
    <div class="flex-1 flex flex-col overflow-hidden">
      <main class="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100 p-4 md:p-8">
        <!-- 路由视图：根据URL显示不同的页面组件 -->
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import Sidebar from "./components/Sidebar.vue";

const route = useRoute();

// 计算属性：判断当前是否为登录页面
const isLoginPage = computed(() => {
  return route.path === '/login' || route.name === 'login';
});
</script>