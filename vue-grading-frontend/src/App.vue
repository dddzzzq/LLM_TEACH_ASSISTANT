<template>
  <!-- 登录页面不使用侧边栏布局 -->
  <div v-if="isLoginPage" class="login-shell">
    <router-view />
  </div>

  <!-- 其他页面使用带侧边栏的布局 -->
  <div v-else class="app-shell">
    <!-- 侧边栏 -->
    <Sidebar />

    <!-- 主内容区 -->
    <div class="workspace">
      <header class="workspace-header">
        <div class="workspace-breadcrumb">
          <span>教学工作台</span><span class="breadcrumb-divider" aria-hidden="true">/</span>
          <span class="breadcrumb-current">{{
            {
              home: '作业工作台',
              assignments: '作业工作台',
              'create-assignment': '新建作业',
              'assignment-detail': '作业详情',
              'ai-assistant': 'AI 助手',
              'grade-homework': '导入与自动批改',
              logs: '日志查阅',
              materials: '作业材料',
              'portal-settings': '教务账号',
            }[route.name] || '作业工作台'
          }}</span>
        </div>
        <router-link class="workspace-caption" to="/settings/portal">教务连接</router-link>
      </header>
      <main class="workspace-content">
        <div class="page-content"><router-view /></div>
      </main>
    </div>
    <RpaAttentionCenter />
    <FloatingAssistant :key="assistantOwner" />
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import Sidebar from './components/Sidebar.vue';
import RpaAttentionCenter from './components/RpaAttentionCenter.vue';
import FloatingAssistant from './components/FloatingAssistant.vue';
import authApi from './services/authApi';

const route = useRoute();
const assistantOwner = computed(() => {
  // Re-evaluate on navigation, keeping one instance for the same signed-in teacher.
  void route.path;
  const user = authApi.getCurrentUser();
  return user?.user_id || user?.id || user?.username || 'teacher';
});

// 计算属性：判断当前是否为登录页面
const isLoginPage = computed(() => {
  return route.path === '/login' || route.name === 'login';
});
</script>
