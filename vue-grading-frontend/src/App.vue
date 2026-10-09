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
          <span class="breadcrumb-current">{{ {
            home: '教学概览', assignments: '作业管理', 'create-assignment': '新建作业',
            'assignment-detail': '作业详情', exams: '试卷管理', 'create-exam': '新建试卷',
            'exam-detail': '试卷详情', 'student-report': '学生报告',
            'ai-assistant': 'AI 教学助手', 'skills-admin': '工具管理', 'grade-homework': '作业批改'
          }[route.name] || '教学概览' }}</span>
        </div>
        <span class="workspace-caption">让技术辅助教学，让教师专注育人</span>
      </header>
      <main class="workspace-content">
        <!-- 路由视图：根据URL显示不同的页面组件 -->
        <div class="page-content"><router-view /></div>
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
