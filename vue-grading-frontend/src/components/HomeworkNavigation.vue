<template>
  <nav class="homework-navigation" aria-label="作业批改导航">
    <router-link
      v-for="item in homeworkNavigation"
      :key="item.path"
      :to="item.path"
      :class="{
        selected:
          route.path === item.path ||
          (item.path === '/assignments' && route.path.startsWith('/assignments/')),
      }"
      >{{ item.name }}</router-link
    >
    <details class="workspace-more"
      ><summary>更多设置</summary
      ><div>
        <router-link to="/tools-admin">助手工具配置</router-link>
        <router-link v-if="admin" to="/logs">运行日志</router-link>
      </div></details
    >
  </nav>
</template>
<script setup>
import { useRoute } from 'vue-router';
import authApi from '../services/authApi';
import { homeworkNavigation } from '../services/teacherWorkspace';
const route = useRoute();
const admin = authApi.getCurrentUser()?.role === 'admin';
</script>
<style scoped>
.homework-navigation {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 24px;
  border-bottom: 1px solid #dce5e9;
  padding-bottom: 14px;
}
.homework-navigation > a,
summary {
  padding: 9px 14px;
  border-radius: 8px;
  font-size: 13px;
  color: #4c6072;
  cursor: pointer;
}
.homework-navigation .selected {
  background: #eaf4f2;
  color: #14655d;
  font-weight: 600;
}
.workspace-more {
  margin-left: auto;
  position: relative;
}
.workspace-more div {
  position: absolute;
  right: 0;
  z-index: 20;
  min-width: 150px;
  display: grid;
  gap: 8px;
  background: white;
  padding: 14px;
  border: 1px solid #dce5e9;
  border-radius: 10px;
  box-shadow: 0 8px 24px #142d4014;
}
.workspace-more a {
  white-space: nowrap;
  font-size: 13px;
  padding: 5px;
}
@media (max-width: 600px) {
  .homework-navigation {
    gap: 2px;
  }
  .homework-navigation > a,
  summary {
    padding: 8px;
    font-size: 12px;
  }
}
</style>
