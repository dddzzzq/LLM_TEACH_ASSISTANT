<template>
  <div class="flex h-screen bg-gray-100">
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
import { RouterView } from "vue-router";
import Sidebar from "./components/Sidebar.vue"; // 导入新的侧边栏组件
</script>
