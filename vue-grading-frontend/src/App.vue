<template>
  <div class="min-h-screen flex flex-col">
    <AppHeader />
    <main class="flex-grow container mx-auto p-4 md:p-8">
      <!-- 路由视图：根据URL显示不同的页面组件 -->
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { RouterView } from 'vue-router'
import AppHeader from './components/AppHeader.vue'
</script>