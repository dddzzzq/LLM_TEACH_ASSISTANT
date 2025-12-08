<template>
  <header class="bg-white shadow-md">
    <nav class="container mx-auto px-4 md:px-8 py-4 flex justify-between items-center">
      <router-link to="/" class="text-xl font-bold text-indigo-600">
        作业&amp;试题智能评分系统
      </router-link>
      <div class="space-x-4 md:space-x-6">
        <router-link
          to="/assignments"
          class="text-gray-600 hover:text-indigo-600"
          active-class="text-indigo-700 font-semibold"
        >
          作业自动查重
        </router-link>
        <router-link
          to="/assignments"
          class="text-gray-600 hover:text-indigo-600"
          active-class="text-indigo-700 font-semibold"
        >
          作业自动评分
        </router-link>
        <router-link
          to="/assignments"
          class="text-gray-600 hover:text-indigo-600"
          active-class="text-indigo-700 font-semibold"
        >
          主观题自动评分
        </router-link>
      </div>
    </nav>
  </header>
</template>

<script setup>
import { RouterLink } from "vue-router";
</script>

<style scoped>
/* 简单的 active-class 样式 */
.text-indigo-700 {
  --tw-text-opacity: 1;
  color: rgba(67, 56, 202, var(--tw-text-opacity));
}
.font-semibold {
  font-weight: 600;
}
</style>
