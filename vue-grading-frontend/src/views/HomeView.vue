<template>
  <div class="text-center">
    <h1 class="text-4xl font-bold text-gray-800 mb-4">欢迎使用智能化作业试卷批改助手</h1>
    <p class="text-lg text-gray-600 mb-8">请选择您需要使用的功能</p>
    <div class="flex justify-center gap-8">
      <router-link to="/grade-exam" class="bg-teal-500 text-white font-bold py-4 px-8 rounded-lg shadow-lg hover:bg-teal-600 transition-colors">
        试卷批改(等待开发中...)
      </router-link>
      <router-link to="/assignments" class="bg-indigo-600 text-white font-bold py-4 px-8 rounded-lg shadow-lg hover:bg-indigo-700 transition-colors text-lg">
        作业批改
      </router-link>
    </div>
  </div>
</template>

<script setup>

import { RouterLink } from 'vue-router';
</script>