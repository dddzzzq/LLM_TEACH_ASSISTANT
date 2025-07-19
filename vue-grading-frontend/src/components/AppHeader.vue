<template>
  <header class="bg-white shadow-md">
    <nav class="container mx-auto px-4 md:px-8 py-4 flex justify-between items-center">
      <router-link to="/" class="text-xl font-bold text-indigo-600">
        AI助教系统
      </router-link>
      <div class="space-x-4">
        <!-- <router-link to="/grade-homework" class="text-gray-600 hover:text-indigo-600">作业批改</router-link> -->
        <router-link to="/grade-exam" class="text-gray-600 hover:text-indigo-600">试卷批改</router-link>
        <router-link to="/assignments" class="text-gray-600 hover:text-indigo-600">作业列表</router-link>
        <router-link to="/assignments/new" class="text-gray-600 hover:text-indigo-600">新建作业</router-link>
      </div>
    </nav>
  </header>
</template>

<script setup>
import { RouterLink } from 'vue-router'
</script>