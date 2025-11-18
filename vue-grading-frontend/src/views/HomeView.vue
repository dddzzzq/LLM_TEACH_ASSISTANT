<template>
  <div
    class="h-full flex flex-col justify-center items-center bg-white rounded-lg shadow-xl p-8 md:p-12"
  >
    <h1 class="text-4xl md:text-5xl font-bold text-gray-800 mb-8 tracking-wide">
      欢迎使用作业&amp;试题智能评分系统
    </h1>

    <p class="text-xl text-gray-600 mb-12">请使用左侧菜单栏选择您需要的功能</p>

    <div class="mb-12 transform transition hover:scale-105 duration-300">
      <img :src="logo" alt="西电logo" class="h-48 w-auto mx-auto object-contain" />
    </div>

    <div class="max-w-3xl text-center">
      <p class="text-gray-500 text-lg leading-loose">
        本系统基于大语言模型（LLM）技术构建，<br />
        旨在为教师提供高效的
        <span class="font-bold text-indigo-600">作业自动查重</span>、
        <span class="font-bold text-indigo-600">作业智能评分</span>
        以及
        <span class="font-bold text-indigo-600">主观题自动批改</span>
        辅助功能。
      </p>
    </div>
  </div>
</template>

<script setup>
// 导入本地 logo 图片
import logo from "@/assets/logo.png";
</script>
