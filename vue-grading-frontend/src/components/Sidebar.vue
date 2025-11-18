<template>
  <div class="flex flex-col w-64 bg-gray-800 text-gray-100 h-screen shadow-xl">
    <!-- 主标题 -->
    <div class="h-16 flex items-center justify-center px-4 shadow-md">
      <h1 class="text-lg font-bold text-white">作业&试题智能评分系统</h1>
    </div>

    <!-- 导航菜单 -->
    <nav class="flex-1 mt-4 px-2 space-y-2">
      <router-link
        v-for="item in menuItems"
        :key="item.name"
        :to="item.path"
        class="flex items-center px-3 py-3 text-sm font-medium rounded-md transition-colors duration-150"
        active-class="bg-gray-900 text-white"
        hover-class="bg-gray-700 text-white"
      >
        <component :is="item.icon" class="h-5 w-5 mr-3" aria-hidden="true" />
        {{ item.name }}
      </router-link>
    </nav>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { RouterLink } from "vue-router";

// 简易的内联 SVG 图标
const HomeIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M2.25 12l8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25" /></svg>`,
};
const DocumentTextIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m-1.125 0H6.75A2.25 2.25 0 004.5 4.5v15A2.25 2.25 0 006.75 21.75h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75m0 0l3.03-3.03m-3.03 3.03l-3.03-3.03m3.03 3.03L9.78 4.47m-3.03 0l3.03 3.03m-3.03-3.03L3.72 4.47m3.03 0L9.78 7.5M6.75 21.75v-4.125a3.375 3.375 0 013.375-3.375h2.25a3.375 3.375 0 013.375 3.375v4.125" /></svg>`,
};
const PencilSquareIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" /></svg>`,
};
// 搜索和评分图标
const MagnifyingGlassIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-5.4-5.4m2.7-4.1A7.5 7.5 0 1111.1 3.6a7.5 7.5 0 017.2 7.5z" /></svg>`,
};
const CheckBadgeIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12c0 1.268-.63 2.4-1.683 3.036M15.75 9c0 2.485-2.015 4.5-4.5 4.5S6.75 11.485 6.75 9c0-2.485 2.015-4.5 4.5-4.5S15.75 6.515 15.75 9c0 .825-.223 1.58-.62 2.22m-1.121-3.662a9.046 9.046 0 015.13 5.13m0 0a8.997 8.997 0 01-5.13 8.28m-5.13 0a8.997 8.997 0 01-8.28-5.13m0 0c-.114-.378-.23-.75-.353-1.122m1.121 3.662a8.997 8.997 0 01-5.13-8.28m5.13 0a8.997 8.997 0 018.28-5.13M3.75 9c0 2.485 2.015 4.5 4.5 4.5S12.75 11.485 12.75 9c0-2.485-2.015-4.5-4.5-4.5S3.75 6.515 3.75 9z" /></svg>`,
};

const menuItems = ref([
  { name: "主页", path: "/home", icon: HomeIcon },
  { name: "作业自动查重", path: "/assignments", icon: MagnifyingGlassIcon },
  { name: "作业自动评分", path: "/assignments", icon: CheckBadgeIcon },
  { name: "主观题自动评分", path: "/exams", icon: PencilSquareIcon },
]);
</script>

<style scoped>
/* 自定义 active-class 和 hover-class 的行为 */
.router-link-active {
  background-color: #1f2937; /* bg-gray-900 */
  color: #ffffff; /* text-white */
}
a:hover {
  background-color: #374151; /* bg-gray-700 */
  color: #ffffff; /* text-white */
}
</style>
