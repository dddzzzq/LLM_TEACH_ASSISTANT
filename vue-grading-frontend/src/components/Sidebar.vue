<template>
  <aside class="app-sidebar" aria-label="主导航">
    <div class="sidebar-brand">
      <div class="brand-symbol"><AppIcon name="book" /></div>
      <div><h1>智能教学助理</h1><p>教学有方，批改有助</p></div>
    </div>
    <div class="sidebar-nav-label">工作空间</div>
    <nav class="sidebar-navigation">
      <router-link v-for="item in filteredMenuItems" :key="item.name" :to="item.path"
        class="sidebar-link" :class="{ 'is-active': $route.path === item.path }">
        <AppIcon :name="item.icon" /><span>{{ item.name }}</span>
        <span v-if="item.path === '/ai-assistant'" class="ai-label">AI</span>
      </router-link>
    </nav>
    <div class="sidebar-note"><AppIcon name="sparkles" /><p>让繁琐交给技术<br /><span>让用心回归教学</span></p></div>
    <div class="sidebar-footer">
      <div v-if="currentUser" class="sidebar-user">
        <div class="user-avatar">{{ (currentUser.name || currentUser.username || 'U').charAt(0).toUpperCase() }}</div>
        <div class="user-info"><p>{{ currentUser.name || currentUser.username }}</p><span>{{ getRoleName(currentUser.role) }}</span></div>
        <span class="user-role">{{ getRoleName(currentUser.role) }}</span>
      </div>
      <button @click="handleLogout" class="logout-button"><AppIcon name="logout" /><span>退出登录</span></button>
    </div>
  </aside>
</template>

<script setup>
import { computed } from "vue";
import { useRouter, useRoute } from "vue-router";
import authApi from "../services/authApi";
import AppIcon from "./AppIcon.vue";

const router = useRouter();
const route = useRoute();

// 获取当前用户信息
const currentUser = computed(() => authApi.getCurrentUser());

// 获取角色名称
const getRoleName = (role) => {
  const roleMap = {
    'student': '学生',
    'teacher': '教师',
    'admin': '管理员'
  }
  return roleMap[role] || '用户'
}

// 图标组件
const HomeIcon = "home";

const MagnifyingGlassIcon = "search";

const CheckBadgeIcon = "check";

const PencilSquareIcon = "pen";

const ChatBubbleLeftRightIcon = "chat";

const WrenchScrewdriverIcon = "settings";

// 完整菜单项
const allMenuItems = [
  { name: "主页", path: "/home", icon: HomeIcon },
  { name: "作业自动查重", path: "/assignments", icon: MagnifyingGlassIcon },
  { name: "作业自动评分", path: "/assignments", icon: CheckBadgeIcon },
  { name: "主观题自动评分", path: "/exams", icon: PencilSquareIcon },
  { name: "AI教学助手", path: "/ai-assistant", icon: ChatBubbleLeftRightIcon },
  { name: "工具管理", path: "/tools-admin", icon: WrenchScrewdriverIcon },
]

// 根据角色过滤菜单项
const filteredMenuItems = computed(() => {
  const userRole = currentUser.value?.role || 'student'
  
  // 学生只能看到AI教学助手
  if (userRole === 'student') {
    return allMenuItems.filter(item => item.path === '/ai-assistant')
  }
  
  // 教师/管理员可见；其中 工具管理建议仅管理员可见
  if (userRole === 'teacher') {
    return allMenuItems.filter(item => item.path !== '/tools-admin')
  }
  return allMenuItems
})

// 处理退出登录
const handleLogout = async () => {
  try {
    authApi.logout();
    router.push('/login');
    alert('已成功退出登录');
  } catch (error) {
    console.error('退出登录失败:', error);
    alert('退出登录失败，请重试');
  }
};
</script>

<style scoped>
.app-sidebar { display: flex; flex-direction: column; flex-shrink: 0; width: 244px; height: 100%; background: #fff; border-right: 1px solid #e0e7ee; }
.sidebar-brand { display: flex; align-items: center; gap: 11px; min-height: 100px; padding: 21px 24px; }
.brand-symbol { display: grid; place-items: center; width: 39px; height: 39px; flex-shrink: 0; background: #197b70; border-radius: 11px; color: #fff; }
.brand-symbol svg { width: 23px; height: 23px; }
.sidebar-brand h1 { color: #142d40; font-size: 16px; font-weight: 700; letter-spacing: -.4px; }
.sidebar-brand p { color: #66788a; font-size: 10px; margin-top: 5px; }
.sidebar-nav-label { font-size: 10px; color: #66788a; padding: 17px 28px 13px; }
.sidebar-navigation { display: flex; flex-direction: column; gap: 6px; padding: 0 15px; overflow-y: auto; }
.sidebar-link { position: relative; display: flex; align-items: center; gap: 12px; padding: 13px 14px; border-radius: 8px; color: #4c6072; font-size: 13px; font-weight: 500; white-space: nowrap; transition: background 150ms, color 150ms; }
.sidebar-link > svg { width: 19px; height: 19px; flex-shrink: 0; color: #66788a; }
.sidebar-link:hover { color: #197b70; background: #f3f8f7; }
.sidebar-link.is-active, .sidebar-link.router-link-exact-active { color: #14655d; background: #eaf4f2; font-weight: 600; }
.sidebar-link.is-active > svg, .sidebar-link.router-link-exact-active > svg { color: #197b70; }
.sidebar-link.is-active::before { content: ''; position: absolute; left: -15px; top: 12px; bottom: 12px; width: 3px; border-radius: 0 3px 3px 0; background: #197b70; }
.ai-label { margin-left: auto; padding: 1px 5px; border: 1px solid #d9e7e4; border-radius: 4px; font-size: 9px; color: #197b70; }
.sidebar-note { display: flex; gap: 11px; margin: auto 25px 28px; padding-top: 40px; color: #197b70; }
.sidebar-note > svg { width: 20px; height: 20px; margin-top: 2px; flex-shrink: 0; }
.sidebar-note p { font-size: 12px; line-height: 1.9; }
.sidebar-note span { color: #66788a; }
.sidebar-footer { border-top: 1px solid #e0e7ee; padding: 20px 21px 17px; }
.sidebar-user { display: flex; align-items: center; gap: 10px; }
.user-avatar { display: grid; place-items: center; width: 35px; height: 35px; flex-shrink: 0; background: #eaf4f2; border: 1px solid #d9e7e4; border-radius: 10px; color: #197b70; font-size: 13px; font-weight: 600; }
.user-info { min-width: 0; }
.user-info p { color: #20384c; font-size: 12px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.user-info > span { color: #66788a; font-size: 10px; }
.user-role { color: #66788a; margin-left: auto; font-size: 9px; background: #f4f7fa; border-radius: 4px; padding: 3px 6px; flex-shrink: 0; }
.logout-button { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 17px; width: 100%; padding: 9px 12px; border: 1px solid #e0e7ee; border-radius: 7px; color: #66788a; font-size: 11px; transition: background 150ms, color 150ms; }
.logout-button:hover { background: #fff3f3; color: #b44040; border-color: #f2dada; }
.logout-button svg { width: 15px; height: 15px; }
@media (max-width: 1100px) and (min-width: 761px) { .app-sidebar { width: 216px; } .sidebar-brand { padding-left: 19px; gap: 9px; } .sidebar-brand h1 { font-size: 15px; } }
@media (max-width: 760px) {
  .app-sidebar { position: relative; width: 100%; height: auto; border-right: 0; border-bottom: 1px solid #e0e7ee; }
  .sidebar-brand { min-height: 70px; padding: 15px 18px; }
  .brand-symbol { width: 34px; height: 34px; border-radius: 9px; }
  .sidebar-brand h1 { font-size: 15px; }
  .sidebar-brand p, .sidebar-nav-label, .sidebar-note, .sidebar-user { display: none; }
  .sidebar-navigation { flex-direction: row; gap: 5px; padding: 0 14px 11px; overflow-x: auto; scrollbar-width: thin; }
  .sidebar-link { padding: 10px; font-size: 11px; gap: 6px; }
  .sidebar-link > svg { width: 16px; height: 16px; }
  .sidebar-link.is-active::before { display: none; }
  .ai-label { display: none; }
  .sidebar-footer { position: absolute; top: 18px; right: 18px; border: 0; padding: 0; }
  .logout-button { margin: 0; padding: 7px 9px; font-size: 10px; }
}
</style>
