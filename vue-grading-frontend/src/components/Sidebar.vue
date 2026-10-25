<template>
  <aside class="app-sidebar" aria-label="主导航">
    <div class="sidebar-brand">
      <div class="brand-symbol"><AppIcon name="book" /></div>
      <div><h1>教师作业助手</h1><p>AI 协作 · 作业批改</p></div>
    </div>
    <div class="sidebar-nav-label">工作空间</div>
    <nav class="sidebar-navigation">
      <router-link
        v-for="item in filteredMenuItems"
        :key="item.name"
        :to="item.path"
        :aria-label="item.name"
        class="sidebar-link"
        :class="{
          'is-active':
            item.path === '/assignments'
              ? route.path.startsWith('/assignments') || route.path === '/grade-homework'
              : route.path === item.path,
        }"
      >
        <AppIcon :name="item.icon" /><span>{{ item.name }}</span>
        <span v-if="item.path === '/ai-assistant'" class="ai-label">AI</span>
      </router-link>
    </nav>
    <details class="sidebar-resources" open>
      <summary>资源与设置</summary>
      <nav aria-label="资源与设置">
        <router-link to="/materials" class="sidebar-link"
          ><AppIcon name="file" />作业材料</router-link
        >
        <router-link to="/settings/portal" class="sidebar-link"
          ><AppIcon name="settings" />教务账号</router-link
        >
        <details class="sidebar-advanced"
          ><summary>更多设置</summary>
          <router-link to="/tools-admin" class="sidebar-link">工具与技能</router-link>
          <router-link v-if="currentUser?.role === 'admin'" to="/logs" class="sidebar-link"
            >运行日志</router-link
          >
        </details>
      </nav>
    </details>
    <div class="sidebar-footer">
      <div v-if="currentUser" class="sidebar-user">
        <div class="user-avatar">{{ displayName.charAt(0).toUpperCase() }}</div>
        <div class="user-info"
          ><p :title="hasRealName ? '真实姓名' : '登录账号'">{{ displayName }}</p
          ><span>{{ currentUser.username }}</span></div
        >
        <span class="user-role">{{ getRoleName(currentUser.role) }}</span>
      </div>
      <button @click="handleLogout" class="logout-button"
        ><AppIcon name="logout" /><span>退出登录</span></button
      >
    </div>
  </aside>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import authApi from '../services/authApi';
import AppIcon from './AppIcon.vue';
import { mainNavigation } from '../services/teacherWorkspace';

const router = useRouter();
const route = useRoute();

// 获取当前用户信息
const currentUser = ref(authApi.getCurrentUser());
const profileVerified = ref(false);
// Confirm the server profile before displaying a name from an older browser cache.
const hasRealName = computed(
  () => profileVerified.value && Boolean(currentUser.value?.name?.trim())
);
const displayName = computed(() =>
  hasRealName.value ? currentUser.value.name.trim() : currentUser.value?.username || '用户'
);
onMounted(async () => {
  if (!authApi.isAuthenticated()) return;
  try {
    const { data } = await authApi.getProfile();
    if (!data?.username || !['teacher', 'admin'].includes(data.role)) return;
    const account = {
      ...(authApi.getCurrentUser() || {}),
      user_id: data.id,
      username: data.username,
      name: data.name,
      role: data.role,
    };
    authApi.setUserInfo(account);
    currentUser.value = account;
    profileVerified.value = true;
  } catch (error) {
    console.warn('账号信息同步失败:', error.response?.status || 'network');
  }
});

const getRoleName = () => '教师工作区';
const filteredMenuItems = computed(() => mainNavigation);
// 处理退出登录
const handleLogout = async () => {
  try {
    authApi.logout();
    router.push('/login');
  } catch (error) {
    console.error('退出登录失败:', error);
    alert('退出登录失败，请重试');
  }
};
</script>

<style scoped>
.app-sidebar {
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  width: 244px;
  height: 100%;
  background: #fff;
  border-right: 1px solid #e0e7ee;
}
.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 11px;
  min-height: 100px;
  padding: 21px 24px;
}
.brand-symbol {
  display: grid;
  place-items: center;
  width: 39px;
  height: 39px;
  flex-shrink: 0;
  background: var(--color-accent);
  border-radius: 11px;
  color: #fff;
}
.brand-symbol svg {
  width: 23px;
  height: 23px;
}
.sidebar-brand h1 {
  color: #142d40;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: -0.4px;
}
.sidebar-brand p {
  color: #66788a;
  font-size: 10px;
  margin-top: 5px;
}
.sidebar-nav-label {
  font-size: 10px;
  color: #66788a;
  padding: 17px 28px 13px;
}
.sidebar-navigation {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 0 15px;
  overflow-y: auto;
}
.sidebar-link {
  position: relative;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 13px 14px;
  border-radius: 8px;
  color: #4c6072;
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
  transition:
    background 150ms,
    color 150ms;
}
.sidebar-link > svg {
  width: 19px;
  height: 19px;
  flex-shrink: 0;
  color: #66788a;
}
.sidebar-link:hover {
  color: var(--color-accent);
  background: #f5f6ff;
}
.sidebar-link.is-active,
.sidebar-link.router-link-exact-active {
  color: #4455cf;
  background: #eef0ff;
  font-weight: 600;
}
.sidebar-link.is-active > svg,
.sidebar-link.router-link-exact-active > svg {
  color: var(--color-accent);
}
.sidebar-link.is-active::before {
  content: '';
  position: absolute;
  left: -15px;
  top: 12px;
  bottom: 12px;
  width: 3px;
  border-radius: 0 3px 3px 0;
  background: var(--color-accent);
}
.ai-label {
  margin-left: auto;
  padding: 1px 5px;
  border: 1px solid #d9e7e4;
  border-radius: 4px;
  font-size: 9px;
  color: var(--color-accent);
}
.sidebar-note {
  display: flex;
  gap: 11px;
  margin: auto 25px 28px;
  padding-top: 40px;
  color: var(--color-accent);
}
.sidebar-note > svg {
  width: 20px;
  height: 20px;
  margin-top: 2px;
  flex-shrink: 0;
}
.sidebar-note p {
  font-size: 12px;
  line-height: 1.9;
}
.sidebar-note span {
  color: #66788a;
}
.sidebar-footer {
  margin-top: auto;
  border-top: 1px solid #e0e7ee;
  padding: 20px 21px 17px;
}
.sidebar-user {
  display: flex;
  align-items: center;
  gap: 10px;
}
.user-avatar {
  display: grid;
  place-items: center;
  width: 35px;
  height: 35px;
  flex-shrink: 0;
  background: #eef0ff;
  border: 1px solid #d9e7e4;
  border-radius: 10px;
  color: var(--color-accent);
  font-size: 13px;
  font-weight: 600;
}
.user-info {
  min-width: 0;
}
.user-info p {
  color: #20384c;
  font-size: 12px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.user-info > span {
  color: #66788a;
  font-size: 10px;
}
.user-role {
  color: #66788a;
  margin-left: auto;
  font-size: 9px;
  background: #f4f7fa;
  border-radius: 4px;
  padding: 3px 6px;
  flex-shrink: 0;
}
.logout-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: 17px;
  width: 100%;
  padding: 9px 12px;
  border: 1px solid #e0e7ee;
  border-radius: 7px;
  color: #66788a;
  font-size: 11px;
  transition:
    background 150ms,
    color 150ms;
}
.logout-button:hover {
  background: #fff3f3;
  color: #b44040;
  border-color: #f2dada;
}
.logout-button svg {
  width: 15px;
  height: 15px;
}
@media (max-width: 1100px) and (min-width: 761px) {
  .app-sidebar {
    width: 216px;
  }
  .sidebar-brand {
    padding-left: 19px;
    gap: 9px;
  }
  .sidebar-brand h1 {
    font-size: 15px;
  }
}
@media (max-width: 760px) {
  .app-sidebar {
    position: relative;
    width: 100%;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid #e0e7ee;
  }
  .sidebar-brand {
    min-height: 70px;
    padding: 15px 18px;
  }
  .brand-symbol {
    width: 34px;
    height: 34px;
    border-radius: 9px;
  }
  .sidebar-brand h1 {
    font-size: 15px;
  }
  .sidebar-brand p,
  .sidebar-nav-label,
  .sidebar-note,
  .sidebar-user {
    display: none;
  }
  .sidebar-navigation {
    flex-direction: row;
    gap: 5px;
    padding: 0 14px 11px;
    overflow-x: auto;
    scrollbar-width: thin;
  }
  .sidebar-link {
    padding: 10px;
    font-size: 11px;
    gap: 6px;
  }
  .sidebar-link > svg {
    width: 16px;
    height: 16px;
  }
  .sidebar-link.is-active::before {
    display: none;
  }
  .ai-label {
    display: none;
  }
  .sidebar-footer {
    position: absolute;
    top: 18px;
    right: 18px;
    border: 0;
    padding: 0;
  }
  .logout-button {
    margin: 0;
    padding: 7px 9px;
    font-size: 10px;
  }
}

.sidebar-resources {
  padding: 28px 15px 20px;
}
.sidebar-resources > summary {
  list-style: none;
  font-size: 10px;
  color: var(--color-muted);
  padding: 0 13px 14px;
  cursor: pointer;
}
.sidebar-resources .sidebar-link {
  font-size: 12px;
  padding: 11px 14px;
}
.sidebar-advanced {
  font-size: 11px;
  color: var(--color-muted);
  margin-top: 8px;
}
.sidebar-advanced > summary {
  padding: 10px 14px;
  cursor: pointer;
}
.sidebar-advanced .sidebar-link {
  padding-left: 26px;
  font-size: 11px;
}
@media (max-width: 760px) {
  .sidebar-resources {
    padding: 0 14px 8px;
  }
  .sidebar-resources > summary {
    font-size: 11px;
    padding: 3px 10px 8px;
  }
  .sidebar-resources:not([open]) {
    padding-bottom: 0;
  }
  .sidebar-resources nav {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    flex-wrap: wrap;
  }
  .sidebar-resources .sidebar-link {
    padding: 6px 10px;
    font-size: 11px;
  }
  .sidebar-advanced {
    margin-top: 0;
  }
  .sidebar-advanced > summary {
    padding: 6px 10px;
  }
}
</style>
