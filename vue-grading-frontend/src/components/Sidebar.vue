<template>
  <aside class="app-sidebar" aria-label="主导航">
    <router-link to="/assignments" class="sidebar-brand" aria-label="教师作业助手首页">
      <span class="brand-symbol"><AppIcon name="book" /></span>
      <span><strong>教师作业助手</strong><small>让教学更从容</small></span>
    </router-link>
    <nav class="sidebar-navigation" aria-label="工作区">
      <router-link
        v-for="item in filteredMenuItems"
        :key="item.path"
        :to="item.path"
        class="sidebar-link"
        :class="{
          'is-active': route.path.startsWith('/assignments') || route.path === '/grade-homework',
        }"
        :aria-label="item.name"
        ><AppIcon :name="item.icon" />{{ item.name }}</router-link
      >
      <div class="sidebar-views" v-if="route.path.startsWith('/assignments')">
        <router-link
          :to="{ path: '/assignments', query: { view: 'attention' } }"
          :class="{ 'is-current': route.query.view === 'attention' }"
          >需要处理</router-link
        >
        <router-link
          :to="{ path: '/assignments', query: { view: 'processing' } }"
          :class="{ 'is-current': route.query.view === 'processing' }"
          >正在批改</router-link
        >
        <router-link
          :to="{ path: '/assignments', query: { view: 'graded' } }"
          :class="{ 'is-current': route.query.view === 'graded' }"
          >已有成绩</router-link
        >
      </div>
    </nav>
    <div class="sidebar-helper"
      ><AppIcon name="chat" /><p>AI 助手随时在侧<br /><span>点击右下角图标开始对话</span></p></div
    >
    <nav class="sidebar-resources" aria-label="辅助功能">
      <router-link to="/materials" class="sidebar-link"
        ><AppIcon name="file" />导入记录</router-link
      >
      <router-link to="/settings/portal" class="sidebar-link"
        ><AppIcon name="settings" />教务连接</router-link
      >
      <router-link v-if="currentUser?.role === 'admin'" to="/logs" class="sidebar-link"
        ><AppIcon name="shield" />运行日志</router-link
      >
    </nav>
    <div class="sidebar-footer">
      <div v-if="currentUser" class="sidebar-user"
        ><span class="user-avatar">{{ displayName.charAt(0).toUpperCase() }}</span
        ><div class="user-info"
          ><p :title="hasRealName ? '真实姓名' : '登录账号'">{{ displayName }}</p
          ><span>{{ currentUser.username }}</span></div
        ></div
      >
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
  width: 204px;
  height: 100%;
  background: var(--color-sidebar);
  border-right: 1px solid var(--color-line);
}
.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 28px 18px 32px;
  color: var(--color-ink);
}
.brand-symbol {
  display: grid;
  place-items: center;
  width: 36px;
  height: 39px;
  flex-shrink: 0;
  color: var(--color-accent);
}
.brand-symbol svg {
  width: 30px;
  height: 30px;
}
.sidebar-brand strong {
  font-size: 15px;
  letter-spacing: -0.3px;
}
.sidebar-brand small {
  display: block;
  margin-top: 5px;
  font-size: 12px;
  color: var(--color-muted);
}
.sidebar-navigation {
  padding: 0 12px;
}
.sidebar-link {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 44px;
  padding: 11px 12px;
  border-radius: 8px;
  font-size: 13px;
  color: var(--color-muted);
  transition:
    background 150ms,
    color 150ms;
}
.sidebar-link svg {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
}
.sidebar-link:hover {
  background: var(--color-paper);
  color: var(--color-ink);
}
.sidebar-link.is-active,
.sidebar-link.router-link-exact-active {
  background: var(--color-paper);
  color: var(--color-accent);
  font-weight: 650;
  box-shadow: var(--shadow-subtle);
}
.sidebar-views {
  display: grid;
  padding: 8px 0 0 40px;
  gap: 2px;
}
.sidebar-views a {
  padding: 9px 0;
  font-size: 12px;
  min-height: 36px;
  color: var(--color-muted);
}
.sidebar-views a:hover,
.sidebar-views .is-current {
  color: var(--color-accent);
}
.sidebar-helper {
  display: flex;
  gap: 9px;
  margin: auto 18px 24px;
  padding-top: 28px;
  color: var(--color-muted);
}
.sidebar-helper svg {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
  margin-top: 2px;
}
.sidebar-helper p {
  font-size: 12px;
  line-height: 1.8;
}
.sidebar-helper span {
  font-size: 11px;
  color: var(--color-muted);
}
.sidebar-resources {
  display: grid;
  padding: 0 12px 12px;
  gap: 3px;
}
.sidebar-footer {
  border-top: 1px solid var(--color-line);
  padding: 18px;
}
.sidebar-user {
  display: flex;
  align-items: center;
  gap: 9px;
}
.user-avatar {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: var(--color-accent-soft);
  color: var(--color-accent);
  font-size: 13px;
}
.user-info {
  min-width: 0;
}
.user-info p {
  font-size: 13px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.user-info span {
  font-size: 11px;
  color: var(--color-muted);
}
.logout-button {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 14px;
  color: var(--color-muted);
  font-size: 12px;
  min-height: 32px;
}
.logout-button svg {
  width: 15px;
  height: 15px;
}
.logout-button:hover {
  color: var(--color-danger);
}
@media (max-width: 1000px) and (min-width: 761px) {
  .app-sidebar {
    width: 188px;
  }
  .sidebar-brand {
    padding-left: 14px;
    padding-right: 14px;
  }
  .sidebar-brand strong {
    font-size: 14px;
  }
}
@media (max-width: 760px) {
  .app-sidebar {
    width: 100%;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--color-line);
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    padding: 10px 16px;
    gap: 6px;
  }
  .sidebar-brand {
    padding: 0;
    gap: 8px;
  }
  .brand-symbol {
    width: 29px;
    height: 32px;
  }
  .brand-symbol svg {
    width: 25px;
    height: 25px;
  }
  .sidebar-brand small,
  .sidebar-views,
  .sidebar-helper,
  .sidebar-user {
    display: none;
  }
  .sidebar-navigation {
    padding: 0;
    grid-column: 2;
    grid-row: 1;
  }
  .sidebar-link {
    font-size: 12px;
    padding: 8px 10px;
    min-height: 36px;
  }
  .sidebar-link svg {
    width: 16px;
    height: 16px;
  }
  .sidebar-resources {
    display: flex;
    padding: 0;
    grid-column: 1/-1;
    flex-wrap: wrap;
    gap: 4px;
    padding-right: 70px;
  }
  .sidebar-resources .sidebar-link {
    font-size: 12px;
    padding: 6px 8px;
    min-height: 32px;
  }
  .sidebar-footer {
    grid-column: 2;
    grid-row: 2;
    border: 0;
    padding: 0;
    align-self: center;
  }
  .logout-button {
    margin: 0;
    min-height: 32px;
    font-size: 11px;
  }
  .logout-button svg {
    width: 13px;
    height: 13px;
  }
}
</style>
