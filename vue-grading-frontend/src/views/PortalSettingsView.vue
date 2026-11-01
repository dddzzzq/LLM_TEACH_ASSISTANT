<template>
  <div class="form-page portal-settings">
    <header
      ><h1 class="page-title">教务连接</h1
      ><p class="page-description"
        >连接本人账号，在专用浏览器完成首次认证后自动同步课程和作业。</p
      ></header
    >
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <p v-if="notice" role="status">{{ notice }}</p>
    <section class="form-surface">
      <div class="status-row"
        ><strong>西安电子科技大学教学平台</strong
        ><span class="connection-state" :class="{ ready: view.status === 'READY' }">{{
          portalStatus(view.status)
        }}</span></div
      >
      <p v-if="view.connection"
        >账号：{{ view.connection.account_mask }} · 最近认证：{{
          date(view.connection.last_authenticated_at)
        }}</p
      >
      <p>最近目录更新：{{ date(view.connection?.last_synced_at) }}</p>
      <p v-if="!view.credential_storage_ready" class="error"
        >服务端尚未配置凭据密钥，请按部署文档配置后连接。</p
      >
      <details class="credentials-panel" :open="!view.connection"
        ><summary>{{ view.connection ? '修改教务账号与连接设置' : '设置教务账号' }}</summary
        ><form class="settings-form" @submit.prevent="save">
          <label
            >教师教务账号<input
              v-model="username"
              name="portal_username"
              autocomplete="off"
              :required="!view.has_password"
              :placeholder="view.has_password ? '留空保持当前账号' : '请输入教师账号'"
              maxlength="255"
          /></label>
          <label
            >教务密码<PasswordInput
              v-model="password"
              name="portal_password"
              autocomplete="new-password"
              :required="!view.has_password"
              :placeholder="view.has_password ? '留空保持原密码' : '请输入教务密码'"
              maxlength="4096"
          /></label>
          <label class="check"
            ><input v-model="trustDevice" type="checkbox" />在本系统专用浏览器中信任该设备</label
          >
          <div class="actions">
            <button class="primary" :disabled="busy || active || !view.credential_storage_ready"
              >保存并连接</button
            >
          </div></form
        ></details
      ><div class="actions connection-actions">
        <button type="button" :disabled="busy || active || !view.has_password" @click="reauth"
          >重新认证</button
        >
        <button type="button" :disabled="busy || active || view.status !== 'READY'" @click="sync"
          >更新课程与作业</button
        >
        <button
          type="button"
          :disabled="busy || !view.connection || view.status === 'DISABLED'"
          @click="disconnect"
          >断开并删除凭据</button
        >
      </div>
    </section>
    <section v-if="operation" class="form-surface operation" aria-live="polite">
      <h2>连接进度：{{ portalStatus(operation.status) }}</h2
      ><p>{{ operation.message }}</p>
      <RpaTaskPanel
        v-if="operation.type !== 'DISCONNECT'"
        :key="operation.id"
        :task-id="operation.id"
        embedded
      />
      <details v-if="events.length"
        ><summary>查看执行记录</summary
        ><ol
          ><li v-for="event in events" :key="event.id">{{ eventText(event) }}</li></ol
        ></details
      >
    </section>
    <section class="form-surface"
      ><h2>课程和作业</h2
      ><p
        >已同步
        {{ courses.length }} 门课程。更新课程与作业会创建或更新本地作业，并自动读取官网要求。</p
      ><router-link to="/assignments">返回作业工作台选择作业 →</router-link
      ><ul
        ><li v-for="course in courses" :key="course.id"
          >{{ course.title }} · {{ course.term || '官网未提供学期' }}</li
        ></ul
      ></section
    >
    <section v-if="isAdmin" class="form-surface"
      ><h2>历史作业归属</h2><p>历史作业需依据实际记录分配给教师；已有成绩保留。</p
      ><button @click="loadLegacy">查询待分配作业</button
      ><div v-for="item in legacy" :key="item.ID" class="actions"
        ><span
          >#{{ item.ID }} {{ item.CourseName }} / {{ item.TaskName }} / {{ item.ClassName }}</span
        ><input
          v-model="legacyOwners[item.ID]"
          type="number"
          min="1"
          aria-label="目标教师用户 ID"
          placeholder="教师用户 ID"
        /><button @click="claim(item.ID)">分配归属</button></div
      ></section
    >
  </div>
</template>

<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue';
import authApi from '../services/authApi';
import { createPortalApi, operationActive, portalStatus } from '../services/portalApi';
import RpaTaskPanel from '../components/RpaTaskPanel.vue';
import PasswordInput from '../components/PasswordInput.vue';
const api = createPortalApi(authApi.getClient());
const view = ref({ status: 'UNCONFIGURED', credential_storage_ready: false }),
  operation = ref(null),
  courses = ref([]),
  events = ref([]);
const username = ref(''),
  password = ref(''),
  trustDevice = ref(true),
  busy = ref(false),
  error = ref(''),
  notice = ref('');
const legacy = ref([]),
  legacyOwners = ref({}),
  isAdmin = authApi.getCurrentUser()?.role === 'admin';
const active = computed(() => operationActive(operation.value));
let disposed = false,
  loading = false,
  initialized = false,
  timer,
  snapshot = '',
  eventOperation = '';
const date = (value) =>
  value
    ? new Date(value).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false })
    : '尚无记录';
const eventText = (event) => {
  try {
    const value = JSON.parse(event.payload);
    return value.message || portalStatus(value.status || value.stage);
  } catch {
    return event.type;
  }
};
async function load() {
  if (disposed || loading) return;
  loading = true;
  try {
    const { data } = await api.connection();
    if (disposed) return;
    view.value = data;
    if (!initialized) {
      trustDevice.value = data.connection?.trust_device ?? true;
      initialized = true;
    }
    operation.value = data.operation || null;
    if (operation.value?.status === 'SUCCEEDED' && operation.value.type === 'CATALOG_SYNC')
      notice.value = '课程与作业目录已更新';
    if (operation.value?.status === 'SUCCEEDED' && operation.value.type === 'DISCONNECT')
      notice.value = '教务凭据与专用浏览器状态已删除';
    if (eventOperation !== operation.value?.id) {
      events.value = [];
      eventOperation = operation.value?.id || '';
    }
    if (operation.value) {
      const res = await api.events(operation.value.id, events.value.at(-1)?.seq || 0);
      if (!disposed) events.value.push(...res.data);
    }
    const current = data.connection?.active_snapshot_id || '';
    if (current !== snapshot) {
      snapshot = current;
      courses.value = current ? (await api.courses()).data : [];
    }
  } catch (e) {
    if (!disposed) error.value = e.response?.data?.detail || '连接状态读取失败';
  } finally {
    loading = false;
  }
}
async function action(fn, success = '') {
  if (busy.value) return;
  busy.value = true;
  error.value = '';
  notice.value = '';
  try {
    await fn();
    notice.value = success;
    await load();
  } catch (e) {
    error.value = e.response?.data?.detail || '操作未完成，请稍后重试';
  } finally {
    busy.value = false;
  }
}
async function save() {
  const body = {
    username: username.value,
    password: password.value,
    trust_device: trustDevice.value,
  };
  password.value = '';
  await action(async () => {
    await api.save(body);
    username.value = '';
  }, '设置已保存，正在连接教务平台');
  body.password = '';
}
const reauth = () => action(() => api.reauth());
const sync = () => action(() => api.sync(), '正在更新课程和作业目录');
const disconnect = () => {
  if (window.confirm('断开后将删除教务凭据和专用浏览器登录状态，保留已完成的评分结果。继续？'))
    action(() => api.disconnect());
};
const loadLegacy = () =>
  action(async () => {
    legacy.value = (await api.legacyAssignments()).data;
  });
const claim = (id) =>
  action(async () => {
    await api.claimAssignment(id, Number(legacyOwners.value[id]));
    legacy.value = legacy.value.filter((item) => item.ID !== id);
  });
onMounted(() => {
  load();
  timer = setInterval(load, 2000);
});
onBeforeUnmount(() => {
  disposed = true;
  clearInterval(timer);
  password.value = '';
  events.value = [];
});
</script>

<style scoped>
.portal-settings {
  display: grid;
  gap: 22px;
  font-size: 13px;
  color: var(--color-ink);
}
.form-surface {
  padding: 26px;
  background: var(--color-paper);
  border: 1px solid var(--color-line);
  border-radius: var(--radius-surface);
}
.form-surface > p {
  font-size: 13px;
  color: var(--color-muted);
  line-height: 1.8;
  margin-top: 10px;
}
.settings-form {
  display: grid;
  gap: 16px;
  margin-top: 18px;
}
.credentials-panel {
  border-top: 1px solid var(--color-line);
  margin-top: 20px;
  padding-top: 16px;
}
.credentials-panel > summary {
  font-size: 13px;
  color: var(--color-accent);
  min-height: 32px;
  line-height: 2;
}
.connection-actions {
  margin-top: 16px;
}
.connection-state {
  font-size: 12px;
  color: var(--color-muted);
  background: var(--color-soft);
  border-radius: 6px;
  padding: 5px 9px;
}
.connection-state.ready {
  color: var(--color-success);
  background: var(--color-success-soft);
}
label {
  display: grid;
  gap: 8px;
  font-size: 13px;
}
.check,
.actions,
.status-row,
.identity {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
.status-row {
  justify-content: space-between;
}
.status-row > strong {
  font-size: 15px;
  font-weight: 600;
}
input:not([type='checkbox']) {
  padding: 10px 12px;
  border: 1px solid var(--color-input-line);
  border-radius: var(--radius-control);
  min-width: 0;
  font-size: 13px;
}
button {
  padding: 9px 14px;
  min-height: 40px;
  border: 1px solid var(--color-line);
  border-radius: var(--radius-control);
  background: var(--color-paper);
  font-size: 13px;
  transition:
    background var(--dur-micro),
    border-color var(--dur-micro);
}
button:hover {
  background: var(--color-soft);
}
button.primary {
  background: var(--color-accent);
  color: var(--color-on-accent);
  border-color: var(--color-accent);
}
button.primary:hover {
  background: var(--color-accent-hover);
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.error,
.form-surface > p.error {
  color: var(--color-danger);
}
.operation {
  display: grid;
  gap: 16px;
}
h2 {
  font-size: 16px;
  font-weight: 600;
}
li {
  margin: 10px 0;
  font-size: 13px;
  color: var(--color-muted);
}
a {
  display: inline-block;
  color: var(--color-accent);
  font-size: 13px;
  margin-top: 12px;
}
.otp-form {
  display: grid;
  gap: 10px;
}
.actions > input {
  max-width: 200px;
}
@media (max-width: 760px) {
  .portal-settings {
    gap: 18px;
  }
  .form-surface {
    padding: 20px;
  }
  .status-row {
    align-items: flex-start;
    gap: 10px;
  }
  .connection-actions {
    gap: 8px;
  }
  .connection-actions button {
    padding: 8px 10px;
    font-size: 12px;
  }
}
</style>
