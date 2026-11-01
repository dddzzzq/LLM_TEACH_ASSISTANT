<template>
  <section class="portal-sync" aria-label="官网作业同步">
    <div class="sync-summary"
      ><AppIcon name="book" /><div
        ><strong>官网课程与作业</strong
        ><span class="connection-label">{{ portalStatus(view?.status || 'UNCONFIGURED') }}</span
        ><p v-if="view?.connection?.last_synced_at"
          >上次同步
          {{
            new Date(view.connection.last_synced_at).toLocaleString('zh-CN', {
              timeZone: 'Asia/Shanghai',
              hour12: false,
            })
          }}</p
        ><p v-else>连接教务平台后同步作业，题目要求自动补全。</p></div
      ></div
    >
    <div class="sync-actions"
      ><button
        type="button"
        class="workspace-button"
        :disabled="busy || view?.status !== 'READY' || operationActive(view?.operation)"
        @click="sync"
        >{{ operationActive(view?.operation) ? '正在同步…' : '更新课程与作业' }}</button
      ><router-link to="/settings/portal">{{
        view?.status === 'READY' ? '连接设置' : '连接教务账号'
      }}</router-link></div
    >
    <div v-if="view?.import?.total" class="sync-detail"
      ><span>已同步 {{ view.import.total }} 份作业</span
      ><span>要求就绪 {{ view.import.ready }}</span
      ><span v-if="view.import.pending">待读取 {{ view.import.pending }}</span
      ><span v-if="view.import.reading">正在读取 {{ view.import.reading }}</span
      ><span v-if="view.import.attention" class="attention">需处理 {{ view.import.attention }}</span
      ><button
        v-if="view.import.pending || view.connection?.imports_paused"
        type="button"
        :disabled="busy"
        @click="pause"
        >{{ view.connection?.imports_paused ? '继续补全要求' : '暂停后续要求同步' }}</button
      ></div
    >
    <p v-if="view?.connection?.imports_paused" class="sync-message"
      >后续要求同步已暂停，当前读取会继续完成。</p
    ><p v-if="notice" role="status" class="sync-message">{{ notice }}</p
    ><p v-if="error" role="alert" class="sync-message error">{{ error }}</p>
    <section v-if="operationActive(view?.operation)" class="current-task" aria-live="polite">
      <header
        ><div
          ><strong>当前同步任务</strong
          ><span>{{
            view.operation.control_mode === 'HUMAN' ? '等待你处理' : '自动处理中'
          }}</span></div
        ><button type="button" class="workspace-button primary" @click="openTask">{{
          view.operation.control_mode === 'HUMAN' ? '处理并交还自动化' : '查看实时进度'
        }}</button></header
      >
      <ol class="task-steps" aria-label="同步进度">
        <li
          v-for="(step, index) in taskSteps"
          :key="step"
          :class="{ done: index < taskStep, current: index === taskStep }"
          >{{ step }}</li
        >
      </ol>
      <p>{{ view.operation.message || taskGuidance }}</p>
      <details class="sync-operation"
        ><summary>高级任务控制与执行记录</summary
        ><RpaTaskPanel :key="view.operation.id" :task-id="view.operation.id" embedded
      /></details>
    </section>
  </section>
</template>
<script setup>
import { computed, nextTick, onMounted, onBeforeUnmount, ref } from 'vue';
import { useRouter } from 'vue-router';
import AppIcon from './AppIcon.vue';
import authApi from '../services/authApi';
import { createPortalApi, operationActive, portalStatus } from '../services/portalApi';
import RpaTaskPanel from './RpaTaskPanel.vue';
const emit = defineEmits(['updated']);
const router = useRouter();
const api = createPortalApi(authApi.getClient()),
  view = ref(null),
  busy = ref(false),
  notice = ref(''),
  error = ref('');
const taskSteps = ['连接官网', '登录验证', '同步作业与要求', '完成'];
const taskStep = computed(() => {
  const stage = view.value?.operation?.stage;
  if (stage === 'DONE') return 3;
  if (['READING_CATALOG', 'READING_DETAIL'].includes(stage)) return 2;
  if (['AUTHENTICATE', 'SOLVE_CAPTCHA', 'FILLING_CREDENTIALS', 'TRUST_DECISION'].includes(stage))
    return 1;
  return 0;
});
const taskGuidance = computed(() =>
  view.value?.operation?.control_mode === 'HUMAN'
    ? '请完成当前页面操作，再交还自动化继续同步。'
    : '系统正在处理，可离开此页面，任务状态会自动更新。'
);
let timer,
  loading = false,
  disposed = false,
  lastImport = null;
async function load() {
  if (loading || disposed) return;
  loading = true;
  try {
    const { data } = await api.connection();
    if (!disposed) {
      view.value = data;
      const signature = JSON.stringify([data.connection?.active_snapshot_id, data.import]);
      if (lastImport !== null && signature !== lastImport) emit('updated');
      lastImport = signature;
      error.value = '';
    }
  } catch (e) {
    if (!disposed) error.value = e.response?.data?.detail || '官网同步状态暂不可用';
  } finally {
    loading = false;
  }
}
async function sync() {
  if (busy.value || operationActive(view.value?.operation)) return;
  busy.value = true;
  error.value = '';
  try {
    await api.sync();
    notice.value = '正在同步课程、作业及作业要求，完成后会自动出现在作业列表';
    await load();
    emit('updated');
  } catch (e) {
    error.value = e.response?.data?.detail || '同步请求未完成';
  } finally {
    busy.value = false;
  }
}
async function pause() {
  busy.value = true;
  try {
    await api.pauseImports(!view.value.connection.imports_paused);
    await load();
  } catch (e) {
    error.value = e.response?.data?.detail || '同步状态未修改';
  } finally {
    busy.value = false;
  }
}
async function syncOrConnect() {
  if (!view.value || view.value.status !== 'READY') return router.push('/settings/portal');
  return sync();
}
async function openTask() {
  const id = view.value?.operation?.id;
  if (!id) return;
  await router.push({ name: 'ai-assistant', query: { task: id } });
  await nextTick();
  window.dispatchEvent(new CustomEvent('rpa-open-console', { detail: { jobId: id } }));
}
defineExpose({ syncOrConnect });
onMounted(() => {
  load();
  timer = setInterval(load, 3000);
});
onBeforeUnmount(() => {
  disposed = true;
  clearInterval(timer);
});
</script>
<style scoped>
.portal-sync {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 14px 20px;
  margin-bottom: 22px;
  padding: 17px 20px;
  border: 1px solid var(--color-line);
  border-radius: 12px;
  background: var(--color-paper);
}
.sync-summary {
  display: flex;
  gap: 12px;
  align-items: center;
}
.sync-summary > svg {
  width: 22px;
  height: 22px;
  color: var(--color-accent);
  flex-shrink: 0;
}
.sync-summary strong {
  font-size: 13px;
  font-weight: 600;
}
.connection-label {
  color: var(--color-muted);
  font-size: 11px;
  margin-left: 12px;
}
.sync-summary p {
  font-size: 12px;
  color: var(--color-muted);
  margin-top: 5px;
}
.sync-actions {
  display: flex;
  gap: 14px;
  align-items: center;
}
.sync-actions > a {
  font-size: 12px;
  color: var(--color-accent);
}
.sync-detail {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  width: 100%;
  padding-top: 12px;
  border-top: 1px solid var(--color-line);
  color: var(--color-muted);
  font-size: 12px;
}
.sync-detail button {
  color: var(--color-accent);
}
.attention {
  color: var(--color-warning);
}
.sync-message {
  flex-basis: 100%;
  font-size: 12px;
  color: var(--color-muted);
}
.sync-message.error {
  color: var(--color-danger);
}
.current-task {
  width: 100%;
  padding: 16px;
  border: 1px solid var(--color-line);
  border-radius: 10px;
  background: var(--color-soft);
}
.current-task > header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.current-task > header > div {
  display: flex;
  align-items: center;
  gap: 10px;
}
.current-task strong {
  font-size: 14px;
}
.current-task header span {
  color: var(--color-warning);
  font-size: 12px;
}
.current-task > p {
  margin-top: 12px;
  color: var(--color-muted);
  font-size: 13px;
  line-height: 1.8;
}
.task-steps {
  display: flex;
  gap: 8px;
  margin-top: 16px;
  padding: 0;
  list-style: none;
}
.task-steps li {
  flex: 1;
  padding-top: 8px;
  border-top: 3px solid var(--color-line);
  color: var(--color-muted);
  font-size: 12px;
}
.task-steps li.done {
  border-color: var(--color-success);
  color: var(--color-success);
}
.task-steps li.current {
  border-color: var(--color-accent);
  color: var(--color-accent);
  font-weight: 600;
}
.current-task .sync-operation {
  margin-top: 12px;
}
.sync-operation {
  width: 100%;
  font-size: 12px;
}
.sync-operation summary {
  color: var(--color-accent);
}
@media (max-width: 760px) {
  .portal-sync {
    padding: 15px;
    gap: 12px;
  }
  .sync-summary {
    gap: 9px;
  }
  .sync-actions {
    width: 100%;
    justify-content: space-between;
  }
  .sync-summary p {
    font-size: 11px;
  }
}
</style>
