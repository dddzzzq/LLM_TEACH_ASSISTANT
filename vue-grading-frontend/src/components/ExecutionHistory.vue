<template>
  <AppModal title="会话执行记录" @close="$emit('close')">
    <p class="text-sm text-slate-600">查看每次请求的处理过程，或从已保存的记录重建上下文。</p>
    <button type="button" class="trace-button" :disabled="loading" @click="loadRuns"
      >刷新记录</button
    >
    <p v-if="error" role="alert" class="text-sm text-red-700">{{ error }}</p>
    <p v-if="!loading && !runs.length" class="text-sm text-slate-500"
      >暂无执行记录。旧聊天记录仍可正常查看。</p
    >
    <ul class="trace-runs">
      <li v-for="run in runs" :key="run.id">
        <button
          type="button"
          class="trace-button"
          :aria-pressed="selected?.id === run.id"
          :disabled="loading"
          @click="select(run)"
        >
          第 {{ run.turn }} 轮 · {{ stateLabel(run.status) }} ·
          {{
            new Date(run.created_at).toLocaleString('zh-CN', {
              timeZone: 'Asia/Shanghai',
              hour12: false,
            })
          }}
        </button>
      </li>
    </ul>
    <section v-if="selected" aria-label="执行详情">
      <p class="text-xs text-slate-500">执行 ID：{{ selected.id }}</p>
      <ol class="trace-events">
        <li v-for="event in events" :key="event.id">
          <details>
            <summary>{{ event.seq }}. {{ eventLabel(event.type) }}</summary>
            <pre>{{ JSON.stringify(event.payload, null, 2) }}</pre>
          </details>
        </li>
      </ol>
      <button
        v-if="hasMore"
        type="button"
        class="trace-button"
        :disabled="loading"
        @click="loadMore"
        >加载更多事件</button
      >
      <button type="button" class="trace-button" :disabled="loading" @click="reconstruct"
        >重建上下文</button
      >
      <div v-if="projection" aria-label="重建结果">
        <p class="text-sm"
          >已重建
          {{ projection.messages.length }} 条消息。本操作只读取记录，不调用模型或执行任务。</p
        >
        <p
          v-if="projection.unresolved_tool_calls.length"
          role="status"
          class="text-sm text-amber-800"
          >有
          {{ projection.unresolved_tool_calls.length }} 次工具调用缺少结果，请先核对任务状态。</p
        >
        <details
          ><summary>查看重建消息</summary
          ><pre>{{ JSON.stringify(projection.messages, null, 2) }}</pre>
        </details>
      </div>
    </section>
  </AppModal>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';
import authApi from '../services/authApi';
import AppModal from './AppModal.vue';

const props = defineProps({ sessionId: { type: String, required: true } });
defineEmits(['close']);
const client = authApi.getClient();
const base = `/api/sessions/${encodeURIComponent(props.sessionId)}/runs`;
const runs = ref([]),
  selected = ref(null),
  events = ref([]),
  projection = ref(null);
const loading = ref(false),
  error = ref(''),
  hasMore = ref(false);
let cursor = 0,
  disposed = false;
onBeforeUnmount(() => {
  disposed = true;
});
const stateLabel = (value) =>
  ({ RUNNING: '执行中', SUCCEEDED: '本轮已完成', FAILED: '执行失败', INTERRUPTED: '执行中断' })[
    value
  ] || value;
const eventLabel = (value) =>
  ({
    'run.started': '开始处理',
    'history.imported': '接入原聊天上下文',
    'user.message': '用户输入',
    'model.request': '模型请求',
    'model.message': '模型回复',
    'model.completed': '模型调用完成',
    'model.failed': '模型调用失败',
    'tool.started': '开始调用工具',
    'tool.bound': '确认执行参数',
    'tool.result': '工具结果',
    'skill.loaded': '加载任务指导',
    'run.finished': '本轮结束',
    'run.interrupted': '本轮中断',
  })[value] || value;

async function perform(action) {
  if (loading.value) return;
  loading.value = true;
  error.value = '';
  try {
    await action();
  } catch (e) {
    if (!disposed) error.value = e.response?.data?.error || '读取执行记录失败';
  } finally {
    if (!disposed) loading.value = false;
  }
}
async function loadRuns() {
  await perform(async () => {
    const { data } = await client.get(base);
    if (!disposed) {
      runs.value = data.runs || [];
      selected.value = null;
      events.value = [];
      projection.value = null;
    }
  });
}
async function loadPage() {
  const { data } = await client.get(`${base}/${encodeURIComponent(selected.value.id)}`, {
    params: { after_seq: cursor, limit: 50 },
  });
  if (disposed) return;
  selected.value = data.run;
  events.value.push(...data.events);
  cursor = data.next_seq;
  hasMore.value = data.has_more;
}
async function select(run) {
  await perform(async () => {
    selected.value = run;
    events.value = [];
    projection.value = null;
    cursor = 0;
    hasMore.value = false;
    await loadPage();
  });
}
async function loadMore() {
  await perform(loadPage);
}
async function reconstruct() {
  await perform(async () => {
    const { data } = await client.get(`${base}/${encodeURIComponent(selected.value.id)}/context`);
    if (!disposed) projection.value = data.projection;
  });
}
onMounted(loadRuns);
</script>

<style scoped>
.trace-button {
  margin: 10px 10px 10px 0;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
}
.trace-button[aria-pressed='true'] {
  background: #edf7f4;
  border-color: #197b70;
}
.trace-button:disabled {
  opacity: 0.5;
}
.trace-runs,
.trace-events {
  display: grid;
  gap: 6px;
  margin: 12px 0;
}
summary {
  cursor: pointer;
  font-size: 13px;
  padding: 6px 0;
}
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  max-height: 360px;
  overflow: auto;
  background: #f1f5f9;
  padding: 12px;
  font-size: 11px;
}
</style>
