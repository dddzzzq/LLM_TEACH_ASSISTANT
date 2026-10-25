<template>
  <div class="logs-page">
    <header class="page-heading">
      <div
        ><h1 class="page-title">日志查阅</h1
        ><p class="page-description"
          >按任务追踪抓取与批改，也可检索服务运行记录。时间统一显示为北京时间。</p
        ></div
      >
      <button
        class="primary-button"
        :disabled="exporting || loading || !applied"
        @click="exportLogs"
        ><AppIcon name="download" />{{ exporting ? '正在导出…' : '导出筛选结果' }}</button
      >
    </header>

    <form class="surface log-filters" @submit.prevent="applyFilters">
      <label class="keyword-field"
        >关键词<input
          v-model="filters.q"
          aria-label="关键词"
          maxlength="300"
          placeholder="搜索错误、文件名或报错代码；多个词同时匹配"
      /></label>
      <label class="task-field"
        >关联任务<select v-model="filters.task_id" aria-label="关联任务"
          ><option value="">{{ taskScoped ? '我的全部任务' : '全部任务' }}</option
          ><option
            v-if="filters.task_id && !tasks.some((t) => t.id === filters.task_id)"
            :value="filters.task_id"
            >{{ filters.task_id }}</option
          ><option v-for="task in tasks" :key="task.id" :value="task.id"
            >{{ task.kind }} · {{ task.course ? task.course + ' / ' : '' }}{{ task.name
            }}{{ task.class_name ? ' · ' + task.class_name : '' }} · {{ statusName(task.status) }} ·
            {{ task.id.slice(0, 8) }}</option
          ></select
        ></label
      >
      <label
        >日志来源<select v-model="filters.source" aria-label="日志来源"
          ><option value="all">全部可见来源</option
          ><option v-for="source in sources" :key="source.id" :value="source.id"
            >{{ source.name }}{{ !source.available ? '（暂无文件）' : '' }}</option
          ></select
        ></label
      >
      <label
        >日志级别<select v-model="filters.level" aria-label="日志级别"
          ><option value="all">全部级别</option
          ><option value="ERROR">错误 ERROR</option
          ><option value="WARN">警告 WARN</option
          ><option value="INFO">信息 INFO</option
          ><option value="DEBUG">调试 DEBUG</option></select
        ></label
      >
      <label
        >时间范围<select v-model="filters.period" aria-label="时间范围"
          ><option value="1h">最近 1 小时</option
          ><option value="24h">最近 24 小时</option
          ><option value="7d">最近 7 天</option
          ><option value="30d">最近 30 天</option
          ><option value="custom">自定义时间</option></select
        ></label
      >
      <template v-if="filters.period === 'custom'"
        ><label
          >开始时间<input
            v-model="filters.from"
            aria-label="开始时间"
            type="datetime-local"
            required /></label
        ><label
          >结束时间<input
            v-model="filters.to"
            aria-label="结束时间"
            type="datetime-local"
            required /></label
      ></template>
      <div class="filter-actions">
        <label class="check-label"
          ><input v-model="filters.hide_http" type="checkbox" />隐藏接口访问记录</label
        >
        <button class="primary-button" :disabled="loading" type="submit"
          ><AppIcon name="search" />{{ loading ? '正在搜索…' : '搜索日志' }}</button
        >
        <button class="text-button" type="button" @click="resetFilters">重置</button>
      </div>
    </form>

    <p class="scope-note"
      ><AppIcon name="shield" />日志查阅仅向管理员开放，检索和导出内容已脱敏。</p
    >
    <p v-if="pendingBytes > 0" class="index-note" role="status"
      >正在整理历史日志，尚有约 {{ size(pendingBytes) }} 待索引。可先搜索已整理的内容。</p
    >
    <p v-if="error" class="log-error" role="alert">{{ error }}</p>
    <p v-if="notice" class="log-notice" role="status">{{ notice }}</p>

    <section class="surface log-results" aria-label="日志搜索结果" :aria-busy="loading">
      <div class="results-toolbar">
        <div class="result-summary"
          ><strong>本页 {{ rows.length }} 条</strong
          ><span class="error-count">错误 {{ rows.filter((r) => r.level === 'ERROR').length }}</span
          ><span>警告 {{ rows.filter((r) => r.level === 'WARN').length }}</span></div
        >
        <div class="refresh-tools"
          ><label class="check-label"
            ><input v-model="autoRefresh" type="checkbox" :disabled="pageIndex > 0" />每 5
            秒刷新</label
          ><button class="text-button" :disabled="loading" @click="refreshLatest"
            >刷新最新</button
          ></div
        >
      </div>
      <div v-if="applied?.task_id" class="task-scope"
        >当前任务 <code>{{ applied.task_id }}</code
        ><button class="text-button" @click="copy(applied.task_id)">复制 ID</button
        ><router-link
          v-if="canOpenConsole"
          :to="{ name: 'ai-assistant', query: { task: parentTaskId } }"
          >打开作业控制台</router-link
        ></div
      >
      <div v-if="!rows.length && !loading && !error" class="logs-empty"
        ><AppIcon name="search" /><h2>未找到匹配日志</h2
        ><p>可扩大时间范围、减少关键词，或选择其他来源。索引保留最近 {{ retention }} 天。</p></div
      >
      <div v-else class="log-table-scroll">
        <table class="log-table"
          ><thead
            ><tr
              ><th>时间（UTC+8）</th><th>级别 / 来源</th><th>日志内容</th><th>查阅</th></tr
            ></thead
          >
          <tbody
            ><tr v-for="row in rows" :key="row.id" :class="'row-' + row.level.toLowerCase()">
              <td class="time-cell"
                ><time :datetime="row.time">{{ date(row.time) }}</time
                ><span
                  v-if="row.estimated_time"
                  class="estimated"
                  title="原日志未记录时间，按相邻记录或采集时间估算"
                  >≈ 时间估算</span
                ></td
              >
              <td
                ><span class="level-tag" :class="'level-' + row.level.toLowerCase()">{{
                  row.level
                }}</span
                ><span class="source-label">{{ sourceName(row.source) }}</span></td
              >
              <td class="message-cell">
                <pre
                  :class="{ expanded: expanded.has(row.id) }"
                ><template v-for="(part,i) in highlight(row.message)" :key="i"><mark v-if="part.hit">{{ part.text }}</mark><template v-else>{{ part.text }}</template></template></pre
                ><button
                  v-if="row.message.length > 180 || row.message.includes('\n')"
                  class="text-button"
                  @click="toggleExpanded(row.id)"
                  >{{ expanded.has(row.id) ? '收起内容' : '展开内容' }}</button
                ><div v-if="row.task_ids?.length" class="task-tags"
                  ><button v-for="id in row.task_ids" :key="id" :title="id" @click="filterTask(id)"
                    >任务 {{ id.slice(0, 8) }}</button
                  ></div
                ></td
              >
              <td class="row-actions"
                ><button class="text-button" @click="openContext(row)">上下文</button
                ><button class="text-button" @click="copy(row.message)">复制</button></td
              >
            </tr></tbody
          >
        </table>
      </div>
      <footer class="results-footer"
        ><span
          >{{ updated ? '查询于 ' + time(updated) : '等待查询' }} · 当前筛选最多导出 2000 条</span
        ><div
          ><button
            class="secondary-button"
            :disabled="loading || pageIndex === 0"
            @click="previousPage"
            >上一页</button
          ><span>第 {{ pageIndex + 1 }} 页</span
          ><button class="secondary-button" :disabled="loading || !nextCursor" @click="nextPage"
            >更早日志</button
          ></div
        ></footer
      >
    </section>

    <AppModal v-if="contextRow" title="日志上下文" @close="closeContext">
      <p class="context-caption"
        >{{ sourceName(contextRow.source) }} · {{ date(contextRow.time)
        }}<span v-if="contextFiltered"> · 已按任务权限过滤</span></p
      >
      <p v-if="contextLoading" role="status" class="mt-4 text-sm">正在读取上下文…</p>
      <p v-if="contextError" role="alert" class="log-error">{{ contextError }}</p>
      <ol class="context-lines"
        ><li
          v-for="line in contextLines"
          :key="line.id"
          :class="{ selected: line.id === contextRow.id }"
          ><span>{{ line.line || '事件' }} · {{ line.level }}</span
          ><pre>{{ line.message }}</pre>
        </li></ol
      >
      <template #footer
        ><button class="secondary-button" @click="closeContext">关闭</button
        ><button
          class="primary-button"
          :disabled="!contextLines.length"
          @click="copy(contextLines.map((l) => l.message).join('\n'))"
          >复制上下文</button
        ></template
      >
    </AppModal>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import authApi from '../services/authApi';
import AppIcon from '../components/AppIcon.vue';
import AppModal from '../components/AppModal.vue';

const api = authApi.getClient(),
  route = useRoute(),
  router = useRouter();
const defaults = () => ({
  q: '',
  task_id: '',
  source: 'all',
  level: 'all',
  period: '24h',
  from: '',
  to: '',
  hide_http: true,
});
const filters = reactive(defaults()),
  sources = ref([]),
  tasks = ref([]),
  rows = ref([]),
  applied = ref(null);
const retention = ref(30);
const loading = ref(false),
  exporting = ref(false),
  error = ref(''),
  notice = ref(''),
  updated = ref(null);
const autoRefresh = ref(false),
  pageIndex = ref(0),
  nextCursor = ref(''),
  expanded = reactive(new Set());
const contextRow = ref(null),
  contextLines = ref([]),
  contextLoading = ref(false),
  contextError = ref(''),
  contextFiltered = ref(false);
let cursors = [''],
  controller,
  contextController,
  timer,
  ticks = 0,
  lastRoute = '',
  appliedDraft = '',
  disposed = false,
  requestId = 0,
  noticeTimer;
const pendingBytes = computed(() =>
  sources.value.reduce((n, s) => n + Math.max(0, (s.size || 0) - (s.indexed_bytes || 0)), 0)
);
const parentTaskId = computed(() => {
  const task = tasks.value.find((t) => t.id === applied.value?.task_id);
  return task?.parent_id || applied.value?.task_id;
});
const canOpenConsole = computed(() => {
  const task = tasks.value.find((t) => t.id === applied.value?.task_id);
  return task?.kind === '抓取' || Boolean(task?.parent_id);
});
const sourceName = (id) =>
  sources.value.find((s) => s.id === id)?.name ||
  {
    task: '任务事件',
    backend: '后端与批改',
    rpa: '教务抓取',
    ai: 'AI 与模型',
    kafka: '消息队列',
    frontend: '前端服务',
  }[id] ||
  id;
const statusName = (s) =>
  ({
    PROCESSING: '批改中',
    PENDING: '排队中',
    CANCELLED: '已取消',
    SUCCEEDED: '下载成功',
    PARTIAL_SUCCESS: '部分下载成功',
    FAILED: '失败',
    SUCCESS: '已完成',
    WAITING_USER: '等待人工',
    RUNNING: '运行中',
    INTERRUPTED: '会话中断',
  })[s] || s;
const date = (v) =>
  new Date(v).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
const time = (v) =>
  new Date(v).toLocaleTimeString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
const size = (n) => (n > 1048576 ? (n / 1048576).toFixed(1) + ' MB' : Math.ceil(n / 1024) + ' KB');
function highlight(message) {
  const terms = (applied.value?.q || '').trim().split(/\s+/).filter(Boolean);
  if (!terms.length) return [{ text: message, hit: false }];
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const regex = new RegExp('(' + escaped.join('|') + ')', 'gi');
  return message.split(regex).map((text, i) => ({ text, hit: i % 2 === 1 }));
}
function toggleExpanded(id) {
  if (expanded.has(id)) expanded.delete(id);
  else expanded.add(id);
}
function flash(text) {
  notice.value = text;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => {
    notice.value = '';
  }, 5000);
}
async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    flash('已复制');
  } catch {
    flash('浏览器未允许复制，请选中文字后复制。');
  }
}
function querySignature(query) {
  return JSON.stringify(
    Object.keys(query)
      .sort()
      .map((k) => [k, query[k]])
  );
}
function readRoute() {
  Object.assign(filters, defaults());
  for (const key of Object.keys(filters))
    if (typeof route.query[key] === 'string')
      filters[key] = key === 'hide_http' ? route.query[key] !== 'false' : route.query[key];
  if (!['1h', '24h', '7d', '30d', 'custom'].includes(filters.period)) filters.period = '24h';
}
function params() {
  let from, to;
  if (filters.period === 'custom') {
    from = new Date(filters.from + ':00+08:00');
    to = new Date(filters.to + ':00+08:00');
  } else {
    to = new Date();
    from = new Date(
      to.getTime() - { '1h': 1, '24h': 24, '7d': 168, '30d': 720 }[filters.period] * 3600000
    );
  }
  if (!Number.isFinite(from.getTime()) || !Number.isFinite(to.getTime()) || from > to)
    throw new Error('请选择有效的开始、结束时间。');
  return {
    q: filters.q.trim(),
    task_id: filters.task_id,
    source: filters.source,
    level: filters.level,
    hide_http: filters.hide_http,
    from: from.toISOString(),
    to: to.toISOString(),
  };
}
async function loadPage(cursor = '') {
  if (!applied.value) return;
  controller?.abort();
  controller = new AbortController();
  const id = ++requestId;
  loading.value = true;
  error.value = '';
  try {
    const { data } = await api.get('/api/logs/search', {
      params: { ...applied.value, cursor, limit: 100 },
      signal: controller.signal,
    });
    if (disposed || id !== requestId) return;
    rows.value = data.items || [];
    nextCursor.value = data.next_cursor || '';
    expanded.clear();
    updated.value = new Date();
  } catch (e) {
    if (!disposed && id === requestId && e.code !== 'ERR_CANCELED')
      error.value = e.response?.data?.error || '读取日志失败，请重试。';
  } finally {
    if (!disposed && id === requestId) loading.value = false;
  }
}
async function applyFilters() {
  try {
    applied.value = params();
  } catch (e) {
    error.value = e.message;
    return;
  }
  const draft = JSON.stringify(filters);
  if (draft !== appliedDraft) {
    rows.value = [];
    expanded.clear();
    appliedDraft = draft;
  }
  pageIndex.value = 0;
  cursors = [''];
  nextCursor.value = '';
  const query = { ...filters };
  if (filters.hide_http) delete query.hide_http;
  else query.hide_http = 'false';
  for (const key of Object.keys(query))
    if (query[key] === '' || (['from', 'to'].includes(key) && filters.period !== 'custom'))
      delete query[key];
  lastRoute = querySignature(query);
  await router.replace({ query });
  await loadPage();
}
function resetFilters() {
  Object.assign(filters, defaults());
  autoRefresh.value = false;
  applyFilters();
}
function filterTask(id) {
  filters.task_id = id;
  applyFilters();
}
async function refreshLatest() {
  pageIndex.value = 0;
  await applyFilters();
  await loadMeta();
}
async function nextPage() {
  if (!nextCursor.value || loading.value) return;
  autoRefresh.value = false;
  const cursor = nextCursor.value;
  cursors = cursors.slice(0, pageIndex.value + 1);
  cursors.push(cursor);
  pageIndex.value++;
  rows.value = [];
  await loadPage(cursor);
}
async function previousPage() {
  if (!pageIndex.value || loading.value) return;
  pageIndex.value--;
  rows.value = [];
  await loadPage(cursors[pageIndex.value]);
}
async function loadMeta() {
  const results = await Promise.allSettled([
    api.get('/api/logs/sources'),
    api.get('/api/logs/tasks'),
  ]);
  if (disposed) return;
  if (results[0].status === 'fulfilled') {
    const data = results[0].value.data;
    sources.value = data.sources || [];
    retention.value = data.retention_days || 30;
  }
  if (results[1].status === 'fulfilled') tasks.value = results[1].value.data || [];
}
async function exportLogs() {
  if (!applied.value || exporting.value) return;
  exporting.value = true;
  error.value = '';
  try {
    const { data } = await api.get('/api/logs/export', {
      params: applied.value,
      responseType: 'blob',
    });
    const url = URL.createObjectURL(data),
      link = document.createElement('a');
    link.href = url;
    link.download = '教学助手日志-' + new Date().toISOString().slice(0, 10) + '.txt';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    flash('已导出当前筛选结果（最多2000条，内容已脱敏）。');
  } catch (e) {
    let message = '日志导出失败，请重试。';
    try {
      if (e.response?.data instanceof Blob)
        message = JSON.parse(await e.response.data.text()).error || message;
    } catch {}
    error.value = message;
  } finally {
    exporting.value = false;
  }
}
async function openContext(row) {
  contextController?.abort();
  contextController = new AbortController();
  contextRow.value = row;
  contextLines.value = [];
  contextError.value = '';
  contextLoading.value = true;
  try {
    const { data } = await api.get(`/api/logs/entries/${row.id}/context`, {
      signal: contextController.signal,
    });
    if (contextRow.value?.id === row.id) {
      contextLines.value = data.items || [];
      contextFiltered.value = data.permission_filtered;
    }
  } catch (e) {
    if (contextRow.value?.id === row.id && e.code !== 'ERR_CANCELED')
      contextError.value = e.response?.data?.error || '上下文读取失败';
  } finally {
    if (contextRow.value?.id === row.id) contextLoading.value = false;
  }
}
function closeContext() {
  contextController?.abort();
  contextRow.value = null;
}
watch(
  () => route.fullPath,
  () => {
    const signature = querySignature(route.query);
    if (signature !== lastRoute) {
      lastRoute = signature;
      readRoute();
      applyFilters();
    }
  }
);
onMounted(async () => {
  readRoute();
  await Promise.all([loadMeta(), applyFilters()]);
  timer = setInterval(() => {
    if (document.hidden) return;
    if (++ticks % 6 === 0) loadMeta();
    if (autoRefresh.value && !loading.value && pageIndex.value === 0 && !contextRow.value)
      applyFilters();
  }, 5000);
});
onBeforeUnmount(() => {
  disposed = true;
  controller?.abort();
  contextController?.abort();
  clearInterval(timer);
  clearTimeout(noticeTimer);
});
</script>

<style scoped>
.logs-page {
  min-width: 0;
}
.log-filters {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  padding: 24px;
}
.log-filters > label {
  display: flex;
  flex-direction: column;
  gap: 8px;
  color: #4c6072;
  font-size: 12px;
  font-weight: 600;
  min-width: 0;
}
.log-filters input:not([type='checkbox']),
.log-filters select {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #c8d3de;
  font-size: 13px;
  font-weight: 400;
}
.keyword-field {
  grid-column: span 2;
}
.filter-actions {
  grid-column: 1/-1;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
  padding-top: 4px;
}
.filter-actions .check-label {
  margin-right: auto;
}
.check-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  font-weight: 400;
  white-space: nowrap;
}
.scope-note,
.index-note {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 14px;
  font-size: 12px;
  color: #66788a;
  line-height: 1.8;
}
.scope-note svg {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}
.index-note {
  padding: 10px 14px;
  border-radius: 8px;
  background: #fff8e9;
  color: #825018;
}
.log-error,
.log-notice {
  margin-top: 16px;
  padding: 12px 16px;
  background: #fff1f1;
  border: 1px solid #f2dada;
  border-radius: 8px;
  font-size: 13px;
  color: #a63737;
}
.log-notice {
  background: #effaf5;
  border-color: #c9e4da;
  color: #14655d;
}
.log-results {
  margin-top: 22px;
  overflow: hidden;
}
.results-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
  padding: 18px 22px;
  border-bottom: 1px solid #e0e7ee;
}
.result-summary,
.refresh-tools {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
  font-size: 12px;
  color: #66788a;
}
.result-summary strong {
  font-size: 14px;
  color: #20384c;
}
.error-count {
  color: #a63737;
}
.text-button {
  color: #197b70;
  font-size: 12px;
  padding: 3px 0;
  text-align: left;
}
.text-button:disabled {
  color: #8b9bac;
}
.task-scope {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  padding: 12px 22px;
  background: #f5faf9;
  font-size: 12px;
}
.task-scope code {
  overflow-wrap: anywhere;
}
.task-scope a {
  color: #197b70;
}
.log-table-scroll {
  overflow-x: auto;
}
.log-table {
  width: 100%;
  table-layout: fixed;
  font-size: 12px;
  text-align: left;
}
.log-table th {
  padding: 13px 18px;
  color: #66788a;
  background: #f8fafc;
  border-bottom: 1px solid #e0e7ee;
}
.log-table th:nth-child(1) {
  width: 170px;
}
.log-table th:nth-child(2) {
  width: 130px;
}
.log-table th:nth-child(4) {
  width: 92px;
}
.log-table td {
  padding: 16px 18px;
  border-bottom: 1px solid #edf1f5;
  vertical-align: top;
}
.log-table tr:last-child td {
  border-bottom: 0;
}
.row-error {
  background: #fffcfc;
}
.row-error td:first-child {
  box-shadow: inset 3px 0 #c94f4f;
}
.time-cell time {
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.estimated {
  display: block;
  margin-top: 6px;
  color: #8b9bac;
  font-size: 10px;
}
.level-tag {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 5px;
  background: #eef3f8;
  color: #526a81;
  font-size: 10px;
  font-weight: 700;
}
.level-error {
  color: #a63737;
  background: #ffebeb;
}
.level-warn {
  color: #92581a;
  background: #fff1db;
}
.level-debug {
  color: #71608e;
  background: #f1edf7;
}
.source-label {
  display: block;
  margin-top: 7px;
  color: #66788a;
  font-size: 11px;
}
.message-cell pre,
.context-lines pre {
  font-family: 'Cascadia Code', Consolas, monospace;
  font-size: 11px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.8;
  max-height: 65px;
  overflow: hidden;
}
.message-cell pre.expanded {
  max-height: none;
}
.message-cell mark {
  background: #fff0ae;
  color: #624508;
  border-radius: 2px;
}
.row-actions button {
  display: block;
  margin-bottom: 6px;
}
.task-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.task-tags button {
  background: #edf7f4;
  color: #197b70;
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 10px;
}
.results-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
  padding: 16px 22px;
  border-top: 1px solid #e0e7ee;
  color: #66788a;
  font-size: 11px;
}
.results-footer > div {
  display: flex;
  align-items: center;
  gap: 12px;
}
.secondary-button {
  padding: 7px 12px;
  border: 1px solid #d9e3e9;
  border-radius: 7px;
  font-size: 12px;
  background: white;
}
.secondary-button:disabled {
  opacity: 0.45;
}
.logs-empty {
  padding: 66px 24px;
  text-align: center;
  color: #66788a;
}
.logs-empty svg {
  width: 32px;
  height: 32px;
  margin: 0 auto 16px;
  color: #8b9bac;
}
.logs-empty h2 {
  font-size: 15px;
  font-weight: 600;
  color: #20384c;
}
.logs-empty p {
  font-size: 12px;
  line-height: 1.8;
  margin-top: 8px;
}
.context-caption {
  font-size: 12px;
  color: #66788a;
}
.context-lines {
  margin-top: 18px;
  display: grid;
  gap: 8px;
}
.context-lines li {
  padding: 10px 12px;
  border-radius: 6px;
  background: #f7f9fb;
}
.context-lines li.selected {
  background: #edf7f4;
  box-shadow: inset 3px 0 #197b70;
}
.context-lines span {
  display: block;
  font-size: 10px;
  color: #66788a;
  margin-bottom: 5px;
}
.context-lines pre {
  max-height: none;
}
@media (max-width: 1050px) {
  .log-filters {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .keyword-field {
    grid-column: span 2;
  }
  .log-table th:nth-child(1) {
    width: 145px;
  }
  .log-table th:nth-child(2) {
    width: 108px;
  }
  .log-table td,
  .log-table th {
    padding: 12px;
  }
}
@media (max-width: 640px) {
  .log-filters {
    grid-template-columns: 1fr;
    padding: 18px;
    gap: 14px;
  }
  .keyword-field {
    grid-column: span 1;
  }
  .log-results {
    margin-top: 16px;
  }
  .log-table {
    min-width: 730px;
  }
  .results-toolbar,
  .results-footer {
    padding: 15px;
  }
  .scope-note {
    align-items: flex-start;
  }
  .page-heading .primary-button {
    width: 100%;
  }
}
</style>
