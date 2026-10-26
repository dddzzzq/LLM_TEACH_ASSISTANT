<template>
  <section class="requirements" aria-label="官网作业要求">
    <p v-if="!items.length">选择作业并加入待批改列表后，系统会自动读取各份作业的官网要求。</p>
    <p v-else>系统按作业分别读取要求，并保留原始图片和附件。读取期间可查看原任务浏览器。</p>
    <article v-for="item in items" :key="item.homework_id">
      <header>
        <strong>{{ item.course_title }} / {{ item.homework_title }}</strong>
        <span>{{ label(records[item.homework_id]) }}</span>
        <button type="button" :disabled="disabled || running" @click="read(item, true)"
          >重新读取</button
        >
        <button
          v-if="operationActive(records[item.homework_id]?.operation)"
          type="button"
          @click="cancel(item.homework_id)"
          >取消读取</button
        >
      </header>
      <p v-if="records[item.homework_id]?.error" role="alert" class="error">{{
        records[item.homework_id].error
      }}</p>
      <template v-if="records[item.homework_id]?.detail">
        <p class="updated"
          >读取于 {{ new Date(records[item.homework_id].updated_at).toLocaleString() }}</p
        >
        <details :open="items.length === 1">
          <summary>查看提取的作业要求</summary>
          <pre>{{ records[item.homework_id].detail.question || '尚未提取到正文' }}</pre>
          <ul v-if="records[item.homework_id].detail.warnings?.length" class="error">
            <li v-for="warning in records[item.homework_id].detail.warnings" :key="warning">{{
              warning
            }}</li>
          </ul>
          <div class="source-list">
            <button
              v-for="file in records[item.homework_id].detail.sources.filter((s) => s.sha256)"
              :key="file.id"
              type="button"
              @click="download(item.homework_id, file)"
            >
              下载原始{{ file.kind === 'image' ? '图片' : '附件' }}：{{ file.name }}
            </button>
          </div>
        </details>
      </template>
      <a v-if="operationActive(records[item.homework_id]?.operation)" href="#requirement-browser"
        >查看读取任务浏览器</a
      >
    </article>
  </section>
</template>
<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import authApi from '../services/authApi';
import { createPortalApi, operationActive, portalStatus } from '../services/portalApi';

const props = defineProps({
  items: { type: Array, default: () => [] },
  snapshotId: String,
  disabled: Boolean,
});
const emit = defineEmits(['ready', 'active-task']);
const api = createPortalApi(authApi.getClient());
const records = ref({}),
  running = ref(false);
let disposed = false,
  generation = 0,
  timer,
  wake;
const ready = computed(
  () =>
    props.items.length > 0 &&
    props.items.every((item) => {
      const record = records.value[item.homework_id];
      return (
        record?.detail?.complete &&
        record?.snapshot_id === props.snapshotId &&
        !operationActive(record?.operation)
      );
    })
);
watch(ready, (value) => emit('ready', value), { immediate: true });
watch(
  () =>
    Object.values(records.value).find((record) => operationActive(record.operation))?.operation
      .id || '',
  (id) => emit('active-task', id),
  { immediate: true }
);
function label(record) {
  if (record?.operation && operationActive(record.operation))
    return `${portalStatus(record.operation.status)} · ${record.operation.message || '正在读取'}`;
  if (record?.error) return '读取未完成';
  return record?.detail ? (record.detail.complete ? '要求已读取' : '部分内容未解析') : '等待读取';
}
function delay() {
  return new Promise((resolve) => {
    wake = resolve;
    timer = setTimeout(() => {
      wake = null;
      resolve();
    }, 2000);
  });
}
async function read(item, force = false) {
  if (disposed || running.value) return;
  const id = item.homework_id,
    snapshot = props.snapshotId,
    turn = generation;
  running.value = true;
  try {
    let data = (await api.homeworkDetail(id)).data;
    if (disposed || turn !== generation) return;
    records.value[id] = data;
    if ((!data.detail || force) && !operationActive(data.operation)) {
      const result = (await api.readHomeworkDetail(id, snapshot)).data;
      records.value[id] = { ...data, operation: result.operation, error: '' };
    }
    while (!disposed && turn === generation && operationActive(records.value[id]?.operation)) {
      await delay();
      if (disposed || turn !== generation) return;
      data = (await api.homeworkDetail(id)).data;
      records.value[id] = data;
    }
    const record = records.value[id];
    if (
      record?.operation &&
      ['FAILED', 'CANCELLED', 'INTERRUPTED', 'EXPIRED'].includes(record.operation.status)
    ) {
      records.value[id] = {
        ...record,
        error: record.operation.message || '读取未完成，可重新读取',
      };
    }
  } catch (error) {
    if (!disposed && turn === generation)
      records.value[id] = {
        ...records.value[id],
        error: error.response?.data?.detail || '作业要求读取失败，请重试',
      };
  } finally {
    running.value = false;
    if (!disposed) void next();
  }
}
async function next() {
  if (disposed || running.value || props.disabled) return;
  const item = props.items.find((item) => !records.value[item.homework_id]);
  if (item) await read(item);
}
watch(
  () => [props.snapshotId, props.items],
  ([snapshot], [oldSnapshot] = []) => {
    if (snapshot !== oldSnapshot) {
      generation++;
      records.value = {};
    }
    void next();
  },
  { immediate: true }
);
watch(
  () => props.disabled,
  () => void next()
);
async function download(homeworkId, file) {
  try {
    const { data } = await api.homeworkDetailSource(homeworkId, file.id);
    const url = URL.createObjectURL(data),
      anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = file.name;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch {
    records.value[homeworkId] = {
      ...records.value[homeworkId],
      error: '原始文件下载失败，请重新读取作业详情',
    };
  }
}
async function cancel(homeworkId) {
  try {
    await api.cancelDetailRead(records.value[homeworkId].operation.id);
  } catch (error) {
    records.value[homeworkId] = {
      ...records.value[homeworkId],
      error: error.response?.data?.detail || '取消失败，请重试',
    };
  }
}
onBeforeUnmount(() => {
  emit('active-task', '');
  disposed = true;
  generation++;
  clearTimeout(timer);
  wake?.();
});
</script>
<style scoped>
.requirements {
  display: grid;
  gap: 14px;
}
article {
  padding: 16px;
  border: 1px solid #dbe3ee;
  border-radius: 10px;
  background: #f8fafc;
}
header,
.source-list {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
header span,
.updated {
  color: #64748b;
  font-size: 13px;
}
header button {
  margin-left: auto;
}
button {
  padding: 7px 12px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  background: white;
}
button:disabled {
  opacity: 0.5;
}
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font: inherit;
  line-height: 1.7;
  max-height: 480px;
  overflow-y: auto;
}
summary {
  cursor: pointer;
  margin: 12px 0;
  color: #2563eb;
}
.error {
  color: #b91c1c;
}
</style>
