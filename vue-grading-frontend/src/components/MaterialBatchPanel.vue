<template>
  <section class="materials-panel" aria-label="作业材料与批改">
    <header class="materials-heading"
      ><div
        ><h2>作业材料与批改</h2
        ><p>先整理材料，再主动开始批改。可指定本机目录，或使用官网下载、上传的批次。</p></div
      ><button class="secondary-button" :disabled="loading" @click="load">刷新批次</button></header
    >
    <form v-if="assignmentId" class="directory-form" @submit.prevent="prepareDirectory">
      <label
        >作业所在目录<input
          v-model="sourcePath"
          aria-label="作业所在目录"
          required
          placeholder="例如 D:\dzq\homework\第一次作业"
      /></label>
      <label
        >保存根目录（可选）<input
          v-model="outputRoot"
          aria-label="保存根目录"
          placeholder="留空使用项目 homework 目录"
      /></label>
      <label class="check"><input v-model="autoGrade" type="checkbox" />整理完成后继续批改</label>
      <button class="primary-button" :disabled="busy || !sourcePath.trim()">{{
        busy ? '正在提交…' : autoGrade ? '整理并批改' : '仅导入并整理'
      }}</button>
    </form>
    <p v-if="error" role="alert" class="material-error">{{ error }}</p
    ><p v-if="notice" role="status" class="material-notice">{{ notice }}</p>
    <p v-if="!batches.length && !loading" class="empty-materials"
      >暂无材料批次。导入目录、上传压缩包，或从教务系统拉取后在这里查看。</p
    >
    <div v-else class="batch-list">
      <article v-for="batch in batches" :key="batch.id" class="batch-card">
        <div class="batch-title"
          ><strong
            >{{ batch.class_name || batch.assignment_name }} · {{ date(batch.created_at) }}</strong
          ><span class="batch-state">{{ status(batch.status) }}</span></div
        >
        <p>{{ batch.message }}</p
        ><p class="batch-meta"
          >已识别 {{ batch.student_count }} 名学生 · {{ batch.issue_count }} 项异常 ·
          {{ batch.auto_grade ? '已请求整理后批改' : '独立材料批次' }}</p
        >
        <p class="batch-path">保存位置：{{ batch.output_dir }}</p>
        <p v-if="batch.grading_job_id" class="batch-meta"
          >批改：{{ status(grades[batch.grading_job_id]?.status || batch.grading_state) }} ·
          {{ batch.grading_job_id }}</p
        >
        <p v-else-if="batch.grading_state === 'WAITING_REQUIREMENTS'" class="material-warning"
          >已请求批改，等待补齐本作业的题目与评分标准。</p
        >
        <div class="batch-actions">
          <button class="text-button" @click="inspect(batch)">查看学生与文件清单</button>
          <button
            v-if="['FAILED', 'NEEDS_REVIEW'].includes(batch.status) && !batch.grading_job_id"
            class="text-button"
            :disabled="busy"
            @click="retry(batch)"
            >重新解压（不批改）</button
          >
          <router-link
            v-if="canViewLogs"
            :to="{ name: 'logs', query: { task_id: batch.id, period: '7d' } }"
            >查看日志</router-link
          >
          <router-link
            v-if="!assignmentId && batch.assignment_id"
            :to="'/assignments/' + batch.assignment_id"
            >打开作业</router-link
          >
          <button
            v-if="assignmentId && batch.grading_job_id"
            class="text-button"
            @click="emit('results', batch.id)"
            >查看本批成绩</button
          >
          <button
            v-if="!batch.grading_job_id && ['READY', 'NEEDS_REVIEW'].includes(batch.status)"
            class="primary-button"
            :disabled="busy || (batch.status === 'NEEDS_REVIEW' && !partial[batch.id])"
            @click="grade(batch)"
            >开始批改</button
          >
          <button
            v-if="!batch.grading_job_id && batch.status !== 'CANCELLED'"
            class="text-button danger"
            :disabled="busy"
            @click="cancel(batch)"
            >取消后续处理</button
          >
        </div>
        <label
          v-if="batch.status === 'NEEDS_REVIEW' && !batch.grading_job_id"
          class="check partial-choice"
          ><input
            v-model="partial[batch.id]"
            type="checkbox"
          />已查看异常，同意仅批改成功识别的学生；冲突及失败材料不计入</label
        >
      </article>
    </div>
    <AppModal v-if="detail" title="作业材料清单" @close="detail = null">
      <p class="batch-path">解压位置：{{ detail.prepared_dir || detail.batch.output_dir }}</p
      ><p class="batch-meta">批次：{{ detail.batch.id }}</p>
      <p
        v-if="detail.manifest.filtered_entries || detail.manifest.skipped_symlinks"
        class="batch-meta"
        >已过滤 {{ detail.manifest.filtered_entries || 0 }} 个依赖或缓存条目，跳过
        {{ detail.manifest.skipped_symlinks || 0 }} 个符号链接；原始附件保留。</p
      >
      <ul v-if="detail.manifest.issues?.length" class="material-issues"
        ><li v-for="(issue, i) in detail.manifest.issues" :key="i"
          >{{ issue.path }}：{{ issue.message }}</li
        ></ul
      >
      <p v-if="!detail.manifest.students?.length" class="empty-materials">{{
        detail.batch.message
      }}</p>
      <details
        v-for="student in detail.manifest.students || []"
        :key="student.student_id"
        class="student-files"
        ><summary
          >{{ student.student_id }} {{ student.student_name }} ·
          {{ student.files.length }} 个文件</summary
        ><p v-for="warning in student.warnings" :key="warning" class="material-warning">{{
          warning
        }}</p
        ><ul
          ><li v-for="file in student.files" :key="file.path"
            ><span>{{ { code: '程序', report: '报告', other: '其他' }[file.kind] }}</span>
            {{ file.path }}</li
          ></ul
        ></details
      >
      <template #footer
        ><button class="secondary-button" @click="detail = null">关闭</button></template
      >
    </AppModal>
  </section>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onBeforeUnmount, watch } from 'vue';
import authApi from '../services/authApi';
import { createMaterialsApi } from '../services/materialsApi.js';
import AppModal from './AppModal.vue';
const props = defineProps({
  assignmentId: { type: [String, Number], default: '' },
  sourceJobId: { type: String, default: '' },
  refreshKey: { type: Number, default: 0 },
});
const emit = defineEmits(['results']);
const materialsApi = createMaterialsApi(authApi.getClient());
const batches = ref([]),
  grades = ref({}),
  loading = ref(false),
  busy = ref(false),
  error = ref(''),
  notice = ref(''),
  detail = ref(null);
const sourcePath = ref(''),
  outputRoot = ref(''),
  autoGrade = ref(false),
  partial = reactive({});
const canViewLogs = computed(() => authApi.getCurrentUser()?.role === 'admin');
let timer,
  disposed = false,
  pending;
const date = (v) =>
  new Date(v).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
const status = (v) =>
  ({
    PENDING: '排队中',
    PROCESSING: '处理中',
    READY: '材料就绪',
    NEEDS_REVIEW: '需要核对',
    FAILED: '失败',
    CANCELLED: '已取消',
    SUCCESS: '批改完成',
    NOT_REQUESTED: '未发起批改',
    REQUESTED: '待整理后批改',
    WAITING_REQUIREMENTS: '等待评分配置',
  })[v] || v;
async function load() {
  if (loading.value) return;
  loading.value = true;
  try {
    const { data } = await materialsApi.list({
      assignmentId: props.assignmentId,
      sourceJobId: props.sourceJobId,
    });
    if (disposed) return;
    batches.value = data.items || [];
    grades.value = Object.fromEntries((data.grading_jobs || []).map((j) => [j.id, j]));
  } catch (e) {
    if (!disposed) error.value = e.response?.data?.error || '读取材料批次失败';
  } finally {
    loading.value = false;
  }
}
async function prepareDirectory() {
  if (busy.value) return;
  busy.value = true;
  error.value = '';
  const body = {
    assignment_id: Number(props.assignmentId),
    source_path: sourcePath.value.trim(),
    output_root: outputRoot.value.trim(),
    auto_grade: autoGrade.value,
  };
  const signature = JSON.stringify(body);
  if (pending?.signature !== signature) pending = { signature, key: crypto.randomUUID() };
  try {
    await materialsApi.create(body, pending.key);
    pending = null;
    notice.value = autoGrade.value
      ? '已创建材料整理任务，完成后按本次要求批改。'
      : '已创建材料整理任务，完成后等待你主动开始批改。';
    await load();
  } catch (e) {
    error.value = e.response?.data?.error || '导入失败';
  } finally {
    busy.value = false;
  }
}
async function inspect(batch) {
  error.value = '';
  try {
    detail.value = (await materialsApi.get(batch.id)).data;
  } catch (e) {
    error.value = e.response?.data?.error || '读取材料清单失败';
  }
}
async function retry(batch) {
  if (busy.value) return;
  busy.value = true;
  error.value = '';
  try {
    await materialsApi.retry(batch.id);
    partial[batch.id] = false;
    notice.value = '已使用原附件重新解压，无需重新拉取；完成后等待你主动开始批改。';
    await load();
  } catch (e) {
    error.value = e.response?.data?.error || '重新解压失败';
  } finally {
    busy.value = false;
  }
}
async function grade(batch) {
  busy.value = true;
  error.value = '';
  try {
    const { data } = await materialsApi.grade(batch.id, Boolean(partial[batch.id]));
    notice.value = data.message;
    await load();
  } catch (e) {
    error.value = e.response?.data?.error || e.response?.data?.message || '批改提交失败';
  } finally {
    busy.value = false;
  }
}
async function cancel(batch) {
  busy.value = true;
  error.value = '';
  try {
    await materialsApi.cancel(batch.id);
    notice.value = '已取消后续整理和自动批改，保存的文件保留。';
    await load();
  } catch (e) {
    error.value = e.response?.data?.error || '取消失败';
  } finally {
    busy.value = false;
  }
}
watch(
  () => [props.assignmentId, props.sourceJobId, props.refreshKey],
  () => {
    load();
  }
);
onMounted(() => {
  load();
  timer = setInterval(() => {
    if (!document.hidden) load();
  }, 4000);
});
onBeforeUnmount(() => {
  disposed = true;
  clearInterval(timer);
});
</script>

<style scoped>
.materials-panel {
  margin-top: 24px;
}
.materials-heading {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}
.materials-heading h2 {
  font-size: 20px;
  font-weight: 700;
  color: #20384c;
}
.materials-heading p,
.batch-card p {
  font-size: 12px;
  line-height: 1.8;
  color: #62778a;
  margin-top: 8px;
}
.directory-form {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin: 20px 0;
  padding: 18px;
  background: #f6faf9;
  border-radius: 10px;
}
.directory-form > label:not(.check) {
  font-size: 12px;
  font-weight: 600;
  color: #456075;
}
.directory-form input:not([type='checkbox']) {
  display: block;
  width: 100%;
  margin-top: 8px;
  padding: 10px;
  border: 1px solid #c8d3de;
  border-radius: 7px;
  font-size: 13px;
  font-weight: 400;
}
.check {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}
.directory-form button {
  justify-self: end;
}
.batch-list {
  display: grid;
  gap: 12px;
  margin-top: 18px;
}
.batch-card {
  padding: 18px;
  border: 1px solid #dce5ec;
  border-radius: 10px;
  background: #fff;
}
.batch-title {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  color: #20384c;
}
.batch-state {
  font-size: 11px;
  color: #197b70;
  white-space: nowrap;
}
.batch-card .batch-meta,
.batch-meta {
  font-size: 11px;
  color: #718499;
  margin-top: 8px;
}
.batch-path {
  font-size: 11px;
  overflow-wrap: anywhere;
  line-height: 1.8;
  color: #62778a;
}
.batch-actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 14px;
  font-size: 12px;
  color: #197b70;
}
.batch-actions .primary-button {
  margin-left: auto;
  padding: 8px 14px;
}
.text-button {
  font-size: 12px;
  color: #197b70;
}
.danger {
  color: #a63737;
}
.partial-choice {
  margin-top: 14px;
  line-height: 1.7;
}
.material-error,
.material-notice {
  font-size: 12px;
  padding: 12px;
  border-radius: 7px;
  margin-top: 14px;
  line-height: 1.8;
  background: #fff1f1;
  color: #a63737;
}
.material-notice {
  background: #edf7f4;
  color: #197b70;
}
.empty-materials {
  padding: 24px 0;
  font-size: 12px;
  color: #718499;
}
.material-warning {
  font-size: 11px;
  color: #92581a;
  line-height: 1.8;
  margin-top: 8px;
}
.material-issues {
  font-size: 12px;
  color: #92581a;
  line-height: 1.8;
  padding: 14px;
  background: #fff8e9;
  border-radius: 8px;
  margin: 16px 0;
}
.student-files {
  font-size: 12px;
  border-top: 1px solid #e0e7ee;
  padding: 12px 0;
}
.student-files summary {
  cursor: pointer;
  font-weight: 600;
  color: #20384c;
}
.student-files li {
  font-size: 11px;
  overflow-wrap: anywhere;
  line-height: 1.8;
  margin-top: 6px;
}
.student-files li span {
  color: #197b70;
}
.secondary-button {
  font-size: 12px;
  white-space: nowrap;
  padding: 8px 12px;
  border: 1px solid #c8d3de;
  border-radius: 7px;
}
.primary-button:disabled {
  opacity: 0.5;
}
@media (max-width: 640px) {
  .directory-form {
    grid-template-columns: 1fr;
  }
  .directory-form button {
    justify-self: stretch;
  }
  .materials-heading {
    flex-wrap: wrap;
  }
  .batch-title {
    flex-wrap: wrap;
  }
  .batch-actions .primary-button {
    margin-left: 0;
  }
}
</style>
