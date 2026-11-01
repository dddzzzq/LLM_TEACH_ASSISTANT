<template>
  <section class="materials-panel" aria-label="作业材料与批改">
    <header class="materials-heading"
      ><div
        ><h2>导入记录与批改</h2
        ><p>查看学生文件、处理材料异常，开始或重新批改，也可查看本批成绩。</p></div
      ><button class="secondary-button" :disabled="loading" @click="load">刷新批次</button></header
    >
    <form
      v-if="assignmentId && showDirectory"
      class="directory-form"
      @submit.prevent="prepareDirectory"
    >
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
      <p class="directory-hint"
        >目录必须能由运行服务的机器读取；这里不是上传浏览器所在电脑的文件夹。</p
      >
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
        <p v-if="grades[batch.grading_job_id]?.message" class="batch-meta">{{
          grades[batch.grading_job_id].message
        }}</p>
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
            v-if="['READY', 'NEEDS_REVIEW'].includes(batch.status)"
            class="primary-button"
            :disabled="
              busy ||
              isGrading(batch) ||
              (batch.grading_job_id && !canRegrade(batch)) ||
              (batch.status === 'NEEDS_REVIEW' && !partial[batch.id])
            "
            @click="grade(batch)"
            >{{
              isGrading(batch) ? '正在批改，请等待' : batch.grading_job_id ? '重新批改' : '开始批改'
            }}</button
          >
          <button
            v-if="!batch.grading_job_id && batch.status !== 'CANCELLED'"
            class="text-button danger"
            :disabled="busy"
            @click="cancel(batch)"
            >取消后续处理</button
          >
        </div>
        <p
          v-if="canRegrade(batch) && ['READY', 'NEEDS_REVIEW'].includes(batch.status)"
          class="batch-meta"
          >可复用本批材料重新批改，无需再次下载；更新 AI 评分，保留人工复核。</p
        >
        <label
          v-if="batch.status === 'NEEDS_REVIEW' && !isGrading(batch)"
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
        ><p v-for="source in studentSources(student)" :key="source.id" class="batch-meta"
          >附件来源：{{ source.name }}（{{ source.format.toUpperCase() }}）</p
        >
        <p
          v-for="skipped in student.skipped_files || []"
          :key="skipped.path"
          class="material-warning"
          >{{ skipped.path }}：{{ skipped.reason }}</p
        >
        <ul
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
  classId: { type: String, default: '' },
  importClassId: { type: String, default: null },
  showDirectory: { type: Boolean, default: true },
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
const regradeIntents = new Map();
const gradingStatus = (batch) => grades.value[batch.grading_job_id]?.status || batch.grading_state;
const isGrading = (batch) => ['PENDING', 'PROCESSING', 'REQUESTED'].includes(gradingStatus(batch));
const canRegrade = (batch) =>
  Boolean(batch.grading_job_id) &&
  ['SUCCESS', 'FAILED', 'CANCELLED'].includes(gradingStatus(batch));
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
      classId: props.classId,
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
    ...((props.importClassId ?? props.classId)
      ? { class_id: props.importClassId ?? props.classId }
      : {}),
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
function studentSources(student) {
  return (detail.value?.manifest?.sources || []).filter(
    (source) => source.student_id && student.source_ids?.includes(source.id)
  );
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
  if (busy.value || isGrading(batch)) return;
  const regrade = Boolean(batch.grading_job_id);
  if (
    regrade &&
    (!canRegrade(batch) ||
      !confirm(
        '确定使用本批已准备的材料重新批改？将按当前题目与评分标准更新本批 AI 评分，已有人工复核分数和评语保留。'
      ))
  )
    return;
  busy.value = true;
  error.value = '';
  try {
    let response;
    if (regrade) {
      let intent = regradeIntents.get(batch.id);
      if (!intent || intent.previous !== batch.grading_job_id) {
        intent = { previous: batch.grading_job_id, key: crypto.randomUUID() };
        regradeIntents.set(batch.id, intent);
      }
      response = await materialsApi.regrade(
        batch.id,
        Boolean(partial[batch.id]),
        intent.previous,
        intent.key
      );
      regradeIntents.delete(batch.id);
    } else response = await materialsApi.grade(batch.id, Boolean(partial[batch.id]));
    const { data } = response;
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
  () => [props.assignmentId, props.sourceJobId, props.refreshKey, props.classId],
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
  font-size: 16px;
  font-weight: 700;
  color: var(--color-ink);
}
.materials-heading p,
.batch-card p {
  font-size: 12px;
  line-height: 1.8;
  color: var(--color-muted);
  margin-top: 8px;
}
.directory-form {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin: 20px 0;
  padding: 18px;
  background: var(--color-soft);
  border-radius: 10px;
}
.directory-form > label:not(.check) {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-ink);
}
.directory-form input:not([type='checkbox']) {
  display: block;
  width: 100%;
  margin-top: 8px;
  padding: 10px;
  border: 1px solid var(--color-input-line);
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
  border: 1px solid var(--color-line);
  border-radius: 10px;
  background: var(--color-paper);
}
.batch-title {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  color: var(--color-ink);
}
.batch-state {
  font-size: 12px;
  color: var(--color-accent);
  white-space: nowrap;
}
.batch-card .batch-meta,
.batch-meta {
  font-size: 12px;
  color: var(--color-muted);
  margin-top: 8px;
}
.batch-path {
  font-size: 12px;
  overflow-wrap: anywhere;
  line-height: 1.8;
  color: var(--color-muted);
}
.batch-actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 14px;
  font-size: 12px;
  color: var(--color-accent);
}
.batch-actions .primary-button {
  margin-left: auto;
  padding: 8px 14px;
}
.text-button {
  font-size: 12px;
  color: var(--color-accent);
}
.danger {
  color: var(--color-danger);
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
  background: var(--color-danger-soft);
  color: var(--color-danger);
}
.material-notice {
  background: var(--color-success-soft);
  color: var(--color-accent);
}
.empty-materials {
  padding: 24px 0;
  font-size: 12px;
  color: var(--color-muted);
}
.material-warning {
  font-size: 12px;
  color: var(--color-warning);
  line-height: 1.8;
  margin-top: 8px;
}
.material-issues {
  font-size: 12px;
  color: var(--color-warning);
  line-height: 1.8;
  padding: 14px;
  background: var(--color-warning-soft);
  border-radius: 8px;
  margin: 16px 0;
}
.student-files {
  font-size: 12px;
  border-top: 1px solid var(--color-line);
  padding: 12px 0;
}
.student-files summary {
  cursor: pointer;
  font-weight: 600;
  color: var(--color-ink);
}
.student-files li {
  font-size: 12px;
  overflow-wrap: anywhere;
  line-height: 1.8;
  margin-top: 6px;
}
.student-files li span {
  color: var(--color-accent);
}
.secondary-button {
  font-size: 12px;
  white-space: nowrap;
  padding: 8px 12px;
  border: 1px solid var(--color-input-line);
  border-radius: 7px;
}
.primary-button:disabled {
  background: var(--color-sidebar);
  color: var(--color-muted);
  cursor: not-allowed;
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
.directory-hint {
  grid-column: 1 / -1;
  font-size: 12px;
  line-height: 1.8;
  color: var(--color-muted);
}
</style>
