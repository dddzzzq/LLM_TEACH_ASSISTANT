<template>
  <section class="portal-assignment" aria-label="按班级批改官网作业">
    <div class="portal-summary"
      ><div><h3>从官网获取并批改</h3><p>使用已保存的题目和评分标准，依次下载、整理并批改。</p></div
      ><button type="button" class="text-action" @click="load">刷新状态</button></div
    >
    <p class="portal-note" role="status">题目要求：{{ requirementLabel }}</p>
    <p v-if="error" role="alert" class="workspace-error">{{ error }}</p
    ><p v-if="notice" role="status" class="portal-notice">{{ notice }}</p>
    <template v-if="context?.available">
      <div class="portal-scope"
        ><strong
          >本次执行：{{
            scopeMode === 'all'
              ? `全部 ${classes.length} 个官网班级`
              : bulkMode
                ? '批量处理所选班级'
                : currentClass?.name || '请先选择官网班级'
          }}</strong
        ><div class="scope-actions"
          ><button
            type="button"
            class="text-action"
            :disabled="busy || running"
            :aria-pressed="scopeMode === 'current'"
            @click="scopeMode = 'current'"
            >当前班级</button
          >
          <button
            type="button"
            class="text-action"
            :disabled="busy || running || !classes.length"
            :aria-pressed="scopeMode === 'all'"
            @click="scopeMode = 'all'"
            >全部班级（{{ classes.length }}）</button
          >
          <button
            type="button"
            class="text-action"
            :disabled="busy || running"
            @click="bulkMode = !bulkMode"
            >{{ bulkMode ? '返回当前班级' : '批量处理班级' }}</button
          ></div
        ></div
      >
      <fieldset v-if="bulkMode" :disabled="busy || running">
        <legend>选择本次批改的班级</legend>
        <button type="button" class="text-action" @click="selected = classes.map((item) => item.id)"
          >选择全部班级</button
        >
        <label v-for="item in classes" :key="item.id"
          ><input v-model="selected" type="checkbox" :value="item.id" />{{ item.name }}</label
        >
      </fieldset>
      <p v-if="!classes.length" class="portal-note">当前目录没有班级，请更新课程与作业。</p>
      <p v-if="scopeMode === 'current' && !currentClass" class="portal-note"
        >当前为历史或未分班记录，请在班级列表选择有效官网班级；也可以使用批量处理。</p
      >
      <button
        class="workspace-button primary"
        type="button"
        :disabled="
          busy || !executionClasses.length || context.requirements_status !== 'READY' || running
        "
        @click="grade"
        >{{
          busy
            ? '正在提交…'
            : scopeMode === 'all'
              ? `下载并批改全部 ${classes.length} 个班级`
              : bulkMode
                ? '按所选班级开始批改'
                : '下载并批改本班作业'
        }}</button
      >
      <p v-if="context.requirements_status !== 'READY'" class="portal-note"
        >请先在“题目与评分标准”核对并补全要求。</p
      >
    </template>
    <p v-else-if="context?.linked" class="portal-note">官网目录已变化，请更新课程与作业后重试。</p>
    <RpaTaskPanel v-if="detailTask" :task-id="detailTask" embedded />
    <section v-if="workflow" class="class-progress" aria-live="polite">
      <header
        ><strong>本轮批改：{{ portalStatus(workflow.workflow.status) }}</strong
        ><button v-if="running" type="button" class="text-action" :disabled="busy" @click="cancel"
          >取消后续批改</button
        ><button
          v-if="['FAILED', 'PARTIAL_SUCCESS'].includes(workflow.workflow.status)"
          type="button"
          class="text-action"
          :disabled="busy"
          @click="retry"
          >重试中断的下载</button
        ></header
      >
      <div v-for="item in workflow.items" :key="item.id" class="class-progress-row"
        ><div
          ><strong>{{ item.class_name }}</strong
          ><p>{{ portalStatus(item.status) }}：{{ item.message }}</p></div
        ><button type="button" class="text-action" @click="$emit('results', item.class_id)"
          >查看本班成绩</button
        ></div
      >
      <RpaTaskPanel
        v-for="item in workflow.items.filter(
          (item) => item.status === 'WAITING_AUTH' && item.rpa_job_id
        )"
        :key="item.id"
        :task-id="item.rpa_job_id"
        embedded
      />
    </section>
  </section>
</template>
<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import authApi from '../services/authApi';
import { createPortalApi, operationActive, portalStatus } from '../services/portalApi';
import RpaTaskPanel from './RpaTaskPanel.vue';
const props = defineProps({
  assignmentId: { type: [Number, String], required: true },
  rubric: { type: Object, required: true },
  classId: { type: String, default: '' },
});
const emit = defineEmits(['context', 'results']);
const api = createPortalApi(authApi.getClient()),
  route = useRoute(),
  router = useRouter();
const context = ref(null),
  selected = ref([]),
  scopeMode = ref('all'),
  workflow = ref(null),
  detailTask = ref(''),
  busy = ref(false),
  error = ref(''),
  notice = ref('');
const classes = computed(() => context.value?.classes || []);
const currentClass = computed(() => classes.value.find((item) => item.id === props.classId));
const bulkMode = computed({
  get: () => scopeMode.value === 'selected',
  set: (value) => {
    scopeMode.value = value ? 'selected' : 'current';
  },
});
const executionClasses = computed(() =>
  scopeMode.value === 'all'
    ? classes.value.map((item) => item.id)
    : bulkMode.value
      ? selected.value.filter((id) => classes.value.some((item) => item.id === id))
      : currentClass.value
        ? [currentClass.value.id]
        : []
);
const running = computed(
  () =>
    workflow.value &&
    !['SUCCEEDED', 'PARTIAL_SUCCESS', 'FAILED', 'CANCELLED'].includes(
      workflow.value.workflow.status
    )
);
const requirementLabel = computed(
  () =>
    ({
      PENDING: '等待读取官网要求',
      READING: '正在读取和整理官网要求',
      READY: '已就绪',
      REVIEW_REQUIRED: '部分内容需要核对',
      FAILED: '读取失败，可重新同步',
      CANCELLED: '已取消',
      INTERRUPTED: '已中断',
    })[context.value?.requirements_status] || '正在检查'
);
let timer,
  disposed = false,
  loading = false,
  key = '',
  lastSelection = '';
async function load() {
  if (disposed || loading) return;
  loading = true;
  try {
    const { data } = await api.assignment(props.assignmentId);
    if (disposed) return;
    context.value = data;
    error.value = '';
    detailTask.value = operationActive(data.operation) ? data.operation.id : '';
    selected.value = selected.value.filter((id) =>
      (data.classes || []).some((item) => item.id === id)
    );
    emit('context', data);
    const workflowId = route.query.class_workflow || workflow.value?.workflow.id;
    if (workflowId) {
      const { data: state } = await api.workflow(workflowId);
      if (
        !disposed &&
        state.items.every((item) => String(item.assignment_id) === String(props.assignmentId))
      )
        workflow.value = state;
    }
  } catch (e) {
    if (!disposed) error.value = e.response?.data?.detail || '官网状态读取失败，请重试。';
  } finally {
    loading = false;
  }
}
async function grade() {
  if (busy.value || running.value || !executionClasses.value.length) return;
  const classIds = [...executionClasses.value];
  busy.value = true;
  error.value = '';
  try {
    const selection = JSON.stringify({
      mode: scopeMode.value,
      classes: classIds.sort(),
      snapshot: context.value?.snapshot_id,
    });
    if (!key || selection !== lastSelection || (workflow.value && !running.value)) {
      key = crypto.randomUUID();
      lastSelection = selection;
    }
    const { data } = await api.gradeClasses(
      props.assignmentId,
      scopeMode.value === 'all' ? [] : classIds,
      key,
      scopeMode.value === 'all'
    );
    await router.replace({ query: { ...route.query, class_workflow: data.id } });
    workflow.value = (await api.workflow(data.id)).data;
    notice.value = '已提交批改，材料与结果按各班分别保存。';
  } catch (e) {
    error.value = e.response?.data?.detail || '批改提交失败';
  } finally {
    busy.value = false;
  }
}
async function cancel() {
  busy.value = true;
  try {
    await api.cancelWorkflow(workflow.value.workflow.id);
    await load();
  } catch (e) {
    error.value = e.response?.data?.detail || '取消失败';
  } finally {
    busy.value = false;
  }
}
async function retry() {
  busy.value = true;
  try {
    await api.retryWorkflow(workflow.value.workflow.id);
    await load();
  } catch (e) {
    error.value = e.response?.data?.detail || '重试失败';
  } finally {
    busy.value = false;
  }
}
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
.portal-assignment {
  display: grid;
  gap: 16px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--color-line);
}
.portal-summary,
.portal-scope,
.class-progress header,
.class-progress-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}
h3 {
  font-size: 15px;
  font-weight: 600;
}
.portal-summary p,
.portal-note,
.class-progress-row p {
  color: var(--color-muted);
  font-size: 13px;
  line-height: 1.8;
  margin-top: 6px;
}
.portal-note {
  margin: 0;
}
.portal-scope {
  font-size: 13px;
  padding: 14px 0;
}
.scope-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
.portal-notice {
  font-size: 13px;
  color: var(--color-accent);
}
fieldset {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  padding: 16px;
  border: 1px solid var(--color-line);
  border-radius: 8px;
  font-size: 13px;
}
fieldset label {
  display: flex;
  align-items: center;
  gap: 7px;
}
legend {
  color: var(--color-muted);
  padding: 0 5px;
}
.workspace-button {
  justify-self: start;
}
.class-progress {
  display: grid;
  gap: 14px;
  border-top: 1px solid var(--color-line);
  padding-top: 20px;
  font-size: 13px;
}
.class-progress-row {
  padding: 12px 0;
  border-bottom: 1px solid var(--color-line);
}
</style>
