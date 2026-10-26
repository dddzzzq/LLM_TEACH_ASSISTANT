<template>
  <section class="portal-assignment" aria-label="按班级批改官网作业">
    <header><h2>按班级批改</h2><button type="button" @click="load">刷新同步状态</button></header>
    <p>这份作业已随官网同步创建。各班级共用当前题目和评分标准，材料与结果按班级区分。</p>
    <p role="status">官网要求：{{ requirementLabel }}</p>
    <p v-if="context?.course?.term">学期：{{ context.course.term }}</p>
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <p v-if="notice" role="status">{{ notice }}</p>
    <template v-if="context?.available">
      <button
        v-if="!['PENDING', 'READING'].includes(context.requirements_status)"
        type="button"
        :disabled="busy"
        @click="readRequirements"
        >重新同步官网要求</button
      >
      <details
        ><summary>编辑评分标准（默认模板可修改）</summary>
        <form @submit.prevent="saveRubric"
          ><RubricEditor v-model="rubricText" /><button
            class="primary"
            :disabled="busy || !rubricText"
            >保存评分标准</button
          ></form
        >
      </details>
      <fieldset :disabled="busy"
        ><legend>选择本次批改的班级</legend>
        <button type="button" @click="selected = classes.map((c) => c.id)">选择全部班级</button>
        <label v-for="item in classes" :key="item.id"
          ><input v-model="selected" type="checkbox" :value="item.id" />{{ item.name }}</label
        >
        <p v-if="!classes.length">当前目录未读取到班级，请更新课程与作业后重试。</p>
      </fieldset>
      <button
        class="primary"
        type="button"
        :disabled="busy || !selected.length || context.requirements_status !== 'READY' || running"
        @click="grade"
        >按所选班级开始批改</button
      >
    </template>
    <p v-else-if="context?.linked">官网目录已变化，请更新课程与作业后重试。</p>
    <RpaTaskPanel v-if="detailTask" :task-id="detailTask" embedded />
    <section v-if="workflow" class="class-progress" aria-live="polite">
      <header
        ><strong>本轮批改：{{ portalStatus(workflow.workflow.status) }}</strong
        ><button v-if="running" type="button" @click="cancel">取消后续批改</button></header
      >
      <p v-for="item in workflow.items" :key="item.id"
        >{{ item.class_name }}：{{ portalStatus(item.status) }} · {{ item.message }}</p
      >
      <RpaTaskPanel
        v-for="item in workflow.items.filter((i) => i.status === 'WAITING_AUTH' && i.rpa_job_id)"
        :key="item.id"
        :task-id="item.rpa_job_id"
        embedded
      />
      <button type="button" @click="$emit('results')">查看本作业评分结果</button>
    </section>
  </section>
</template>
<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import authApi from '../services/authApi';
import { createPortalApi, operationActive, portalStatus } from '../services/portalApi';
import RubricEditor from './RubricEditor.vue';
import RpaTaskPanel from './RpaTaskPanel.vue';
const props = defineProps({
  assignmentId: { type: [Number, String], required: true },
  rubric: { type: Object, required: true },
});
const emit = defineEmits(['context', 'rubric', 'results']);
const api = createPortalApi(authApi.getClient());
const route = useRoute(),
  router = useRouter();
const context = ref(null),
  selected = ref([]),
  rubricText = ref(JSON.stringify(props.rubric)),
  workflow = ref(null),
  detailTask = ref(''),
  busy = ref(false),
  error = ref(''),
  notice = ref('');
const classes = computed(() => context.value?.classes || []);
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
      PENDING: '已创建作业，等待自动读取官网要求',
      READING: '正在读取和整理官网要求',
      READY: '同步完成',
      REVIEW_REQUIRED: '部分内容需要核对',
      FAILED: '读取失败，可重新同步',
      CANCELLED: '已取消读取',
      INTERRUPTED: '读取已中断',
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
    if (operationActive(data.operation)) detailTask.value = data.operation.id;
    selected.value = selected.value.filter((id) => (data.classes || []).some((c) => c.id === id));
    emit('context', data);
    if (detailTask.value && data.available) {
      const { data: state } = await api.homeworkDetail(data.homework.id);
      if (!operationActive(state.operation)) detailTask.value = '';
    }
    const workflowId = workflow.value?.workflow.id || route.query.class_workflow;
    if (workflowId) {
      const result = (await api.workflow(workflowId)).data;
      if (result.items.every((item) => String(item.assignment_id) === String(props.assignmentId)))
        workflow.value = result;
    }
  } catch (e) {
    if (!disposed) error.value = e.response?.data?.detail || '作业同步状态读取失败';
  } finally {
    loading = false;
  }
}
async function saveRubric() {
  busy.value = true;
  error.value = '';
  try {
    const rubric = JSON.parse(rubricText.value);
    await api.saveRubric(props.assignmentId, rubric);
    emit('rubric', rubric);
    notice.value = '评分标准已保存，后续批改使用此标准';
  } catch (e) {
    error.value = e.response?.data?.detail || '评分标准无效或保存失败';
  } finally {
    busy.value = false;
  }
}
async function readRequirements() {
  busy.value = true;
  error.value = '';
  try {
    const { data } = await api.readHomeworkDetail(
      context.value.homework.id,
      context.value.snapshot_id
    );
    detailTask.value = data.operation_id;
    await load();
  } catch (e) {
    error.value = e.response?.data?.detail || '重新读取未完成';
  } finally {
    busy.value = false;
  }
}
async function grade() {
  busy.value = true;
  error.value = '';
  try {
    const selection = JSON.stringify([...selected.value].sort());
    if (!key || selection !== lastSelection || (workflow.value && !running.value)) {
      key = crypto.randomUUID();
      lastSelection = selection;
    }
    const { data } = await api.gradeClasses(props.assignmentId, selected.value, key);
    await router.replace({ query: { ...route.query, class_workflow: data.id } });
    workflow.value = (await api.workflow(data.id)).data;
    notice.value = '已按班级提交批改，结果归入当前作业';
  } catch (e) {
    error.value = e.response?.data?.detail || '班级批改提交失败';
  } finally {
    busy.value = false;
  }
}
async function cancel() {
  try {
    await api.cancelWorkflow(workflow.value.workflow.id);
    await load();
  } catch (e) {
    error.value = e.response?.data?.detail || '取消失败';
  }
}
onMounted(() => {
  load();
  timer = setInterval(load, 4000);
});
onBeforeUnmount(() => {
  disposed = true;
  clearInterval(timer);
});
</script>
<style scoped>
.portal-assignment {
  display: grid;
  gap: 14px;
  padding: 20px;
  border: 1px solid #dbe3ee;
  border-radius: 12px;
  background: #f8fafc;
  margin-bottom: 22px;
}
header {
  display: flex;
  gap: 12px;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
}
h2 {
  font-size: 20px;
  font-weight: 600;
}
fieldset {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  padding: 14px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
}
label {
  display: flex;
  gap: 7px;
  align-items: center;
}
button {
  padding: 8px 12px;
  border: 1px solid #cbd5e1;
  border-radius: 7px;
  background: white;
}
.primary {
  background: #4f46e5;
  color: white;
  justify-self: start;
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
summary {
  cursor: pointer;
  color: #4338ca;
  margin-bottom: 12px;
}
.class-progress {
  display: grid;
  gap: 12px;
  border-top: 1px solid #dbe3ee;
  padding-top: 14px;
}
.error {
  color: #b91c1c;
}
</style>
