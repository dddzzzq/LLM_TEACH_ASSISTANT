<template>
  <div class="form-page grading-workbench">
    <header
      ><h1 class="page-title">作业批改工作台</h1
      ><p class="page-description"
        >按三个步骤开始：选择作业、填写要求、提交批改。全部班级默认一并处理。</p
      ></header
    >
    <p class="workflow-step">提交后可在“批改记录”查看结果，在“作业材料”处理异常。</p>
    <form class="form-surface workbench-form" @submit.prevent="submit">
      <fieldset class="source-options"
        ><legend>1. 选择作业来源</legend
        ><label v-for="option in sources" :key="option.value"
          ><input
            v-model="source"
            type="radio"
            name="material_source"
            :value="option.value"
            :disabled="busy"
          />{{ option.label }}</label
        ></fieldset
      >
      <PortalAssignmentPicker
        v-if="source === 'portal'"
        :disabled="busy"
        @change="selection = $event"
      />
      <template v-else>
        <div class="local-target"
          ><label>课程名称<input v-model="courseName" required maxlength="255" /></label
          ><label>作业名称<input v-model="taskName" required maxlength="255" /></label
        ></div>
        <label v-if="source === 'upload'"
          >上传作业压缩包<input
            type="file"
            accept=".zip,.rar,.7z"
            required
            @change="file = $event.target.files[0] || null"
        /></label>
        <label v-else
          >作业所在目录<input
            v-model="directory"
            required
            placeholder="例如 D:\dzq\homework\第一次作业"
          /><small>填写运行本系统的服务器可读取的绝对路径。</small></label
        >
      </template>
      <label
        >2. 填写题目要求<textarea
          aria-label="题目要求"
          v-model="question"
          rows="5"
          required
          placeholder="描述本次作业要求，作为批改依据"
        />
      </label>
      <RubricEditor v-model="rubric" />
      <h2 class="submission-title">3. 提交批改</h2>
      <p
        >点击开始批改后，将按本次选定范围自动下载或读取附件、整理材料并评分；需核对的异常会保留在材料批次中。</p
      >
      <button
        class="primary"
        type="submit"
        :disabled="busy || (source === 'portal' && !selection.items.length)"
        >{{ busy ? '正在提交…' : '开始批改' }}</button
      >
    </form>
    <p v-if="error" role="alert" class="error">{{ error }}</p
    ><p v-if="notice" role="status">{{ notice }}</p>
    <section v-if="workflow" class="form-surface workflow-progress" aria-live="polite">
      <header
        ><h2>批改流程：{{ portalStatus(workflow.workflow.status) }}</h2
        ><button v-if="!finished" :disabled="busy" @click="cancelWorkflow">取消后续处理</button
        ><button
          v-if="['FAILED', 'PARTIAL_SUCCESS'].includes(workflow.workflow.status)"
          :disabled="busy"
          @click="retryWorkflow"
          >重试中断的下载</button
        ><button v-if="finished" :disabled="busy" @click="newRound">新建下一轮</button></header
      >
      <p>流程编号：{{ workflow.workflow.id }}</p>
      <article v-for="item in workflow.items" :key="item.id"
        ><strong>{{ item.course_name }} / {{ item.homework_name }} / {{ item.class_name }}</strong
        ><p>{{ portalStatus(item.status) }} · {{ item.message }}</p
        ><router-link v-if="item.assignment_id" :to="'/assignments/' + item.assignment_id"
          >查看材料和批改结果</router-link
        ><router-link
          v-if="item.status === 'WAITING_AUTH' && !item.rpa_job_id"
          to="/settings/portal"
          >完成教务认证</router-link
        ><RpaTaskPanel
          v-if="item.rpa_job_id && item.status === 'WAITING_AUTH'"
          :task-id="item.rpa_job_id"
          embedded
      /></article>
    </section>
    <section v-if="assignmentId" class="form-surface"
      ><router-link :to="'/assignments/' + assignmentId">查看本次作业与批改结果 →</router-link
      ><MaterialBatchPanel :assignment-id="assignmentId"
    /></section>
  </div>
</template>
<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import authApi from '../services/authApi';
import gradingApi from '../services/gradingApi';
import { createMaterialsApi } from '../services/materialsApi';
import { createPortalApi, portalStatus } from '../services/portalApi';
import PortalAssignmentPicker from '../components/PortalAssignmentPicker.vue';
import MaterialBatchPanel from '../components/MaterialBatchPanel.vue';
import RpaTaskPanel from '../components/RpaTaskPanel.vue';
import RubricEditor from '../components/RubricEditor.vue';
const route = useRoute(),
  router = useRouter(),
  api = createPortalApi(authApi.getClient()),
  materials = createMaterialsApi(authApi.getClient());
const sources = [
  { value: 'upload', label: '上传压缩包' },
  { value: 'directory', label: '指定目录' },
  { value: 'portal', label: '选择官网作业' },
];
const source = ref('portal'),
  courseName = ref(''),
  className = ref('全部班级'),
  taskName = ref(''),
  question = ref(''),
  rubric = ref(
    JSON.stringify({
      正确性: { description: '程序运行正确，符合题目要求', score: 60 },
      完成度: { description: '功能完整，提交材料齐全', score: 20 },
      代码质量: { description: '结构清晰，命名合理', score: 20 },
    })
  ),
  directory = ref(''),
  file = ref(null),
  selection = ref({ snapshot_id: '', items: [] });
const busy = ref(false),
  error = ref(''),
  notice = ref(''),
  workflow = ref(null),
  assignmentId = ref(null);
let timer,
  disposed = false,
  polling = false,
  submissionKey = '',
  submissionBody = '',
  localConfig = '';
const finished = computed(() =>
  ['SUCCEEDED', 'PARTIAL_SUCCESS', 'FAILED', 'CANCELLED'].includes(workflow.value?.workflow.status)
);
async function loadWorkflow() {
  const id = route.query.workflow;
  if (!id || disposed || polling) return;
  polling = true;
  try {
    const { data } = await api.workflow(id);
    if (!disposed) workflow.value = data;
  } catch (e) {
    if (!disposed) error.value = e.response?.data?.detail || '批改流程读取失败';
  } finally {
    polling = false;
  }
}
async function submit() {
  busy.value = true;
  error.value = '';
  notice.value = '';
  try {
    const parsed = JSON.parse(rubric.value);
    if (!parsed || typeof parsed !== 'object' || !Object.keys(parsed).length)
      throw new Error('评分标准必须为非空 JSON 对象或数组');
    if (source.value === 'portal') {
      const body = {
          source: 'portal',
          ...selection.value,
          question: question.value,
          rubric: parsed,
        },
        serialized = JSON.stringify(body);
      if (serialized !== submissionBody) {
        submissionKey = crypto.randomUUID();
        submissionBody = serialized;
      }
      const { data } = await api.createWorkflow(body, submissionKey);
      await router.replace({ query: { workflow: data.id } });
      await loadWorkflow();
      notice.value = '批改流程已受理，可离开页面，后台会继续执行';
    } else {
      if (source.value === 'upload' && !file.value) throw new Error('请选择作业压缩包');
      const config = JSON.stringify({
        course_name: courseName.value,
        class_name: className.value,
        task_name: taskName.value,
        question: question.value,
        rubric: parsed,
      });
      if (!assignmentId.value || localConfig !== config) {
        const { data } = await gradingApi.createAssignment(JSON.parse(config));
        assignmentId.value = data.id;
        localConfig = config;
      }
      if (source.value === 'upload') {
        const body = new FormData();
        body.append('batch_file', file.value);
        body.append('auto_grade', 'true');
        await gradingApi.submitBatch(assignmentId.value, body);
      } else
        await materials.create(
          { source_path: directory.value, assignment_id: assignmentId.value, auto_grade: true },
          crypto.randomUUID()
        );
      notice.value = '材料已受理，整理完成后自动批改';
    }
  } catch (e) {
    error.value =
      e.response?.data?.detail ||
      (e instanceof SyntaxError ? '评分标准不是有效 JSON' : e.message) ||
      '提交失败';
  } finally {
    busy.value = false;
  }
}
async function cancelWorkflow() {
  busy.value = true;
  try {
    await api.cancelWorkflow(workflow.value.workflow.id);
    await loadWorkflow();
  } catch (e) {
    error.value = e.response?.data?.detail || '取消未完成';
  } finally {
    busy.value = false;
  }
}
async function retryWorkflow() {
  busy.value = true;
  error.value = '';
  try {
    await api.retryWorkflow(workflow.value.workflow.id);
    await loadWorkflow();
  } catch (e) {
    error.value = e.response?.data?.detail || '重试未完成';
  } finally {
    busy.value = false;
  }
}
async function newRound() {
  submissionKey = '';
  submissionBody = '';
  workflow.value = null;
  notice.value = '';
  await router.replace({ query: {} });
}
onMounted(() => {
  loadWorkflow();
  timer = setInterval(loadWorkflow, 3000);
});
onBeforeUnmount(() => {
  disposed = true;
  clearInterval(timer);
});
</script>
<style scoped>
.grading-workbench {
  display: grid;
  gap: 22px;
}
.form-surface {
  padding: 24px;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  background: white;
}
.workbench-form {
  display: grid;
  gap: 20px;
}
label {
  display: grid;
  gap: 8px;
}
input:not([type='radio']),
textarea {
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 10px;
  width: 100%;
}
.source-options {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}
.source-options label {
  display: flex;
  align-items: center;
  gap: 8px;
}
.source-options legend {
  margin-bottom: 10px;
  font-weight: 600;
}
.local-target {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}
.rubric {
  font-family: monospace;
}
button {
  padding: 10px 16px;
  border-radius: 8px;
  border: 1px solid #cbd5e1;
}
button.primary {
  background: #2563eb;
  color: white;
}
button:disabled {
  opacity: 0.5;
}
.error {
  color: #b91c1c;
}
a {
  color: #2563eb;
}
.workflow-progress header {
  display: flex;
  justify-content: space-between;
  gap: 14px;
}
.workflow-progress article {
  padding: 18px 0;
  border-top: 1px solid #e2e8f0;
  margin-top: 16px;
}
small,
p {
  color: #64748b;
}
</style>
