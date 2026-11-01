<template>
  <button v-if="notices.length" class="attention-launcher" type="button" @click="showInbox = true">
    作业待处理 <span>{{ notices.length }}</span>
  </button>
  <p v-if="success" class="attention-success" role="status">{{ success }}</p>
  <AppModal v-if="showInbox" title="作业待处理" @close="showInbox = false">
    <ul class="space-y-3">
      <li v-for="notice in notices" :key="notice.key" class="rounded-lg border p-3">
        <p class="text-sm font-semibold"
          >{{ notice.course_name }} / {{ notice.assignment_name
          }}<span v-if="notice.class_name"> · {{ notice.class_name }}</span></p
        >
        <p class="mt-1 text-sm text-slate-600">{{
          notice.kind === 'rubric'
            ? '需要补充题目或评分标准'
            : notice.kind === 'class'
              ? '需要核对班级'
              : notice.message
        }}</p>
        <button
          type="button"
          class="mt-2 text-sm text-blue-700"
          @click="
            showInbox = false;
            openNotice(notice);
          "
          >处理此项</button
        >
      </li>
    </ul>
    <template #footer
      ><button type="button" class="secondary-button" @click="showInbox = false"
        >关闭列表</button
      ></template
    >
  </AppModal>
  <AppModal
    v-if="current"
    :key="current.key"
    :title="title"
    :close-disabled="saving"
    @close="dismiss"
  >
    <p class="task-label"
      >{{ current.course_name }} / {{ current.assignment_name
      }}<span v-if="current.class_name"> · {{ current.class_name }}</span></p
    >
    <template v-if="current.kind === 'rubric'">
      <p class="notice-message"
        >附件已下载并校验完成。请补充{{
          missingText
        }}。仅已明确请求自动批改的流程会继续；普通材料仍需主动开始批改。</p
      >
      <p v-if="loading" role="status" class="mt-4 text-sm text-slate-600"
        >正在读取已保存的题目与评分标准…</p
      >
      <form
        v-else
        id="rpa-requirements-form"
        class="requirements-form"
        @submit.prevent="saveRequirements"
      >
        <label for="rpa-question"
          >题目要求<textarea
            id="rpa-question"
            v-model="question"
            :disabled="saving || !loaded"
            rows="4"
            required
            placeholder="请填写本次作业的题目、提交要求等"
            @input="error = ''"
          />
        </label>
        <RubricEditor v-model="rubric" :disabled="saving || !loaded" />
        <p class="text-xs text-slate-500"
          >按评分项填写分值和要求；已有内容会保留。JSON 仅作为高级编辑方式。</p
        >
      </form>
    </template>
    <RpaTargetResolver
      v-else-if="current.kind === 'target' && current.target_resolution"
      :key="current.target_resolution.id"
      :resolution="current.target_resolution"
      :busy="saving"
      @resolve="resolveTarget"
    />
    <template v-else>
      <p class="notice-message">{{ current.message || '任务需要你检查当前页面并继续处理。' }}</p>
      <p class="mt-4 text-sm text-slate-600">{{ guidance }}</p>
    </template>
    <p v-if="error" role="alert" class="mt-4 text-sm text-red-700">{{ error }}</p>
    <p v-if="notices.length > 1" class="mt-4 text-xs text-slate-500"
      >还有 {{ notices.length - 1 }} 项待处理，可通过右下角“作业待处理”查看。</p
    >
    <template #footer>
      <button type="button" class="secondary-button" :disabled="saving" @click="dismiss"
        >稍后处理</button
      >
      <button
        v-if="current.kind === 'rubric'"
        type="submit"
        form="rpa-requirements-form"
        class="primary-button"
        :disabled="loading || saving || !loaded"
        >{{ saving ? '正在保存…' : '保存评分配置' }}</button
      >
      <button
        v-else-if="current.reason === 'BROWSER_STARTUP'"
        type="button"
        class="primary-button"
        :disabled="saving"
        @click="reloadLogin"
        >{{ saving ? '正在请求…' : '重新加载登录页' }}</button
      >
      <button
        v-else-if="current.kind !== 'target'"
        type="button"
        class="primary-button"
        @click="handleInConsole"
        >打开控制台处理</button
      >
    </template>
  </AppModal>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue';
import { useRouter } from 'vue-router';
import authApi from '../services/authApi';
import AppModal from './AppModal.vue';
import RpaTargetResolver from './RpaTargetResolver.vue';
import RubricEditor from './RubricEditor.vue';

const router = useRouter(),
  api = authApi.getClient();
const notices = ref([]),
  current = ref(null),
  question = ref(''),
  rubric = ref('');
const showInbox = ref(false);
const loading = ref(false),
  loaded = ref(false),
  saving = ref(false),
  error = ref(''),
  success = ref('');
const user = authApi.getCurrentUser();
const allowed = ['teacher', 'admin'].includes(user?.role);
const storageKey = `rpa-attention-seen-${user?.user_id || user?.id || ''}`;
let seen = new Set(),
  timer,
  successTimer,
  inFlight = false,
  disposed = false;
const drafts = new Map();
try {
  seen = new Set(JSON.parse(sessionStorage.getItem(storageKey) || '[]'));
} catch {
  /* Storage can be unavailable. */
}
const title = computed(() =>
  current.value?.reason === 'COURSE_OR_TERM'
    ? '请确认课程名或学期'
    : current.value?.kind === 'target'
      ? current.value.reason === 'TARGET_NOT_FOUND'
        ? '请核查课程或作业名称'
        : '请确认找到的课程或作业'
      : current.value?.kind === 'rubric'
        ? '请补充题目与评分标准'
        : current.value?.kind === 'class'
          ? '请核对附件班级'
          : '作业抓取需要人工处理'
);
const missingText = computed(
  () =>
    [current.value?.question_missing && '题目要求', current.value?.rubric_missing && '评分标准']
      .filter(Boolean)
      .join('和') || '题目要求和评分标准'
);
const guidance = computed(() => {
  const reason = current.value?.reason;
  if (reason === 'BROWSER_STARTUP')
    return '系统已尝试恢复连接。点击“重新加载登录页”可在当前任务中重试，无需重新创建抓取任务。';
  if (reason === 'COURSE_OR_TERM')
    return '请核对课程名和当前学期。如需切换学期，请在任务浏览器中手动操作，完成后点击“交还自动化”。';
  if (reason === 'LOGIN') return '请在任务浏览器中完成登录和验证，再点击“交还自动化”。';
  if (reason === 'LOGIN_COMPLETE') return '官网登录已完成，请打开控制台，点击“交还自动化”继续。';
  if (['CLASS_SCOPE', 'SCOPE_SELECTION'].includes(reason))
    return '请在控制台核对导出班级范围，确认后继续。';
  if (reason === 'EXPORT_FORMAT')
    return '请在官网选中“导出提交附件”，然后交还自动化；已确认的班级范围会保留。';
  if (current.value?.kind === 'class')
    return '请在控制台核对或更正附件所属班级，系统将继续匹配作业。';
  return '请打开控制台查看当前页面，按提示处理后再交还自动化。';
});
function remember() {
  try {
    sessionStorage.setItem(storageKey, JSON.stringify([...seen]));
  } catch {
    /* No persistence required to operate. */
  }
}
function dismiss() {
  if (!current.value || saving.value) return;
  seen.add(current.value.key);
  if (current.value.kind === 'rubric' && loaded.value)
    drafts.set(current.value.key, { question: question.value, rubric: rubric.value });
  remember();
  current.value = null;
}
async function openNotice(notice) {
  if (saving.value || !notice || current.value?.key === notice.key) return;
  if (current.value) dismiss();
  current.value = notice;
  error.value = '';
  loaded.value = false;
  loading.value = false;
  question.value = '';
  rubric.value = '';
  if (notice.kind !== 'rubric') return;
  loading.value = true;
  try {
    const { data } = await api.get(`/api/rpa/jobs/${notice.job_id}/requirements`, {
      params: { file_id: notice.file_id },
    });
    if (disposed || current.value?.key !== notice.key) return;
    const draft = drafts.get(notice.key);
    question.value = draft?.question ?? data.question ?? '';
    rubric.value = draft?.rubric ?? JSON.stringify(data.rubric || {}, null, 2);
    loaded.value = true;
  } catch (e) {
    if (current.value?.key === notice.key)
      error.value = e.response?.data?.error || '读取补充信息失败，请稍后重新打开提醒。';
  } finally {
    if (current.value?.key === notice.key) loading.value = false;
  }
}
async function refresh() {
  if (!allowed || disposed || inFlight || document.hidden) return;
  inFlight = true;
  try {
    const { data } = await api.get('/api/rpa/jobs/attention');
    if (disposed) return;
    notices.value = Array.isArray(data) ? data : [];
    const keys = new Set(notices.value.map((n) => n.key));
    seen = new Set([...seen].filter((key) => keys.has(key)));
    for (const key of drafts.keys()) if (!keys.has(key)) drafts.delete(key);
    remember();
    if (current.value && !keys.has(current.value.key) && !saving.value) current.value = null;
    // Keep teachers in their current editing flow; the persistent badge opens the inbox on demand.
  } catch {
    /* A network failure must not create a human-intervention prompt. */
  } finally {
    inFlight = false;
  }
}
async function saveRequirements() {
  if (!loaded.value || saving.value || current.value?.kind !== 'rubric') return;
  error.value = '';
  let parsed;
  try {
    parsed = JSON.parse(rubric.value);
  } catch {
    error.value = '评分标准不是有效 JSON，请检查引号、逗号和括号。';
    return;
  }
  if (
    !question.value.trim() ||
    !parsed ||
    Array.isArray(parsed) ||
    typeof parsed !== 'object' ||
    !Object.keys(parsed).length
  ) {
    error.value = '请填写题目要求，评分标准需为非空 JSON 对象。';
    return;
  }
  const notice = current.value;
  saving.value = true;
  try {
    await api.post(`/api/rpa/jobs/${notice.job_id}/requirements`, {
      file_id: notice.file_id,
      question: question.value.trim(),
      rubric: parsed,
    });
    seen.add(notice.key);
    remember();
    drafts.delete(notice.key);
    current.value = null;
    // Do not re-open the old snapshot while the scheduler is consuming the update.
    notices.value = notices.value.filter((n) => n.key !== notice.key);
    success.value = '题目与评分标准已保存；后续按本次已选择的处理范围执行。';
    clearTimeout(successTimer);
    successTimer = setTimeout(() => {
      success.value = '';
    }, 8000);
  } catch (e) {
    error.value = e.response?.data?.error || '保存失败，已保留填写内容，请重试。';
  } finally {
    saving.value = false;
  }
  await refresh();
}
async function handleInConsole() {
  const id = current.value.job_id;
  dismiss();
  await router.push({ name: 'ai-assistant', query: { task: id } });
  await nextTick();
  window.dispatchEvent(new CustomEvent('rpa-open-console', { detail: { jobId: id } }));
}
async function reloadLogin() {
  if (saving.value || current.value?.reason !== 'BROWSER_STARTUP') return;
  const notice = current.value;
  saving.value = true;
  error.value = '';
  try {
    await api.post(`/api/rpa/jobs/${notice.job_id}/resume`);
    seen.add(notice.key);
    remember();
    current.value = null;
    notices.value = notices.value.filter((n) => n.key !== notice.key);
    await router.push({ name: 'ai-assistant', query: { task: notice.job_id } });
    await nextTick();
    window.dispatchEvent(new CustomEvent('rpa-open-console', { detail: { jobId: notice.job_id } }));
  } catch (e) {
    error.value = e.response?.data?.error || '重新加载未完成，请重试。';
  } finally {
    saving.value = false;
  }
  await refresh();
}
async function resolveTarget(payload) {
  if (saving.value || current.value?.kind !== 'target') return;
  const notice = current.value;
  saving.value = true;
  error.value = '';
  try {
    await api.post(`/api/rpa/jobs/${notice.job_id}/resolve-target`, payload);
    seen.add(notice.key);
    remember();
    current.value = null;
    notices.value = notices.value.filter((n) => n.key !== notice.key);
    success.value = '已保存确认，正在继续查找和拉取。';
    clearTimeout(successTimer);
    successTimer = setTimeout(() => {
      success.value = '';
    }, 8000);
    window.dispatchEvent(
      new CustomEvent('rpa-target-resolved', { detail: { jobId: notice.job_id } })
    );
  } catch (e) {
    error.value = e.response?.data?.error || '确认未完成，请重试。';
  } finally {
    saving.value = false;
  }
  await refresh();
}
async function openRequested(event) {
  if (!allowed) return;
  await refresh();
  const notice = notices.value.find(
    (n) => n.job_id === event.detail?.jobId && n.file_id === event.detail?.fileId
  );
  if (notice) await openNotice(notice);
}
onMounted(() => {
  if (!allowed) return;
  refresh();
  timer = setInterval(refresh, 5000);
  window.addEventListener('focus', refresh);
  document.addEventListener('visibilitychange', refresh);
  window.addEventListener('rpa-open-requirements', openRequested);
});
onBeforeUnmount(() => {
  disposed = true;
  clearInterval(timer);
  clearTimeout(successTimer);
  window.removeEventListener('focus', refresh);
  document.removeEventListener('visibilitychange', refresh);
  window.removeEventListener('rpa-open-requirements', openRequested);
});
</script>

<style scoped>
.attention-launcher {
  position: fixed;
  right: 24px;
  bottom: 92px;
  z-index: 60;
  padding: 12px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  border: 1px solid #e5c591;
  border-radius: 12px;
  background: #fff8e9;
  color: #825018;
  font-size: 13px;
  box-shadow: 0 4px 20px #142d4014;
}
@media (max-width: 760px) {
  .attention-launcher {
    right: 16px;
    bottom: 86px;
    max-width: calc(100vw - 32px);
  }
}
.attention-launcher span {
  min-width: 24px;
  padding: 2px 7px;
  border-radius: 12px;
  background: #a26218;
  color: white;
  font-size: 12px;
}
.attention-success {
  position: fixed;
  right: 24px;
  top: 24px;
  max-width: calc(100vw - 48px);
  z-index: 70;
  padding: 14px 18px;
  border: 1px solid #90c9bc;
  border-radius: 10px;
  background: #effaf5;
  color: #14655d;
  font-size: 13px;
}
.task-label {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.8;
  overflow-wrap: anywhere;
}
.notice-message {
  margin-top: 12px;
  padding: 12px 14px;
  border-radius: 8px;
  background: #fff8e9;
  color: #825018;
  font-size: 13px;
  line-height: 1.8;
}
.requirements-form {
  display: grid;
  gap: 16px;
  margin-top: 18px;
}
.requirements-form label {
  display: grid;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
}
.requirements-form textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #c8d3de;
  font-size: 13px;
  font-weight: 400;
  resize: vertical;
}
.secondary-button {
  padding: 10px 16px;
  border: 1px solid #c8d3de;
  border-radius: 8px;
  background: white;
  font-size: 13px;
}
button:disabled {
  opacity: 0.55;
}
</style>
