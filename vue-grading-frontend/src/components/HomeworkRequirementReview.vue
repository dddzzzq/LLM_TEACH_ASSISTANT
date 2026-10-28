<template>
  <section class="requirement-review" aria-label="作业要求核对" :aria-busy="loading">
    <p v-if="loading && !record" role="status" class="review-note">正在读取已提取的要求…</p>
    <div v-if="error" class="workspace-error" role="alert"
      >{{ error }} <button type="button" @click="load">重新加载</button></div
    >
    <template v-if="record?.detail">
      <div v-if="!record.detail.complete" class="review-warning" role="status">
        <h3>已读取内容，等待教师核对</h3>
        <p>下面是已保存的要求，尚未用于批改。请核对疑点，必要时补充课件或其他要求。</p>
        <ul
          ><li v-for="warning in record.detail.warnings || []" :key="warning">{{ warning }}</li></ul
        >
        <p v-if="!record.detail.warnings?.length"
          >原始内容未被标记为完整，请核对正文及来源后再采用。</p
        >
      </div>
      <p v-if="record.detail.review" class="review-note" role="status"
        >已由教师核对确认。已有成绩不会重新计算。</p
      >
      <RequirementContent
        v-if="record.detail.question"
        class="full-question"
        :text="record.detail.question"
      />
      <p v-else class="review-note">本次未提取到可用正文。请查看原始文字或附件，并在下方补充。</p>
      <details
        v-if="record.detail.raw_question || record.detail.sources?.length"
        class="original-sources"
      >
        <summary>原始文字与来源</summary>
        <pre v-if="record.detail.raw_question">{{ record.detail.raw_question }}</pre>
        <div
          v-for="(source, index) in record.detail.sources || []"
          :key="source.id || index"
          class="source-item"
        >
          <span>{{
            source.name || (source.kind === 'text' ? '官网页面文字' : '来源 ' + (index + 1))
          }}</span>
          <p v-if="source.error">{{ source.error }}</p>
          <pre v-if="source.text && !record.detail.raw_question">{{ source.text }}</pre>
          <button
            v-if="source.sha256 && source.id"
            class="text-action"
            type="button"
            @click="download(source)"
            >下载原始{{ source.kind === 'image' ? '图片' : '附件' }}</button
          >
        </div>
        <p v-if="!record.detail.sources?.some((source) => source.sha256)" class="review-note"
          >本次来源为文字，没有可下载的图片或附件。</p
        >
      </details>
      <form v-if="!record.detail.complete" class="review-form" @submit.prevent="submit">
        <label :for="editorId"
          >核对后的作业要求<textarea
            :id="editorId"
            v-model="draft"
            rows="8"
            required
            :disabled="saving || readActive"
            @input="error = ''"
          />
        </label>
        <label class="review-check"
          ><input
            v-model="confirmed"
            type="checkbox"
            :disabled="saving || readActive"
          />我已核对上述疑点，确认这份正文可作为本次作业的批改依据</label
        >
        <p class="review-note"
          >保存将使要求就绪，不会新建批改任务；此前明确请求、正在等待要求的自动流程可能继续。</p
        >
        <p v-if="readActive" class="review-note"
          >新的读取任务正在进行，请等待完成后核对最新内容。</p
        >
        <button
          class="workspace-button primary"
          :disabled="saving || readActive || !confirmed || !draft.trim()"
          >{{ saving ? '保存中…' : '保存并确认要求' }}</button
        >
      </form>
    </template>
    <template v-else-if="!loading">
      <RequirementContent v-if="fallbackQuestion" class="full-question" :text="fallbackQuestion" />
      <p v-else class="review-note"
        >暂无可核对的提取结果。请重新同步要求；如仍为空，请检查读取任务的错误提示。</p
      >
    </template>
    <p v-if="notice" class="review-notice" role="status">{{ notice }}</p>
  </section>
</template>
<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import authApi from '../services/authApi';
import { createPortalApi, operationActive } from '../services/portalApi';
import RequirementContent from './RequirementContent.vue';
const props = defineProps({
  homeworkId: { type: String, required: true },
  requirementsStatus: String,
  fallbackQuestion: String,
});
const emit = defineEmits(['confirmed']);
const api = createPortalApi(authApi.getClient());
const record = ref(null),
  draft = ref(''),
  confirmed = ref(false),
  loading = ref(false),
  saving = ref(false),
  error = ref(''),
  notice = ref('');
const readActive = computed(
  () =>
    operationActive(record.value?.operation) ||
    ['PENDING', 'READING'].includes(props.requirementsStatus)
);
const editorId = 'requirement-review-' + globalThis.crypto.randomUUID();
let generation = 0,
  disposed = false;
async function load() {
  const turn = ++generation,
    id = props.homeworkId;
  loading.value = true;
  error.value = '';
  try {
    const { data } = await api.homeworkDetail(id);
    if (disposed || turn !== generation) return;
    const sameRevision = record.value?.operation_id === data.operation_id;
    record.value = data;
    if (!sameRevision || !draft.value) {
      draft.value = data.detail?.question || data.detail?.raw_question || '';
      confirmed.value = false;
    }
  } catch (e) {
    if (!disposed && turn === generation)
      error.value = e.response?.data?.detail || '已提取要求暂时无法读取，请重试。';
  } finally {
    if (turn === generation) loading.value = false;
  }
}
async function submit() {
  if (saving.value || readActive.value || !confirmed.value || !draft.value.trim()) return;
  const turn = generation,
    id = props.homeworkId,
    question = draft.value.trim();
  saving.value = true;
  error.value = '';
  try {
    await api.reviewHomeworkDetail(id, {
      snapshot_id: record.value.snapshot_id,
      operation_id: record.value.operation_id,
      question,
      confirmed: true,
    });
    if (disposed || turn !== generation) return;
    emit('confirmed', question);
    await load();
    notice.value = '要求已核对并保存，后续批改使用确认后的正文。';
  } catch (e) {
    if (!disposed && turn === generation)
      error.value =
        e.response?.status === 409
          ? '这份要求已更新或正在重新读取。请重新加载最新结果再核对；当前编辑内容已保留。'
          : e.response?.data?.detail || '保存失败，编辑内容已保留，请重试。';
  } finally {
    saving.value = false;
  }
}
async function download(source) {
  try {
    const { data } = await api.homeworkDetailSource(props.homeworkId, source.id);
    const url = URL.createObjectURL(data),
      anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = source.name || '原始材料';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch {
    error.value = '原始材料读取失败，请重试。';
  }
}
watch(
  () => props.homeworkId,
  () => {
    record.value = null;
    draft.value = '';
    notice.value = '';
    void load();
  },
  { immediate: true }
);
watch(
  () => props.requirementsStatus,
  () => void load()
);
onBeforeUnmount(() => {
  disposed = true;
  generation++;
});
</script>
<style scoped>
.requirement-review {
  min-width: 0;
}
.review-warning {
  padding: 18px;
  margin-bottom: 22px;
  border-radius: var(--radius-control);
  background: var(--color-warning-soft);
  color: var(--color-warning);
  font-size: 13px;
  line-height: 1.9;
}
.review-warning h3 {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 8px;
}
.review-warning ul {
  list-style: disc;
  padding-left: 20px;
  margin-top: 8px;
}
.full-question {
  min-height: 100px;
  line-height: 1.9;
  font-size: 16px;
}
.review-note {
  color: var(--color-muted);
  font-size: 13px;
  line-height: 1.8;
  margin: 12px 0;
}
.original-sources {
  margin: 22px 0;
  padding: 16px 0;
  border-top: 1px solid var(--color-line);
  border-bottom: 1px solid var(--color-line);
  font-size: 13px;
}
.original-sources pre {
  white-space: pre-wrap;
  overflow-wrap: break-word;
  font-family: var(--font-body);
  color: var(--color-muted);
  line-height: 1.8;
  padding: 16px 0;
}
.source-item {
  padding-top: 12px;
}
.source-item button {
  display: block;
  margin-top: 8px;
}
.review-form,
.review-form > label:first-child {
  display: grid;
  gap: 12px;
}
.review-form {
  margin-top: 22px;
  font-size: 13px;
}
.review-form textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid var(--color-line);
}
.review-check {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  line-height: 1.8;
}
.review-check input {
  flex-shrink: 0;
  margin-top: 5px;
}
.review-form button {
  justify-self: start;
}
.review-notice {
  color: var(--color-success);
  padding-top: 16px;
  font-size: 13px;
}
</style>
