<template>
  <AppModal
    :title="'学生复核 / ' + (submission.student_name || submission.student_id)"
    :close-disabled="saving"
    drawer
    @close="close"
  >
    <div class="review-navigation" aria-label="切换复核学生">
      <button type="button" :disabled="saving || !canPrevious" @click="navigate(-1)">上一位</button>
      <span>第 {{ position }} / {{ total }} 位</span>
      <button type="button" :disabled="saving || !canNext" @click="navigate(1)">下一位</button>
    </div>
    <p class="review-intro"
      >学号 {{ submission.student_id }}。结合原始评分和检测证据，保存教师最终判断。</p
    >
    <section class="original-review">
      <div
        ><span>AI 原始评分</span
        ><strong>{{
          submission.reading_status === 'NEEDS_REVIEW'
            ? '待复核'
            : submission.score < 0
              ? '评分失败'
              : Number(submission.score).toFixed(1)
        }}</strong></div
      >
      <p>{{ submission.feedback || '暂无原始评语' }}</p>
    </section>
    <div class="review-evidence">
      <details
        v-if="submission.reading_report && submission.reading_status"
        :open="submission.reading_status === 'NEEDS_REVIEW'"
      >
        <summary>材料读取与原始文件</summary>
        <p>{{
          submission.reading_report.review_stage === 'SCORING'
            ? '材料已读取，评分结果尚未通过核验，请复核。'
            : submission.reading_status === 'ERROR'
              ? '材料读取记录已保留，评分服务调用失败。'
              : submission.reading_status === 'COMPLETE'
                ? '相关材料已读取；下载产物与依赖保留清单。'
                : '部分材料未完成读取，请核对原文件，系统未将读取失败判为缺交。'
        }}</p>
        <ul
          ><li v-for="(warning, index) in submission.reading_report.warnings || []" :key="index">{{
            warning
          }}</li></ul
        >
        <div
          v-if="submission.reading_report.submission_archives?.some((source) => source.student_id)"
        >
          <p>实际附件来源（平台导出可能包含多层压缩）：</p>
          <p
            v-for="source in submission.reading_report.submission_archives.filter(
              (source) => source.student_id
            )"
            :key="source.id"
            class="source-file"
            >{{ source.name }}（{{ source.format.toUpperCase() }}）</p
          >
        </div>
        <div
          v-for="file in submission.reading_report.files || []"
          :key="file.id"
          class="source-file"
        >
          <p
            >{{ file.id }} · {{ file.path }} ·
            {{
              file.status === 'INVENTORY_ONLY'
                ? '仅保留清单'
                : file.status === 'READ'
                  ? '已读取'
                  : '待复核'
            }}</p
          >
          <button
            v-if="submission.material_batch_id"
            type="button"
            class="workspace-button"
            @click="openEvidence(file.path)"
            >查看原文件</button
          >
        </div>
      </details>
      <details v-if="submission.score_details?.length" open>
        <summary>逐项评分与材料依据</summary>
        <div
          v-for="item in submission.score_details"
          :key="item.criterion"
          class="criterion-evidence"
        >
          <p
            ><strong>{{ item.criterion }}：{{ item.score }} / {{ item.max_score }}</strong></p
          >
          <p>{{ item.feedback }}</p>
          <p v-for="evidence in item.evidence || []" :key="evidence.id">
            {{ evidence.id }} · {{ evidence.path }} ·
            {{
              evidence.pages?.length
                ? '第 ' + evidence.pages.join('、') + ' 页'
                : '行 ' + evidence.line_start + '–' + evidence.line_end
            }}
            <button
              v-if="submission.material_batch_id"
              type="button"
              class="workspace-button"
              @click="openEvidence(evidence.path, evidence.pages?.[0])"
              >查看原文件</button
            >
          </p>
        </div>
      </details>
      <p v-if="evidenceError" role="alert">{{ evidenceError }}</p>
      <details
        ><summary>查重报告与片段对比</summary
        ><ReportModal :submission="submission" report-type="plagiarism" embedded
      /></details>
      <details
        ><summary>AIGC 检测</summary
        ><template v-if="submission.aigc_report"
          ><p
            >AI 生成概率：{{
              Number.isFinite(submission.aigc_report.ai_probability)
                ? (submission.aigc_report.ai_probability * 100).toFixed(1) + '%'
                : '未提供'
            }}</p
          ><p>检测来源：{{ submission.aigc_report.detection_source || '未提供' }}</p
          ><p v-if="submission.aigc_report.predicted_label"
            >检测标签：{{ submission.aigc_report.predicted_label }}</p
          ></template
        ><p v-else>未检测</p></details
      >
      <details
        ><summary>代码与文档匹配</summary
        ><template v-if="submission.code_doc_match_report"
          ><p>匹配得分 {{ submission.code_doc_match_report.score }}</p
          ><p>{{ submission.code_doc_match_report.reasoning }}</p></template
        ><p v-else>未检测或不适用</p></details
      >
    </div>
    <form id="human-review-form" class="review-form" @submit.prevent="saveReview">
      <h3>教师最终评分</h3>
      <label for="human_score"
        >最终分数（满分 {{ maxScore }} 分）<input
          id="human_score"
          v-model.number="editableScore"
          type="number"
          min="0"
          :max="maxScore"
          step="0.1"
          required
          :disabled="saving"
      /></label>
      <label for="human_feedback"
        >最终评语<textarea
          id="human_feedback"
          v-model="editableFeedback"
          rows="6"
          required
          :disabled="saving"
        />
      </label>
      <p v-if="error" role="alert" class="workspace-error">{{ error }}</p>
    </form>
    <template #footer
      ><button type="button" class="workspace-button" :disabled="saving" @click="close">取消</button
      ><button form="human-review-form" type="submit" class="workspace-button" :disabled="saving">{{
        saving ? '保存中…' : '仅保存'
      }}</button
      ><button
        v-if="hasNextUnreviewed"
        form="human-review-form"
        type="submit"
        value="next"
        class="workspace-button primary"
        :disabled="saving"
        >{{ saving ? '保存中…' : '保存并复核下一位' }}</button
      ></template
    >
  </AppModal>
</template>
<script setup>
import { ref } from 'vue';
import AppModal from './AppModal.vue';
import ReportModal from './ReportModal.vue';
import { finalScore } from '../services/classWorkspace';
import authApi from '../services/authApi';
import { createMaterialsApi } from '../services/materialsApi';
const props = defineProps({
  submission: { type: Object, required: true },
  saving: Boolean,
  error: String,
  maxScore: { type: Number, default: 100 },
  position: { type: Number, default: 1 },
  total: { type: Number, default: 1 },
  canPrevious: Boolean,
  canNext: Boolean,
  hasNextUnreviewed: Boolean,
});
const emit = defineEmits(['close', 'save', 'navigate']);
const evidenceError = ref('');
const materialApi = createMaterialsApi(authApi.getClient());
async function openEvidence(path, page) {
  evidenceError.value = '';
  const download = /\.docx?$/i.test(path);
  const preview = download ? null : window.open('about:blank', '_blank');
  if (preview) preview.opener = null;
  try {
    const response = await materialApi.file(props.submission.material_batch_id, path);
    const url = URL.createObjectURL(response.data);
    const link = document.createElement('a');
    link.href = url;
    if (preview)
      preview.location.href =
        url + (/\.pdf$/i.test(path) && Number.isInteger(page) ? '#page=' + page : '');
    else {
      link.download = path.split('/').pop();
      link.click();
    }
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch {
    if (preview) preview.close();
    evidenceError.value = '原文件无法读取，文件可能已变化或当前无访问权限。';
  }
}
const initialScore = () => {
  const score = finalScore(props.submission);
  return Number.isFinite(score) && score >= 0 ? score : null;
};
const editableScore = ref(initialScore());
const editableFeedback = ref(
  props.submission.is_human_reviewed
    ? (props.submission.human_feedback ?? props.submission.feedback)
    : props.submission.feedback
);
function confirmDiscard() {
  const changed =
    editableScore.value !== initialScore() ||
    editableFeedback.value !==
      (props.submission.is_human_reviewed
        ? (props.submission.human_feedback ?? props.submission.feedback)
        : props.submission.feedback);
  return !changed || confirm('复核内容尚未保存，确定放弃修改？');
}
function close() {
  if (!props.saving && confirmDiscard()) emit('close');
}
function navigate(offset) {
  if (!props.saving && confirmDiscard()) emit('navigate', offset);
}
function saveReview(event) {
  emit('save', {
    id: props.submission.id,
    score: editableScore.value,
    human_feedback: editableFeedback.value,
    continueNext: event.submitter?.value === 'next',
  });
}
</script>
<style scoped>
.review-navigation {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 18px;
  color: var(--color-muted);
  font-size: 13px;
}
.review-navigation button {
  min-height: 36px;
  padding: 6px 12px;
  border: 1px solid var(--color-line);
  border-radius: var(--radius-control);
  color: var(--color-accent);
  background: var(--color-paper);
}
.review-navigation button:disabled {
  color: var(--color-subtle);
}
.review-intro {
  color: var(--color-muted);
  font-size: 13px;
  line-height: 1.8;
  margin-bottom: 20px;
}
.original-review {
  padding: 20px;
  background: var(--color-soft);
  border-radius: 10px;
}
.original-review > div {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  font-size: 13px;
  color: var(--color-muted);
}
.original-review strong {
  font-size: 24px;
  color: var(--color-ink);
  font-variant-numeric: tabular-nums;
}
.original-review p {
  white-space: pre-wrap;
  line-height: 1.8;
  margin-top: 16px;
  font-size: 14px;
}
.review-evidence {
  margin: 24px 0;
  font-size: 13px;
}
.review-evidence details {
  padding: 15px 0;
  border-bottom: 1px solid var(--color-line);
}
.source-file,
.criterion-evidence {
  margin-top: 14px;
  overflow-wrap: anywhere;
}
.review-evidence summary {
  font-weight: 500;
}
.review-evidence p {
  white-space: pre-wrap;
  line-height: 1.8;
  color: var(--color-muted);
  margin-top: 12px;
}
.review-form {
  display: grid;
  gap: 18px;
}
.review-form h3 {
  font-size: 16px;
  font-weight: 600;
}
.review-form label {
  display: grid;
  gap: 9px;
  font-size: 13px;
}
.review-form input,
.review-form textarea {
  width: 100%;
  border: 1px solid var(--color-line);
  padding: 10px 12px;
}
.review-form input {
  max-width: 180px;
}
</style>
