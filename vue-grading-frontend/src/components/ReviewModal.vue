<template>
  <AppModal
    :title="'学生复核 / ' + (submission.student_name || submission.student_id)"
    :close-disabled="saving"
    drawer
    @close="close"
  >
    <p class="review-intro"
      >学号 {{ submission.student_id }}。结合原始评分和检测证据，保存教师最终判断。</p
    >
    <section class="original-review">
      <div
        ><span>AI 原始评分</span
        ><strong>{{
          submission.score < 0 ? '评分失败' : Number(submission.score).toFixed(1)
        }}</strong></div
      >
      <p>{{ submission.feedback || '暂无原始评语' }}</p>
    </section>
    <div class="review-evidence">
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
        >最终分数<input
          id="human_score"
          v-model.number="editableScore"
          type="number"
          min="0"
          max="100"
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
      ><button
        form="human-review-form"
        type="submit"
        class="workspace-button primary"
        :disabled="saving"
        >{{ saving ? '保存中…' : '保存复核' }}</button
      ></template
    >
  </AppModal>
</template>
<script setup>
import { ref } from 'vue';
import AppModal from './AppModal.vue';
import ReportModal from './ReportModal.vue';
import { finalScore } from '../services/classWorkspace';
const props = defineProps({
  submission: { type: Object, required: true },
  saving: Boolean,
  error: String,
});
const emit = defineEmits(['close', 'save']);
const editableScore = ref(Math.max(0, finalScore(props.submission)));
const editableFeedback = ref(
  props.submission.is_human_reviewed
    ? (props.submission.human_feedback ?? props.submission.feedback)
    : props.submission.feedback
);
function close() {
  if (props.saving) return;
  const changed =
    editableScore.value !== Math.max(0, finalScore(props.submission)) ||
    editableFeedback.value !==
      (props.submission.is_human_reviewed
        ? (props.submission.human_feedback ?? props.submission.feedback)
        : props.submission.feedback);
  if (!changed || confirm('复核内容尚未保存，确定关闭？')) emit('close');
}
function saveReview() {
  emit('save', {
    id: props.submission.id,
    score: editableScore.value,
    human_feedback: editableFeedback.value,
  });
}
</script>
<style scoped>
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
