<template>
  <div class="rubric-summary">
    <div v-for="(row, index) in rows" :key="index" class="rubric-summary-row">
      <div
        ><strong>{{ row.name }}</strong
        ><p v-if="row.description">{{ row.description }}</p></div
      >
      <span v-if="row.score !== null" class="rubric-score">{{ row.score }} <small>分</small></span>
    </div>
    <p v-if="!rows.length" class="muted">尚未设置评分标准。</p>
  </div>
</template>
<script setup>
import { computed } from 'vue';
import { rubricRows } from '../services/assignmentWorkspace';
const props = defineProps({ rubric: { type: [Object, Array, String], default: () => ({}) } });
const rows = computed(() => rubricRows(props.rubric));
</script>
<style scoped>
.rubric-summary-row {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  padding: 16px 0;
  border-bottom: 1px solid var(--color-line);
}
.rubric-summary-row:last-child {
  border-bottom: 0;
}
strong {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-ink);
}
p {
  font-size: 12px;
  line-height: 1.8;
  color: var(--color-muted);
  white-space: pre-wrap;
  margin-top: 5px;
  overflow-wrap: anywhere;
}
.rubric-score {
  flex-shrink: 0;
  font-size: 17px;
  font-weight: 600;
  color: var(--color-accent);
}
small {
  font-size: 11px;
  font-weight: 400;
}
</style>
