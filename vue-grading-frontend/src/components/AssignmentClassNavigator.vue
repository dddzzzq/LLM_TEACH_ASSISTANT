<template>
  <nav class="class-navigator" aria-label="作业班级">
    <div class="class-nav-heading"
      ><h2>班级</h2><span>{{ classes.length }}</span></div
    >
    <p v-if="loading" role="status">正在读取班级…</p>
    <p v-else-if="error" class="workspace-error" role="alert"
      >{{ error }}<button @click="$emit('retry')">重试</button></p
    >
    <div v-else class="class-nav-list">
      <button
        v-for="item in classes"
        :key="item.id"
        type="button"
        :aria-current="modelValue === item.id ? 'page' : undefined"
        :class="{ selected: modelValue === item.id }"
        @click="$emit('update:modelValue', item.id)"
      >
        <span class="class-monogram" aria-hidden="true">{{
          item.source === 'unassigned' ? '未' : '班'
        }}</span>
        <span class="class-copy"
          >{{ item.name
          }}<small>{{
            item.source === 'unassigned'
              ? '保留的未分班记录'
              : item.source === 'portal'
                ? '官网班级'
                : '历史班级'
          }}</small
          ><em v-if="item.material_status" class="class-summary" :class="classSummary(item).tone"
            >{{ classSummary(item).label
            }}<span v-if="classSummary(item).detail"> · {{ classSummary(item).detail }}</span></em
          ></span
        >
      </button>
      <p v-if="!classes.length">暂无可查看班级</p>
    </div>
  </nav>
</template>
<script setup>
import { classSummary } from '../services/classWorkspace';
defineProps({
  classes: { type: Array, default: () => [] },
  modelValue: String,
  loading: Boolean,
  error: String,
});
defineEmits(['update:modelValue', 'retry']);
</script>
<style scoped>
.class-navigator {
  min-width: 0;
  align-self: start;
  padding: var(--space-lg, 20px) 0;
}
.class-nav-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0 12px 16px;
  color: var(--color-muted);
  font-size: 13px;
}
.class-nav-heading h2 {
  font-weight: 600;
}
.class-nav-list {
  display: grid;
  gap: 6px;
  max-height: 65dvh;
  overflow-y: auto;
}
.class-nav-list button {
  display: flex;
  align-items: center;
  text-align: left;
  gap: 10px;
  padding: 12px;
  border-radius: 9px;
  min-width: 0;
  color: var(--color-ink);
  font-size: 13px;
}
.class-nav-list button > span:last-child {
  min-width: 0;
  overflow-wrap: break-word;
}
.class-nav-list button:hover {
  background: var(--color-paper);
}
.class-nav-list button.selected {
  color: var(--color-accent);
  background: var(--color-accent-soft);
  font-weight: 600;
}
.class-monogram {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  flex-shrink: 0;
  background: var(--color-paper);
  border: 1px solid var(--color-line);
  border-radius: 8px;
  font-size: 12px;
}
.class-nav-list small {
  display: block;
  font-weight: 400;
  font-size: 12px;
  color: var(--color-muted);
  margin-top: 4px;
}
.class-summary {
  display: block;
  margin-top: 6px;
  color: var(--color-muted);
  font-size: 11px;
  font-style: normal;
  font-weight: 500;
  line-height: 1.6;
}
.class-summary.warning {
  color: var(--color-warning);
}
.class-summary.active {
  color: var(--color-accent);
}
.class-summary.success {
  color: var(--color-success);
}
.class-summary span {
  color: var(--color-muted);
  font-weight: 400;
}
@media (max-width: 1000px) {
  .class-navigator {
    padding: 0;
  }
  .class-nav-heading {
    margin: 0 0 10px;
  }
  .class-nav-list {
    display: flex;
    flex-wrap: nowrap;
    max-height: none;
    padding-bottom: 6px;
    overflow-x: auto;
  }
  .class-nav-list button {
    flex: 0 0 auto;
    max-width: 230px;
    border: 1px solid var(--color-line);
  }
}
</style>
