<template>
  <section class="assistant-entry surface">
    <div class="assistant-entry-symbol"><AppIcon name="chat" /></div>
    <h1 class="page-title">AI 助手</h1>
    <p>助手已移至右下角。在任意页面都可以打开对话，继续拉取作业、批改和查询成绩。</p>
    <div class="assistant-entry-actions"
      ><button type="button" class="workspace-button primary" @click="openAssistant"
        >展开对话</button
      ><router-link class="workspace-button" to="/assignments">返回作业工作台</router-link></div
    >
  </section>
</template>

<script setup>
import { onMounted } from 'vue';
import { useRoute } from 'vue-router';
import AppIcon from '../components/AppIcon.vue';
const route = useRoute();
function openAssistant() {
  window.dispatchEvent(
    new CustomEvent('teacher-open-assistant', {
      detail: {
        expanded: true,
        taskId: typeof route.query.task === 'string' ? route.query.task : '',
      },
    })
  );
}
onMounted(openAssistant);
</script>

<style scoped>
.assistant-entry {
  padding: 48px;
  max-width: 720px;
  margin: 36px auto;
}
.assistant-entry-symbol {
  width: 44px;
  height: 44px;
  display: grid;
  place-items: center;
  border-radius: 12px;
  color: var(--color-accent);
  background: var(--color-accent-soft);
  margin-bottom: 24px;
}
.assistant-entry-symbol svg {
  width: 25px;
  height: 25px;
}
.assistant-entry p {
  margin-top: 16px;
  color: var(--color-muted);
  font-size: 15px;
  line-height: 1.8;
  max-width: 46ch;
}
.assistant-entry-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 28px;
}
@media (max-width: 640px) {
  .assistant-entry {
    padding: 24px;
    margin: 12px auto;
  }
}
</style>
