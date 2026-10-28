<template><div class="requirement-content" v-html="html"></div></template>
<script setup>
import { computed } from 'vue';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
const props = defineProps({ text: { type: String, default: '' } });
const html = computed(() =>
  DOMPurify.sanitize(marked.parse(props.text, { breaks: true }), {
    FORBID_TAGS: ['img', 'iframe', 'object', 'embed', 'form', 'input', 'button', 'video', 'audio'],
    FORBID_ATTR: ['style'],
  })
);
</script>
<style scoped>
.requirement-content {
  font-size: 16px;
  line-height: 1.9;
  color: var(--color-ink);
  overflow-wrap: anywhere;
  max-height: 600px;
  overflow: auto;
}
.requirement-content :deep(pre) {
  padding: 16px;
  font-size: 13px;
  border-radius: 6px;
  background: var(--color-soft);
  white-space: pre-wrap;
  font-family: ui-monospace, monospace;
}
.requirement-content :deep(h1),
.requirement-content :deep(h2),
.requirement-content :deep(h3),
.requirement-content :deep(h4) {
  font-weight: 600;
  margin: 16px 0 8px;
}
.requirement-content :deep(h1) {
  font-size: 20px;
}
.requirement-content :deep(h2) {
  font-size: 18px;
}
.requirement-content :deep(h3) {
  font-size: 16px;
}
.requirement-content :deep(p) {
  margin: 8px 0;
}
.requirement-content :deep(ol),
.requirement-content :deep(ul) {
  padding-left: 24px;
}
.requirement-content :deep(ol) {
  list-style: decimal;
}
.requirement-content :deep(ul) {
  list-style: disc;
}
.requirement-content :deep(table) {
  border-collapse: collapse;
  font-size: 14px;
}
.requirement-content :deep(td),
.requirement-content :deep(th) {
  border: 1px solid var(--color-line);
  padding: 6px 10px;
}

.requirement-content :deep(a) {
  color: var(--color-accent);
  text-decoration: underline;
  text-underline-offset: 3px;
}
.requirement-content :deep(blockquote) {
  border-left: 2px solid var(--color-line);
  padding-left: 16px;
  color: var(--color-muted);
  margin: 16px 0;
}
.requirement-content :deep(code) {
  font-size: 0.9em;
}
.requirement-content :deep(th) {
  background: var(--color-soft);
  font-weight: 600;
}
@media (max-width: 760px) {
  .requirement-content {
    font-size: 14px;
  }
}
</style>
