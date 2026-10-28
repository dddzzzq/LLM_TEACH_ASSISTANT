<template>
  <Teleport to="body">
    <dialog
      ref="dialog"
      class="app-modal"
      :class="{ 'is-drawer': drawer }"
      :aria-labelledby="titleId"
      tabindex="-1"
      @cancel.prevent="close"
      @click="backdropClick"
      @keydown="trapFocus"
    >
      <header class="modal-heading">
        <h2 :id="titleId">{{ title }}</h2>
        <button type="button" :disabled="closeDisabled" aria-label="关闭提醒" @click="close"
          >×</button
        >
      </header>
      <div class="modal-body"><slot /></div>
      <footer class="modal-footer"><slot name="footer" /></footer>
    </dialog>
  </Teleport>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';
const props = defineProps({
  title: { type: String, required: true },
  closeDisabled: Boolean,
  drawer: Boolean,
});
const emit = defineEmits(['close']);
const dialog = ref(null);
const titleId = `modal-${globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)}`;
let previousFocus;
function close() {
  if (!props.closeDisabled) emit('close');
}
function trapFocus(event) {
  if (event.key !== 'Tab') return;
  const controls = [
    ...dialog.value.querySelectorAll('button, input, select, textarea, a[href], [tabindex]'),
  ].filter(
    (element) => !element.disabled && element.tabIndex >= 0 && element.getClientRects().length
  );
  const first = controls[0],
    last = controls.at(-1);
  if (!first) {
    event.preventDefault();
    dialog.value.focus();
    return;
  }
  if (
    event.shiftKey &&
    (document.activeElement === first || document.activeElement === dialog.value)
  ) {
    event.preventDefault();
    last.focus();
  } else if (
    !event.shiftKey &&
    (document.activeElement === last || document.activeElement === dialog.value)
  ) {
    event.preventDefault();
    first.focus();
  }
}
function backdropClick(event) {
  if (event.target !== dialog.value) return;
  const box = dialog.value.getBoundingClientRect();
  if (
    event.clientX < box.left ||
    event.clientX > box.right ||
    event.clientY < box.top ||
    event.clientY > box.bottom
  )
    close();
}
onMounted(() => {
  previousFocus = document.activeElement;
  dialog.value.showModal();
});
onBeforeUnmount(() => {
  dialog.value?.close();
  if (previousFocus?.isConnected) previousFocus.focus();
});
</script>

<style scoped>
.app-modal {
  width: min(640px, calc(100% - 32px));
  max-height: calc(100dvh - 48px);
  padding: 0;
  margin: auto;
  border: 1px solid var(--color-line);
  border-radius: 16px;
  color: var(--color-ink);
  background: var(--color-paper);
  box-shadow: var(--shadow-panel);
  overflow-y: auto;
}
.app-modal::backdrop {
  background: var(--color-scrim);
}
.modal-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 20px 24px;
  border-bottom: 1px solid var(--color-line);
}
.modal-heading h2 {
  font-size: 18px;
  font-weight: 700;
}
.modal-heading button {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  color: var(--color-muted);
  font-size: 24px;
}
.modal-heading button:hover {
  background: var(--color-soft);
}
.modal-body {
  padding: 22px 24px;
}
.modal-footer {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  flex-wrap: wrap;
  gap: 12px;
  padding: 16px 24px;
  border-top: 1px solid var(--color-line);
  background: var(--color-soft);
}
.app-modal.is-drawer {
  width: min(760px, 100%);
  height: 100dvh;
  max-height: 100dvh;
  margin: 0 0 0 auto;
  border-radius: 16px 0 0 16px;
}
@media (max-width: 600px) {
  .modal-heading,
  .modal-body,
  .modal-footer {
    padding: 16px;
  }
}
</style>
