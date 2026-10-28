<template>
  <Teleport to="body">
    <button
      ref="launcher"
      type="button"
      class="assistant-launcher"
      :class="{ 'is-dragging': drag?.moved }"
      :style="launcherStyle"
      aria-label="AI 教学助手"
      aria-controls="floating-assistant-panel"
      :aria-expanded="open"
      title="AI 教学助手，可拖动移动"
      @pointerdown="startDrag"
      @pointermove="moveDrag"
      @pointerup="endDrag"
      @pointercancel="cancelDrag"
      @click="toggle"
    >
      <AppIcon name="chat" />
      <span class="sr-only">点击展开对话，拖动调整位置</span>
    </button>

    <section
      v-if="activated"
      v-show="open"
      id="floating-assistant-panel"
      ref="panel"
      class="floating-assistant-panel"
      :class="{ 'is-expanded': expanded }"
      :style="panelStyle"
      role="dialog"
      aria-modal="false"
      aria-labelledby="floating-assistant-title"
      tabindex="-1"
      @keydown.esc="panelEscape"
    >
      <header class="floating-assistant-heading">
        <div class="assistant-title"
          ><AppIcon name="sparkles" /><h2 id="floating-assistant-title">AI 教学助手</h2></div
        >
        <div class="assistant-panel-actions">
          <button
            type="button"
            title="恢复图标默认位置"
            aria-label="恢复助手默认位置"
            @click="resetPosition"
            ><AppIcon name="history"
          /></button>
          <button
            type="button"
            :aria-label="expanded ? '缩小对话面板' : '放大对话面板'"
            :title="expanded ? '缩小' : '放大'"
            @click="expanded = !expanded"
            ><span aria-hidden="true">{{ expanded ? '↙' : '↗' }}</span></button
          >
          <button
            ref="closeButton"
            type="button"
            aria-label="收起 AI 助手"
            title="收起"
            @click="close"
            ><span aria-hidden="true">×</span></button
          >
        </div>
      </header>
      <div class="assistant-task-toolbar"
        ><span>拉取作业、批改与成绩查询</span
        ><button v-if="canFetch" type="button" @click="openConsole()"
          ><AppIcon name="download" />任务控制台</button
        ></div
      >
      <div class="floating-chat-body"><AgentChatWidget @fetch-task="openConsole" /></div>
    </section>

    <div
      v-if="consoleActivated && canFetch"
      v-show="consoleOpen"
      class="assistant-console-layer"
      @keydown="consoleKeydown"
    >
      <div class="assistant-console-backdrop" aria-hidden="true" @click="closeConsole" />
      <aside
        id="homework-console"
        ref="consolePanel"
        class="assistant-console-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="homework-console-title"
      >
        <header class="assistant-console-heading"
          ><div><p>任务控制台</p><h2 id="homework-console-title">作业拉取与材料</h2></div
          ><button
            ref="consoleClose"
            type="button"
            aria-label="收起作业控制台"
            @click="closeConsole"
            >收起控制台 <span aria-hidden="true">×</span></button
          ></header
        >
        <div class="assistant-console-content"
          ><RpaTaskPanel :active="consoleOpen" :task-id="consoleTaskId" embedded
        /></div>
      </aside>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useRoute } from 'vue-router';
import authApi from '../services/authApi';
import {
  assistantPanelRect,
  assistantPositionKey,
  clampAssistantPosition,
  defaultAssistantPosition,
  isAssistantDrag,
} from '../services/floatingAssistant';
import AgentChatWidget from './AgentChatWidget.vue';
import AppIcon from './AppIcon.vue';
import RpaTaskPanel from './RpaTaskPanel.vue';

const route = useRoute();
const user = authApi.getCurrentUser();
const canFetch = ['teacher', 'admin'].includes(user?.role);
const positionKey = assistantPositionKey(user);
const viewport = ref(readViewport());
const position = ref(readPosition());
const drag = ref(null);
const open = ref(false),
  activated = ref(false),
  expanded = ref(false);
const launcher = ref(null),
  panel = ref(null),
  closeButton = ref(null);
const consoleOpen = ref(false),
  consoleActivated = ref(false),
  consoleTaskId = ref('');
const consolePanel = ref(null),
  consoleClose = ref(null);
let ignoreClick = false,
  clickReset,
  previousConsoleFocus;
const launcherStyle = computed(() => ({
  left: `${position.value.x}px`,
  top: `${position.value.y}px`,
}));
const panelStyle = computed(() => {
  const box = assistantPanelRect(position.value, viewport.value, expanded.value);
  return Object.fromEntries(Object.entries(box).map(([key, value]) => [key, `${value}px`]));
});

function readViewport() {
  return {
    width: window.visualViewport?.width || window.innerWidth,
    height: window.visualViewport?.height || window.innerHeight,
  };
}
function readPosition() {
  try {
    const saved = JSON.parse(localStorage.getItem(positionKey));
    if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y))
      return clampAssistantPosition(saved, viewport.value);
  } catch {
    /* A blocked or stale preference does not prevent opening chat. */
  }
  return defaultAssistantPosition(viewport.value);
}
function savePosition() {
  try {
    localStorage.setItem(positionKey, JSON.stringify(position.value));
  } catch {
    /* Preference storage is optional. */
  }
}
function resize() {
  viewport.value = readViewport();
  position.value = clampAssistantPosition(position.value, viewport.value);
}
function resetPosition() {
  position.value = defaultAssistantPosition(viewport.value);
  savePosition();
}
function startDrag(event) {
  if (event.button !== 0 || !event.isPrimary) return;
  clearTimeout(clickReset);
  ignoreClick = false;
  drag.value = {
    pointerId: event.pointerId,
    start: { x: event.clientX, y: event.clientY },
    origin: { ...position.value },
    moved: false,
  };
  event.currentTarget.setPointerCapture(event.pointerId);
}
function moveDrag(event) {
  if (!drag.value || drag.value.pointerId !== event.pointerId) return;
  const point = { x: event.clientX, y: event.clientY };
  if (!drag.value.moved && !isAssistantDrag(drag.value.start, point)) return;
  drag.value.moved = true;
  position.value = clampAssistantPosition(
    {
      x: drag.value.origin.x + point.x - drag.value.start.x,
      y: drag.value.origin.y + point.y - drag.value.start.y,
    },
    viewport.value
  );
}
function endDrag(event) {
  if (!drag.value || drag.value.pointerId !== event.pointerId) return;
  ignoreClick = drag.value.moved;
  if (ignoreClick) savePosition();
  drag.value = null;
  if (event.currentTarget.hasPointerCapture(event.pointerId))
    event.currentTarget.releasePointerCapture(event.pointerId);
  clickReset = setTimeout(() => {
    ignoreClick = false;
  }, 300);
}
function cancelDrag() {
  drag.value = null;
  ignoreClick = true;
  clickReset = setTimeout(() => {
    ignoreClick = false;
  }, 300);
}
function toggle(event) {
  if (ignoreClick && event.detail !== 0) {
    ignoreClick = false;
    return;
  }
  if (open.value) close();
  else openAssistant();
}
async function openAssistant(options = {}) {
  activated.value = true;
  open.value = true;
  if (options.expanded) expanded.value = true;
  await nextTick();
  panel.value?.querySelector('input[aria-label="消息内容"]:not(:disabled)')?.focus();
  if (!panel.value?.contains(document.activeElement)) panel.value?.focus();
  if (typeof options.taskId === 'string' && options.taskId) await openConsole(options.taskId);
}
function close() {
  open.value = false;
  nextTick(() => launcher.value?.focus());
}
function panelEscape(event) {
  if (event.defaultPrevented || consoleOpen.value) return;
  event.preventDefault();
  close();
}
async function openConsole(taskId) {
  if (!canFetch) return;
  if (!consoleOpen.value) previousConsoleFocus = document.activeElement;
  consoleActivated.value = true;
  consoleTaskId.value = typeof taskId === 'string' ? taskId : '';
  consoleOpen.value = true;
  await nextTick();
  consoleClose.value?.focus();
}
function closeConsole() {
  consoleOpen.value = false;
  nextTick(() => {
    (previousConsoleFocus?.isConnected && previousConsoleFocus.getClientRects().length
      ? previousConsoleFocus
      : launcher.value
    )?.focus();
  });
}
function consoleKeydown(event) {
  if (event.defaultPrevented) return;
  if (event.key === 'Escape') {
    event.preventDefault();
    closeConsole();
  } else if (event.key === 'Tab') {
    const controls = [
      ...consolePanel.value.querySelectorAll(
        'button, input, select, textarea, a[href], [tabindex]'
      ),
    ].filter(
      (element) => !element.disabled && element.tabIndex >= 0 && element.getClientRects().length
    );
    const first = controls[0],
      last = controls.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first?.focus();
    }
  }
}
const handleOpenRequest = (event) => openAssistant(event.detail || {});
const handleConsoleRequest = (event) => openConsole(event.detail?.jobId);
watch(
  () => [route.name, route.query.task],
  ([name, task]) => {
    if (name === 'ai-assistant')
      openAssistant({ expanded: true, taskId: typeof task === 'string' ? task : '' });
  },
  { immediate: true }
);
onMounted(() => {
  window.addEventListener('resize', resize);
  window.visualViewport?.addEventListener('resize', resize);
  window.addEventListener('teacher-open-assistant', handleOpenRequest);
  window.addEventListener('rpa-open-console', handleConsoleRequest);
});
onBeforeUnmount(() => {
  clearTimeout(clickReset);
  window.removeEventListener('resize', resize);
  window.visualViewport?.removeEventListener('resize', resize);
  window.removeEventListener('teacher-open-assistant', handleOpenRequest);
  window.removeEventListener('rpa-open-console', handleConsoleRequest);
});
</script>

<style scoped>
.assistant-launcher {
  position: fixed;
  z-index: 38;
  display: grid;
  place-items: center;
  width: 52px;
  height: 52px;
  padding: 0;
  border: 1px solid var(--color-accent);
  border-radius: 18px;
  background: var(--color-accent);
  color: var(--color-on-accent);
  box-shadow: 0 8px 24px color-mix(in srgb, var(--color-accent) 28%, transparent);
  cursor: grab;
  touch-action: none;
  user-select: none;
}
.assistant-launcher:hover {
  filter: brightness(1.08);
}
.assistant-launcher.is-dragging {
  cursor: grabbing;
}
.assistant-launcher svg {
  width: 25px;
  height: 25px;
}
.floating-assistant-panel {
  position: fixed;
  z-index: 39;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--color-line);
  border-radius: 16px;
  background: var(--color-paper);
  color: var(--color-ink);
  box-shadow: 0 20px 70px color-mix(in srgb, var(--color-ink) 18%, transparent);
}
.floating-assistant-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  background: var(--color-paper);
  border-bottom: 1px solid var(--color-line);
  flex-shrink: 0;
}
.assistant-title {
  display: flex;
  align-items: center;
  gap: 9px;
}
.assistant-title > svg {
  color: var(--color-accent);
  width: 20px;
  height: 20px;
}
.assistant-title h2 {
  font-size: 15px;
  font-weight: 650;
}
.assistant-panel-actions {
  display: flex;
  align-items: center;
  gap: 2px;
}
.assistant-panel-actions button {
  width: 36px;
  min-height: 36px;
  display: grid;
  place-items: center;
  color: var(--color-muted);
  border-radius: 8px;
  font-size: 24px;
  padding: 0;
}
.assistant-panel-actions button svg {
  width: 16px;
  height: 16px;
}
.assistant-panel-actions button:hover {
  background: var(--color-canvas);
  color: var(--color-accent);
}
.assistant-task-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 5px 16px;
  color: var(--color-muted);
  font-size: 12px;
  border-bottom: 1px solid var(--color-line);
  flex-shrink: 0;
}
.assistant-task-toolbar button {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 34px;
  color: var(--color-accent);
  white-space: nowrap;
}
.assistant-task-toolbar button svg {
  width: 15px;
  height: 15px;
}
.floating-chat-body {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  display: flex;
}
.floating-chat-body :deep(.agent-chat-widget) {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  min-width: 0;
  width: 100%;
  border: 0;
  border-radius: 0;
  box-shadow: none;
}
.floating-chat-body :deep(.chat-header) {
  display: none;
}
.floating-chat-body :deep(.toolbar) {
  flex-shrink: 0;
  padding: 8px 12px;
  gap: 6px;
  border-top: 0;
}
.floating-chat-body :deep(.toolbar > div:last-child) {
  display: none;
}
.floating-chat-body :deep(.toolbar button) {
  min-height: 34px;
  font-size: 12px;
}
.floating-chat-body :deep(.messages-container),
.floating-chat-body :deep(.messages-container.messages-idle) {
  flex: 1;
  min-height: 70px;
  height: auto;
  padding: 18px 14px;
  background: var(--color-canvas);
}
.floating-chat-body :deep(.message-content) {
  font-size: 13px;
  max-width: calc(100% - 38px);
}
.floating-chat-body :deep(.message-content.bg-indigo-500) {
  background: var(--color-accent);
}
.floating-chat-body :deep(.message-time) {
  font-size: 12px;
}
.floating-chat-body :deep(.message-avatar) {
  color: var(--color-accent);
  border-color: var(--color-line);
}
.floating-chat-body :deep(.quick-commands) {
  padding: 10px 12px;
  flex-shrink: 0;
  border-color: var(--color-line);
}
.floating-chat-body :deep(.command-heading) {
  margin-bottom: 7px;
  font-size: 12px;
}
.floating-chat-body :deep(.command-options) {
  gap: 6px;
}
.floating-chat-body :deep(.quick-cmd-btn) {
  padding: 7px 6px;
  min-height: 38px;
  justify-content: center;
  align-items: center;
  gap: 5px;
  background: var(--color-paper);
  color: var(--color-accent);
  border-color: var(--color-line);
  flex-direction: row;
  text-align: left;
}
.floating-chat-body :deep(.quick-cmd-btn.command-selected) {
  background: var(--color-accent-soft, var(--color-canvas));
  border-color: var(--color-accent);
}
.floating-chat-body :deep(.command-icon) {
  display: none;
}
.floating-chat-body :deep(.command-copy strong) {
  font-size: 12px;
}
.floating-chat-body :deep(.command-copy > span) {
  display: none;
}
.floating-chat-body :deep(.command-guidance) {
  margin-top: 7px;
  font-size: 12px;
  gap: 5px;
  max-height: 100px;
  overflow-y: auto;
}
.floating-chat-body :deep(.input-area) {
  flex-shrink: 0;
  padding: 12px;
  border-color: var(--color-line);
}
.floating-chat-body :deep(.input-area input) {
  min-height: 42px;
  font-size: 13px;
}
.floating-chat-body :deep(.input-area button[type='submit']) {
  padding: 8px 13px;
  background: var(--color-accent);
}
.floating-chat-body :deep(.input-area > div:last-child) {
  margin-top: 6px;
  font-size: 12px;
}
.floating-chat-body :deep(.session-dropdown) {
  flex-shrink: 0;
  max-height: 120px;
  overflow-y: auto;
}
.assistant-console-layer {
  position: fixed;
  inset: 0;
  z-index: 48;
}
.assistant-console-backdrop {
  position: absolute;
  inset: 0;
  background: color-mix(in srgb, var(--color-ink) 30%, transparent);
}
.assistant-console-panel {
  position: absolute;
  inset: 0 0 0 auto;
  width: min(896px, 100%);
  display: flex;
  flex-direction: column;
  background: var(--color-paper);
  color: var(--color-ink);
  box-shadow: -16px 0 60px color-mix(in srgb, var(--color-ink) 14%, transparent);
}
.assistant-console-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 18px 22px;
  border-bottom: 1px solid var(--color-line);
}
.assistant-console-heading p {
  font-size: 12px;
  color: var(--color-muted);
  margin-bottom: 4px;
}
.assistant-console-heading h2 {
  font-size: 18px;
  font-weight: 650;
}
.assistant-console-heading button {
  min-height: 40px;
  padding: 6px 12px;
  border-radius: 8px;
  background: var(--color-canvas);
  font-size: 13px;
}
.assistant-console-content {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
}
.assistant-launcher:focus-visible,
.floating-assistant-panel button:focus-visible,
.assistant-console-panel button:focus-visible {
  outline: 3px solid var(--color-accent);
  outline-offset: 3px;
}
@media (max-height: 550px) {
  .floating-chat-body :deep(.quick-commands) {
    max-height: 72px;
    overflow-y: auto;
  }
  .floating-chat-body {
    overflow-y: auto;
  }
  .floating-chat-body :deep(.agent-chat-widget) {
    flex: 1 0 auto;
    min-height: 320px;
  }
  .floating-chat-body :deep(.input-area > div:last-child) {
    display: none;
  }
}
@media (prefers-reduced-motion: reduce) {
  .floating-chat-body :deep(.animate-pulse) {
    animation: none;
  }
  .floating-chat-body :deep(*) {
    scroll-behavior: auto;
  }
}
</style>
