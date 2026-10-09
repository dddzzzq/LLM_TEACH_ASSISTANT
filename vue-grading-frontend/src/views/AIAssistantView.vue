<template>
  <div class="ai-assistant-view">
    <header class="text-center mb-8">
      <h1 class="text-3xl md:text-4xl font-bold text-gray-800">AI 教学助手</h1>
      <p class="text-gray-500 mt-2">作业下载与批改、学生成绩查询</p>
    </header>

    <main class="max-w-5xl mx-auto">
      <div v-if="canFetch" class="flex justify-end mb-3">
        <button ref="consoleTrigger" type="button" class="px-4 py-2 text-sm text-indigo-700 bg-white border border-indigo-200 rounded-lg hover:bg-indigo-50"
          aria-controls="homework-console" :aria-expanded="consoleOpen" @click="openConsole()">
          打开作业控制台
        </button>
      </div>
      <AgentChatWidget @fetch-task="openConsole" />
    </main>

    <Teleport to="body">
      <Transition name="console-drawer">
        <div v-if="canFetch" v-show="consoleOpen" class="fixed inset-0 z-50" @keydown="consoleKeydown">
          <div class="console-backdrop absolute inset-0 bg-slate-900/30" aria-hidden="true" @click="closeConsole" />
          <aside id="homework-console" ref="consolePanel" role="dialog" aria-modal="true" aria-labelledby="homework-console-title"
            class="console-panel absolute inset-y-0 right-0 w-full bg-white shadow-2xl flex flex-col">
            <div class="flex items-center justify-between gap-3 px-5 py-4 border-b shrink-0">
              <h2 id="homework-console-title" class="text-lg font-semibold text-gray-800">作业下载与批改</h2>
              <button ref="consoleClose" type="button" class="px-3 py-2 text-sm text-slate-600 rounded-lg hover:bg-slate-100"
                aria-label="收起作业控制台" @click="closeConsole">收起控制台 ×</button>
            </div>
            <div class="flex-1 min-h-0 overflow-y-auto overscroll-contain">
              <RpaTaskPanel :active="consoleOpen" :task-id="consoleTaskId" embedded />
            </div>
          </aside>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import AgentChatWidget from '../components/AgentChatWidget.vue'
import RpaTaskPanel from '../components/RpaTaskPanel.vue'
import authApi from '../services/authApi'

const canFetch = ['teacher', 'admin'].includes(authApi.getCurrentUser()?.role)
const consoleOpen = ref(false), consoleTaskId = ref('')
const consoleTrigger = ref(null), consoleClose = ref(null), consolePanel = ref(null)
let previousFocus

async function openConsole(taskId) {
  if (!canFetch) return
  if (!consoleOpen.value) previousFocus = document.activeElement
  consoleTaskId.value = typeof taskId === 'string' ? taskId : ''
  consoleOpen.value = true
  await nextTick()
  consoleClose.value?.focus()
}
function closeConsole() {
  consoleOpen.value = false
  nextTick(() => {
    const target = previousFocus?.isConnected && previousFocus !== document.body ? previousFocus : consoleTrigger.value
    target?.focus()
  })
}
function consoleKeydown(event) {
  // Keyboard events handled by the remote browser should stay in that browser.
  if (event.defaultPrevented) return
  if (event.key === 'Escape') {
    event.preventDefault()
    closeConsole()
  } else if (event.key === 'Tab') {
    const controls = [...consolePanel.value.querySelectorAll('button, input, select, textarea, a[href], [tabindex]')]
      .filter(element => !element.disabled && element.tabIndex >= 0 && element.getClientRects().length)
    const first = controls[0], last = controls.at(-1)
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault(); last?.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault(); first?.focus()
    }
  }
}
</script>

<style scoped>
.ai-assistant-view {
  min-height: calc(100vh - 8rem);
}
.console-panel {
  max-width: 56rem;
}
.console-drawer-enter-active .console-panel,
.console-drawer-leave-active .console-panel {
  transition: transform 220ms ease;
}
.console-drawer-enter-active .console-backdrop,
.console-drawer-leave-active .console-backdrop {
  transition: opacity 220ms ease;
}
.console-drawer-enter-from .console-panel,
.console-drawer-leave-to .console-panel {
  transform: translateX(100%);
}
.console-drawer-enter-from .console-backdrop,
.console-drawer-leave-to .console-backdrop {
  opacity: 0;
}
@media (prefers-reduced-motion: reduce) {
  .console-drawer-enter-active .console-panel,
  .console-drawer-leave-active .console-panel,
  .console-drawer-enter-active .console-backdrop,
  .console-drawer-leave-active .console-backdrop {
    transition: none;
  }
}
</style>
