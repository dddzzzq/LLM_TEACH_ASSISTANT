<template>
  <section class="rounded-xl border bg-slate-50 overflow-hidden" aria-label="任务浏览器">
    <div class="flex flex-wrap items-center gap-2 p-3 border-b bg-white">
      <strong class="text-sm">任务浏览器</strong>
      <span class="text-xs" :class="human ? 'text-amber-700' : 'text-blue-700'">{{ human ? '人工操作中' : '自动执行中，仅查看' }}</span>
      <span class="text-xs text-slate-500">{{ connected ? '实时连接' : '正在连接…' }}</span>
      <select v-if="frame?.pages?.length" :value="frame.page_id" :disabled="!human || !connected || switching !== null" class="ml-auto border rounded px-2 py-1 text-sm max-w-60" @change="switchPage($event.target.value)">
        <option v-for="page in frame.pages" :key="page.id" :value="page.id">{{ page.title || '页面 ' + page.id }}</option>
      </select>
    </div>
    <p class="px-3 py-2 text-xs text-slate-600">点击画面后可输入、滚动和拖动。登录完成后点击“交还自动化”。密码及验证码不会进入聊天。</p>
    <div ref="surface" tabindex="0" role="application" aria-label="远程官网浏览器，可键盘操作"
      class="outline-none focus:ring-2 focus:ring-blue-500 touch-none select-none min-h-32"
      @keydown="keyDown" @pointerdown.prevent="pointerDown" @pointermove="pointerMove"
      @pointerup.prevent="pointerUp" @pointercancel="releaseDrag" @lostpointercapture="releaseDrag" @wheel.prevent="wheel">
      <img v-if="frame" :src="'data:image/jpeg;base64,' + frame.image" alt="官网浏览器实时画面" draggable="false" class="w-full block" />
      <p v-else class="p-8 text-center text-sm text-slate-500">{{ frameError || '正在连接浏览器…' }}</p>
    </div>
    <form v-if="human" class="p-3 border-t flex gap-2 bg-white" @submit.prevent="sendText">
      <input v-model="inputText" type="password" autocomplete="off" placeholder="可在这里输入中文、密码或验证码，再填写到网页当前输入框"
        class="min-w-0 flex-1 border rounded px-3 py-2 text-sm" aria-label="填写到网页当前输入框的内容" />
      <button class="px-3 py-2 rounded bg-blue-600 text-white text-sm" :disabled="!inputText || !connected || !frame">填写并清空</button>
    </form>
    <p v-if="frameError" class="px-3 pb-2 text-sm text-red-700" role="alert">{{ frameError }}</p>
  </section>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import authApi from '../services/authApi'
import { createBrowserInputQueue } from '../services/browserInputQueue'
import { createBrowserStream } from '../services/browserStream'
const props = defineProps({ job: { type: Object, required: true } })
const human = computed(() => props.job.control_mode === 'HUMAN')
const frame = ref(null), frameError = ref(''), inputText = ref(''), surface = ref(null)
const connected = ref(false), switching = ref(null)
let dragging = false, lastMove = 0, pointerId, lastPoint, disposed = false
const stream = createBrowserStream({
  url: () => authApi.getClient().getUri({ url: `/api/rpa/jobs/${props.job.job_id}/stream` }),
  token: () => authApi.getAccessToken(),
  prepare: () => authApi.getProfile(),
  onState: state => { connected.value = state === 'connected' },
  onReset: () => { inputs.clear(); clearDrag(); frame.value = null; switching.value = null; inputText.value = '' },
  onFrame: data => {
    if (data.epoch < props.job.control_epoch) return
    if (switching.value !== null && data.page_id !== switching.value) return
    switching.value = null
    frame.value = data
    frameError.value = ''
  },
  onError: message => { frameError.value = message },
})
const inputs = createBrowserInputQueue({
  send: payload => stream.send(payload),
  onError: () => { frameError.value = '操作结果未确认，请检查画面后重试' },
})
const terminal = () => ['SUCCEEDED', 'PARTIAL_SUCCESS', 'FAILED', 'CANCELLED', 'INTERRUPTED'].includes(props.job.status)
function enqueue(payload) {
  if (disposed || !connected.value || !human.value || !frame.value || switching.value !== null
      || frame.value.epoch !== props.job.control_epoch) return Promise.resolve(false)
  const data = { ...payload, job_id: props.job.job_id, epoch: frame.value.epoch, page_id: frame.value.page_id, revision: frame.value.revision }
  if (payload.kind === 'switch_page') data.page_id = payload.page_id
  return inputs.enqueue(data)
}
function coords(event) {
  const box = surface.value.getBoundingClientRect()
  const width = frame.value.width, height = frame.value.height
  return { x: Math.max(0, Math.min(width - 1, (event.clientX - box.left) / box.width * width)),
    y: Math.max(0, Math.min(height - 1, (event.clientY - box.top) / box.height * height)) }
}
function pointerDown(event) {
  if (!human.value || !connected.value || !frame.value || frame.value.epoch !== props.job.control_epoch
      || dragging || event.button !== 0 || switching.value !== null) return
  surface.value.focus()
  surface.value.setPointerCapture(event.pointerId)
  pointerId = event.pointerId
  lastPoint = coords(event)
  dragging = true
  enqueue({ kind: 'pointer_down', ...lastPoint })
}
function pointerMove(event) {
  if (!dragging || event.pointerId !== pointerId || !frame.value || Date.now() - lastMove < 16) return
  lastMove = Date.now()
  lastPoint = coords(event)
  enqueue({ kind: 'pointer_move', ...lastPoint })
}
function clearDrag() {
  dragging = false
  if (pointerId !== undefined && surface.value?.hasPointerCapture(pointerId)) surface.value.releasePointerCapture(pointerId)
  pointerId = undefined
}
function releaseDrag() {
  if (!dragging) return
  enqueue({ kind: 'pointer_cancel' })
  clearDrag()
}
function pointerUp(event) {
  if (!dragging || event.pointerId !== pointerId) return
  if (frame.value) lastPoint = coords(event)
  enqueue({ kind: 'pointer_up', ...lastPoint })
  clearDrag()
}
function wheel(event) { enqueue({ kind: 'scroll', delta_y: Math.sign(event.deltaY) * 500 }) }
function keyDown(event) {
  if (!human.value || event.isComposing) return
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a') {
    event.preventDefault(); enqueue({ kind: 'key', key: 'Control+A' }); return
  }
  if (['Enter', 'Tab', 'Backspace', 'Delete', 'Escape', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
    event.preventDefault(); enqueue({ kind: 'key', key: event.key })
  } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey) {
    event.preventDefault(); enqueue({ kind: 'text', text: event.key })
  }
}
async function sendText() {
  const text = inputText.value
  inputText.value = ''
  await enqueue({ kind: 'text', text })
  surface.value?.focus()
}
async function switchPage(page_id) {
  inputs.clear()
  clearDrag()
  const sent = enqueue({ kind: 'switch_page', page_id })
  switching.value = page_id
  if (!await sent) switching.value = null
}
function reset() { stream.stop(); frameError.value = ''; wake() }
function wake() {
  if (!disposed && !document.hidden && !terminal()) stream.start()
  else stream.stop()
}
watch([() => props.job.job_id, () => props.job.control_epoch, () => props.job.control_mode], reset)
watch(() => props.job.status, wake)
onMounted(() => { wake(); document.addEventListener('visibilitychange', wake); window.addEventListener('blur', releaseDrag) })
onBeforeUnmount(() => { disposed = true; stream.stop(); document.removeEventListener('visibilitychange', wake); window.removeEventListener('blur', releaseDrag) })
</script>
