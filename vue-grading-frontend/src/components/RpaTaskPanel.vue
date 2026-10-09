<template>
  <section :class="embedded ? 'bg-white p-5' : 'rounded-2xl border bg-white shadow-sm p-5 mb-6'" aria-label="作业抓取控制台">
    <div class="flex items-center justify-between gap-3 mb-4">
      <div v-if="!embedded"><h2 class="text-lg font-semibold">作业下载与批改</h2></div>
      <button class="text-sm text-blue-700 ml-auto" @click="loadJobs">刷新任务</button>
    </div>
    <form class="flex flex-wrap items-end gap-3" @submit.prevent="create">
      <label class="flex-1 min-w-40 text-sm">课程名称<input v-model="course" required maxlength="255" class="block mt-1 w-full border rounded px-3 py-2" /></label>
      <label class="flex-1 min-w-40 text-sm">作业名称<input v-model="assignment" required maxlength="255" class="block mt-1 w-full border rounded px-3 py-2" /></label>
      <label class="w-40 text-sm">学期（可选）<input v-model="term" maxlength="255" class="block mt-1 w-full border rounded px-3 py-2" /></label>
      <button :disabled="busy" class="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50">启动抓取</button>
    </form>
    <div v-if="jobs.length" class="mt-4 flex flex-wrap items-center gap-2">
      <label for="rpa-jobs" class="text-sm">我的任务</label>
      <select id="rpa-jobs" :value="selectedId" class="border rounded px-2 py-2 text-sm max-w-full" @change="select($event.target.value)">
        <option v-for="item in jobs" :key="item.job_id" :value="item.job_id">{{ item.course_name }} / {{ item.assignment_name }} · {{ statusName(item.status) }}</option>
      </select>
      <button v-if="job && !showBrowser" class="text-sm text-blue-700" @click="showBrowser = true">打开任务浏览器</button>
    </div>
    <p v-if="error" role="alert" class="mt-3 text-sm text-red-700">{{ error }}</p>
    <div v-if="job" class="mt-4 space-y-4">
      <div class="rounded-lg bg-slate-50 p-3">
        <div class="flex flex-wrap gap-x-5 gap-y-2 text-sm"><span>状态：{{ loginWaiting ? (loginComplete ? '已登录，等待继续' : '等待登录完成并交还控制') : statusName(job.status) }}</span><span>阶段：{{ loginWaiting ? (loginComplete ? '登录完成' : '人工登录与交接') : stageName(job.stage) }}</span><span>控制：{{ controlName(job.control_mode) }}</span></div>
        <p class="mt-2 text-sm">{{ loginWaiting ? (loginComplete ? '官网登录已完成，点击下方“交还自动化”即可继续。' : '官网右上角显示用户信息后，请点击下方“交还自动化”。人工控制期间任务会等待你的确认。') : job.message }}</p><p class="mt-1 text-xs text-slate-500 break-all">任务 ID：{{ job.job_id }}</p>
        <p v-if="job.stage === 'DOWNLOAD' && job.download_progress" class="mt-2 text-xs text-slate-600">已接收 {{ Math.round(job.download_progress.received_bytes / 1024) }} KB<span v-if="job.download_progress.total_bytes"> / {{ Math.round(job.download_progress.total_bytes / 1024) }} KB</span></p>
      </div>
      <div class="flex flex-wrap gap-2 text-sm">
        <button v-if="!ended && job.control_mode !== 'HUMAN'" :disabled="busy || job.control_mode === 'PAUSING'" class="px-3 py-2 rounded border" @click="control('pause')">暂停并接管</button>
        <button v-if="!ended && job.control_mode === 'HUMAN'" :disabled="busy" class="px-3 py-2 rounded bg-blue-600 text-white" @click="control('resume')">交还自动化</button>
        <button v-if="['INTERRUPTED','FAILED','PARTIAL_SUCCESS'].includes(job.status)" :disabled="busy" class="px-3 py-2 rounded border" @click="control('restart')">重新登录并重试，保留成功附件</button>
        <button v-if="!ended || job.status === 'INTERRUPTED'" :disabled="busy" class="px-3 py-2 rounded border text-red-700" @click="control('cancel')">取消任务</button>
        <button v-if="showBrowser" class="px-3 py-2 rounded border" @click="showBrowser = false">收起浏览器</button>
      </div>
      <RpaClassSelector v-if="job.waiting_reason === 'CLASS_SCOPE'" :key="job.job_id + JSON.stringify(job.classes)" :classes="job.classes || []" @confirm="confirmScope" />
      <div v-if="job.control_mode === 'HUMAN' && !job.export_attempted && ['TARGET_CONFIRMATION','RECOVERY','USER_TAKEOVER'].includes(job.waiting_reason)" class="border rounded-lg p-3 text-sm">
        <label class="flex gap-2"><input v-model="targetConfirmed" type="checkbox" />我已核对当前官网页面属于“{{ job.target.course_name }} / {{ job.target.assignment_name }}”</label>
        <button :disabled="!targetConfirmed || busy" class="mt-2 text-blue-700 disabled:opacity-50" @click="confirmTarget">确认当前作业并继续</button>
      </div>
      <div v-if="job.waiting_reason === 'EXPORT_ASSOCIATION' && job.export_candidates?.length" class="border rounded-lg p-4 space-y-2">
        <strong class="text-sm">请核对并选择本次导出的全部记录</strong>
        <label v-for="entry in job.export_candidates" :key="entry.ref" class="flex gap-2 text-sm"><input v-model="exportRefs" type="checkbox" :value="entry.ref" />{{ entry.text }}</label>
        <button :disabled="!exportRefs.length || busy" class="px-3 py-2 rounded bg-blue-600 text-white text-sm" @click="bindExports">确认记录并继续</button>
      </div>
      <RpaBrowserPanel v-if="active && showBrowser && !ended" :key="job.job_id" :job="job" />
      <div v-if="job.grading_files?.length" class="overflow-x-auto">
        <table class="w-full text-sm text-left"><thead><tr class="border-b"><th class="p-2">班级 / 附件</th><th class="p-2">批改状态</th><th class="p-2">说明</th></tr></thead>
          <tbody><tr v-for="file in job.grading_files" :key="file.id" class="border-b"><td class="p-2">{{ file.class_name || '待核对班级' }}<div class="text-xs text-slate-500 break-all">{{ file.path.split('/').pop() }}</div></td><td class="p-2">{{ statusName(file.grading_status) }}</td><td class="p-2"><router-link v-if="file.assignment_id" :to="'/assignments/' + file.assignment_id" class="text-blue-700">查看作业</router-link><div>{{ file.message }}</div><details v-if="file.grading_status === 'WAITING_RUBRIC'" class="mt-2"><summary class="cursor-pointer text-blue-700">核对或更正班级</summary><input v-model="classNames[file.id]" maxlength="255" placeholder="本地作业中的准确班级名称" class="border rounded px-2 py-1 mt-2" /><button :disabled="!classNames[file.id] || busy" class="ml-2 text-blue-700" @click="control('map-file', { file_id:file.id, class_name:classNames[file.id] })">关联班级</button></details></td></tr></tbody>
        </table>
      </div>
      <p v-for="file in (job.files || []).filter(f => f.status === 'FAILED')" :key="file.export_ref" class="text-sm text-red-700">附件失败：{{ file.error }}</p>
      <details v-if="job.events?.length" class="text-sm"><summary class="cursor-pointer text-slate-600">执行记录</summary><ol class="mt-2 space-y-1"><li v-for="(event, i) in job.events.slice(-12)" :key="i">{{ new Date(event.time * 1000).toLocaleTimeString() }} · {{ event.message }}</li></ol></details>
    </div>
  </section>
</template>

<script setup>
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
import authApi from '../services/authApi'
import RpaBrowserPanel from './RpaBrowserPanel.vue'
import RpaClassSelector from './RpaClassSelector.vue'
const props = defineProps({
  active: { type: Boolean, default: true },
  taskId: { type: String, default: '' },
  embedded: { type: Boolean, default: false },
})
const course = ref(''), assignment = ref(''), term = ref(''), jobs = ref([]), job = ref(null)
const selectedId = ref(''), showBrowser = ref(false), busy = ref(false), error = ref(''), exportRefs = ref([])
const targetConfirmed = ref(false)
const classNames = ref({})
const api = authApi.getClient()
let timer, pollingId = '', pendingCreate = null
const storageKey = 'rpa-job-' + (authApi.getCurrentUser()?.user_id || '')
const ended = computed(() => ['SUCCEEDED', 'PARTIAL_SUCCESS', 'FAILED', 'CANCELLED', 'INTERRUPTED'].includes(job.value?.status))
const loginWaiting = computed(() => !ended.value && job.value?.control_mode === 'HUMAN' && ['LOGIN', 'LOGIN_COMPLETE'].includes(job.value?.waiting_reason))
const loginComplete = computed(() => job.value?.auth_status === 'AUTHENTICATED')
const statuses = { QUEUED:'排队中',RUNNING:'运行中',WAITING_USER:'等待人工操作',SUCCEEDED:'下载成功',SUCCESS:'批改完成',PARTIAL_SUCCESS:'部分下载成功',FAILED:'失败',CANCELLED:'已取消',INTERRUPTED:'会话中断',WAITING_RUBRIC:'等待题目与评分标准',READY:'等待批改队列',PUBLISHED:'已投递批改',PENDING:'等待批改',PROCESSING:'批改中' }
const stages = { OPEN_PORTAL:'打开官网',AUTHENTICATE:'登录与验证',LOCATE_COURSE:'定位课程',LOCATE_ASSIGNMENT:'定位作业',SELECT_CLASSES:'选择班级',EXPORT:'提交导出',WAIT_EXPORT:'等待打包',DOWNLOAD:'下载附件',VERIFY:'校验附件',DONE:'下载结束' }
const statusName = value => statuses[value] || value
const stageName = value => stages[value] || value
const controlName = value => ({AUTO:'自动执行',HUMAN:'人工操作',PAUSING:'正在暂停',RESUMING:'恢复检查'})[value] || '等待启动'
async function loadJobs() {
  try {
    jobs.value = (await api.get('/api/rpa/jobs')).data
    if (!selectedId.value && jobs.value.length) {
      const saved = localStorage.getItem(storageKey)
      await select(jobs.value.some(item => item.job_id === saved) ? saved : jobs.value[0].job_id)
    }
  } catch (e) { error.value = e.response?.data?.error || '读取任务失败' }
}
async function poll() {
  if (!props.active || !selectedId.value || pollingId === selectedId.value) return
  const id = selectedId.value
  pollingId = id
  try { const { data } = await api.get(`/api/rpa/jobs/${id}`); if (selectedId.value === id) job.value = data }
  catch (e) { if (selectedId.value === id) error.value = e.response?.data?.error || '读取任务状态失败' }
  finally { if (pollingId === id) pollingId = '' }
}
async function select(id) {
  selectedId.value = id; job.value = null; exportRefs.value = []; error.value = ''
  targetConfirmed.value = false; classNames.value = {}
  localStorage.setItem(storageKey, id)
  await poll()
}
async function create() {
  if (busy.value) return
  busy.value = true; error.value = ''
  const target = { course_name:course.value, assignment_name:assignment.value, term:term.value }
  const fingerprint = JSON.stringify(target)
  if (!pendingCreate || pendingCreate.fingerprint !== fingerprint) {
    pendingCreate = { fingerprint, key: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}` }
  }
  try {
    const { data } = await api.post('/api/rpa/jobs', target, { headers: { 'Idempotency-Key': pendingCreate.key } })
    pendingCreate = null
    await select(data.job_id); showBrowser.value = true; await loadJobs()
  } catch (e) { error.value = e.response?.data?.error || '创建任务失败' }
  finally { busy.value = false }
}
async function control(action, payload) {
  busy.value = true; error.value = ''
  try { await api.post(`/api/rpa/jobs/${selectedId.value}/${action}`, payload); await poll(); await loadJobs(); showBrowser.value = true; return true }
  catch (e) { error.value = e.response?.data?.error || '操作未完成'; return false }
  finally { busy.value = false }
}
async function confirmScope(scope) { if (await control('class-scope', scope)) await control('resume') }
async function bindExports() { if (await control('bind-exports', { refs:exportRefs.value })) await control('resume') }
async function confirmTarget() { if (await control('confirm-target', job.value.target)) { targetConfirmed.value = false; await control('resume') } }
watch([() => props.active, () => props.taskId], async ([active, taskId], previous = []) => {
  if (!active) return
  if (taskId && (!previous[0] || taskId !== previous[1])) await select(taskId)
  await loadJobs()
  await poll()
  showBrowser.value = true
}, { immediate: true })
onMounted(() => { timer = setInterval(poll, 2000) })
onBeforeUnmount(() => { clearInterval(timer) })
</script>
