<template>
  <section class="portal-picker" aria-label="官网课程和作业选择">
    <header
      ><strong>官网课程和作业</strong
      ><button type="button" :disabled="refreshing || !ready" @click="sync">{{
        refreshing ? '正在更新目录…' : '更新课程'
      }}</button
      ><router-link to="/settings/portal">教务账号设置</router-link></header
    >
    <p v-if="!ready"
      >{{ connectionMessage }}。<router-link to="/settings/portal"
        >前往设置并完成认证</router-link
      ></p
    >
    <p v-if="error" role="alert" class="error">{{ error }}</p
    ><p v-if="notice" role="status">{{ notice }}</p>
    <p v-if="lastSynced">目录更新于 {{ new Date(lastSynced).toLocaleString() }}</p>
    <label
      >筛选课程或学期<input v-model="filter" placeholder="输入名称或学期筛选已同步目录"
    /></label>
    <label
      >课程<select v-model="courseId" aria-label="课程" :disabled="disabled" @change="loadCourse"
        ><option value="">请选择课程</option
        ><option v-for="course in filtered" :key="course.id" :value="course.id"
          >{{ course.title }} · {{ course.term || '未提供学期' }}</option
        ></select
      ></label
    >
    <div v-if="detail" class="selection-grid">
      <label
        >作业<select v-model="homeworkIds" aria-label="作业" :disabled="disabled" multiple size="6"
          ><option v-for="work in detail.homeworks" :key="work.id" :value="work.id"
            >{{ work.title }}{{ work.deadline ? ` · ${work.deadline}` : '' }}</option
          ></select
        ></label
      >
      <p>官网作业已自动创建。进入作业可按班级批改；此处的批量入口处理全部班级。</p>
      <p
        v-for="work in detail.homeworks.filter(
          (w) => w.assignment_id && homeworkIds.includes(w.id)
        )"
        :key="work.id"
        ><router-link :to="`/assignments/${work.assignment_id}?tab=materials`"
          >进入“{{ work.title }}”按班级批改 →</router-link
        ></p
      >
      <button type="button" :disabled="!homeworkIds.length || disabled" @click="add"
        >加入待批改列表</button
      >
    </div>
    <ul v-if="selected.length"
      ><li v-for="item in selected" :key="item.homework_id"
        >{{ item.course_title }} / {{ item.homework_title }} / 全部班级<button
          type="button"
          :disabled="disabled"
          @click="remove(item)"
          >移除</button
        ></li
      ></ul
    >
    <p v-else>尚未选择作业。选择课程后，可按住 Ctrl 多选作业，将自动批改全部班级。</p>
  </section>
</template>
<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue';
import authApi from '../services/authApi';
import { createPortalApi, operationActive, portalStatus } from '../services/portalApi';
defineProps({ disabled: Boolean });
const emit = defineEmits(['change']);
const api = createPortalApi(authApi.getClient());
const courses = ref([]),
  courseId = ref(''),
  detail = ref(null),
  homeworkIds = ref([]),
  selected = ref([]),
  filter = ref(''),
  snapshotId = ref(''),
  lastSynced = ref(null);
const error = ref(''),
  notice = ref(''),
  ready = ref(false),
  refreshing = ref(false),
  connectionMessage = ref('请先连接教务账号');
const filtered = computed(() =>
  courses.value.filter((course) => `${course.title} ${course.term}`.includes(filter.value.trim()))
);
let timer,
  disposed = false,
  polling = false;
function publish() {
  emit('change', {
    snapshot_id: snapshotId.value,
    items: selected.value.map(({ homework_id }) => ({ homework_id })),
    titles: selected.value.map(({ homework_id, course_title, homework_title }) => ({
      homework_id,
      course_title,
      homework_title,
    })),
  });
}
async function load() {
  if (disposed || polling) return;
  polling = true;
  try {
    const { data } = await api.connection();
    if (disposed) return;
    ready.value = data.status === 'READY';
    connectionMessage.value = portalStatus(data.status);
    refreshing.value = Boolean(operationActive(data.operation));
    lastSynced.value = data.connection?.last_synced_at;
    const current = data.connection?.active_snapshot_id || '';
    if (current !== snapshotId.value) {
      if (selected.value.length) notice.value = '目录已更新，请根据最新课程和作业重新选择';
      snapshotId.value = current;
      selected.value = [];
      courseId.value = '';
      detail.value = null;
      homeworkIds.value = [];
      publish();
      courses.value = current ? (await api.courses()).data : [];
    }
    if (data.operation?.status === 'FAILED') error.value = data.operation.message;
  } catch (e) {
    if (!disposed) error.value = e.response?.data?.detail || '课程目录读取失败';
  } finally {
    polling = false;
  }
}
async function loadCourse() {
  detail.value = null;
  homeworkIds.value = [];
  error.value = '';
  if (!courseId.value) return;
  const id = courseId.value;
  try {
    const { data } = await api.homeworks(id);
    if (!disposed && id === courseId.value) detail.value = data;
  } catch (e) {
    error.value = e.response?.data?.detail || '作业目录读取失败';
  }
}
async function sync() {
  refreshing.value = true;
  error.value = '';
  try {
    await api.sync();
    notice.value = '目录刷新已排队，完成后会自动显示最新内容';
  } catch (e) {
    refreshing.value = false;
    error.value = e.response?.data?.detail || '目录刷新未完成';
  }
}
function add() {
  if (!detail.value || detail.value.snapshot_id !== snapshotId.value) {
    error.value = '目录已变化，请重新选择';
    return;
  }
  const next = [...selected.value];
  for (const workId of homeworkIds.value) {
    if (next.some((item) => item.homework_id === workId)) continue;
    const work = detail.value.homeworks.find((item) => item.id === workId);
    if (!work) continue;
    next.push({
      homework_id: work.id,
      course_title: detail.value.course.title,
      homework_title: work.title,
    });
  }
  if (next.length > 50) {
    error.value = '一次最多选择 50 个作业';
    return;
  }
  selected.value = next;
  error.value = '';
  publish();
}
function remove(item) {
  selected.value = selected.value.filter((value) => value !== item);
  publish();
}
onMounted(() => {
  load();
  timer = setInterval(load, 3000);
});
onBeforeUnmount(() => {
  disposed = true;
  clearInterval(timer);
});
</script>
<style scoped>
.portal-picker {
  display: grid;
  gap: 14px;
  padding: 18px;
  border: 1px solid #cbd5e1;
  border-radius: 12px;
}
header {
  display: flex;
  gap: 14px;
  align-items: center;
  flex-wrap: wrap;
}
label {
  display: grid;
  gap: 7px;
}
input,
select {
  width: 100%;
  padding: 9px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
}
button {
  padding: 7px 12px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
}
button:disabled {
  opacity: 0.5;
}
.selection-grid {
  display: grid;
  gap: 14px;
}
li {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin: 8px 0;
}
.error {
  color: #b91c1c;
}
a {
  color: #2563eb;
}
</style>
