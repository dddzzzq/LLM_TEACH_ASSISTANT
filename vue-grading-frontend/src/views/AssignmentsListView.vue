<template>
  <div class="assignment-workspace">
    <header class="assignment-heading">
      <div
        ><h1 class="page-title">作业工作台</h1
        ><p class="page-description">从一份作业开始，按班级完成批改与复核。</p></div
      >
      <div class="heading-actions">
        <details class="action-menu"
          ><summary class="workspace-button">更多操作</summary
          ><div class="action-menu-content"
            ><router-link to="/grade-homework">导入与自动批改 / 批量处理</router-link
            ><router-link to="/materials">全部导入记录</router-link></div
          ></details
        >
        <details ref="addMenu" class="action-menu"
          ><summary class="workspace-button primary"><AppIcon name="plus" />添加作业</summary
          ><div class="action-menu-content"
            ><button type="button" @click="syncFromMenu">从教务同步</button
            ><router-link to="/assignments/new">新建本地作业</router-link></div
          ></details
        >
      </div>
    </header>
    <PortalSyncBar ref="syncBar" @updated="loadAssignments" />
    <div v-if="error" class="workspace-error" role="alert"
      >{{ error }} <button @click="loadAssignments">重新加载</button></div
    >
    <section class="assignment-list surface" aria-label="作业列表" :aria-busy="loading">
      <nav class="assignment-views" aria-label="作业状态视图">
        <button
          v-for="item in views"
          :key="item.key"
          :class="{ active: view === item.key }"
          :aria-pressed="view === item.key"
          @click="view = item.key"
          >{{ item.label }}<span>{{ item.count }}</span></button
        >
      </nav>
      <div class="list-toolbar">
        <label class="search-field"
          ><AppIcon name="search" /><input
            v-model="search"
            aria-label="搜索作业"
            placeholder="搜索作业、课程或题目…"
        /></label>
        <div class="list-filters">
          <select v-model="course" aria-label="筛选课程"
            ><option value="">全部课程</option
            ><option v-for="name in courses" :key="name" :value="name">{{ name }}</option></select
          >
          <select v-model="source" aria-label="筛选来源"
            ><option value="">全部来源</option
            ><option value="portal">官网作业</option
            ><option value="local">本地作业</option></select
          >
          <select v-model="status" aria-label="筛选状态"
            ><option value="">全部状态</option
            ><option v-for="(state, key) in assignmentStates" :key="key" :value="key">{{
              state.label
            }}</option></select
          >
          <button
            class="icon-button"
            aria-label="刷新作业列表"
            :disabled="loading"
            @click="loadAssignments"
            ><AppIcon name="history"
          /></button>
        </div>
      </div>
      <div v-if="loading && !assignments.length" class="workspace-empty"
        ><Loader /><p>正在读取作业…</p></div
      >
      <template v-else-if="filtered.length">
        <div class="assignment-table-scroll"
          ><table class="assignment-table"
            ><thead
              ><tr
                ><th>作业名称 / 课程</th><th>要求与状态</th><th>成绩情况</th><th>最近处理</th
                ><th>下一步</th><th><span class="sr-only">预览</span></th></tr
              ></thead
            ><tbody>
              <tr v-for="item in filtered" :key="item.id">
                <td
                  ><router-link class="assignment-title" :to="detailLocation(item)">{{
                    item.task_name || '未命名作业'
                  }}</router-link
                  ><span class="course-label"
                    >{{ item.course_name || '未命名课程'
                    }}<span class="source-label">{{
                      item.portal_homework_id ? '官网' : '本地'
                    }}</span></span
                  ></td
                >
                <td
                  ><span class="status-badge" :class="assignmentState(item).tone">{{
                    assignmentState(item).label
                  }}</span
                  ><small class="requirement-state" v-if="item.portal_homework_id">{{
                    requirementLabel(item)
                  }}</small
                  ><small
                    class="needs-attention"
                    v-if="item.summary?.issue_count && item.summary?.status !== 'ATTENTION'"
                    >{{ item.summary.issue_count }} 项材料异常</small
                  ></td
                >
                <td class="tabular"
                  ><span>{{ item.summary?.graded_count ?? '—' }} <small>已评分</small></span
                  ><small class="reviewed-count"
                    >{{ item.summary?.reviewed_count ?? '—' }} 已复核</small
                  ></td
                >
                <td class="activity-cell">{{ formatActivity(item.summary?.updated_at) }}</td>
                <td
                  ><router-link class="row-open" :to="detailLocation(item, nextAction(item).tab)">{{
                    nextAction(item).label
                  }}</router-link></td
                >
                <td
                  ><button
                    class="preview-button"
                    :aria-label="`预览${item.task_name}`"
                    @click="selectedId = item.id"
                    ><AppIcon name="file" /><span>预览</span></button
                  ></td
                >
              </tr>
            </tbody></table
          ></div
        >
      </template>
      <div v-else class="workspace-empty"
        ><AppIcon name="book" /><h2>{{
          assignments.length ? '没有符合条件的作业' : '开始整理您的第一份作业'
        }}</h2
        ><p>{{
          assignments.length
            ? '试试其他关键词，或清除筛选。'
            : '从教务平台同步作业，或创建一份本地作业。'
        }}</p
        ><div class="empty-actions" v-if="!assignments.length"
          ><button class="workspace-button primary" @click="syncFromMenu">同步官网作业</button
          ><router-link class="workspace-button" to="/assignments/new">新建作业</router-link></div
        ></div
      >
      <footer class="list-footer"
        ><span>{{ filtered.length }} 项作业</span
        ><button
          v-if="search || course || status || source || view"
          class="text-action"
          @click="clearFilters"
          >清除筛选</button
        ><span class="list-tip">选择作业后，按班级查看材料与成绩</span></footer
      >
    </section>
    <AppModal v-if="selected" title="作业预览" @close="selectedId = null">
      <aside class="assignment-preview" aria-label="作业预览"
        ><p class="preview-course">{{ selected.course_name }}</p
        ><h2>{{ selected.task_name }}</h2
        ><div class="preview-meta"
          ><span class="status-badge" :class="assignmentState(selected).tone">{{
            assignmentState(selected).label
          }}</span
          ><span>已评分 {{ selected.summary?.graded_count ?? '—' }} 人</span
          ><span>{{ selected.summary?.batch_count ?? '—' }} 次导入</span></div
        ><h3>题目要求</h3
        ><p class="question-preview">{{
          selected.question ||
          (selected.portal_homework_id ? '作业已创建，正在读取官网要求…' : '尚未填写题目要求。')
        }}</p
        ><h3>评分标准</h3><RubricSummary :rubric="selected.rubric" /><details
          class="assignment-more"
          ><summary>更多操作</summary
          ><button class="danger-action" :disabled="deleting" @click="removeAssignment(selected)">{{
            deleting ? '删除中…' : '删除此作业'
          }}</button></details
        ></aside
      >
      <template #footer
        ><router-link class="workspace-button" :to="detailLocation(selected, 'results')"
          >查看评分结果</router-link
        ><router-link class="workspace-button primary" :to="detailLocation(selected)"
          >查看完整作业</router-link
        ></template
      >
    </AppModal>
  </div>
</template>
<script setup>
import { computed, ref, onMounted, onUnmounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import gradingApi from '../services/gradingApi';
import AppIcon from '../components/AppIcon.vue';
import Loader from '../components/Loader.vue';
import AppModal from '../components/AppModal.vue';
import RubricSummary from '../components/RubricSummary.vue';
import PortalSyncBar from '../components/PortalSyncBar.vue';
import {
  assignmentStates,
  assignmentState,
  filterAssignments,
  formatActivity,
} from '../services/assignmentWorkspace';
const route = useRoute(),
  router = useRouter();
const assignments = ref([]),
  selectedId = ref(null),
  loading = ref(false),
  deleting = ref(false),
  error = ref(''),
  syncBar = ref(null),
  addMenu = ref(null);
const queryModel = (key) =>
  computed({
    get: () => (typeof route.query[key] === 'string' ? route.query[key] : ''),
    set: (value) => {
      const query = { ...route.query };
      if (value) query[key] = value;
      else delete query[key];
      router.replace({ query });
    },
  });
const search = queryModel('search'),
  course = queryModel('course'),
  status = queryModel('status'),
  source = queryModel('source'),
  view = queryModel('view');
const courses = computed(() =>
  [...new Set(assignments.value.map((item) => item.course_name).filter(Boolean))].sort()
);
const summariesAvailable = computed(() => assignments.value.every((item) => item.summary));
const needsAttention = (item) =>
  item.summary?.status === 'ATTENTION' ||
  item.summary?.issue_count > 0 ||
  ['FAILED', 'REVIEW_REQUIRED', 'INTERRUPTED'].includes(item.requirements_status);
const isProcessing = (item) =>
  ['GRADING', 'PREPARING'].includes(item.summary?.status) ||
  ['PENDING', 'READING'].includes(item.requirements_status);
const processingCount = computed(() => assignments.value.filter(isProcessing).length);
const views = computed(() => [
  { key: '', label: '全部作业', count: assignments.value.length },
  {
    key: 'attention',
    label: '需要处理',
    count: summariesAvailable.value ? assignments.value.filter(needsAttention).length : '—',
  },
  {
    key: 'processing',
    label: '处理中',
    count: summariesAvailable.value ? processingCount.value : '—',
  },
  {
    key: 'graded',
    label: '已有成绩',
    count: summariesAvailable.value
      ? assignments.value.filter((item) => item.summary?.graded_count > 0).length
      : '—',
  },
]);
const filtered = computed(() =>
  filterAssignments(assignments.value, {
    search: search.value,
    course: course.value,
    status: status.value,
  }).filter(
    (item) =>
      (!source.value ||
        (source.value === 'portal' ? !!item.portal_homework_id : !item.portal_homework_id)) &&
      (!view.value ||
        (view.value === 'attention'
          ? needsAttention(item)
          : view.value === 'processing'
            ? isProcessing(item)
            : view.value === 'graded'
              ? item.summary?.graded_count > 0
              : true))
  )
);
const selected = computed(() => assignments.value.find((item) => item.id === selectedId.value));
function clearFilters() {
  const query = { ...route.query };
  for (const key of ['search', 'course', 'status', 'source', 'view']) delete query[key];
  router.replace({ query });
}
function detailLocation(item, tab = 'content') {
  return { path: `/assignments/${item.id}`, query: { tab, return_to: route.fullPath } };
}
function nextAction(item) {
  if (item.portal_homework_id && item.requirements_status !== 'READY')
    return { label: '核对要求', tab: 'content' };
  if (isProcessing(item)) return { label: '查看进度', tab: 'materials' };
  if (needsAttention(item)) return { label: '处理异常', tab: 'materials' };
  if (item.summary?.graded_count > 0) return { label: '复核成绩', tab: 'results' };
  return { label: item.summary?.status === 'READY' ? '开始批改' : '导入材料', tab: 'materials' };
}
function requirementLabel(item) {
  return (
    {
      READY: '题目要求就绪',
      PENDING: '题目待读取',
      READING: '题目同步中',
      REVIEW_REQUIRED: '题目需要核对',
      FAILED: '题目读取失败',
      INTERRUPTED: '题目读取中断',
    }[item.requirements_status] || '题目待同步'
  );
}
async function syncFromMenu() {
  if (addMenu.value) addMenu.value.open = false;
  await syncBar.value?.syncOrConnect();
}
async function loadAssignments() {
  if (loading.value) return;
  loading.value = true;
  try {
    const { data } = await gradingApi.getAssignments();
    assignments.value = Array.isArray(data) ? data : [];
    error.value = '';
  } catch {
    error.value = '暂时无法读取作业，请检查服务连接后重试。';
  } finally {
    loading.value = false;
  }
}
async function removeAssignment(item) {
  if (!confirm(`确定删除“${item.task_name}”及其评分记录？此操作无法撤销。`)) return;
  deleting.value = true;
  try {
    await gradingApi.deleteAssignment(item.id);
    assignments.value = assignments.value.filter((row) => row.id !== item.id);
    selectedId.value = null;
  } catch {
    error.value = '删除失败，请稍后重试。';
  } finally {
    deleting.value = false;
  }
}
let timer;
onMounted(() => {
  loadAssignments();
  timer = setInterval(() => {
    if (processingCount.value && !document.hidden) loadAssignments();
  }, 10000);
});
onUnmounted(() => clearInterval(timer));
</script>
<style scoped>
.assignment-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 24px;
}
.heading-actions {
  display: flex;
  gap: 10px;
  align-items: center;
}
.action-menu {
  position: relative;
}
.action-menu > summary {
  list-style: none;
  cursor: pointer;
}
.action-menu > summary::-webkit-details-marker {
  display: none;
}
.action-menu-content {
  position: absolute;
  right: 0;
  top: calc(100% + 8px);
  min-width: 206px;
  padding: 6px;
  background: var(--color-paper);
  border: 1px solid var(--color-line);
  border-radius: 12px;
  box-shadow: var(--shadow-panel);
  z-index: 20;
}
.action-menu-content > a,
.action-menu-content > button {
  display: block;
  width: 100%;
  text-align: left;
  padding: 12px;
  border-radius: 6px;
  font-size: 13px;
  color: var(--color-ink);
}
.action-menu-content > a:hover,
.action-menu-content > button:hover {
  background: var(--color-soft);
}
.assignment-list {
  overflow: clip;
}
.assignment-views {
  display: flex;
  gap: 22px;
  padding: 0 22px;
  border-bottom: 1px solid var(--color-line);
  overflow-x: auto;
}
.assignment-views button {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 18px 0 15px;
  border-bottom: 2px solid transparent;
  color: var(--color-muted);
  font-size: 13px;
  white-space: nowrap;
  min-height: 52px;
}
.assignment-views button.active {
  color: var(--color-accent);
  border-color: var(--color-accent);
  font-weight: 600;
}
.assignment-views span {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 21px;
  height: 20px;
  padding: 0 5px;
  border-radius: 5px;
  background: var(--color-soft);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.assignment-views .active span {
  background: var(--color-accent-soft);
}
.list-toolbar {
  display: flex;
  gap: 14px;
  padding: 18px 22px;
  align-items: center;
  justify-content: space-between;
}
.search-field {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 180px;
  flex: 1;
  max-width: 360px;
}
.search-field svg {
  width: 17px;
  height: 17px;
  color: var(--color-muted);
  flex-shrink: 0;
}
.search-field input {
  border: 0;
  box-shadow: none;
  background: transparent;
  padding: 0;
  min-width: 0;
  width: 100%;
  font-size: 13px;
}
.search-field input:focus {
  box-shadow: none;
}
.list-filters {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.list-filters select {
  padding: 7px 25px 7px 10px;
  max-width: 175px;
  font-size: 12px;
  border: 1px solid var(--color-line);
}
.assignment-table-scroll {
  overflow-x: auto;
}
.assignment-table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
}
.assignment-table th {
  padding: 12px 20px;
  background: var(--color-soft);
  font-size: 12px;
  font-weight: 500;
  color: var(--color-muted);
}
.assignment-table td {
  padding: 20px;
  border-bottom: 1px solid var(--color-line);
  font-size: 13px;
}
.assignment-table tr:last-child td {
  border-bottom: 0;
}
.assignment-table tbody tr:hover {
  background: var(--color-soft);
}
.assignment-table th:first-child,
.assignment-table td:first-child {
  padding-left: 24px;
  min-width: 240px;
}
.assignment-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-ink);
  line-height: 1.6;
  display: inline-block;
}
.assignment-title:hover {
  color: var(--color-accent);
}
.course-label {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-muted);
  font-size: 12px;
  margin-top: 8px;
}
.source-label {
  font-size: 10px;
  border: 1px solid var(--color-line);
  border-radius: 4px;
  padding: 0 4px;
}
.requirement-state,
.needs-attention {
  display: block;
  font-size: 11px;
  margin-top: 7px;
  color: var(--color-muted);
}
.needs-attention {
  color: var(--color-warning);
}
.tabular {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.tabular > span {
  font-size: 15px;
  font-weight: 600;
}
.tabular small {
  font-size: 11px;
  color: var(--color-muted);
  font-weight: 400;
}
.reviewed-count {
  display: block;
  margin-top: 7px;
}
.activity-cell {
  font-size: 12px !important;
  color: var(--color-muted);
  white-space: nowrap;
}
.row-open {
  display: inline-block;
  font-size: 12px;
  color: var(--color-accent);
  white-space: nowrap;
  padding: 8px 0;
}
.preview-button {
  display: flex;
  gap: 5px;
  align-items: center;
  color: var(--color-muted);
  font-size: 12px;
  min-height: 36px;
  white-space: nowrap;
}
.preview-button svg {
  width: 16px;
  height: 16px;
}
.preview-button:hover {
  color: var(--color-accent);
}
.list-footer {
  border-top: 1px solid var(--color-line);
  padding: 14px 22px;
  display: flex;
  align-items: center;
  gap: 14px;
  color: var(--color-muted);
  font-size: 12px;
}
.list-tip {
  margin-left: auto;
}
.workspace-empty {
  padding: 68px 22px;
  text-align: center;
  color: var(--color-muted);
  display: grid;
  justify-items: center;
  gap: 12px;
}
.workspace-empty > svg {
  width: 36px;
  height: 36px;
  color: var(--color-accent);
}
.workspace-empty h2 {
  font-size: 17px;
  font-weight: 600;
  color: var(--color-ink);
}
.workspace-empty p {
  font-size: 13px;
}
.empty-actions {
  display: flex;
  gap: 10px;
  margin-top: 6px;
}
.preview-course {
  color: var(--color-muted);
  font-size: 12px;
}
.assignment-preview h2 {
  font-size: 22px;
  font-weight: 650;
  margin-top: 9px;
}
.preview-meta {
  display: flex;
  gap: 14px;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 18px;
  font-size: 12px;
  color: var(--color-muted);
}
.assignment-preview h3 {
  font-size: 14px;
  font-weight: 600;
  margin: 24px 0 12px;
}
.question-preview {
  font-size: 14px;
  line-height: 1.9;
  white-space: pre-wrap;
  max-height: 240px;
  overflow: auto;
}
.assignment-more {
  margin-top: 24px;
  color: var(--color-muted);
  font-size: 12px;
}
.assignment-more summary {
  padding: 8px 0;
}
.danger-action {
  color: var(--color-danger);
  padding: 8px 0;
}
@media (max-width: 1100px) {
  .list-toolbar {
    flex-wrap: wrap;
  }
  .search-field {
    max-width: none;
  }
  .assignment-table th,
  .assignment-table td {
    padding: 15px;
  }
  .assignment-table td:first-child {
    min-width: 210px;
  }
  .activity-cell {
    display: none;
  }
  .assignment-table th:nth-child(4) {
    display: none;
  }
}
@media (max-width: 760px) {
  .assignment-heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 16px;
  }
  .heading-actions {
    width: 100%;
    justify-content: flex-end;
  }
  .assignment-views {
    gap: 18px;
    padding: 0 14px;
  }
  .assignment-views button {
    font-size: 12px;
  }
  .list-toolbar {
    padding: 12px 14px;
    gap: 8px;
  }
  .list-filters {
    width: 100%;
    gap: 6px;
  }
  .list-filters select {
    max-width: 125px;
    font-size: 11px;
    padding: 6px;
    min-height: 36px;
  }
  .search-field {
    flex-basis: 100%;
  }
  .assignment-table th,
  .assignment-table td {
    padding: 14px 10px;
  }
  .assignment-table th:first-child,
  .assignment-table td:first-child {
    padding-left: 14px;
    min-width: 175px;
  }
  .assignment-table th:nth-child(3),
  .assignment-table td:nth-child(3) {
    display: none;
  }
  .assignment-table th:last-child,
  .assignment-table td:last-child {
    display: none;
  }
  .assignment-title {
    font-size: 13px;
  }
  .course-label {
    font-size: 11px;
  }
  .requirement-state {
    display: none;
  }
  .list-footer {
    padding: 12px 14px;
  }
  .list-tip {
    display: none;
  }
  .empty-actions {
    flex-wrap: wrap;
    justify-content: center;
  }
}
</style>
