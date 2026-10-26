<template>
  <div class="assignment-workspace">
    <header class="assignment-heading">
      <div
        ><div class="eyebrow">ASSIGNMENTS</div><h1 class="page-title">作业批改</h1
        ><p class="page-description"
          >同步官网作业后，进入作业查看要求、调整评分标准，再按班级批改。</p
        ></div
      >
      <div class="heading-actions"
        ><router-link class="workspace-button" to="/grade-homework"
          ><AppIcon name="download" />导入本地作业</router-link
        ><router-link class="workspace-button primary" to="/assignments/new"
          ><AppIcon name="plus" />新建作业</router-link
        ></div
      >
    </header>
    <PortalSyncBar @updated="loadAssignments" />
    <div class="workspace-metrics" aria-label="作业概况">
      <div
        ><span>全部作业</span><strong>{{ assignments.length }}</strong></div
      >
      <div
        ><span><i class="metric-dot blue"></i>处理中</span
        ><strong>{{ summariesAvailable ? processingCount : '—' }}</strong></div
      >
      <div
        ><span><i class="metric-dot amber"></i>待处理</span
        ><strong>{{ summariesAvailable ? attentionCount : '—' }}</strong></div
      >
      <div
        ><span><i class="metric-dot green"></i>已有评分结果</span
        ><strong>{{ summariesAvailable ? gradedCount : '—' }}</strong></div
      >
    </div>
    <div v-if="error" class="workspace-error" role="alert"
      >{{ error }} <button @click="loadAssignments">重新加载</button></div
    >
    <div class="assignment-layout">
      <section class="assignment-list surface" aria-label="作业列表" :aria-busy="loading">
        <div class="list-toolbar"
          ><label class="search-field"
            ><AppIcon name="search" /><input
              v-model="search"
              aria-label="搜索作业"
              placeholder="搜索作业、课程或题目…" /></label
          ><button
            class="icon-button"
            aria-label="刷新作业列表"
            :disabled="loading"
            @click="loadAssignments"
            ><AppIcon name="history" /></button
        ></div>
        <div class="list-filters"
          ><select v-model="course" aria-label="筛选课程"
            ><option value="">全部课程</option
            ><option v-for="name in courses" :key="name" :value="name">{{ name }}</option></select
          ><select v-model="status" aria-label="筛选状态"
            ><option value="">全部状态</option
            ><option v-for="(state, key) in assignmentStates" :key="key" :value="key">{{
              state.label
            }}</option></select
          ><span>{{ filtered.length }} 项作业</span
          ><button
            v-if="search || course || status"
            class="text-action"
            @click="
              search = '';
              course = '';
              status = '';
            "
            >清除筛选</button
          ></div
        >
        <div v-if="loading && !assignments.length" class="workspace-empty"
          ><Loader /><p>正在读取作业…</p></div
        >
        <template v-else-if="filtered.length">
          <div class="assignment-table-scroll"
            ><table class="assignment-table"
              ><thead
                ><tr
                  ><th>作业名称 / 课程</th><th>状态</th><th>已评分</th><th>最近处理</th
                  ><th><span class="sr-only">操作</span></th></tr
                ></thead
              ><tbody>
                <tr
                  v-for="item in filtered"
                  :key="item.id"
                  :class="{ selected: selected?.id === item.id }"
                  @click="selectedId = item.id"
                >
                  <td
                    ><button
                      class="assignment-title"
                      :aria-pressed="selected?.id === item.id"
                      @click.stop="selectedId = item.id"
                      >{{ item.task_name || '未命名作业' }}</button
                    ><span class="course-label">{{ item.course_name || '未命名课程' }}</span></td
                  >
                  <td
                    ><span class="status-badge" :class="assignmentState(item).tone"
                      ><i></i>{{ assignmentState(item).label }}</span
                    ></td
                  ><td class="tabular"
                    >{{ item.summary ? item.summary.graded_count : '—'
                    }}<small v-if="item.summary"> 人</small></td
                  ><td class="activity-cell">{{ formatActivity(item.summary?.updated_at) }}</td
                  ><td
                    ><router-link
                      class="row-open"
                      :to="`/assignments/${item.id}?tab=materials`"
                      :aria-label="`批改${item.task_name}`"
                      >进入 <span aria-hidden="true">↗</span></router-link
                    ></td
                  >
                </tr>
              </tbody></table
            ></div
          >
          <div class="list-footer">点击作业名称预览内容，进入作业后导入材料或查看评分。</div>
        </template>
        <div v-else class="workspace-empty"
          ><div class="empty-symbol"><AppIcon name="book" /></div
          ><h2>{{ assignments.length ? '没有符合条件的作业' : '从第一份作业开始' }}</h2
          ><p>{{
            assignments.length
              ? '试试其他关键词，或清除筛选。'
              : '先创建题目和评分标准，也可以直接导入并自动批改。'
          }}</p
          ><router-link
            v-if="!assignments.length"
            class="workspace-button primary"
            to="/assignments/new"
            >新建作业</router-link
          ></div
        >
      </section>
      <aside class="assignment-preview surface" aria-label="作业预览">
        <div class="preview-heading"><span>作业预览</span><AppIcon name="file" /></div>
        <template v-if="selected"
          ><div class="preview-course">{{ selected.course_name }}</div
          ><h2>{{ selected.task_name }}</h2
          ><span class="status-badge" :class="assignmentState(selected).tone"
            ><i></i>{{ assignmentState(selected).label }}</span
          ><div class="preview-stats"
            ><div
              ><strong>{{ selected.summary?.graded_count ?? '—' }}</strong
              ><span>已评分学生</span></div
            ><div
              ><strong>{{ selected.summary?.batch_count ?? '—' }}</strong
              ><span>材料批次</span></div
            ></div
          ><h3>题目要求</h3
          ><p class="question-preview">{{
            selected.question ||
            (selected.portal_homework_id
              ? '作业已创建，正在自动读取官网要求…'
              : '尚未填写题目要求。')
          }}</p
          ><h3>评分标准</h3><RubricSummary :rubric="selected.rubric" /><router-link
            class="workspace-button primary preview-open"
            :to="`/assignments/${selected.id}`"
            >查看完整作业 <span aria-hidden="true">→</span></router-link
          ><router-link
            class="text-action preview-results"
            :to="`/assignments/${selected.id}?tab=results`"
            >查看评分结果</router-link
          ><details class="assignment-more"
            ><summary>更多操作</summary
            ><button
              class="danger-action"
              :disabled="deleting"
              @click="removeAssignment(selected)"
              >{{ deleting ? '删除中…' : '删除此作业' }}</button
            ></details
          ></template
        >
        <div v-else class="preview-placeholder"
          ><AppIcon name="file" /><p>选择一份作业<br />在这里查看题目和评分标准</p></div
        >
      </aside>
    </div>
  </div>
</template>
<script setup>
import { computed, ref, onMounted, onUnmounted } from 'vue';
import gradingApi from '../services/gradingApi';
import AppIcon from '../components/AppIcon.vue';
import Loader from '../components/Loader.vue';
import RubricSummary from '../components/RubricSummary.vue';
import PortalSyncBar from '../components/PortalSyncBar.vue';
import {
  assignmentStates,
  assignmentState,
  filterAssignments,
  formatActivity,
} from '../services/assignmentWorkspace';
const assignments = ref([]),
  search = ref(''),
  course = ref(''),
  status = ref(''),
  selectedId = ref(null),
  loading = ref(false),
  deleting = ref(false),
  error = ref('');
const courses = computed(() =>
  [...new Set(assignments.value.map((item) => item.course_name).filter(Boolean))].sort()
);
const filtered = computed(() =>
  filterAssignments(assignments.value, {
    search: search.value,
    course: course.value,
    status: status.value,
  })
);
const selected = computed(
  () => filtered.value.find((item) => item.id === selectedId.value) || filtered.value[0]
);
const summariesAvailable = computed(() => assignments.value.every((item) => item.summary));
const processingCount = computed(
  () =>
    assignments.value.filter((item) => ['GRADING', 'PREPARING'].includes(item.summary?.status))
      .length
);
const attentionCount = computed(
  () => assignments.value.filter((item) => item.summary?.status === 'ATTENTION').length
);
const gradedCount = computed(
  () => assignments.value.filter((item) => item.summary?.graded_count > 0).length
);
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
  justify-content: space-between;
  align-items: center;
  gap: 20px;
  margin-bottom: 28px;
}
.eyebrow {
  color: var(--color-muted);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 2px;
  margin-bottom: 8px;
}
.heading-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.workspace-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  border: 1px solid var(--color-line);
  border-radius: 12px;
  overflow: hidden;
  background: var(--color-line);
  margin-bottom: 26px;
}
.workspace-metrics > div {
  background: white;
  padding: 19px 24px;
}
.workspace-metrics span {
  display: flex;
  gap: 7px;
  align-items: center;
  font-size: 12px;
  color: var(--color-muted);
}
.workspace-metrics strong {
  display: block;
  margin-top: 9px;
  font-size: 27px;
  font-weight: 600;
  line-height: 1.2;
  color: var(--color-ink);
  font-variant-numeric: tabular-nums;
}
.metric-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #6679ed;
}
.metric-dot.amber {
  background: #d99531;
}
.metric-dot.green {
  background: #2d9b7b;
}
.assignment-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 310px;
  gap: 22px;
  align-items: start;
}
.assignment-list {
  min-width: 0;
  overflow: hidden;
}
.list-toolbar {
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 17px 20px 10px;
}
.search-field {
  display: flex;
  align-items: center;
  gap: 9px;
  flex: 1;
  min-width: 0;
}
.search-field svg {
  width: 17px;
  color: #8490a2;
}
.search-field input {
  width: 100%;
  min-width: 0;
  border: 0 !important;
  box-shadow: none !important;
  font-size: 13px;
  padding: 0;
  min-height: 34px !important;
  background: transparent;
}
.list-filters {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 0 20px 16px;
  flex-wrap: wrap;
}
.list-filters select {
  max-width: 200px;
  min-height: 32px;
  font-size: 11px;
  padding: 4px 25px 4px 10px;
  border: 1px solid var(--color-line);
  background: #fafbfe;
}
.list-filters > span {
  margin-left: auto;
  color: var(--color-muted);
  font-size: 11px;
}
.assignment-table-scroll {
  overflow-x: auto;
}
.assignment-table {
  width: 100%;
  border-collapse: collapse;
  white-space: nowrap;
}
.assignment-table th {
  text-align: left;
  font-size: 10px;
  color: var(--color-muted);
  font-weight: 500;
  background: #fafbfe;
  border-block: 1px solid var(--color-line);
  padding: 12px 16px;
}
.assignment-table td {
  padding: 19px 16px;
  border-bottom: 1px solid #edf0f5;
  font-size: 12px;
}
.assignment-table th:first-child,
.assignment-table td:first-child {
  padding-left: 20px;
}
.assignment-table tr.selected td {
  background: #f5f6ff;
}
.assignment-table tbody tr:hover td {
  background: #f8f9ff;
}
.assignment-title {
  display: block;
  text-align: left;
  font-size: 13px;
  font-weight: 600;
  color: #27334b;
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.assignment-title:hover {
  color: var(--color-accent);
}
.course-label {
  display: block;
  margin-top: 7px;
  color: var(--color-muted);
  font-size: 11px;
}
.tabular {
  font-variant-numeric: tabular-nums;
}
.tabular small {
  font-size: 10px;
  color: var(--color-muted);
}
.activity-cell {
  font-size: 11px !important;
  color: var(--color-muted);
}
.row-open {
  font-size: 11px;
  color: var(--color-accent);
}
.row-open span {
  margin-left: 3px;
}
.list-footer {
  padding: 15px 20px;
  font-size: 10px;
  color: #8993a6;
  background: #fcfcfe;
}
.assignment-preview {
  padding: 23px;
  min-width: 0;
}
.preview-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #687387;
  font-size: 11px;
  border-bottom: 1px solid var(--color-line);
  padding-bottom: 16px;
  margin-bottom: 22px;
}
.preview-heading svg {
  width: 16px;
}
.preview-course {
  font-size: 11px;
  color: var(--color-accent);
  margin-bottom: 7px;
}
.assignment-preview h2 {
  font-size: 19px;
  font-weight: 600;
  line-height: 1.6;
  margin-bottom: 12px;
  overflow-wrap: anywhere;
}
.preview-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  background: #f8f9fc;
  border-radius: 9px;
  padding: 17px 12px;
  margin: 22px 0;
}
.preview-stats > div {
  display: flex;
  flex-direction: column;
  gap: 5px;
  text-align: center;
}
.preview-stats > div + div {
  border-left: 1px solid var(--color-line);
}
.preview-stats strong {
  font-size: 21px;
  font-weight: 600;
}
.preview-stats span {
  font-size: 10px;
  color: var(--color-muted);
}
.assignment-preview h3 {
  font-size: 12px;
  font-weight: 600;
  margin-top: 23px;
}
.question-preview {
  font-size: 12px;
  color: #667086;
  line-height: 1.9;
  white-space: pre-wrap;
  display: -webkit-box;
  -webkit-line-clamp: 7;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin-top: 10px;
  overflow-wrap: anywhere;
}
.preview-open {
  display: flex;
  width: 100%;
  margin-top: 22px;
  justify-content: space-between;
}
.preview-results {
  display: block;
  text-align: center;
  margin-top: 15px;
  font-size: 12px;
}
.assignment-more {
  border-top: 1px solid var(--color-line);
  margin-top: 23px;
  padding-top: 15px;
  font-size: 11px;
  color: var(--color-muted);
}
.assignment-more summary {
  cursor: pointer;
}
.danger-action {
  color: #bf435b;
  padding-top: 12px;
}
.preview-placeholder {
  text-align: center;
  padding: 65px 0;
  color: var(--color-muted);
  font-size: 12px;
  line-height: 2;
}
.preview-placeholder svg {
  width: 36px;
  margin: 0 auto 18px;
  color: #bcc3d3;
}
.workspace-empty {
  text-align: center;
  padding: 65px 22px;
  color: var(--color-muted);
  font-size: 12px;
}
.workspace-empty h2 {
  color: var(--color-ink);
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 10px;
}
.workspace-empty p {
  line-height: 1.8;
  margin-bottom: 20px;
}
.empty-symbol {
  width: 48px;
  height: 48px;
  display: grid;
  place-items: center;
  background: #f0f2ff;
  border-radius: 13px;
  color: var(--color-accent);
  margin: 0 auto 20px;
}
.empty-symbol svg {
  width: 24px;
}
@media (min-width: 1500px) {
  .assignment-layout {
    grid-template-columns: minmax(0, 1fr) 340px;
  }
}
@media (max-width: 1200px) {
  .assignment-layout {
    grid-template-columns: minmax(0, 1fr);
  }
  .assignment-preview {
    display: block;
  }
  .preview-stats {
    max-width: 360px;
  }
  .question-preview {
    -webkit-line-clamp: 5;
  }
  .preview-open {
    max-width: 300px;
  }
  .preview-results {
    text-align: left;
  }
  .assignment-more {
    max-width: 300px;
  }
}
@media (max-width: 760px) {
  .assignment-heading {
    align-items: flex-start;
    flex-direction: column;
    margin-bottom: 20px;
  }
  .heading-actions {
    width: 100%;
  }
  .heading-actions a {
    flex: 1;
    justify-content: center;
    font-size: 11px;
    padding: 10px;
  }
  .workspace-metrics {
    grid-template-columns: repeat(2, 1fr);
    margin-bottom: 18px;
  }
  .workspace-metrics > div {
    padding: 15px 18px;
  }
  .workspace-metrics strong {
    font-size: 23px;
  }
  .list-filters select {
    max-width: 145px;
  }
  .list-toolbar {
    padding: 12px 14px 6px;
  }
  .list-filters {
    padding: 0 14px 12px;
  }
  .assignment-title {
    max-width: 190px;
  }
  .assignment-preview {
    padding: 20px;
  }
  .assignment-table td {
    padding: 16px 12px;
  }
  .assignment-layout {
    gap: 16px;
  }
}

@media (max-width: 760px) {
  .assignment-table th:nth-child(4),
  .assignment-table td:nth-child(4) {
    display: none;
  }
  .assignment-table th,
  .assignment-table td {
    padding-left: 8px;
    padding-right: 8px;
  }
  .assignment-table th:first-child,
  .assignment-table td:first-child {
    padding-left: 14px;
  }
  .assignment-title,
  .course-label {
    max-width: 125px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
}
</style>
