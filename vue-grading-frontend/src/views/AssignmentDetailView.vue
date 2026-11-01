<template>
  <div class="detail-page">
    <div v-if="loading" class="workspace-empty"><Loader /><p>正在读取作业…</p></div>
    <div v-else-if="error" class="workspace-error" role="alert"
      >{{ error }} <button @click="loadAssignment">重试</button></div
    >
    <template v-else-if="assignment">
      <router-link :to="returnLocation" class="back-link">返回作业列表</router-link>
      <header class="detail-heading">
        <div
          ><p class="detail-course"
            >{{ assignment.course_name
            }}<span class="source-label">{{
              assignment.portal_homework_id ? '官网作业' : '本地作业'
            }}</span></p
          ><h1 class="page-title">{{ assignment.task_name }}</h1
          ><p class="page-description">题目与标准共用，学生材料与成绩按班级分别管理。</p></div
        >
        <button
          v-if="activeTab === 'content'"
          class="workspace-button primary"
          @click="selectTab('materials')"
          >导入与批改</button
        >
      </header>
      <nav class="detail-tabs" aria-label="作业详情导航">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          :class="{ active: activeTab === tab.key }"
          :aria-current="activeTab === tab.key ? 'page' : undefined"
          @click="selectTab(tab.key)"
          >{{ tab.label }}</button
        >
      </nav>
      <section v-if="activeTab === 'content'" class="assignment-content-grid">
        <div class="surface content-section">
          <div class="section-heading"
            ><h2>题目要求</h2
            ><button
              v-if="assignment.portal_homework_id"
              class="text-action"
              :disabled="reading"
              @click="readRequirements"
              >{{ reading ? '正在提交…' : '重新同步要求' }}</button
            ></div
          >
          <p v-if="assignment.portal_homework_id" class="requirement-state" role="status">{{
            requirementLabel
          }}</p>
          <HomeworkRequirementReview
            v-if="assignment.portal_homework_id"
            :key="assignment.portal_homework_id"
            :homework-id="assignment.portal_homework_id"
            :requirements-status="assignment.requirements_status"
            :fallback-question="assignment.question"
            @confirmed="
              assignment.question = $event;
              assignment.requirements_status = 'READY';
            "
          />
          <p v-else class="full-question">{{ assignment.question || '等待读取题目要求…' }}</p>
          <RpaTaskPanel v-if="requirementTask" :task-id="requirementTask" embedded />
          <p v-if="requirementError" class="workspace-error" role="alert">{{ requirementError }}</p>
        </div>
        <aside class="surface content-section rubric-section">
          <div class="section-heading"
            ><h2>评分标准</h2
            ><button v-if="!editingRubric" class="text-action" @click="editRubric"
              >编辑标准</button
            ></div
          >
          <form v-if="editingRubric" @submit.prevent="saveRubric">
            <RubricEditor v-model="rubricDraft" :disabled="savingRubric" />
            <p v-if="rubricError" class="workspace-error" role="alert">{{ rubricError }}</p>
            <div class="form-actions"
              ><button class="workspace-button primary" :disabled="savingRubric || !rubricDraft">{{
                savingRubric ? '保存中…' : '保存评分标准'
              }}</button
              ><button
                type="button"
                class="workspace-button"
                :disabled="savingRubric"
                @click="editingRubric = false"
                >取消</button
              ></div
            >
          </form>
          <RubricSummary v-else :rubric="assignment.rubric" />
          <p class="section-note">所有班级共用此标准。修改用于后续批改，已有成绩不会自动重算。</p>
          <p v-if="rubricNotice" class="inline-notice" role="status">{{ rubricNotice }}</p>
        </aside>
      </section>
      <div v-else class="class-workspace">
        <AssignmentClassNavigator
          :classes="classes"
          :model-value="selectedClass"
          :loading="classesLoading"
          :error="classesError"
          @update:model-value="selectClass"
          @retry="loadClasses"
        />
        <div class="class-content">
          <header class="class-content-heading"
            ><div
              ><h2>{{ currentClass?.name || '选择班级' }}</h2
              ><p>{{
                activeTab === 'materials'
                  ? '本班的材料、导入记录与批改进度'
                  : '仅展示本班学生成绩与复核记录'
              }}</p></div
            ><button
              v-if="activeTab === 'results'"
              class="workspace-button"
              :disabled="!selectedClass || resultsLoading"
              @click="fetchResults"
              >刷新结果</button
            ></header
          >
          <p v-if="!selectedClass" class="workspace-empty">请先选择一个班级。</p>
          <section v-else-if="activeTab === 'materials'" class="surface material-section">
            <div class="section-heading"
              ><div
                ><h2>导入学生作业</h2
                ><p class="section-note">导入到：{{ currentClass?.name }}</p></div
              ></div
            >
            <nav class="source-switch" aria-label="材料来源">
              <button
                v-for="item in availableSources"
                :key="item.value"
                :aria-pressed="materialSource === item.value"
                :class="{ active: materialSource === item.value }"
                @click="materialSource = item.value"
                >{{ item.label }}</button
              >
            </nav>
            <AssignmentPortalPanel
              v-if="assignment.portal_homework_id"
              v-show="materialSource === 'portal'"
              :assignment-id="id"
              :rubric="assignment.rubric"
              :class-id="selectedClass"
              @context="updatePortalContext"
              @results="openWorkflowResults"
            />
            <form
              v-if="materialSource === 'upload' && canWriteClass"
              class="upload-form"
              @submit.prevent="submitBatchFile"
            >
              <label for="file-upload">学生作业压缩包 <span>ZIP / RAR / 7z</span></label>
              <input
                id="file-upload"
                type="file"
                accept=".zip,.rar,.7z"
                required
                :disabled="submitting"
                @change="handleFileChange"
              />
              <p v-if="fileName" class="section-note">已选择：{{ fileName }}</p>
              <label class="check"
                ><input
                  v-model="autoGrade"
                  type="checkbox"
                  :disabled="submitting"
                />材料整理完成后自动批改</label
              >
              <button class="workspace-button primary" :disabled="submitting">{{
                submitting ? '正在提交…' : autoGrade ? '上传并自动批改' : '上传并整理材料'
              }}</button>
            </form>
            <p v-if="materialSource !== 'portal' && !canWriteClass" class="section-note"
              >此班级为历史记录，不能导入到已失效的班级。仍可查看已有材料、批改结果与复核成绩。</p
            >
            <p
              v-if="submissionMessage"
              :class="submissionError ? 'workspace-error' : 'inline-notice'"
              role="status"
              >{{ submissionMessage }}</p
            >
            <MaterialBatchPanel
              :key="id + ':' + selectedClass"
              :assignment-id="id"
              :class-id="selectedClass"
              :import-class-id="writeClassId"
              :show-directory="materialSource === 'directory' && canWriteClass"
              :refresh-key="materialsVersion"
              @results="showBatchResults"
            />
          </section>
          <section v-else class="surface results-section">
            <div class="results-toolbar">
              <label class="student-search"
                ><AppIcon name="search" /><input
                  v-model="studentSearch"
                  aria-label="搜索本班学生"
                  placeholder="搜索姓名或学号"
              /></label>
              <select v-model="reviewFilter" aria-label="复核状态"
                ><option value="">全部复核状态</option
                ><option value="pending">未复核</option
                ><option value="reviewed">已复核</option></select
              >
              <select v-model="resultSort" aria-label="成绩排序"
                ><option value="review">未复核优先</option
                ><option value="score_desc">分数从高到低</option
                ><option value="score_asc">分数从低到高</option></select
              >
              <button
                class="workspace-button primary"
                :disabled="exporting || !results.length"
                @click="exportResults(false)"
                >{{
                  exporting ? '导出中…' : resultsBatchId ? '导出本班本批成绩' : '导出本班成绩'
                }}</button
              >
              <details class="results-more"
                ><summary class="workspace-button">更多</summary
                ><div class="result-menu"
                  ><button :disabled="exporting" @click="exportResults(true)"
                    >导出各班成绩（分表）</button
                  ><button class="danger-action" @click="deleteAllResults"
                    >清空所有班级结果</button
                  ></div
                ></details
              >
              <button class="workspace-button" @click="selectTab('materials')"
                >使用已有材料批改</button
              >
              <button
                class="workspace-button danger-action"
                :disabled="deletingResults || !selectedClass || resultsLoading"
                @click="deleteClassResults"
                >{{ deletingResults ? '正在删除…' : '删除当前班级作业' }}</button
              >
            </div>
            <p class="section-note"
              >当前班级：{{
                currentClass?.name
              }}。导出按班级与导入记录，不受姓名、复核及排序筛选影响。</p
            >
            <div v-if="resultsBatchId" class="batch-filter"
              >正在查看一条导入记录
              <button class="text-action" @click="showBatchResults('')"
                >查看本班全部记录</button
              ></div
            >
            <p v-if="resultsError" class="workspace-error" role="alert"
              >{{ resultsError }} <button @click="fetchResults">重新加载</button></p
            >
            <p v-if="resultsNotice" class="inline-notice" role="status">{{ resultsNotice }}</p>
            <div v-if="resultsLoading && !results.length" class="workspace-empty"
              ><Loader /><p>正在读取本班成绩…</p></div
            >
            <template v-else>
              <div class="results-caption"
                ><span>共 {{ results.length }} 条成绩</span
                ><span>已复核 {{ reviewedCount }} 条</span
                ><span>未复核 {{ results.length - reviewedCount }} 条</span
                ><span v-if="studentSearch || reviewFilter"
                  >筛选后 {{ filteredResults.length }} 条</span
                ><span v-if="resultsLoading" role="status">正在刷新，当前数据仍可查看</span
                ><button
                  v-if="studentSearch || reviewFilter"
                  class="text-action"
                  @click="clearResultFilters"
                  >清除筛选</button
                ></div
              >
              <div v-if="filteredResults.length" class="result-table-scroll">
                <table class="results-table"
                  ><thead
                    ><tr
                      ><th>学生</th><th>最终分数</th><th>复核状态</th><th>检测提示</th
                      ><th>操作</th></tr
                    ></thead
                  >
                  <tbody
                    ><tr v-for="result in filteredResults" :key="result.id">
                      <td
                        ><strong>{{ result.student_name || result.student_id }}</strong
                        ><small>{{ result.student_id }}</small></td
                      >
                      <td
                        ><span class="score" :class="finalScore(result) < 0 ? 'score-error' : ''">{{
                          finalScore(result) < 0
                            ? result.reading_status === 'NEEDS_REVIEW'
                              ? '待复核'
                              : '评分失败'
                            : Number(finalScore(result)).toFixed(1)
                        }}</span></td
                      >
                      <td
                        ><span
                          class="review-state"
                          :class="{ reviewed: result.is_human_reviewed }"
                          >{{ result.is_human_reviewed ? '已复核' : '未复核' }}</span
                        ></td
                      >
                      <td
                        ><button class="risk-link" @click="showReport(result, 'plagiarism')">{{
                          plagiarismLabel(result)
                        }}</button
                        ><small
                          >{{ aigcLabel(result)
                          }}<span v-if="result.code_doc_match_report"
                            >；代码文档匹配 {{ result.code_doc_match_report.score }} 分</span
                          ></small
                        ></td
                      >
                      <td
                        ><button class="text-action" @click="openReview(result)">查看并复核</button
                        ><details class="row-more"
                          ><summary aria-label="更多学生操作">更多</summary
                          ><div
                            ><button @click="showReport(result, 'feedback')">查看完整评语</button
                            ><button class="danger-action" @click="deleteSingleResult(result)"
                              >删除此条成绩</button
                            ></div
                          ></details
                        ></td
                      >
                    </tr></tbody
                  >
                </table>
              </div>
              <div v-if="filteredResults.length" class="result-card-list">
                <article v-for="result in filteredResults" :key="result.id" class="result-card">
                  <header>
                    <div
                      ><strong>{{ result.student_name || result.student_id }}</strong
                      ><small>{{ result.student_id }}</small></div
                    >
                    <span class="score" :class="finalScore(result) < 0 ? 'score-error' : ''">{{
                      finalScore(result) < 0
                        ? result.reading_status === 'NEEDS_REVIEW'
                          ? '待复核'
                          : '评分失败'
                        : Number(finalScore(result)).toFixed(1)
                    }}</span>
                  </header>
                  <div class="result-card-state">
                    <span class="review-state" :class="{ reviewed: result.is_human_reviewed }">{{
                      result.is_human_reviewed ? '已复核' : '未复核'
                    }}</span>
                    <button class="risk-link" @click="showReport(result, 'plagiarism')">{{
                      plagiarismLabel(result)
                    }}</button>
                  </div>
                  <p
                    >{{ aigcLabel(result)
                    }}<span v-if="result.code_doc_match_report"
                      >；代码文档匹配 {{ result.code_doc_match_report.score }} 分</span
                    ></p
                  >
                  <footer>
                    <button class="text-action" @click="showReport(result, 'feedback')"
                      >查看完整评语</button
                    >
                    <button class="workspace-button primary" @click="openReview(result)"
                      >查看并复核</button
                    >
                  </footer>
                </article>
              </div>
              <div v-else class="workspace-empty"
                ><AppIcon name="file" /><h3>{{
                  results.length ? '没有符合条件的学生' : '本班暂无成绩'
                }}</h3
                ><p>{{
                  results.length
                    ? '调整搜索或复核筛选条件。'
                    : '已有材料可直接开始或重新批改，也可以导入新的学生作业。'
                }}</p
                ><button
                  v-if="!results.length"
                  class="workspace-button"
                  @click="selectTab('materials')"
                  >前往导入与批改</button
                ></div
              >
            </template>
          </section>
        </div>
      </div>
    </template>
    <ReportModal
      v-if="reportSubmission"
      :submission="reportSubmission"
      :report-type="reportType"
      @close="reportSubmission = null"
    />
    <ReviewModal
      v-if="reviewSubmission"
      :key="reviewSubmission.id"
      :submission="reviewSubmission"
      :saving="savingReview"
      :error="reviewError"
      :max-score="maxReviewScore"
      :position="reviewPosition"
      :total="filteredResults.length"
      :can-previous="reviewPosition > 1"
      :can-next="reviewPosition < filteredResults.length"
      :has-next-unreviewed="Boolean(nextUnreviewed)"
      @close="reviewSubmission = null"
      @save="saveReview"
      @navigate="navigateReview"
    />
  </div>
</template>
<script setup>
import { ref, computed, watch, onBeforeUnmount } from 'vue';
import { useRoute, useRouter, onBeforeRouteLeave, onBeforeRouteUpdate } from 'vue-router';
import gradingApi from '../services/gradingApi';
import authApi from '../services/authApi';
import { createPortalApi } from '../services/portalApi';
import {
  chooseClass,
  finalScore,
  filterClassResults,
  canImportClass,
} from '../services/classWorkspace';
import Loader from '../components/Loader.vue';
import AppIcon from '../components/AppIcon.vue';
import AssignmentClassNavigator from '../components/AssignmentClassNavigator.vue';
import AssignmentPortalPanel from '../components/AssignmentPortalPanel.vue';
import MaterialBatchPanel from '../components/MaterialBatchPanel.vue';
import HomeworkRequirementReview from '../components/HomeworkRequirementReview.vue';
import RubricEditor from '../components/RubricEditor.vue';
import RubricSummary from '../components/RubricSummary.vue';
import RpaTaskPanel from '../components/RpaTaskPanel.vue';
import ReviewModal from '../components/ReviewModal.vue';
import ReportModal from '../components/ReportModal.vue';
import { rubricRows } from '../services/assignmentWorkspace';
const props = defineProps({ id: { type: String, required: true } });
const route = useRoute(),
  router = useRouter(),
  portalApi = createPortalApi(authApi.getClient());
const tabs = [
  { key: 'content', label: '题目与评分标准' },
  { key: 'materials', label: '学生作业与批改' },
  { key: 'results', label: '成绩与复核' },
];
const activeTab = computed(() =>
  tabs.some((item) => item.key === route.query.tab) ? route.query.tab : 'content'
);
const assignment = ref(null),
  loading = ref(true),
  error = ref('');
const classes = ref([]),
  selectedClass = ref(''),
  classesLoading = ref(false),
  classesError = ref('');
const currentClass = computed(() => classes.value.find((item) => item.id === selectedClass.value));
const writeClassId = computed(() => (currentClass.value?.can_import ? selectedClass.value : ''));
const canWriteClass = computed(() => canImportClass(assignment.value, currentClass.value));
const materialSource = ref('upload'),
  autoGrade = ref(false),
  file = ref(null),
  fileName = ref(''),
  submitting = ref(false),
  materialsVersion = ref(0),
  submissionMessage = ref(''),
  submissionError = ref(false);
const availableSources = computed(() => [
  ...(assignment.value?.portal_homework_id ? [{ value: 'portal', label: '官网获取' }] : []),
  { value: 'upload', label: '上传压缩包' },
  { value: 'directory', label: '读取目录' },
]);
const editingRubric = ref(false),
  rubricDraft = ref(''),
  savingRubric = ref(false),
  rubricError = ref(''),
  rubricNotice = ref('');
const reading = ref(false),
  requirementTask = ref(''),
  requirementError = ref('');
const requirementLabel = computed(
  () =>
    ({
      READY: '官网要求已就绪',
      PENDING: '等待读取官网要求',
      READING: '正在读取官网要求',
      REVIEW_REQUIRED: '部分要求需要核对',
      FAILED: '要求读取失败，请重试',
      CANCELLED: '要求读取已取消',
      INTERRUPTED: '要求读取已中断',
    })[assignment.value?.requirements_status] || '正在检查官网要求'
);
const results = ref([]),
  resultsLoading = ref(false),
  resultsError = ref(''),
  resultsNotice = ref(''),
  deletingResults = ref(false),
  studentSearch = ref(''),
  reviewFilter = ref(''),
  resultSort = ref('review'),
  exporting = ref(false);
const resultsBatchId = computed(() =>
  typeof route.query.batch_id === 'string' ? route.query.batch_id : ''
);
const filteredResults = computed(() =>
  filterClassResults(results.value, studentSearch.value, reviewFilter.value, resultSort.value)
);
const reviewedCount = computed(() => results.value.filter((item) => item.is_human_reviewed).length);
const maxReviewScore = computed(() => {
  const total = rubricRows(assignment.value?.rubric).reduce(
    (sum, row) => sum + (Number.isFinite(row.score) ? row.score : 0),
    0
  );
  return total > 0 ? Math.round(total * 10) / 10 : 100;
});
const reportSubmission = ref(null),
  reportType = ref('feedback'),
  reviewSubmission = ref(null),
  savingReview = ref(false),
  reviewError = ref('');
const reviewPosition = computed(() => {
  const index = filteredResults.value.findIndex((item) => item.id === reviewSubmission.value?.id);
  return index < 0 ? 0 : index + 1;
});
const nextUnreviewed = computed(() => {
  if (!reviewSubmission.value) return null;
  const rows = filteredResults.value;
  const index = rows.findIndex((item) => item.id === reviewSubmission.value.id);
  const candidates = index < 0 ? rows : [...rows.slice(index + 1), ...rows.slice(0, index)];
  return (
    candidates.find((item) => item.id !== reviewSubmission.value.id && !item.is_human_reviewed) ||
    null
  );
});
let assignmentEpoch = 0,
  resultEpoch = 0,
  disposed = false,
  timer;
const returnLocation = computed(() =>
  typeof route.query.return_to === 'string' && /^\/assignments(?:\?|$)/.test(route.query.return_to)
    ? route.query.return_to
    : '/assignments'
);
const memoryKey = () =>
  'teacher-class:' +
  (authApi.getCurrentUser()?.user_id || authApi.getCurrentUser()?.username || '') +
  ':' +
  props.id;
const selectTab = (tab) => router.replace({ query: { ...route.query, tab } });
function clearResultFilters() {
  studentSearch.value = '';
  reviewFilter.value = '';
}
async function selectClass(id) {
  if (!classes.value.some((item) => item.id === id) || id === selectedClass.value) return;
  if ((file.value || submitting.value) && !confirm('切换班级会清除未提交的文件选择。继续？'))
    return;
  file.value = null;
  fileName.value = '';
  submissionMessage.value = '';
  resultsNotice.value = '';
  results.value = [];
  reviewSubmission.value = null;
  clearResultFilters();
  resultEpoch++;
  selectedClass.value = id;
  try {
    localStorage.setItem(memoryKey(), id);
  } catch {}
  await router.replace({ query: { ...route.query, class_id: id, batch_id: undefined } });
}
async function loadClasses() {
  const id = props.id,
    epoch = assignmentEpoch;
  classesLoading.value = true;
  classesError.value = '';
  try {
    const { data } = await gradingApi.getAssignmentClasses(id);
    if (disposed || id !== props.id || epoch !== assignmentEpoch) return;
    classes.value = Array.isArray(data) ? data : [];
    let remembered = '';
    try {
      remembered = localStorage.getItem(memoryKey());
    } catch {}
    selectedClass.value = chooseClass(classes.value, route.query.class_id, remembered);
    if (selectedClass.value && route.query.class_id !== selectedClass.value)
      await router.replace({ query: { ...route.query, class_id: selectedClass.value } });
  } catch {
    if (epoch === assignmentEpoch) classesError.value = '班级读取失败，请重试。';
  } finally {
    if (epoch === assignmentEpoch) classesLoading.value = false;
  }
}
async function loadAssignment() {
  const epoch = ++assignmentEpoch,
    id = props.id;
  loading.value = true;
  error.value = '';
  assignment.value = null;
  classes.value = [];
  selectedClass.value = '';
  results.value = [];
  resultEpoch++;
  editingRubric.value = false;
  rubricNotice.value = '';
  requirementTask.value = '';
  file.value = null;
  fileName.value = '';
  reportSubmission.value = null;
  reviewSubmission.value = null;
  try {
    const { data } = await gradingApi.getAssignment(id);
    if (disposed || epoch !== assignmentEpoch) return;
    assignment.value = data;
    materialSource.value = data.portal_homework_id ? 'portal' : 'upload';
    await loadClasses();
    if (data.portal_homework_id) await refreshPortal();
  } catch {
    if (epoch === assignmentEpoch) error.value = '无法加载作业详情，请检查连接。';
  } finally {
    if (epoch === assignmentEpoch) loading.value = false;
  }
}
function updatePortalContext(data) {
  if (assignment.value && data.available) {
    assignment.value.question = data.question || assignment.value.question;
    assignment.value.requirements_status = data.requirements_status;
  }
}
async function refreshPortal() {
  const id = props.id;
  try {
    const { data } = await portalApi.assignment(id);
    if (!disposed && props.id === id) updatePortalContext(data);
  } catch {
    /* Main content and class list remain usable if the portal is unavailable. */
  }
}
async function readRequirements() {
  reading.value = true;
  requirementError.value = '';
  try {
    const { data } = await portalApi.assignment(props.id);
    if (!data.available) throw new Error('官网目录已变化，请先更新课程与作业。');
    const response = await portalApi.readHomeworkDetail(data.homework.id, data.snapshot_id);
    requirementTask.value = response.data.operation_id;
    assignment.value.requirements_status = 'READING';
  } catch (e) {
    requirementError.value = e.response?.data?.detail || e.message || '读取失败';
  } finally {
    reading.value = false;
  }
}
function editRubric() {
  rubricDraft.value = JSON.stringify(assignment.value.rubric);
  rubricError.value = '';
  editingRubric.value = true;
}
async function saveRubric() {
  const assignmentId = props.id;
  savingRubric.value = true;
  rubricError.value = '';
  try {
    const rubric = JSON.parse(rubricDraft.value);
    await portalApi.saveRubric(assignmentId, rubric);
    if (disposed || props.id !== assignmentId) return;
    assignment.value.rubric = rubric;
    editingRubric.value = false;
    rubricNotice.value = '评分标准已保存，后续批改使用此标准。';
  } catch (e) {
    rubricError.value = e.response?.data?.detail || '评分标准无效或保存失败，请检查后重试。';
  } finally {
    savingRubric.value = false;
  }
}
async function fetchResults() {
  if (!selectedClass.value || disposed) return;
  const epoch = ++resultEpoch,
    classId = selectedClass.value,
    assignmentId = props.id,
    batchId = resultsBatchId.value;
  const hadResults = results.value.length > 0;
  resultsLoading.value = true;
  resultsError.value = '';
  try {
    const { data } = await gradingApi.getResultsForAssignment(assignmentId, batchId, classId);
    if (
      !disposed &&
      epoch === resultEpoch &&
      classId === selectedClass.value &&
      assignmentId === props.id
    )
      results.value = Array.isArray(data) ? data : [];
  } catch {
    if (epoch === resultEpoch)
      resultsError.value = hadResults
        ? '刷新失败，当前显示的是上次读取的本班成绩。'
        : '本班成绩暂时无法读取。';
  } finally {
    if (epoch === resultEpoch) resultsLoading.value = false;
  }
}
async function showBatchResults(batchId) {
  await router.replace({
    query: { ...route.query, tab: 'results', batch_id: batchId || undefined },
  });
}
async function openWorkflowResults(classId) {
  if (classId && classes.value.some((item) => item.id === classId)) await selectClass(classId);
  await showBatchResults('');
}
function handleFileChange(event) {
  file.value = event.target.files?.[0] || null;
  fileName.value = file.value?.name || '';
}
async function submitBatchFile() {
  if (!file.value || submitting.value || !canWriteClass.value) return;
  const id = props.id,
    classId = selectedClass.value;
  submitting.value = true;
  submissionMessage.value = '';
  submissionError.value = false;
  const body = new FormData();
  body.append('batch_file', file.value);
  body.append('auto_grade', String(autoGrade.value));
  if (writeClassId.value) body.append('class_id', writeClassId.value);
  try {
    const { data } = await gradingApi.submitBatch(id, body);
    if (props.id === id && selectedClass.value === classId) {
      submissionMessage.value = data.message || '材料已受理';
      materialsVersion.value++;
      file.value = null;
      fileName.value = '';
      const input = document.getElementById('file-upload');
      if (input) input.value = '';
    }
  } catch (e) {
    if (props.id === id && selectedClass.value === classId) {
      submissionError.value = true;
      submissionMessage.value = e.response?.data?.detail || '上传失败，请重试。';
    }
  } finally {
    submitting.value = false;
  }
}
async function exportResults(allClasses) {
  if (exporting.value) return;
  exporting.value = true;
  try {
    const response = await gradingApi.exportAssignment(
      props.id,
      allClasses ? '' : resultsBatchId.value,
      allClasses ? '' : selectedClass.value
    );
    const url = URL.createObjectURL(
      new Blob([response.data], { type: response.headers['content-type'] })
    );
    const link = document.createElement('a');
    const disposition = response.headers['content-disposition'] || '';
    const match = disposition.match(/filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i);
    link.download = match
      ? decodeURIComponent(match[1] || match[2])
      : assignment.value.task_name +
        '_' +
        (allClasses ? '各班成绩' : currentClass.value.name) +
        '.xlsx';
    link.href = url;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch {
    resultsError.value = '导出失败，请确认所选范围已有成绩后重试。';
  } finally {
    exporting.value = false;
  }
}
async function deleteSingleResult(result) {
  if (!confirm('确定删除学生 ' + result.student_id + ' 的这条成绩？')) return;
  try {
    await gradingApi.deleteSubmission(result.id);
    await fetchResults();
  } catch {
    resultsError.value = '删除失败，请重试。';
  }
}
async function deleteAllResults() {
  if (deletingResults.value) return;
  if (
    !confirm(
      '将清空“' +
        assignment.value.task_name +
        '”所有班级的评分记录，包括当前未显示的班级。此操作无法撤销，是否继续？'
    )
  )
    return;
  deletingResults.value = true;
  try {
    await gradingApi.deleteAllSubmissions(props.id);
    await fetchResults();
    await loadClasses();
  } catch {
    resultsError.value = '清空失败，请重试。';
  } finally {
    deletingResults.value = false;
  }
}
async function deleteClassResults() {
  if (deletingResults.value || !selectedClass.value || !currentClass.value) return;
  const id = props.id,
    classId = selectedClass.value,
    name = currentClass.value.name;
  if (
    !confirm(
      '确定删除“' +
        name +
        '”在“' +
        assignment.value.task_name +
        '”下全部批次的评分和人工复核记录？原始材料保留，可重新批改。此操作无法撤销。'
    )
  )
    return;
  deletingResults.value = true;
  resultsError.value = '';
  resultsNotice.value = '';
  try {
    await gradingApi.deleteClassSubmissions(id, classId);
    if (props.id === id && selectedClass.value === classId) {
      resultsNotice.value =
        '已删除当前班级的评分记录，原始材料保留。可点击“使用已有材料批改”重新批改。';
      await fetchResults();
      await loadClasses();
      materialsVersion.value++;
    }
  } catch (e) {
    if (props.id === id && selectedClass.value === classId)
      resultsError.value = e.response?.data?.detail || '删除本班作业失败，请重试。';
  } finally {
    deletingResults.value = false;
  }
}
function showReport(result, type) {
  reportSubmission.value = result;
  reportType.value = type;
}
function openReview(result) {
  reviewSubmission.value = result;
  reviewError.value = '';
}
function navigateReview(offset) {
  const index = filteredResults.value.findIndex((item) => item.id === reviewSubmission.value?.id);
  const target = filteredResults.value[index + offset];
  if (target) openReview(target);
}
async function saveReview(data) {
  if (savingReview.value) return;
  savingReview.value = true;
  reviewError.value = '';
  try {
    const { data: saved } = await gradingApi.updateSubmission(data.id, {
      human_score: data.score,
      human_feedback: data.human_feedback,
      is_human_reviewed: true,
    });
    const index = results.value.findIndex((item) => item.id === data.id);
    if (index >= 0) results.value[index] = saved;
    const next = data.continueNext ? nextUnreviewed.value : null;
    reviewSubmission.value = null;
    if (next) openReview(next);
  } catch (e) {
    reviewError.value = e.response?.data?.detail || '保存失败，输入已保留，请重试。';
  } finally {
    savingReview.value = false;
  }
}
function plagiarismLabel(result) {
  if (!result.plagiarism_reports?.length) return '暂无查重提示';
  const max = Math.max(
    ...result.plagiarism_reports.map((item) => item.llm_analysis?.similarity_score || 0)
  );
  return max >= 85 ? '查重需关注 (' + max + '分)' : '查看查重报告';
}
function aigcLabel(result) {
  return Number.isFinite(result.aigc_report?.ai_probability)
    ? 'AI 生成概率 ' + (result.aigc_report.ai_probability * 100).toFixed(1) + '%'
    : 'AIGC 未检测';
}
watch(() => props.id, loadAssignment, { immediate: true });
watch(
  () => route.query.class_id,
  (value) => {
    if (classes.value.some((item) => item.id === value)) {
      if (selectedClass.value !== value) {
        resultEpoch++;
        results.value = [];
      }
      selectedClass.value = value;
    }
  }
);
watch([activeTab, selectedClass, resultsBatchId], () => {
  if (activeTab.value === 'results' && selectedClass.value) fetchResults();
});
function confirmLeave() {
  if (savingRubric.value || submitting.value) return confirm('提交仍在进行中，确定离开当前作业？');
  return (
    !editingRubric.value ||
    rubricDraft.value === JSON.stringify(assignment.value?.rubric) ||
    confirm('评分标准尚未保存，确定离开？')
  );
}
onBeforeRouteLeave(confirmLeave);
onBeforeRouteUpdate((to, from) => to.params.id === from.params.id || confirmLeave());
timer = setInterval(() => {
  if (
    !document.hidden &&
    assignment.value?.portal_homework_id &&
    assignment.value.requirements_status !== 'READY'
  )
    refreshPortal();
}, 5000);
onBeforeUnmount(() => {
  disposed = true;
  assignmentEpoch++;
  resultEpoch++;
  clearInterval(timer);
});
</script>
<style scoped>
.detail-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 28px;
}
.detail-course {
  color: var(--color-muted);
  font-size: 13px;
  margin-bottom: 8px;
}
.source-label {
  margin-left: 12px;
  color: var(--color-accent);
  background: var(--color-accent-soft);
  padding: 3px 8px;
  border-radius: 5px;
  font-size: 12px;
}
.detail-tabs {
  display: flex;
  gap: 28px;
  border-bottom: 1px solid var(--color-line);
  margin-bottom: 28px;
  overflow-x: auto;
}
.detail-tabs button {
  flex-shrink: 0;
  padding: 0 0 16px;
  border-bottom: 2px solid transparent;
  color: var(--color-muted);
  font-size: 14px;
}
.detail-tabs .active {
  color: var(--color-accent);
  border-color: var(--color-accent);
  font-weight: 600;
}
.assignment-content-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(320px, 1fr);
  gap: 24px;
  align-items: start;
}
.content-section {
  padding: 28px;
  min-width: 0;
}
.section-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 22px;
}
.section-heading h2 {
  font-size: 16px;
  font-weight: 600;
}
.full-question {
  white-space: pre-wrap;
  line-height: 1.9;
  font-size: 16px;
  color: var(--color-ink);
  overflow-wrap: break-word;
  min-height: 240px;
}
.full-question.requirement-content {
  white-space: normal;
}
.section-note,
.requirement-state {
  color: var(--color-muted);
  font-size: 13px;
  line-height: 1.8;
  margin: 12px 0;
}
.requirement-state {
  color: var(--color-accent);
}
.form-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}
.inline-notice {
  background: var(--color-accent-soft);
  color: var(--color-accent);
  padding: 12px 16px;
  margin: 16px 0;
  border-radius: 8px;
  font-size: 13px;
}
.class-workspace {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  gap: 26px;
  align-items: start;
}
.class-content {
  min-width: 0;
}
.class-content-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 4px 0 22px;
}
.class-content-heading h2 {
  font-size: 20px;
  font-weight: 600;
}
.class-content-heading p {
  margin-top: 6px;
  font-size: 13px;
  color: var(--color-muted);
}
.material-section,
.results-section {
  padding: 24px;
  min-width: 0;
}
.source-switch {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 4px;
  padding: 4px;
  background: var(--color-canvas);
  border-radius: 9px;
  margin-bottom: 24px;
}
.source-switch button {
  padding: 8px 16px;
  font-size: 13px;
  border-radius: 6px;
  color: var(--color-muted);
}
.source-switch .active {
  color: var(--color-accent);
  background: var(--color-paper);
  font-weight: 600;
}
.upload-form {
  display: grid;
  gap: 18px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--color-line);
}
.upload-form > label:first-child {
  font-size: 14px;
  font-weight: 500;
}
.upload-form > label span {
  color: var(--color-muted);
  margin-left: 10px;
  font-size: 12px;
}
.upload-form input[type='file'] {
  padding: 20px;
  width: 100%;
  border: 1px dashed var(--color-line);
  background: var(--color-canvas);
  border-radius: 8px;
}
.upload-form button {
  justify-self: start;
}
.check {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}
.results-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}
.student-search {
  display: flex;
  align-items: center;
  flex: 1;
  min-width: 160px;
  position: relative;
}
.student-search svg {
  position: absolute;
  left: 12px;
  width: 16px;
  color: var(--color-muted);
}
.student-search input {
  width: 100%;
  padding: 9px 12px 9px 36px;
  border: 1px solid var(--color-line);
}
.results-toolbar select {
  padding: 8px;
  font-size: 13px;
}
.results-more {
  position: relative;
}
.results-more summary {
  list-style: none;
}
.result-menu,
.row-more > div {
  display: grid;
  gap: 8px;
  position: absolute;
  right: 0;
  z-index: 5;
  width: 210px;
  padding: 12px;
  background: var(--color-paper);
  border: 1px solid var(--color-line);
  border-radius: 9px;
  box-shadow: 0 12px 30px var(--color-scrim);
}
.result-menu button,
.row-more button {
  text-align: left;
  padding: 8px;
  font-size: 13px;
}
.results-caption {
  display: flex;
  gap: 20px;
  margin: 20px 0 12px;
  color: var(--color-muted);
  font-size: 12px;
}
.result-table-scroll {
  overflow-x: auto;
}
.results-table {
  width: 100%;
  text-align: left;
  font-size: 13px;
}
.results-table th {
  padding: 12px;
  background: var(--color-canvas);
  color: var(--color-muted);
  font-size: 12px;
  font-weight: 500;
}
.results-table td {
  padding: 18px 12px;
  border-bottom: 1px solid var(--color-line);
}
.results-table strong {
  font-weight: 600;
}
.results-table small {
  display: block;
  margin-top: 5px;
  color: var(--color-muted);
  font-size: 12px;
  max-width: 260px;
}
.results-table td:last-child {
  white-space: nowrap;
}
.score {
  font-size: 18px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.score-error,
.danger-action {
  color: var(--color-danger);
}
.review-state {
  color: var(--color-muted);
  font-size: 12px;
}
.review-state.reviewed {
  color: var(--color-success);
}
.risk-link {
  text-align: left;
  color: var(--color-accent);
}
.row-more {
  display: inline-block;
  margin-left: 14px;
  position: relative;
  font-size: 12px;
  color: var(--color-muted);
}
.row-more > div {
  position: static;
  box-shadow: none;
  margin-top: var(--space-2);
}
.batch-filter {
  display: flex;
  align-items: center;
  gap: 16px;
  color: var(--color-accent);
  font-size: 13px;
  padding: 12px 0;
}
.result-card-list {
  display: none;
}
.workspace-empty {
  display: grid;
  justify-items: center;
  gap: 12px;
  padding: 32px 16px;
  text-align: center;
  color: var(--color-muted);
}
.workspace-empty > svg {
  width: 36px;
  height: 36px;
}
.workspace-empty h3 {
  font-size: 15px;
  color: var(--color-ink);
}
.result-card {
  padding: 16px;
  border: 1px solid var(--color-line);
  border-radius: 10px;
  background: var(--color-paper);
}
.result-card > header,
.result-card-state,
.result-card > footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.result-card strong {
  font-size: 14px;
}
.result-card small,
.result-card > p {
  display: block;
  margin-top: 4px;
  color: var(--color-muted);
  font-size: 12px;
  line-height: 1.7;
}
.result-card-state {
  justify-content: flex-start;
  margin-top: 12px;
  font-size: 13px;
}
.result-card > footer {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--color-line);
}
@media (max-width: 1100px) {
  .assignment-content-grid {
    grid-template-columns: 1fr;
  }
  .class-workspace {
    gap: 18px;
    grid-template-columns: 150px minmax(0, 1fr);
  }
}
@media (max-width: 1000px) {
  .class-workspace {
    grid-template-columns: minmax(0, 1fr);
  }
}
@media (max-width: 760px) {
  .detail-heading {
    align-items: flex-start;
    flex-direction: column;
    margin-bottom: 18px;
  }
  .detail-heading .page-description {
    display: none;
  }
  .detail-tabs {
    gap: 20px;
    margin-bottom: 20px;
  }
  .detail-tabs button {
    font-size: 13px;
  }
  .class-content-heading {
    padding-bottom: 16px;
  }
  .content-section,
  .material-section,
  .results-section {
    padding: 16px;
  }
  .results-toolbar {
    align-items: stretch;
    gap: 8px;
  }
  .results-toolbar select {
    flex: 1;
    min-width: 120px;
  }
  .student-search {
    flex-basis: 100%;
  }
  .results-caption {
    flex-wrap: wrap;
    gap: 6px 14px;
  }
  .result-table-scroll {
    display: none;
  }
  .result-card-list {
    display: grid;
    gap: 12px;
  }
}
</style>
