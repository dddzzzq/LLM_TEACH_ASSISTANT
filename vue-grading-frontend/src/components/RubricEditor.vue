<template>
  <fieldset class="rubric-editor" :disabled="disabled">
    <legend>评分标准</legend>
    <div class="template-picker">
      <div class="template-actions">
        <label>
          评分模板
          <select v-model="selectedTemplateId" aria-label="评分模板" @change="applyTemplate">
            <option value="" disabled>选择评分模板</option>
            <option v-for="item in rubricTemplates" :key="item.id" :value="item.id">{{
              item.name
            }}</option>
          </select>
        </label>
        <button type="button" :disabled="!selectedTemplate" @click="applyTemplate"
          >恢复模板初始标准</button
        >
      </div>
      <template v-if="selectedTemplate">
        <p class="template-description">{{ selectedTemplate.description }}</p>
        <ul class="template-preview" aria-label="模板评分项预览">
          <li v-for="item in selectedTemplate.criteria" :key="item.name"
            >{{ item.name }} {{ item.score }} 分</li
          >
        </ul>
        <p class="rubric-hint"
          >模板合计 100 分。切换模板会填入对应的初始评分标准，可继续修改，保存后生效。</p
        >
      </template>
      <p v-else class="rubric-hint">选择模板即可替换当前草稿，填入对应的评分项、分值和要求。</p>
      <p v-if="templateNotice" class="template-notice" role="status">{{ templateNotice }}</p>
    </div>
    <p class="rubric-hint"
      >当前标准：{{
        currentTemplate?.name || '自定义标准'
      }}。可按本次作业调整评分项、分值和要求。<span v-if="!advanced"
        >当前合计 {{ total }} 分。</span
      ></p
    >
    <template v-if="!advanced">
      <p v-if="!rows.length" class="rubric-hint">请应用一个模板，或添加评分项。</p>
      <div class="criterion" v-for="(row, index) in rows" :key="index">
        <label
          >评分项<input
            v-model="row.name"
            :aria-label="`评分项${index + 1}名称`"
            required
            maxlength="100"
            @input="publish"
        /></label>
        <label
          >分值<input
            v-model.number="row.score"
            :aria-label="`评分项${index + 1}分值`"
            type="number"
            min="0.1"
            step="0.1"
            required
            @input="publish"
        /></label>
        <label class="criterion-description"
          >要求<textarea
            v-model="row.description"
            :aria-label="`评分项${index + 1}要求`"
            rows="3"
            placeholder="填写本项的评分依据与扣分规则"
            @input="publish"
          />
        </label>
        <button
          type="button"
          :aria-label="`删除评分项${index + 1}`"
          :disabled="rows.length === 1"
          @click="remove(index)"
          >删除</button
        >
      </div>
      <button type="button" @click="add">添加评分项</button>
    </template>
    <label v-else
      >评分标准（JSON）<textarea
        :value="modelValue"
        rows="7"
        @input="publishJson($event.target.value)"
        required
      />
    </label>
    <button type="button" class="mode-toggle" @click="toggle">{{
      advanced ? '返回表格编辑' : '编辑 JSON（高级）'
    }}</button>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
  </fieldset>
</template>
<script setup>
import { computed, ref, watch } from 'vue';
import {
  rubricTemplates,
  createRubricFromTemplate,
  matchingRubricTemplate,
} from '../services/rubricTemplates';
const props = defineProps({
  modelValue: { type: String, required: true },
  disabled: Boolean,
});
const emit = defineEmits(['update:modelValue']);
const advanced = ref(false),
  error = ref(''),
  selectedTemplateId = ref(''),
  templateNotice = ref('');
const selectedTemplate = computed(() =>
  rubricTemplates.find((item) => item.id === selectedTemplateId.value)
);
const currentTemplate = computed(() => matchingRubricTemplate(props.modelValue));
function decode(value) {
  const object = JSON.parse(value);
  if (!object || Array.isArray(object) || typeof object !== 'object' || !Object.keys(object).length)
    throw new Error();
  return Object.entries(object).map(([name, item]) => {
    const score = typeof item === 'number' ? item : item?.score;
    if (!Number.isFinite(score) || score <= 0) throw new Error();
    return { name, score, description: typeof item === 'object' ? item.description || '' : '' };
  });
}
const rows = ref([]);
let publishedValue;
watch(
  () => props.modelValue,
  (value) => {
    // Keep partial table edits when the parent echoes this component's output.
    if (value === publishedValue) return;
    try {
      rows.value = decode(value);
      advanced.value = false;
    } catch {
      rows.value = [];
      advanced.value = Boolean(value && value !== '{}');
    }
    selectedTemplateId.value = matchingRubricTemplate(value)?.id || '';
    error.value = '';
    templateNotice.value = '';
  },
  { immediate: true }
);
const total = computed(
  () => Math.round(rows.value.reduce((sum, row) => sum + (Number(row.score) || 0), 0) * 10) / 10
);
function publish() {
  templateNotice.value = '';
  const names = rows.value.map((row) => row.name.trim());
  error.value =
    names.some((name) => !name) || new Set(names).size !== names.length
      ? '请填写不重复的评分项名称。'
      : '';
  const valid =
    !error.value && rows.value.every((row) => Number.isFinite(row.score) && row.score > 0);
  publishedValue = valid
    ? JSON.stringify(
        Object.fromEntries(
          rows.value.map((row) => [
            row.name.trim(),
            { description: row.description, score: row.score },
          ])
        )
      )
    : '';
  emit('update:modelValue', publishedValue);
}
function publishJson(value) {
  publishedValue = value;
  templateNotice.value = '';
  emit('update:modelValue', value);
}
function applyTemplate() {
  if (!selectedTemplate.value) return;
  rows.value = decode(createRubricFromTemplate(selectedTemplateId.value));
  advanced.value = false;
  publish();
  templateNotice.value = `已填入“${selectedTemplate.value.name}”的初始评分标准，可继续调整，保存后生效。`;
}
function add() {
  rows.value.push({ name: '', score: 10, description: '' });
  publish();
}
function remove(index) {
  rows.value.splice(index, 1);
  publish();
}
function toggle() {
  if (advanced.value) {
    try {
      rows.value = decode(props.modelValue);
    } catch {
      error.value = '当前 JSON 无法转换为评分项表格，请继续使用 JSON 编辑。';
      return;
    }
  }
  advanced.value = !advanced.value;
  error.value = '';
}
</script>
<style scoped>
.rubric-editor {
  border: 0;
  border-top: 1px solid var(--color-line);
  border-radius: 0;
  padding: 20px 0 0;
  display: grid;
  gap: 12px;
}
legend {
  font-weight: 600;
  padding: 0 6px;
}
.rubric-hint {
  color: var(--color-muted);
  font-size: 13px;
}
.template-picker {
  display: grid;
  gap: 10px;
  padding: 14px;
  background: var(--color-accent-soft);
  border: 1px solid var(--color-line);
  border-radius: 8px;
}
.template-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  gap: 10px;
}
.template-actions label {
  flex: 1 1 140px;
}
.template-description,
.template-notice {
  font-size: 13px;
  line-height: 1.7;
}
.template-notice {
  color: var(--color-accent);
}
.template-preview {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  list-style: none;
  padding: 0;
  font-size: 12px;
  color: var(--color-muted);
}
.criterion {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 84px auto;
  gap: 10px;
  align-items: end;
}
.criterion-description {
  grid-column: 1/-1;
  grid-row: 2;
}
label {
  display: grid;
  min-width: 0;
  gap: 6px;
  font-size: 13px;
}
input,
textarea,
select {
  min-width: 0;
  width: 100%;
  padding: 9px;
  border: 1px solid var(--color-input-line);
  border-radius: 6px;
  background: var(--color-paper);
}
textarea {
  resize: vertical;
  line-height: 1.6;
}
button {
  justify-self: start;
  padding: 8px 11px;
  border: 1px solid var(--color-input-line);
  border-radius: 6px;
  font-size: 13px;
}
.mode-toggle {
  color: var(--color-accent);
}
.error {
  color: var(--color-danger);
}
@media (max-width: 600px) {
  .criterion {
    grid-template-columns: minmax(0, 1fr) 70px auto;
  }
}
</style>
