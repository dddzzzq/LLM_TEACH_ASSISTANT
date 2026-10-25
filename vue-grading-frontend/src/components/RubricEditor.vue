<template>
  <fieldset class="rubric-editor">
    <legend>评分标准</legend>
    <p class="rubric-hint"
      >请按本次作业调整评分项、分值和要求。<span v-if="!advanced"
        >当前合计 {{ total }} 分。</span
      ></p
    >
    <template v-if="!advanced">
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
          >要求<input
            v-model="row.description"
            :aria-label="`评分项${index + 1}要求`"
            placeholder="例如：程序运行正确，覆盖主要测试用例"
            @input="publish"
        /></label>
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
        @input="$emit('update:modelValue', $event.target.value)"
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
import { computed, ref } from 'vue';
const props = defineProps({ modelValue: { type: String, required: true } });
const emit = defineEmits(['update:modelValue']);
const advanced = ref(false),
  error = ref('');
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
let initial;
try {
  initial = decode(props.modelValue);
} catch {
  initial = [];
  advanced.value = true;
}
const rows = ref(initial);
const total = computed(
  () => Math.round(rows.value.reduce((sum, row) => sum + (Number(row.score) || 0), 0) * 10) / 10
);
function publish() {
  const names = rows.value.map((row) => row.name.trim());
  error.value =
    names.some((name) => !name) || new Set(names).size !== names.length
      ? '请填写不重复的评分项名称。'
      : '';
  const valid =
    !error.value && rows.value.every((row) => Number.isFinite(row.score) && row.score > 0);
  emit(
    'update:modelValue',
    valid
      ? JSON.stringify(
          Object.fromEntries(
            rows.value.map((row) => [
              row.name.trim(),
              { description: row.description, score: row.score },
            ])
          )
        )
      : ''
  );
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
  border: 1px solid #dce5e9;
  border-radius: 10px;
  padding: 16px;
  display: grid;
  gap: 12px;
}
legend {
  font-weight: 600;
  padding: 0 6px;
}
.rubric-hint {
  color: #66788a;
  font-size: 13px;
}
.criterion {
  display: grid;
  grid-template-columns: minmax(110px, 1fr) 90px minmax(140px, 2fr) auto;
  gap: 10px;
  align-items: end;
}
label {
  display: grid;
  min-width: 0;
  gap: 6px;
  font-size: 13px;
}
input,
textarea {
  min-width: 0;
  width: 100%;
  padding: 9px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
}
button {
  justify-self: start;
  padding: 8px 11px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-size: 12px;
}
.mode-toggle {
  color: #197b70;
}
.error {
  color: #b91c1c;
}
@media (max-width: 600px) {
  .criterion {
    grid-template-columns: minmax(0, 1fr) 70px auto;
  }
  .criterion-description {
    grid-column: 1/-1;
    grid-row: 2;
  }
}
</style>
