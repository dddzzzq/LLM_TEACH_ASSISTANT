<template>
  <form class="target-resolver space-y-4 text-sm" @submit.prevent="submit">
    <p class="rounded-lg bg-amber-50 p-3 text-amber-900">{{ resolution.message }}</p>
    <p>输入的{{ label }}：<strong>{{ resolution.requested_name }}</strong></p>
    <p v-if="resolution.term">所在学期：<strong>{{ resolution.term }}</strong></p>
    <fieldset v-if="resolution.candidates?.length" :disabled="busy" class="space-y-2">
      <legend class="mb-2 font-semibold">请选择实际要拉取的{{ label }}</legend>
      <label v-for="candidate in resolution.candidates" :key="candidate.id" class="flex items-start gap-2 rounded-lg border p-3">
        <input v-model="selected" type="radio" :value="candidate.id" class="mt-1" @change="editing = false" />
        <span class="min-w-0 break-words"><strong>{{ candidate.name }}</strong><span v-if="candidate.context" class="mt-1 block text-xs text-slate-500">{{ candidate.context }}</span></span>
      </label>
    </fieldset>
    <template v-if="editing">
      <label class="block font-semibold">正确的{{ label }}名称
        <input v-model="name" :disabled="busy" maxlength="255" required class="mt-2 block w-full rounded border px-3 py-2 font-normal" />
      </label>
      <p class="text-xs text-slate-500">将在当前页面重新查找此名称。找到近似名称时仍会请你确认；学期需要你手动核查或切换。</p>
    </template>
    <div class="flex flex-wrap gap-3">
      <button v-if="!editing" type="submit" :disabled="busy || !selected" class="rounded-lg bg-blue-600 px-4 py-2 text-white disabled:opacity-50">{{ busy ? '正在确认…' : '确认所选并继续' }}</button>
      <button v-if="!editing" type="button" :disabled="busy" class="rounded-lg border px-4 py-2" @click="editing = true">都不是，修改名称</button>
      <button v-if="editing" type="submit" :disabled="busy || !name.trim()" class="rounded-lg bg-blue-600 px-4 py-2 text-white disabled:opacity-50">{{ busy ? '正在提交…' : '修改名称并重新查找' }}</button>
    </div>
  </form>
</template>

<script setup>
import { computed, ref } from 'vue'
const props = defineProps({ resolution: {type:Object, required:true}, busy:Boolean })
const emit = defineEmits(['resolve'])
const selected = ref(props.resolution.candidates?.[0]?.id || '')
const editing = ref(!props.resolution.candidates?.length)
const name = ref(props.resolution.requested_name || '')
const label = computed(() => props.resolution.kind === 'course' ? '课程' : '作业')
function submit() {
  if (props.busy) return
  emit('resolve', { resolution_id:props.resolution.id, ...(editing.value
    ? {choice:'edit',name:name.value.trim()} : {choice:'confirm',candidate_id:selected.value}) })
}
</script>
