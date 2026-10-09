<template>
  <div class="p-4 rounded-lg border border-amber-200 bg-amber-50 space-y-3">
    <strong class="text-sm">确认本次导出班级</strong>
    <p v-if="!classes.length" class="text-sm">尚未识别到可验证的班级选项，请在浏览器中打开作业附件的导出设置后交还自动化。</p>
    <template v-else>
      <label v-if="classes.some(c => c.all)" class="flex gap-2 text-sm"><input v-model="mode" type="radio" value="all" />全部班级</label>
      <label v-if="classes.some(c => !c.all)" class="flex gap-2 text-sm"><input v-model="mode" type="radio" value="selected" />指定班级</label>
      <div v-if="mode === 'selected'" class="space-y-2 pl-4">
        <label v-for="item in classes.filter(c => !c.all)" :key="item.id" class="flex gap-2 text-sm"><input v-model="ids" type="checkbox" :value="item.id" />{{ item.name }}</label>
      </div>
      <button class="px-3 py-2 bg-amber-700 rounded text-white text-sm disabled:opacity-50" :disabled="!mode || (mode === 'selected' && !ids.length)" @click="$emit('confirm', { mode, class_ids: ids })">确认范围并继续</button>
    </template>
  </div>
</template>
<script setup>
import { ref } from 'vue'
defineProps({ classes: { type: Array, default: () => [] } })
defineEmits(['confirm'])
const mode = ref(''), ids = ref([])
</script>
