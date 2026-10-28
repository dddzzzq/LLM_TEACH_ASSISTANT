<template>
  <div class="materials-index"
    ><header class="page-heading"
      ><div
        ><h1 class="page-title">作业材料</h1
        ><p class="page-description"
          >查看全部作业的导入记录与材料异常。日常导入和批改请进入对应作业，按班级处理。</p
        ></div
      ></header
    ><div class="surface p-5 md:p-7"><MaterialBatchPanel /></div
  ></div>
</template>
<script setup>
import MaterialBatchPanel from '../components/MaterialBatchPanel.vue';
</script>

<style scoped>
.materials-index > .surface {
  padding: 0;
  border: 0;
  background: transparent;
  box-shadow: none;
}
</style>
