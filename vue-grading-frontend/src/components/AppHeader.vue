<template>
  <header class="simple-header"
    ><router-link to="/ai-assistant">教师作业助手</router-link
    ><nav aria-label="主导航"
      ><router-link to="/ai-assistant">AI 助手</router-link
      ><router-link to="/grade-homework">作业批改</router-link></nav
    ></header
  >
</template>
<style scoped>
.simple-header,
nav {
  display: flex;
  gap: 24px;
  align-items: center;
}
.simple-header {
  padding: 18px 24px;
  background: white;
  justify-content: space-between;
}
</style>
