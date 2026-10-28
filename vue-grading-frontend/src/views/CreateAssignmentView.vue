<template>
  <div class="create-assignment">
    <router-link to="/assignments" class="back-link">返回作业工作台</router-link>
    <header class="page-heading"
      ><div
        ><h1 class="page-title">新建本地作业</h1
        ><p class="page-description">保存题目和评分标准，随后按班级导入学生材料。</p></div
      ></header
    >
    <form class="create-form surface" @submit.prevent="createAssignment($event)">
      <section
        ><div class="section-title"><span>01</span><h2>基本信息</h2></div
        ><div class="form-columns"
          ><label
            >课程名称<input
              v-model.trim="courseName"
              required
              maxlength="255"
              placeholder="例如：移动互联网导论" /></label
          ><label
            >作业名称<input
              v-model.trim="taskName"
              required
              maxlength="255"
              placeholder="例如：编程小作业 1" /></label></div
      ></section>
      <section
        ><div class="section-title"><span>02</span><h2>作业内容</h2></div
        ><label
          >题目要求<textarea
            v-model.trim="question"
            required
            rows="8"
            placeholder="填写题目、提交要求及需要完成的内容，作为批改依据。"
          /></label
      ></section>
      <section
        ><div class="section-title"><span>03</span><h2>评分标准</h2></div
        ><RubricEditor v-model="rubric"
      /></section>
      <div v-if="error" class="workspace-error" role="alert">{{ error }}</div>
      <footer
        ><router-link to="/assignments" class="workspace-button">取消</router-link
        ><button
          class="workspace-button"
          :disabled="isLoading || !rubric"
          type="submit"
          value="content"
          >仅创建作业</button
        ><button
          class="workspace-button primary"
          :disabled="isLoading || !rubric"
          type="submit"
          value="materials"
          >{{ isLoading ? '保存中…' : '创建并导入材料' }}</button
        ></footer
      >
    </form>
  </div>
</template>
<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import gradingApi from '../services/gradingApi';
import RubricEditor from '../components/RubricEditor.vue';
const router = useRouter();
const courseName = ref(''),
  taskName = ref(''),
  question = ref(''),
  error = ref(''),
  isLoading = ref(false);
const rubric = ref(
  JSON.stringify({
    功能实现: { description: '核心功能完整无误', score: 60 },
    代码质量: { description: '代码规范，注释清晰', score: 20 },
    文档报告: { description: '报告内容详实，格式正确', score: 20 },
  })
);
async function createAssignment(event) {
  const nextTab = event?.submitter?.value === 'content' ? 'content' : 'materials';
  if (isLoading.value) return;
  isLoading.value = true;
  error.value = '';
  try {
    const { data } = await gradingApi.createAssignment({
      course_name: courseName.value,
      class_name: '全部班级',
      task_name: taskName.value,
      question: question.value,
      rubric: JSON.parse(rubric.value),
    });
    await router.push(
      data.id ? { path: `/assignments/${data.id}`, query: { tab: nextTab } } : '/assignments'
    );
  } catch (e) {
    error.value = e instanceof SyntaxError ? '请完善评分标准。' : '保存失败，请稍后重试。';
  } finally {
    isLoading.value = false;
  }
}
</script>
<style scoped>
.create-assignment {
  max-width: 960px;
  margin: 0 auto;
}
.create-form {
  padding: 30px;
}
.create-form section + section {
  border-top: 1px solid var(--color-line);
  margin-top: 28px;
  padding-top: 28px;
}
.section-title {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}
.section-title span {
  font-size: 11px;
  color: var(--color-accent);
  background: var(--color-accent-soft);
  border-radius: 6px;
  padding: 5px 7px;
}
.section-title h2 {
  font-size: 14px;
  font-weight: 600;
}
.form-columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 22px;
}
label {
  display: flex;
  flex-direction: column;
  gap: 9px;
  font-size: 12px;
  color: #4c5870;
}
input,
textarea {
  width: 100%;
  border: 1px solid var(--color-line);
  padding: 11px 13px;
  font-size: 13px;
}
footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 28px;
  border-top: 1px solid var(--color-line);
  padding-top: 22px;
}
.workspace-error {
  margin-top: 20px;
}
@media (max-width: 760px) {
  .create-form {
    padding: 20px;
  }
  .form-columns {
    grid-template-columns: 1fr;
    gap: 16px;
  }
  footer {
    justify-content: stretch;
  }
  footer > * {
    flex: 1;
  }
}
</style>
