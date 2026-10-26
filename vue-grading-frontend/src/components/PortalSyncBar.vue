<template>
  <section class="portal-sync" aria-label="官网作业同步">
    <div
      ><strong>官网课程与作业</strong
      ><p>同步后直接生成作业，自动填入官网要求和可编辑的评分模板。</p>
      <p v-if="view?.import?.total"
        >已创建 {{ view.import.total }} 份作业 · 要求就绪 {{ view.import.ready }} · 等待
        {{ view.import.pending }} · 正在读取 {{ view.import.reading }} · 待处理
        {{ view.import.attention }}</p
      >
      <p v-if="view?.connection?.imports_paused"
        >后续要求同步已暂停，当前正在读取的作业会继续完成。</p
      >
    </div>
    <div class="sync-actions">
      <button
        type="button"
        :disabled="busy || view?.status !== 'READY' || operationActive(view?.operation)"
        @click="sync"
        >更新课程与作业</button
      >
      <button v-if="view?.import?.pending" type="button" :disabled="busy" @click="pause">{{
        view.connection?.imports_paused ? '继续补全要求' : '暂停后续要求同步'
      }}</button>
      <router-link to="/settings/portal">教务账号设置</router-link>
    </div>
    <p v-if="notice" role="status">{{ notice }}</p
    ><p v-if="error" role="alert" class="error">{{ error }}</p>
    <details v-if="operationActive(view?.operation)"
      ><summary>查看同步任务：{{ view.operation.message }}</summary
      ><RpaTaskPanel :key="view.operation.id" :task-id="view.operation.id" embedded
    /></details>
  </section>
</template>
<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue';
import authApi from '../services/authApi';
import { createPortalApi, operationActive } from '../services/portalApi';
import RpaTaskPanel from './RpaTaskPanel.vue';
const emit = defineEmits(['updated']);
const api = createPortalApi(authApi.getClient()),
  view = ref(null),
  busy = ref(false),
  notice = ref(''),
  error = ref('');
let timer,
  loading = false,
  disposed = false;
async function load() {
  if (loading || disposed) return;
  loading = true;
  try {
    const { data } = await api.connection();
    if (!disposed) view.value = data;
  } catch (e) {
    if (!disposed) error.value = e.response?.data?.detail || '官网同步状态暂不可用';
  } finally {
    loading = false;
  }
}
async function sync() {
  busy.value = true;
  error.value = '';
  try {
    await api.sync();
    notice.value = '正在同步课程与作业，完成后会自动出现在作业列表';
    await load();
    emit('updated');
  } catch (e) {
    error.value = e.response?.data?.detail || '同步请求未完成';
  } finally {
    busy.value = false;
  }
}
async function pause() {
  busy.value = true;
  try {
    await api.pauseImports(!view.value.connection.imports_paused);
    await load();
  } catch (e) {
    error.value = e.response?.data?.detail || '同步状态未修改';
  } finally {
    busy.value = false;
  }
}
onMounted(() => {
  load();
  timer = setInterval(load, 3000);
});
onBeforeUnmount(() => {
  disposed = true;
  clearInterval(timer);
});
</script>
<style scoped>
.portal-sync {
  display: grid;
  gap: 12px;
  border: 1px solid #dbe3ee;
  border-radius: 12px;
  padding: 18px 22px;
  background: white;
  margin: 20px 0;
}
strong {
  font-weight: 600;
}
p {
  color: #64748b;
  font-size: 14px;
  margin-top: 6px;
}
.sync-actions {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  align-items: center;
}
button {
  padding: 8px 14px;
  border: 1px solid #cbd5e1;
  border-radius: 7px;
}
button:disabled {
  opacity: 0.5;
}
a,
summary {
  color: #4338ca;
  cursor: pointer;
}
.error {
  color: #b91c1c;
}
</style>
