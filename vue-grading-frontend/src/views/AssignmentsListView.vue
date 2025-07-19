<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-3xl font-bold text-gray-800">所有作业任务</h1>
      <router-link
        to="/assignments/new"
        class="px-4 py-2 font-semibold text-white bg-indigo-600 rounded-lg shadow-md transition-colors hover:bg-indigo-700"
      >
        + 新建作业
      </router-link>
    </div>

    <div v-if="isLoading" class="py-10 text-center">
      <Loader />
    </div>

    <div v-else-if="error" class="p-4 text-red-500 bg-red-100 rounded-lg">
      {{ error }}
    </div>

    <div v-else class="overflow-hidden bg-white rounded-lg shadow-xl">
      <ul class="divide-y divide-gray-200">
        <li
          v-for="(assignment, index) in assignments"
          :key="assignment.id"
          class="flex items-center justify-between p-4 transition-colors hover:bg-gray-50"
        >
          <router-link :to="`/assignments/${assignment.id}`" class="flex-grow">
            <p class="text-lg font-semibold text-indigo-700">{{ assignment.task_name }}</p>
            <p class="mt-1 text-sm text-gray-600 truncate">
              {{ assignment.question.slice(0, 110) + '...' }}
            </p>
          </router-link>

          <button
            @click.prevent="deleteAssignment(assignment.id, index)"
            title="删除此作业"
            class="p-2 ml-4 text-gray-400 rounded-full hover:text-red-600"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="w-5 h-5"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fill-rule="evenodd"
                d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z"
                clip-rule="evenodd"
              />
            </svg>
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { RouterLink } from 'vue-router';
import gradingApi from '../services/gradingApi';
import Loader from '../components/Loader.vue';


interface Assignment {
  id: number;
  task_name: string;
  question: string;
}

const assignments = ref<Assignment[]>([]);
const isLoading = ref(true);
const error = ref<string | null>(null);


const fetchAssignments = async () => {
  isLoading.value = true;
  try {
    const response = await gradingApi.getAssignments();
    assignments.value = response.data;
  } catch (e) {
    error.value = '无法加载作业列表。';
    console.error(e);
  } finally {
    isLoading.value = false;
  }
};

const deleteAssignment = async (id: number, index: number) => {
  if (confirm('确定要删除这个作业及其所有评分记录吗？\n此操作不可撤销。')) {
    try {
      await gradingApi.deleteAssignment(id);
      assignments.value.splice(index, 1);
    } catch (e) {
      alert('删除失败，请重试。');
      console.error(e);
    }
  }
};


onMounted(fetchAssignments);
</script>