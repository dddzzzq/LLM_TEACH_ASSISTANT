<template>
  <div class="bg-white rounded-lg shadow p-6">
    <div class="flex items-start justify-between gap-4 mb-6">
      <div>
        <h2 class="text-2xl font-bold text-gray-800">Skills 管理</h2>
        <p class="text-sm text-gray-500 mt-1">
          配置 LLM 可调用的工具：启用状态、允许角色、描述与 JSON Schema。
        </p>
      </div>
      <div class="flex gap-2">
        <button
          @click="refreshCache"
          class="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded border"
          :disabled="loading"
        >
          刷新缓存
        </button>
        <button
          @click="load"
          class="px-3 py-2 text-sm bg-indigo-600 hover:bg-indigo-700 text-white rounded"
          :disabled="loading"
        >
          重新加载
        </button>
      </div>
    </div>

    <div v-if="error" class="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded">
      {{ error }}
    </div>

    <div v-if="loading" class="text-sm text-gray-500">加载中...</div>

    <div v-else class="space-y-4">
      <div
        v-for="skill in skills"
        :key="skill.name"
        class="border rounded-lg p-4"
      >
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div class="min-w-0">
            <div class="flex items-center gap-2">
              <div class="font-semibold text-gray-800 truncate">{{ skill.name }}</div>
              <span
                class="text-xs px-2 py-0.5 rounded-full"
                :class="skill.enabled ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'"
              >
                {{ skill.enabled ? 'ENABLED' : 'DISABLED' }}
              </span>
            </div>
            <div class="text-xs text-gray-500 mt-1">impl_key: {{ skill.impl_key }}</div>
          </div>

          <label class="flex items-center gap-2 text-sm">
            <input type="checkbox" v-model="skill.enabled" />
            启用
          </label>
        </div>

        <div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div class="text-sm font-medium text-gray-700 mb-2">允许角色</div>
            <div class="flex gap-3 text-sm">
              <label class="flex items-center gap-2">
                <input type="checkbox" :value="'student'" v-model="skill.allowed_roles" />
                student
              </label>
              <label class="flex items-center gap-2">
                <input type="checkbox" :value="'teacher'" v-model="skill.allowed_roles" />
                teacher
              </label>
              <label class="flex items-center gap-2">
                <input type="checkbox" :value="'admin'" v-model="skill.allowed_roles" />
                admin
              </label>
            </div>
          </div>

          <div>
            <div class="text-sm font-medium text-gray-700 mb-2">描述（description）</div>
            <textarea
              v-model="skill.description"
              class="w-full border rounded px-3 py-2 text-sm h-24"
            />
          </div>
        </div>

        <div class="mt-4">
          <div class="text-sm font-medium text-gray-700 mb-2">Schema（JSON Schema）</div>
          <textarea
            v-model="skill.schema_json"
            class="w-full border rounded px-3 py-2 text-sm font-mono h-40"
          />
          <div class="text-xs text-gray-500 mt-1">
            注意：这里只校验“是否为合法 JSON”，不做严格 JSON Schema 校验。
          </div>
        </div>

        <div class="mt-4 flex items-center justify-between">
          <div class="text-xs text-gray-400">
            updated_at: {{ skill.updated_at }}
          </div>
          <button
            @click="save(skill)"
            class="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-700 text-white rounded"
            :disabled="savingName === skill.name"
          >
            {{ savingName === skill.name ? '保存中...' : '保存' }}
          </button>
        </div>
      </div>

      <div v-if="skills.length === 0" class="text-sm text-gray-500">
        未查询到任何 skills。
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import skillsApi from '../services/skillsApi'

const skills = ref([])
const loading = ref(false)
const error = ref('')
const savingName = ref('')

function safeParseAllowedRoles(value) {
  try {
    const arr = JSON.parse(value || '[]')
    return Array.isArray(arr) ? arr : []
  } catch {
    return []
  }
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const res = await skillsApi.listSkills()
    const rows = Array.isArray(res.data) ? res.data : []
    skills.value = rows.map((s) => ({
      id: s.id,
      name: s.name,
      impl_key: s.impl_key,
      enabled: !!s.enabled,
      description: s.description || '',
      schema_json: s.schema_json || '',
      allowed_roles: safeParseAllowedRoles(s.allowed_roles),
      updated_at: s.updated_at
    }))
  } catch (e) {
    error.value = e?.response?.data?.error || e?.message || '加载失败'
  } finally {
    loading.value = false
  }
}

async function save(skill) {
  savingName.value = skill.name
  error.value = ''
  try {
    await skillsApi.updateSkill(skill.name, {
      enabled: skill.enabled,
      description: skill.description,
      schema_json: skill.schema_json,
      allowed_roles: skill.allowed_roles
    })
  } catch (e) {
    error.value = e?.response?.data?.error || e?.message || '保存失败'
  } finally {
    savingName.value = ''
  }
}

async function refreshCache() {
  error.value = ''
  try {
    await skillsApi.refreshSkillsCache()
  } catch (e) {
    error.value = e?.response?.data?.error || e?.message || '刷新缓存失败'
  }
}

onMounted(load)
</script>

