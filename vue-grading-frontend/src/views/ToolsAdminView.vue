<template>
  <div class="surface tools-page">
    <div class="flex items-start justify-between gap-4 mb-6">
      <div>
        <h2 class="page-title">工具管理</h2>
        <p class="text-sm text-gray-500 mt-1">
          配置 Agent 可调用工具的启用状态、允许角色和描述；参数定义供查阅。
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
        v-for="tool in tools"
        :key="tool.name"
        class="border tool-card"
      >
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div class="min-w-0">
            <div class="flex items-center gap-2">
              <div class="font-semibold text-gray-800 truncate">{{ tool.name }}</div>
              <span
                class="text-xs px-2 py-0.5 rounded-full"
                :class="tool.enabled ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'"
              >
                {{ tool.enabled ? 'ENABLED' : 'DISABLED' }}
              </span>
            </div>
            <div v-if="!tool.registered" class="text-xs text-amber-700 mt-1">工具暂不可用：执行器未注册</div>
          </div>

          <label class="flex items-center gap-2 text-sm">
            <input type="checkbox" v-model="tool.enabled" :disabled="!tool.registered" />
            启用
          </label>
        </div>

        <div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div class="text-sm font-medium text-gray-700 mb-2">允许角色</div>
            <div class="flex gap-3 text-sm">
              <label class="flex items-center gap-2">
                <input type="checkbox" :value="'student'" v-model="tool.allowed_roles" />
                student
              </label>
              <label class="flex items-center gap-2">
                <input type="checkbox" :value="'teacher'" v-model="tool.allowed_roles" />
                teacher
              </label>
              <label class="flex items-center gap-2">
                <input type="checkbox" :value="'admin'" v-model="tool.allowed_roles" />
                admin
              </label>
            </div>
          </div>

          <div>
            <div class="text-sm font-medium text-gray-700 mb-2">描述（description）</div>
            <textarea
              v-model="tool.description"
              class="w-full border rounded px-3 py-2 text-sm h-24"
            />
          </div>
        </div>

        <div class="mt-4">
          <div class="text-sm font-medium text-gray-700 mb-2">参数定义（只读）</div>
          <textarea
            :value="tool.schema_json"
            readonly
            class="w-full border rounded px-3 py-2 text-sm font-mono h-40"
          />
          <div class="text-xs text-gray-500 mt-1">
            参数定义与执行器保持一致，不支持在此页面修改。
          </div>
        </div>

        <div class="mt-4 flex items-center justify-between">
          <div class="text-xs text-gray-400">
            updated_at: {{ tool.updated_at }}
          </div>
          <button
            @click="save(tool)"
            class="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-700 text-white rounded"
            :disabled="savingName === tool.name || !tool.registered"
          >
            {{ savingName === tool.name ? '保存中...' : '保存' }}
          </button>
        </div>
      </div>

      <div v-if="tools.length === 0" class="text-sm text-gray-500">
        暂无工具配置。
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import toolsApi from '../services/toolsApi'

const tools = ref([])
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
    const res = await toolsApi.listTools()
    const rows = Array.isArray(res.data) ? res.data : []
    tools.value = rows.map((s) => ({
      id: s.id,
      name: s.name,
      registered: s.registered !== false,
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

async function save(tool) {
  savingName.value = tool.name
  error.value = ''
  try {
    const res = await toolsApi.updateTool(tool.name, {
      enabled: tool.enabled,
      description: tool.description,
      allowed_roles: tool.allowed_roles
    })
    tool.updated_at = res.data.updated_at
  } catch (e) {
    error.value = e?.response?.data?.error || e?.message || '保存失败'
  } finally {
    savingName.value = ''
  }
}

async function refreshCache() {
  error.value = ''
  try {
    await toolsApi.refreshToolsCache()
  } catch (e) {
    error.value = e?.response?.data?.error || e?.message || '刷新缓存失败'
  }
}

onMounted(load)
</script>

