<template>
  <div class="password-input">
    <input
      v-bind="$attrs"
      :type="visible ? 'text' : 'password'"
      :value="modelValue"
      @input="$emit('update:modelValue', $event.target.value)"
    />
    <button
      type="button"
      class="password-toggle"
      :aria-label="visible ? '隐藏密码' : '显示密码'"
      :title="visible ? '隐藏密码' : '显示密码'"
      :aria-pressed="visible"
      :disabled="$attrs.disabled"
      @click="visible = !visible"
    >
      <AppIcon :name="visible ? 'eye-off' : 'eye'" />
    </button>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import AppIcon from './AppIcon.vue';

defineOptions({ inheritAttrs: false });
defineProps({ modelValue: { type: String, default: '' } });
defineEmits(['update:modelValue']);

const visible = ref(false);
</script>

<style scoped>
.password-input {
  position: relative;
}

.password-input input {
  padding-right: 2.75rem;
}

.password-toggle {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  display: inline-flex;
  width: 2.5rem;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 0 0.375rem 0.375rem 0;
  background: transparent;
  color: #64748b;
  cursor: pointer;
}

.password-toggle:hover {
  color: #334155;
}

.password-toggle:focus-visible {
  outline: 2px solid #6366f1;
  outline-offset: -2px;
}

.password-toggle:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.password-toggle svg {
  width: 1.15rem;
  height: 1.15rem;
}
</style>
