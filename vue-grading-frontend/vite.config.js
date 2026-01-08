import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path' // 导入 path 模块

export default defineConfig({
  plugins: [vue()],
  server: {
    host: true,
    // allowedHosts: ['http://localhost:5173'],    // 5173端口网址
    // 新增 proxy 配置
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000', // 目标为本地后端服务，8000端口
        changeOrigin: true, // 需要虚拟主机站点
        rewrite: (path) => path.replace(/^\/api/, ''), // 重写请求路径，去掉'/api'
      },
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    }
  }
})