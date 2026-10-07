import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path' // 导入 path 模块

const businessProxy = {
  target: 'http://127.0.0.1:8000',
  changeOrigin: true,
  bypass(req) {
    if (req.method === 'GET' && req.headers.accept?.includes('text/html')) {
      return '/index.html'
    }
  },
}

export default defineConfig({
  plugins: [vue()],
  cacheDir: process.env.VITE_CACHE_DIR || 'node_modules/.vite',
  server: {
    host: true,
    allowedHosts: [new URL(process.env.AutoDLService6006URL || 'http://localhost').hostname],
    // allowedHosts: ['http://localhost:5173'],    // 5173端口网址
    // 新增 proxy 配置
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000', // 目标为本地后端服务，8000端口
        changeOrigin: true, // 需要虚拟主机站点
      },
      '/assignments': businessProxy,
      '/submissions': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/exams': businessProxy,
      '/uploads': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    }
  }
})
