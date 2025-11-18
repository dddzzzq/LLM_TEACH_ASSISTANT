import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path' // 导入 path 模块

// export default defineConfig({
//   plugins: [
//     vue(),
//   ],
//   resolve: {
//     alias: {
//       '@': fileURLToPath(new URL('./src', import.meta.url))
//     }
//   }
// })
export default defineConfig({
  plugins: [vue()],
  server: {
    host: true,
    allowedHosts: ['.ngrok-free.app'],
    // 新增 proxy 配置
    proxy: {
      '/api': {
        target: 'http://localhost:8000', // 目标为本地后端服务
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