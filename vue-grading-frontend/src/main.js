import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router' // 导入路由配置

const app = createApp(App)

app.use(router) // 告诉Vue应用使用我们定义的路由

app.mount('#app')