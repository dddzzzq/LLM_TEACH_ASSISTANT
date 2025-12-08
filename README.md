# LLM_TEACH_ASSISTANT

智能化作业与试卷批改系统

基于 FastAPI (后端) 和 Vue.js (前端) 构建的全栈Web应用，旨在利用大语言模型（LLM）的能力，为教师提供一个高效、智能、可靠的AI助教，实现对学生项目制作业的自动化评估。

# 主要功能
本系统将教师从繁重、重复的批改工作中解放出来，并集成了先进的学术诚信检测功能。


# 技术栈
后端 (Backend):

框架: FastAPI

语言: Python 3.11.13

数据库 ORM: SQLAlchemy (异步模式 with aiosqlite)

数据库: SQLite (便于开发，可轻松迁移至 MySQL/PostgreSQL)

核心库: scikit-learn (用于TF-IDF), python-docx, PyPDF2, rarfile等等

前端 (Frontend):

框架: Vue.js 3

构建工具: Vite

路由: Vue Router

UI: Tailwind CSS

HTTP客户端: Axios

外部服务:

大语言模型: DeepSeek API (可替换)

# 快速开始
## 1. 环境准备

Python: 版本 3.8 或更高。

Node.js: 版本 18.x 或更高，并附带 npm 包管理器。

UnRAR 工具:

Windows: 从 RARLAB官网 下载 UnRAR.exe 并将其路径添加到系统环境变量 PATH 中。

macOS: brew install unrar

Linux (Ubuntu/Debian): sudo apt-get install unrar

## 2. 后端配置与启动

1. 安装所有python依赖库
2. 配置API密钥
3. 启动后端服务器
   1. `uvicorn app.main:app --reload`

## 3. 前端配置与启动

1. 安装所有Node.js依赖库
   1. `npm install`
2. 启动前端开发服务器
   1. `npm run dev`

# 使用流程
启动服务: 确保后端和前端服务都已成功启动。

访问主页: 打开浏览器，访问前端应用的地址（例如 http://localhost:5173）。

新建作业: 点击“新建作业”，填写作业名称、题目要求和JSON格式的评分标准，然后创建。

查看作业: 创建成功后，会自动跳转到“作业列表”页面，您可以看到所有作业。

提交评分: 点击任意一个作业，进入详情页。在此页面，上传一个包含所有学生作业的ZIP或RAR压缩包，然后点击“提交并开始后台评分”。

查看结果: 提交后，系统会提示正在后台处理。您可以稍等片刻后点击“刷新结果”按钮，即可看到包含分数和查重报告的详细评分列表。