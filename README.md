# 教师作业助手

教师专用的 AI 助手与作业批改系统。分支为 `codex/teacher-homework-only`，基于完整系统快照 `88e7ddc` 拆分。

## 两个主入口

### AI 助手

通过对话拉取作业、整理材料、安排批改、查询学生作业成绩。保留聊天会话、可重建执行记录、任务状态和浏览器控制台。

快捷操作为“拉取作业并解压”“批改作业”“查询成绩”，点击仅填写消息草稿，发送后才执行。`after_download` 支持 download（仅下载）、extract（下载并整理，默认）、grade（整理后自动批改）。

### 作业批改

1. 选择来源：官网作业、上传 ZIP/RAR/7z 压缩包或指定本机目录。
2. 填写题目要求，调整评分项名称、分值和要求；需要时切换 JSON 高级编辑。
3. 点击“开始批改”，系统自动下载或读取附件、整理材料、查重、检测 AIGC、评分并保存结果。

默认处理全部班级，无需班级选择。相关页面集中在作业批改入口下：

| 页面 | 用途 |
| --- | --- |
| 开始批改 | 选择材料、配置题目和评分标准 |
| 批改记录 | 查看结果、复核评分、导出 Excel |
| 作业材料 | 查看解压进度、学生材料清单和异常 |
| 教务账号 | 加密保存教务凭据、认证、更新课程目录 |
| 更多设置 | 助手工具配置；已有管理员账号还可查阅运行日志 |

## 账号和功能边界

- 登录后直接进入 AI 助手，不再展示学生端或多功能概览。
- 新注册账号固定为教师，不接受学生或管理员角色注册。学生账号及旧学生令牌无法访问教师业务 API。
- 已有教师和管理员账号进入同一教师工作区，管理员原有后台权限保留。
- 本系统账号与学校教务账号分别管理，教务账号在“作业批改 → 教务账号”中设置。
- 移除试卷页面、HTTP 接口、Skill、评分工具考试参数、试卷队列处理及 Python 试卷评分实现。
- 历史数据库表和 protobuf 消息定义保留兼容，不删除历史数据。旧试卷 RPC 返回 UNIMPLEMENTED，新进程仅订阅作业批改队列。
- 作业查重、AIGC 检测、文档解析、代码文档匹配、教师复核、成绩池化和结果导出继续保留。

## 运行与验证

沿用 Vue 3 前端、Go/Eino 业务后端、Python AI 计算服务和原 RPA Worker，依赖版本及部署配置不变。

- 部署：[Windows 脚本](scripts/windows/) 与 [WSL 启动脚本](scripts/start-wsl.sh)。
- 开发环境与维护：[CONTRIBUTING.md](CONTRIBUTING.md)。
- 教务账号与课程目录：[教务部署说明](docs/teacher-portal-automation.md)。
- 下载及材料整理：[RPA 实现](docs/rpa-implementation.md)、[材料工作流](docs/homework-materials-workflow.md)。

```bash
cd vue-grading-frontend
npm ci
npm test
npm run build
cd ../backend-go
go test ./...
go vet ./...
cd ../ai_engine_python
python -m unittest discover -s tests
```

浏览器回归：`vue-grading-frontend/tests/teacher_workspace_smoke.py` 验证真实路由、教师权限、两个主入口、评分编辑及移动端布局；`portal_automation_smoke.py` 验证教务认证、课程刷新及三种材料来源。均使用模拟接口，不触发真实批改或发送验证码。

## 源码入口

| 内容 | 位置 |
| --- | --- |
| 工作区路由与导航 | `vue-grading-frontend/src/router/index.js`、`src/services/teacherWorkspace.js` |
| 作业与评分表格 | `vue-grading-frontend/src/views/HomeworkGradingView.vue`、`src/components/RubricEditor.vue` |
| 教师账号权限 | `backend-go/internal/handlers/auth_handler.go`、`internal/middleware/auth.go` |
| Agent 与工具 | `backend-go/internal/agent/`、`internal/browseragent/` |
| 下载、整理、批改衔接 | `backend-go/internal/portal/workflows.go`、`internal/rpa/service.go`、`internal/grading/materials.go` |
| 作业计算 | `backend-go/internal/tools/homework_process.go`、`ai_engine_python/app/rpc/compute.py` |

完整版本保存在 Git 历史中，可与拆分前快照比较。本分支不需要额外的版本开关。
