# 教师作业助手

教师专用的 AI 助手与作业批改系统。分支为 `codex/teacher-homework-only`，基于完整系统快照 `88e7ddc` 拆分。

## 作业工作区与悬浮助手

### AI 助手

通过右下角可拖动图标展开对话，拉取作业、整理材料、安排批改、查询学生作业成绩。面板可放大或收起，跨页面保留草稿和当前会话；会话引用按账号隔离。保留历史聊天、可重建执行记录、任务状态和浏览器控制台。旧 `/ai-assistant` 链接仍会打开同一个全局助手。

快捷操作为“拉取作业并解压”“批改作业”“查询成绩”，点击仅填写消息草稿，发送后才执行。`after_download` 支持 download（仅下载）、extract（下载并整理，默认）、grade（整理后自动批改）。

### 作业批改

默认首页为独立的作业工作台：按课程、状态、来源和关键词筛选作业，名称直接进入详情，独立预览入口保留。筛选条件写入 URL，返回列表后恢复。进度与人数来自实际材料批次和批改记录，统计不可用时显示待同步。

同一作业下直接列出各班级，选择班级后只展示对应班级的材料、批改进度和成绩，历史未分班记录单独保留。题目与评分标准由同一作业的各班共用；修改标准只影响后续批改。成绩复核集中查看原始评分和检测依据，人工 0 分可正常保存。Excel 支持单班导出，整作业导出时按班级分表。

要求显示“需要核对”时，题目页会展示已提取正文、原始文字、来源和具体疑点。教师可补充正文，勾选核对确认后保存；服务端校验当前目录及读取任务版本，原始来源和疑点留存。确认使要求就绪，不会新建批改任务；此前明确请求且正在等待要求的自动流程可能继续。

常规流程：新建作业 → 保存题目与评分标准 → 在作业详情导入学生材料 → 开始批改 → 查看结果、复查与导出。新建作业默认处理全部班级，无需选择班级。

“导入与自动批改”保留原来的一步提交方式：选择官网作业、压缩包或本机目录，填写要求后自动整理并批改。它与常规作业详情分开，避免长表单挤占作业内容。

| 页面 | 用途 |
| --- | --- |
| 作业批改 `/assignments` | 作业列表、筛选、进度、内容预览 |
| 新建作业 `/assignments/new` | 保存题目与评分标准，随后进入详情 |
| 作业详情 `/assignments/:id` | 题目与评分标准、学生作业与批改、成绩与复核；`class_id` 定位具体班级 |
| 导入与自动批改 `/grade-homework` | 原有三种来源与自动流程，旧链接继续有效 |
| 作业材料 | 查看解压进度、学生材料清单和异常 |
| 教务账号 | 加密保存教务凭据、认证、更新课程目录 |
| 管理员维护 | 已有管理员账号可查阅运行日志和维护历史作业归属 |

## 账号和功能边界

- 登录后直接进入作业工作台；AI 助手为全局悬浮入口。
- 新注册账号固定为教师，不接受学生或管理员角色注册。学生账号及旧学生令牌无法访问教师业务 API。
- 已有教师和管理员账号进入同一教师工作区，管理员原有后台权限保留。
- 本系统账号与学校教务账号分别管理，教务账号在侧栏“资源与设置 → 教务账号”中设置。
- 移除试卷页面、HTTP 接口、Skill、评分工具考试参数、试卷队列处理及 Python 试卷评分实现。
- 历史数据库表和 protobuf 消息定义保留兼容，不删除历史数据。旧试卷 RPC 返回 UNIMPLEMENTED，新进程仅订阅作业批改队列。
- 作业查重、AIGC 检测、文档解析、代码文档匹配、教师复核、成绩池化和结果导出继续保留。
- 工具与技能配置前端已移除，旧 `/tools-admin` 和 `/skills-admin` 地址返回工作台；后台业务工具继续服务于 AI 助手。

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

本次改版浏览器回归：`vue-grading-frontend/tests/teacher_redesign_smoke.py` 验证列表筛选、题目标准、班级隔离、切班慢响应、三种材料来源、人工 0 分、分班导出、悬浮助手和移动端；`floating_assistant_isolation_smoke.py` 验证助手历史、执行记录、浏览器控制台和账号切换后的迟到回复隔离。原材料处理/失败恢复、助手快捷操作与执行历史回归继续使用。均拦截业务 API，不触发真实批改或发送验证码。旧界面冒烟脚本保留作历史交互参考，部分页面选择器与新版不同。`TestAssignmentWorkspaceSummaryAndOwnership` 及班级集成测试在配置隔离 MySQL DSN 后验证教师数据与班级范围隔离。

## 源码入口

| 内容 | 位置 |
| --- | --- |
| 工作区路由与导航 | `vue-grading-frontend/src/router/index.js`、`src/services/teacherWorkspace.js` |
| 作业列表与内容详情 | `vue-grading-frontend/src/views/AssignmentsListView.vue`、`src/views/AssignmentDetailView.vue` |
| 新建与自动批改表单 | `vue-grading-frontend/src/views/CreateAssignmentView.vue`、`src/views/HomeworkGradingView.vue`、`src/components/RubricEditor.vue` |
| 作业状态和人数统计 | `backend-go/internal/handlers/assignment_summary.go` |
| 教师账号权限 | `backend-go/internal/handlers/auth_handler.go`、`internal/middleware/auth.go` |
| Agent 与工具 | `backend-go/internal/agent/`、`internal/browseragent/` |
| 下载、整理、批改衔接 | `backend-go/internal/portal/workflows.go`、`internal/rpa/service.go`、`internal/grading/materials.go` |
| 作业计算 | `backend-go/internal/tools/homework_process.go`、`ai_engine_python/app/rpc/compute.py` |

完整版本保存在 Git 历史中，可与拆分前快照比较。本分支不需要额外的版本开关。
