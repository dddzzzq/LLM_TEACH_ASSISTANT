# 源码组织与维护约定

本项目有三个独立应用：Go 业务后端、Python AI/浏览器执行服务和 Vue 前端。整理代码时以保持已有 HTTP、WebSocket、gRPC、数据库和任务状态契约为前提；业务调整应单独说明和验证。

## 推荐阅读顺序

1. [README.md](README.md)：系统边界、运行方式和主要流程。
2. [Go 启动入口](backend-go/main.go) → [路由装配](backend-go/internal/server/router.go) → [认证与权限](backend-go/internal/middleware/auth.go)：先确认请求从哪里进入、谁能访问。
3. [HTTP handlers](backend-go/internal/handlers/) → [grading](backend-go/internal/grading/) / [materials](backend-go/internal/materials/) / [rpa](backend-go/internal/rpa/)：追踪一次操作的业务服务和状态变化。
4. [models](backend-go/internal/models/) → [database](backend-go/internal/database/) / [mq](backend-go/internal/mq/)：核对持久化模型、缓存、任务投递与消费。
5. [gRPC 协议](proto/compute_service.proto) → [Go 客户端](backend-go/internal/grpcclient/) → [Python 请求适配](ai_engine_python/app/rpc/compute.py) → [AI 服务](ai_engine_python/app/services/)：理解计算请求、结果和异常返回。
6. [主助手](backend-go/internal/agent/) → [页面 Agent](backend-go/internal/browseragent/) → [浏览器 Worker](ai_engine_python/app/rpa/worker.py) → [任务控制器](ai_engine_python/app/rpa/task_controller.py)：理解模型决策与浏览器执行之间的边界。
7. [前端路由](vue-grading-frontend/src/router/index.js) → [页面](vue-grading-frontend/src/views/) → [组件](vue-grading-frontend/src/components/) → [services](vue-grading-frontend/src/services/)：从用户操作回到 HTTP 请求。

以材料整理为例，可沿 `MaterialBatchPanel.vue` → `materialsApi.js` → `server/router.go` → `handlers/material_handler.go` → `materials/` 追踪；主动批改再进入 `grading/` 和 Kafka 消费流程。

## 各层职责

| 位置 | 职责与约定 |
| --- | --- |
| `backend-go/main.go` | 初始化依赖、启动后台任务、启动 HTTP；保持初始化顺序明确。 |
| `backend-go/internal/server/` | 统一注册 HTTP 路由、中间件与静态资源；构造路由不启动数据库或后台任务。 |
| `backend-go/internal/execution/` | 同步保存助手执行事件，从事件重建结构化上下文及聊天展示；查询和重建不执行业务工具。详见 [执行记录设计](docs/execution-records.md)。 |
| `backend-go/internal/portal/` | 教师凭据保管、本机一次性认证通道、官网目录快照与选择式批改流程。参见 [教务接入部署说明](docs/teacher-portal-automation.md)。 |
| `backend-go/internal/handlers/` | 解析 HTTP 输入、调用业务能力、转换响应；文件使用 `snake_case.go`。 |
| `backend-go/internal/models/` | 按用户、会话、作业、历史考试数据、异步任务、工具、材料和 RPA 分文件；仍属于同一 `models` 包。 |
| `backend-go/internal/tools/` | 文件过滤、文档解析、作业流水线、响应转换和成绩池化；避免继续添加含糊的通用辅助文件。 |
| `ai_engine_python/app/grpc_server.py` | 保留原启动命令，负责模型加载、日志和 gRPC 服务启动。 |
| `ai_engine_python/app/rpc/` | 协议导入和请求适配；依赖显式传入，不在导入时加载模型权重，便于隔离测试。 |
| `ai_engine_python/app/services/` | OCR、查重、AIGC 和模型计算。模型导入顺序涉及本地运行时，不能随意重排。 |
| `ai_engine_python/app/rpa/` | 浏览器操作、下载验证、任务控制和 Worker API。 |
| `vue-grading-frontend/src/services/` | HTTP/流式协议；组件保留交互状态、错误提示与生命周期，接口路径集中维护。 |
| `proto/`、Go `pb/`、Python `app/pb2/` | 协议源及生成结果；不手改、不格式化生成代码。 |
| `skills/` | Markdown 任务知识包，与可执行工具注册及权限配置分开。 |
| `scripts/`、`docs/`、`archive/` | 部署/检查脚本、设计文档、未参与构建的历史快照。 |

`app/services/grading_service.py` 和 `langgraph_agent.py` 是历史实现，当前 Go/Eino 主链路不调用它们；暂保留原路径，避免未经核实删除历史入口。旧 `/api/admin/skills` 路由、`skillsApi.js`、`SkillDefinition` 类型别名和 `skill_definitions` 表名仍承担兼容职责。

## 格式与文件管理

- 编辑器遵循根目录 `.editorconfig`；Go 使用 `gofmt`，Python 使用 Ruff 0.14.1，前端使用锁定的 Prettier 3.6.2。
- 前端配置保留分号，避免 Vue 模板中的多语句事件处理器在格式化后丢失语句分隔；使用严格 HTML 空白规则，减少模板文本间距变化。
- 不手改依赖锁文件中的版本与校验信息；新增开发工具时保留运行依赖版本。
- 新文件按职责放入已有业务模块；外部接口、模型字段名与数据库标签不跟随内部文件名调整。
- `.env`、账号凭据、上传附件、模型缓存、测试产物与运行日志保持在忽略目录；历史界面只存入 `archive/`，不放入 `src/`。
- 纯整理不调整提示词、评分阈值、错误降级语义、任务重试/确认行为或数据库结构；这类变更需要独立回归依据。

在已安装 Go、uv 和 Node 的环境中，先安装前端锁定依赖，再检查或应用格式：

```bash
cd vue-grading-frontend
npm ci
cd ..
python scripts/format-code.py          # 只检查
python scripts/format-code.py --write  # 写入格式化结果
```

脚本可从任意目录执行。Python 格式化仅覆盖 `app/` 的手写代码和本次新增的 RPC 契约测试；既有测试及部署脚本未做全量格式迁移。生成代码和历史界面快照不参与格式化。

本仓库配置的 WSL 环境可先执行 `source scripts/wsl-env.sh`，再使用 `/opt/teaching-runtime/python-env/bin/python` 运行 Python 命令。不要为了代码检查启动或重启正在处理真实任务的业务服务。

## 验证命令

以下命令从对应目录运行：

```bash
# backend-go/
go test ./...
go vet ./...

# ai_engine_python/（项目 Python 环境，浏览器测试需安装 Chromium）
python -m unittest discover -s tests

# vue-grading-frontend/
npm test
npm run build

# 仓库根目录：部署进程管理的纯测试
python -m unittest discover -s scripts/tests
```

Go 中依赖测试数据库的用例在未提供相应环境变量时会跳过；“单元测试通过”不等同于实际数据库、Kafka 和模型端到端验证。新增 RPC 测试通过本机临时端口进行真实 protobuf/gRPC 往返，AI 服务使用替身，不调用付费模型或下载权重。

前端浏览器冒烟测试拦截业务 API，使用 fixture 数据。先在单独终端启动测试用 Vite：

```bash
cd vue-grading-frontend
npm run dev -- --port 6017 --strictPort
```

再从仓库根目录执行，完成后停止这个测试用 Vite 进程：

```bash
RPA_UI_BASE_URL=http://127.0.0.1:6017 python vue-grading-frontend/tests/materials_smoke.py
RPA_UI_BASE_URL=http://127.0.0.1:6017 python vue-grading-frontend/tests/materials_recovery_smoke.py
```

`backend-go/internal/server/testdata/routes.json` 是整理前从真实路由注册代码采集的契约快照，测试同时覆盖保护路由的未认证访问与代表性角色权限。新增接口时应明确检查差异后更新快照，不能为了让测试通过而直接覆盖。
