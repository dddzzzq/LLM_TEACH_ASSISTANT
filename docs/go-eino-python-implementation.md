# Go/Eino/Python 架构实施说明

日期：2026-10-08。本次在工作区已有下载、批改和 Tool/Skill 拆分代码上实施，不覆盖此前未提交的业务改动。

## 改动范围与职责

| 实现位置 | 已完成的内容 |
| --- | --- |
| `backend-go/internal/agent/runtime.go` | 主助手应用入口，调用 Eino 运行时 |
| `backend-go/internal/agent/einoruntime/` | 固定 Eino v0.9.21；ChatModelAgent、Runner、原生 Skill 中间件、Tool 及事件适配 |
| `backend-go/internal/agent/gateway.go` | 角色与参数校验、Skill 前置条件、服务端执行上下文、单轮提交控制；业务 Tool 声明自己的流程要求 |
| `backend-go/internal/agent/modelclient/` | 主助手与页面 Agent 共用的模型 HTTP 客户端，保留原调用方兼容入口 |
| `backend-go/internal/agent/catalog/` | 扫描 Skill 包，按 runtime 选择；校验元数据，计算包哈希 |
| `backend-go/internal/browseragent/` | 页面 Agent 加载导航 Skill，按 Worker 发布的 Schema 生成一次动作提案；不持有浏览器 |
| `backend-go/internal/rpa/navigation.go` | 后台页面调度、单任务数据库租约、并发限制、协议快照、动作意图持久化与回执恢复 |
| `backend-go/internal/rpa/service.go` | 下载创建幂等、导航 Skill 快照、任务状态与既有评分接续 |
| `ai_engine_python/app/rpa/task_controller.py` | 外部动作驱动；观察失效检查、控制权、持久动作回执、确定性导出与文件校验 |
| `ai_engine_python/app/rpa/worker.py` | 内部 capabilities、observation、action、decision-failed 接口；不读取模型配置 |
| `skills/download-homework/` | 新增主助手入口 Skill，与页面导航 Skill 分工 |
| Vue 对话与下载面板 | 请求幂等键、重复点击保护；下载创建响应丢失后保留同目标重试键 |

删除 Python `app/rpa/agent.py`。页面 Worker 不需要模型密钥或 Skill 目录；既有 Python 计算服务继续执行 OCR、Embedding、查重和评分。

Eino 具体类型留在适配包内。RPA 业务调度通过 `DecidePage` 函数依赖调用页面决策，由 `main.go` 注入实现，不依赖 Eino 对象。主助手返回已受理任务后，页面任务仍由后台调度推进。

## 浏览器执行协议

内部接口均沿用 `X-RPA-Token` 认证，只在 Worker 内部网络入口使用：

- `GET /capabilities`：返回 `browser.v1` 与从 Python 代码生成的动作 JSON Schema。
- `GET /tasks/{id}/observation`：返回是否可决策、观察 ID/revision/control epoch、目标、页面摘要和近期动作。
- `POST /tasks/{id}/action`：提交相同观察对应的一个动作；包含稳定 action_id、观察版本、控制 epoch 与 Skill 哈希。
- `POST /tasks/{id}/decision-failed`：在当前 epoch 仍有效时交还人工；模型失败或配置不兼容有明确等待原因。

Worker 执行前重新观察页面并比较指纹，拒绝过期元素引用；人工接管使旧 epoch 失效。指纹内部包含完整 URL，但返回模型的 URL 去除查询字符串。

`completed` 表示一个动作已完成，不表示下载任务完成。`unknown` 表示动作结果需要人工核对；动作意图先落盘，重发同一个 ID 只返回既有回执。导出提交与 ZIP 完成仍由确定性控制器核验。

## 持久化与恢复

`RPAJob` 通过现有 AutoMigrate 增加：

- `navigation_json`：导航 Skill 正文、发布号、内容哈希及动作协议快照。
- `pending_action`：发送前保存的动作请求，Go 重启后优先核对该动作。
- `decision_owner`、`decision_lease_until`：后台任务决策租约；过期或非持有者不能写入/清除动作意图。

模型调用期间不持有数据库事务。单次决策有 90 秒预算、2 分钟租约；当前最多并发处理 4 个页面任务。浏览器 Worker 仍限制会话数。

创建任务支持 `Idempotency-Key`，作用域包含创建用户；同键同目标返回原任务，同键不同目标拒绝。首次创建将 Skill 固定到任务，后续目录修改不会影响该任务。聊天请求键通过工具上下文传播到下载与批改提交。

Go 丢失动作响应时保留待确认请求，后续只重发原请求；确认回执后才允许下一次决策。Worker 重启后任务变为 INTERRUPTED，需用户重启并重新登录，已提交导出关联、有效文件和动作回执继续保留。

## 配置与升级

`AGENT_MODEL/AGENT_MODEL_URL/DEEPSEEK_API_KEY` 配置 Go 默认模型；`PAGE_AGENT_MODEL/PAGE_AGENT_MODEL_URL/PAGE_AGENT_API_KEY` 可独立覆盖页面决策。URL 为完整 Chat Completions 入口。Worker 不再使用 `RPA_AGENT_MODEL`。

`TEACH_SKILLS_DIR` 指向 Skill 根目录，启动脚本默认设置为项目 `skills/`。每个包可用 `skill.yaml` 声明 `version` 与 `runtimes`，正文仍保存在 `SKILL.md`。

Go 与 Worker 应配套更新。启动脚本在启动或复用 Worker 后会使用内部令牌检查 `/health`，要求 `browser.v1` 和 `go-eino`，不兼容时终止启动。启动脚本仍会复用已在运行的 Go 服务，因此升级应先妥善结束旧浏览器任务，执行 `bash scripts/kill.bash`，再执行 `bash scripts/start.sh`，让 Go 重新编译、Worker 配套重启。重启后的浏览器登录需人工重新完成。

当前下载控制台继续使用已有任务轮询、人工操作和结果展示。设计文档中的通用 AgentRun/SSE 接口、Skill 发布管理界面、独立 CLI/MCP 发行包和通用 Embedding RPC 属于后续扩展，不在本次核心架构迁移中声称已实现。业务等待人工由持久任务处理；本次没有为聊天轮次新增 Eino 检查点恢复接口。

## 验证

- Go 全量测试与 Agent/页面 Agent/RPA 的 race 检测。
- Python 真实 Chromium 行为测试：人工登录、班级范围、人工导出、ZIP 下载、控制权切换、过期 DOM 拒绝、重复动作与不确定动作恢复。
- 独立临时 MySQL 集成测试：任务归属、状态版本、评分投递、创建幂等、Skill 快照以及租约下的动作意图读写。未连接业务数据库。
- 跨语言闭环测试：真实 Go/Eino、真实 Python HTTP Worker、真实 Chromium，模型和官网使用本地 fixture；导出次数为一次并验证产物文件与 SHA-256。
- Vue 生产构建。

跨语言测试命令（项目根目录执行，路径按环境调整）：

```bash
cd backend-go
RPA_BROWSER_PYTHON=/root/autodl-tmp/LLM_TEACH_ASSISTANT/ai_engine_python/.venv/bin/python \
PLAYWRIGHT_BROWSERS_PATH=/root/autodl-tmp/.playwright \
go test ./internal/rpa -run TestEinoThroughPythonWorkerDownloadsVerifiedArchive -count=1 -v
```

跨语言测试未调用真实模型 API、未登录真实官网。实际模型的导航表现、真实课程页面与官网导出仍需教师账号验收。
