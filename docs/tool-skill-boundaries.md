# Tool 与 Skill 的职责边界

## 当前实现

| 概念 | 定义 | 实现位置 |
| --- | --- | --- |
| Agent | 根据目标、观察和工具结果作出决策 | Go 对话入口；Python `DownloadAgent` |
| Tool | 代码注册的可执行能力，提供名称、描述、参数协议和执行器 | Go `internal/agent/tools.go`、`tool_*.go`；Python `app/rpa/tools.py` |
| Skill | 提供某类任务的操作知识、判断依据和协作方式 | `skills/{fetch-homework,grade-homework,grade-exam}/SKILL.md` |
| 工作流与控制器 | 推进业务状态，核验范围、控制权、导出关联和下载结果 | RPA 控制器、Go 批改流水线 |

Skill 中提到一个动作不会注册该动作，也不会赋予身份、权限或任务控制权。工具的执行边界始终由代码校验。

Go 的 `internal/tools/` 目前保留原有业务实现（批改流水线、解析等）；它与 `internal/agent/` 下的 LLM Tool 适配器职责不同。

## Go 工具目录与配置

`NewBuiltinToolRegistry(userID, role)` 是聊天与管理接口共用的内置工具目录。聊天入口注入认证用户 ID 和角色；管理接口使用零 ID 只读元数据，不执行用户操作。

- 工具名称、执行器、JSON Schema：以注册代码为准。
- 启用开关、允许角色、描述覆盖：从 MySQL / Redis 读取。
- 配置中没有代码执行器的工具不会暴露给模型；没有注册表时拒绝构造工具列表。
- `impl_key` 是历史元数据，不根据它动态创建执行器。
- 管理接口返回当前执行器的 Schema；编辑参数 Schema 或替换执行器会返回 HTTP 400。旧客户端回传未修改的只读字段仍可保存。

现有聊天执行权限检查、学生查询范围覆盖和 RPA 任务归属检查继续生效。工具配置不是 Markdown Skill 的存储位置。

## 浏览器动作与导航知识

`app/rpa/tools.py` 定义浏览器动作的名称和参数类型，生成模型可见的动作协议，并供 `DownloadAgent` 和 `Task.execute` 校验输出。未知动作、缺少参数、额外参数及类型错误在操作浏览器之前拒绝。

`SKILL.md` 只解释如何定位课程和作业、如何消歧、何时请用户协助，以及导航阶段的完成依据。动作参数格式和运行时约束由代码提供。

Schema 校验之后，`Task` / `PortalAdapter` 继续检查观察引用、页面域名、搜索字段、作业身份、班级范围及人工控制版本。工具协议正确不意味着业务操作自动获准。

主助手与页面 Agent 均使用 Go/Eino 和统一 SkillCatalog，先提供元数据，再通过 Eino 的 load_skill 加载正文。主助手加载 download-homework、grade-homework、grade-exam 等适用 Skill，页面 Agent 加载 fetch-homework。下载任务固定导航 Skill 正文与内容哈希，Python Worker 只提供执行工具。参考文件按需读取尚未实现。

## 批改 Skill 与统一任务入口

`grade-homework` 和 `grade-exam` 规定目标消歧、配置补齐、材料来源、提交及结果解释。`inspect_grading_target`、`start_grading_job`、`get_grading_job` 执行对应操作。`load_skill` 是主助手内部的只读加载工具，不属于数据库中的业务工具配置。

主助手最多执行八轮模型调用，保留原生 assistant/tool 消息；提交批改之前强制要求模型已在前一轮收到相应 Skill 正文。聊天工具、旧 `trigger_async_pipeline` 和网页上传均使用 `internal/grading` 校验并创建持久任务，再交给 Kafka。RPA 仍保留原有独立 outbox 和任务恢复机制。

详见 [批改 Skill 实现与边界](grading-skills.md)。

## 兼容与上线

- 新管理 API：`GET /api/admin/tools`、`PUT /api/admin/tools/:name`、`POST /api/admin/tools/cache/refresh`。
- 原 `/api/admin/skills` 三个接口为同处理器、同权限的兼容别名，不改作 Skill 知识包接口。
- 前端页面使用 `/tools-admin`，原 `/skills-admin` 保留路由别名及原命名路由；旧 `skillsApi` 导入委托给 `toolsApi`。
- 前后端分批更新时，新路由返回 Gin 默认的“路由不存在”404，前端才回退旧接口；参数、权限和资源错误不回退重试。
- `ToolDefinition.TableName()` 继续使用 `skill_definitions`，不重命名或复制数据库表。
- Redis 继续使用原策略缓存键，新旧接口刷新同一份缓存。
- `skill_compat.go` 中保留已弃用 Go 类型和函数别名，新业务代码使用 Tool 命名。
- `fetch_homework`、`fetch_and_grade_homework` 及其他模型可调用名称保持兼容；两个下载入口均在附件核验、评分配置齐备后继续原有批改流程。

旧数据库 Schema 不再决定模型收到的参数协议。如曾通过管理页定制 Schema，需要将参数变更实现到对应工具代码。已有启用状态、允许角色和描述覆盖继续使用。

Go 与 Python 的改动在相应服务重启后生效。浏览器 Worker 重启会终止在内存中的浏览器会话，应在任务结束或安排好恢复后更新；本次代码整理不自动重启服务。

## 后续接口演进

`Tool.Execute` 暂时沿用字符串输入输出，新批改工具支持可选的 `ExecuteContext` 并返回 JSON。有限多轮调用、批改持久任务入口和 Skill 版本记录已实现；通用的强类型 ToolResult、完整 AgentRun 事件存储和跨任务追踪仍待独立改造。

## 验证

- Go：在 `backend-go/` 执行 `go test ./...`。新增用例覆盖配置不能替换执行器 Schema、角色与开关过滤、新旧管理接口认证、旧客户端只读字段回传以及数据库表名兼容。
- Python：在 `ai_engine_python/` 执行 `.venv/bin/python -m unittest discover -s tests -v`。先将 `PLAYWRIGHT_BROWSERS_PATH` 设为启动脚本使用的浏览器目录；当前环境是 `/root/autodl-tmp/.playwright`。
- 前端：在 `vue-grading-frontend/` 执行 `npm run build`。另通过拦截 API 的浏览器检查验证新页面、旧页面别名、旧后端回退和资源错误不回退；测试不读写真实管理配置。

浏览器回归使用本地拦截页面及模拟决策，不代表已验证最新真实官网或在线模型的成功率。需要独立 MySQL 的已有集成测试仅在配置 `RPA_TEST_DSN` 时运行。
