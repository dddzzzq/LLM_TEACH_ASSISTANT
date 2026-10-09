# Tool 与 Skill 管理规范化方案

日期：2026-10-08。本文是拟议设计，尚未修改执行逻辑。

范围：定义、发现、加载、调用、版本、运行记录和对外分发。教师修改全局工具策略、相关角色管理页面和审批流程暂不纳入；沿用现有身份与业务归属校验。

## 1. 设计依据

Codex 的 Skill 以 `SKILL.md` 及可选脚本、参考资料组成，先提供名称和描述，选择后再加载正文。我们采用这种渐进加载方式，把运行时扩展字段放到独立描述文件，保持核心 Markdown 可迁移。[OpenAI 官方 Skill 文档](https://learn.chatgpt.com/docs/build-skills)

Claude Code 区分用户调用与模型自动调用，也提供 Skill 级工具配置。其 `allowed-tools` 有特定的免提示许可语义，并非工具白名单。我们借鉴调用方式可声明的设计，不直接把该字段解释为教学系统的业务授权。[Claude Code Skills](https://code.claude.com/docs/en/skills)

工具服务和任务知识可独立部署，再作为插件一起交付。Codex 与 Claude Code 均提供包含 Skill 和 MCP 配置的插件机制，但宿主清单和字段存在差异。我们维护一个模块源，通过宿主适配生成发行包。[OpenAI 插件打包](https://developers.openai.com/plugins/build/plugins)、[Claude Code Plugins](https://code.claude.com/docs/en/plugins)

以下接口、目录和字段是本项目的建议规范，不是上述产品已经共同采用的标准。

## 2. 当前基础与缺口

已有基础：

- Go ToolRegistry 已拥有执行器与 JSON 参数协议，数据库配置不再替换代码 Schema。
- 作业与试卷 Skill 已按需加载，批改任务记录了 Skill 内容哈希。
- Python 浏览器动作定义与参数校验共用一份代码协议。
- 下载控制器已有人工接管、导出检查点、文件校验及持久任务。

当前缺口：

- Go SkillCatalog 写死两个批改 Skill；Python 单独读取固定路径的导航 Skill。
- 主助手把“工具需要哪个 Skill”的关系写在 `requiredGradingSkill` 条件分支中。
- Tool 的返回值以字符串为主，结构化回执与错误语义不一致。
- 加载 Skill、依赖检查、任务提交及完成报告缺少统一规范。
- Skill 没有统一的安装、启用、依赖就绪和版本发布视图。
- 当前 ZIP 属于源码提取，完整人工界面及宿主接入仍需开发。

## 3. 四项职责和两层运行时

| 对象 | 管理内容 | 示例 |
| --- | --- | --- |
| Skill | 何时使用、判断步骤、工具组合、人工交接、完成依据 | 作业批改指导、页面导航知识 |
| Tool | 可执行名称、参数、输出、超时及执行器 | 创建下载任务、查询批改状态、点击控件 |
| Agent Runtime | 可用能力选择、上下文、模型调用、工具分发及预算 | Go 主助手、Python 页面 Agent |
| Workflow / Controller | 长任务状态、业务校验、检查点、副作用、恢复 | 官网导出下载、作业评分流水线 |

插件是以上资源的安装和分发单位，不再引入一层业务决策逻辑。

两层 Agent 保持各自循环：`teaching-assistant` 处理教师需求与任务入口，`portal-browser` 处理页面导航。统一它们的 Skill 元数据、Tool 描述和运行事件；不要求把两个语言的循环合并成一个进程。

## 4. Tool 定义规范

引入一个代码拥有的 `ToolDescriptor`，供模型协议转换、管理展示、参数验证和文档生成共用。

| 字段 | 要求 |
| --- | --- |
| id | 全局唯一内部标识，例如 `portal/fetch_homework` |
| name / aliases | 保留现有模型工具名称，旧别名映射同一执行器 |
| description | 明确作用、适用前提和关键副作用 |
| provider | builtin、python-worker、mcp 等实现来源 |
| contract_version | 参数和结果协议版本 |
| input_schema / output_schema | JSON Schema；执行前验证参数，执行后验证返回结果 |
| execution_mode | sync 或 job |
| effects | read、local_write、job_create、external_write 等行为分类 |
| timeout / retry_policy | 明确超时与可重试条件，外部提交不可盲目重试 |
| workflow_binding | 可选的业务流程与 Skill 前置关系 |

内部 ID 不直接成为模型函数名；由适配器生成宿主允许的名字。现有 `fetch_homework`、`start_grading_job` 等名称继续可用，避免改名破坏历史提示与客户端。

同一内部 ID 重复注册时报错，不再静默覆盖。兼容别名显式声明，模型默认只看到一个规范入口。

`workflow_binding` 由代码注册的流程策略拥有。比如 homework 提交绑定 grade-homework，exam 提交绑定 grade-exam；参数先验证，再解析前置关系。Skill 作者不能通过自报依赖绕过这层约束。只对确有前置流程要求的操作使用门控，普通查询无须强制加载 Skill。

## 5. 统一调用上下文与结果

统一 Tool 调用接口的概念模型：

```text
Execute(context, validated_input) -> ToolResult

ExecutionContext:
  actor_id, role, run_id, task_id, invocation_id,
  idempotency_key, skill_bindings, deadline
```

身份和幂等标识由服务端取得，不能由模型输入覆盖。网页/API 和 Agent 工具可以复用业务服务；“模型必须先阅读 Skill”属于 Agent Runtime 的执行规则，不强加到正常网页上传上。

建议返回结构：

```json
{
  "outcome": "accepted",
  "message": "下载任务已创建，请完成官网登录",
  "data": {
    "job_id": "...",
    "job_status": "WAITING_USER"
  },
  "interaction": {
    "kind": "human_browser",
    "reason": "login",
    "url": "/tasks/.../browser"
  },
  "artifacts": [],
  "error": null
}
```

`outcome` 建议统一为 completed、accepted、needs_input、failed。调用受理与后台任务状态分开表达；`accepted` 不能解释为下载或批改完成。错误使用稳定 code、面向用户的 message、retryable 和恢复提示，内部调试信息写入关联事件。

长任务返回任务引用，查询和事件更新继续跟踪；人工输入通过 interaction 交接。调用重试沿用同一个 invocation/idempotency 标识；用户明确新建任务才产生新业务幂等键。跨请求幂等和崩溃恢复需要持久仓储支持，不能只用内存 memoization 声称已经实现。

兼容阶段为旧字符串 Tool 包装结果适配器，逐个迁移；保留旧前端需要的 job_id/job_type 字段一段时间。

## 6. Skill 包与元数据规范

每个 Skill 保持以下目录，按需要添加文件：

```text
skills/fetch-homework/
  SKILL.md
  skill.yaml
  references/
  scripts/
  assets/
```

`SKILL.md` 核心要求 name、description，并在正文解释输入、判断步骤、缺项处理、工具组合及完成条件。不要复制完整 JSON Schema、服务地址、认证密钥或站点选择器。

`skill.yaml` 为本系统扩展配置，示意：

```yaml
api_version: teaching.skills/v1
id: teaching/fetch-homework
version: 1.0.0
entry: SKILL.md
runtimes:
  - portal-browser
invocation:
  automatic: true
  explicit: true
requires:
  tools:
    - id: portal-browser/click
      contract: "^1.0"
    - id: portal-browser/open_export_settings
      contract: "^1.0"
```

这里只展示两个依赖，实际发布必须列全必要工具；工具 ID 和上述扩展字段均是拟议协议。需要声明具体工具依赖、可选依赖和运行时兼容范围。依赖表示运行条件，不授予权限。读取正文后，工具仍受当前用户身份、任务归属、阶段和控制权限制。

脚本是可选配套资源，执行仍经过受控执行工具。安装、扫描和读取 Markdown 不执行脚本或安装钩子。站点适配 Python 代码仍作为 provider 发布，避免每个 Skill 复制一份浏览器驱动实现。

保持 `SKILL.md` 的通用部分可复用。导出时再生成 Codex 的可选 `agents/openai.yaml`、Claude Code 对应字段与插件清单；未知自定义字段不能假定被宿主理解。正文可读不代表执行环境兼容，发行包必须声明支持的宿主与依赖。

## 7. 统一 SkillCatalog 与按需加载

统一目录协议，允许 Go 与 Python 保留不同实现，使用同一批解析和行为样例验证一致性。

```text
发现配置目录 → 校验元数据及资源路径 → 建立索引
→ 检查运行时、版本和依赖 → 提供候选名称与描述
→ 选择 Skill → 加载正文 → 按需读取 references / 调用工具
```

发现目录限于部署配置和已安装包，不递归扫描任意上传文件。正文和参考文件必须留在包根目录内；标识冲突直接报告，不静默按扫描顺序替换。

“已安装”“已启用”“依赖就绪”“允许自动选择”“适用当前 Runtime”分别计算。用户可以看见明确的不可用原因，例如 Worker 未连接、所需 Tool 未注册、版本不兼容。依赖已安装但任务当前处于人工模式，应表现为当前阶段不可执行，不误报为未安装。

目录不应一次加载全部正文。现有规模先做名称与描述匹配，无须先引入向量库。显式指定 Skill 时也必须检查可用性，不能伪装成成功加载后让模型改走其他副作用路径。

一次 AgentRun 固定 Skill 包版本与内容快照。收到正文后再调用受其指导的提交工具，延续现有“同一轮加载并提交不能绕过阅读”的检查。

现有 fetch-homework 是页面导航 Skill，先声明适用于 portal-browser。新增用户入口 `download-homework`，指导主助手创建、跟踪和处理下载任务。不要让同一个名称在不同 Runtime 里隐含两种任务语义。

## 8. 版本、存储与运行记录

采用语义版本和内容哈希并存：版本号供人理解兼容性，哈希标识实际安装内容。哈希覆盖正文、扩展元数据、引用及脚本，并按规范路径顺序计算；不能只计算 SKILL.md 后声称整个包已固定。

发布版本不可覆盖；编辑产生新版本。新任务选择当前启用版本，已有任务固定旧版本。发布回滚改变新任务的版本选择；不强行替换运行中的浏览器任务。工具协议升级需要兼容性检查，已经下线的执行器不能假定仍可执行旧版本任务。

建议保留以下数据职责：

| 存储对象 | 保存内容 |
| --- | --- |
| 代码 / Provider 发布物 | Tool 执行器、输入输出协议及默认描述 |
| 不可变 Skill 包目录 | Skill 文本、参考文件、脚本及内容哈希 |
| tool 配置表 | 已有开关、角色、描述覆盖；沿用当前行为 |
| skill_installations / skill_releases | 来源、版本、内容哈希、安装范围、启用及当前版本指针 |
| agent_runs / agent_events | 运行快照、调用记录、人工交接、任务与结果关联 |

不把 Markdown、执行器映射和策略混入一张大表。第一版可合并安装与发布索引表，包清单完整保存；业务规模需要时再拆表。

事件记录 SkillLoaded、ToolRequested、ToolFinished、HumanRequired、TaskSubmitted、TaskUpdated 等，关联 run_id、job_id、tool_id、协议版本和 Skill 版本。记录经过筛选的摘要和错误，不记录账号密码、验证码、Cookie、浏览器输入正文或模型内部推理。

## 9. 管理界面与 API

在现有管理页面旁增加三个视图：

- **Tools**：名称、provider、协议版本、参数/结果 Schema、执行模式、健康情况、最近失败原因和使用它的 Skill。沿用当前配置功能，本轮不改教师的全局策略管理。
- **Skills**：名称、说明、版本、来源、适用 Runtime、调用方式、工具依赖和就绪状态；支持预览、校验、选择版本、启停和导出。
- **Runs**：任务关联、加载的 Skill、执行的 Tool、人工等待原因、产物和失败位置。

新接口建议使用 `/api/skill-catalog`、`/api/skill-catalog/:id/releases`、`/api/agent-runs/:id`。已有 `/api/admin/skills` 是 Tools 配置兼容别名，保持语义，不直接占用为新 Skill 接口。发布和安装操作使用宿主既有管理身份，本轮不扩展教师权限设计。

管理页的“测试 Skill”首先做解析、依赖和模拟调用验证。真实下载另建有明确目标的任务；不能因为预览或校验一个包就自动访问教务平台或触发评分。

## 10. 对外分发：Skill + Provider

官网下载能力建议作为 `xidian-homework` 模块交付：

```text
xidian-homework/
  skills/download-homework/SKILL.md
  skills/fetch-homework/SKILL.md
  provider/                 # 浏览器、控制器、下载校验
  ui/                       # 独立人工登录和班级选择入口
  adapters/                 # CLI / MCP 与宿主打包适配
  install/                  # 初始化和 doctor 检查
  tests/
```

CLI/MCP 都调用同一执行层，不能分别复制下载业务逻辑。MCP 提供跨宿主工具连接，不能自动解决人工登录界面、任务持久化和平台适配；这些能力仍随 provider 交付。Codex MCP 的服务配置也单独管理连接、超时和工具可用性，说明工具接入与 Skill 文本需要分别处理。[OpenAI MCP 文档](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)

两个运行方式可以渐进实现：

1. embedded：复用当前 DownloadAgent，清楚声明模型配置依赖。
2. host-driven：宿主 Agent 读取观察并提交动作，取消必须额外配置 DeepSeek 的要求。

同一任务只能有一个自动决策驱动方；用户接管继续由同一控制器管理。先完成 embedded 的独立 UI、CLI 和安装链路，再开放 host-driven，不在第一版同时维护两套任务控制器。

对外默认交付“下载并返回 ZIP 产物”；自动批改作为显式选择的宿主集成能力。原教学系统保留下载后批改的既有入口，新的外部分发入口通过明确选项控制接续，避免给普通下载用户强加整个评分系统。

## 11. 分阶段实施与验收

| 阶段 | 改动 | 验收标准 |
| --- | --- | --- |
| 第一阶段：定义与目录 | ToolDescriptor、统一 Skill 元数据、扩展 SkillCatalog、兼容旧接口 | 新增 Skill 无需修改名称数组；未注册 Tool 明确报缺依赖；同名冲突可见 |
| 第二阶段：调用与记录 | ExecutionContext、ToolResult、声明式 workflow_binding、运行事件 | 身份不能由模型覆盖；重试不重复提交；accepted 与完成区分；可查 Skill 和 Tool 版本 |
| 第三阶段：管理与发布 | Skill 预览/校验/版本选择、Tool 健康信息、Runs 视图 | 可解释不可用原因；新旧任务版本隔离；旧 /admin/skills 仍正常 |
| 第四阶段：独立分发 | provider 包、独立人工界面、初始化 CLI、MCP/宿主适配 | 全新环境按说明安装，经人工登录确认班级后取得校验 ZIP，无需完整教学系统 |

第一阶段先提供新接口与旧实现适配器，不先搬动整个目录。现有文件可逐步演进：

- `internal/agent/tools.go` → 增加 ToolDescriptor 与重复注册检查。
- `internal/agent/skill_catalog.go` → 配置目录发现、扩展元数据、依赖解析与版本快照。
- `internal/agent/runtime.go` → 使用统一加载和调用接口，移出工具名称条件分支。
- `app/rpa/agent.py` → 从同一 Skill 清单解析适用的页面 Skill。
- `app/rpa/tools.py` → 导出可验证的 provider 描述，保留现有实时动作校验。
- `ToolsAdminView.vue` → 展示定义和健康信息；新增 Skill 和 Run 视图。

测试分为三类：协议/加载单元测试；模拟模型与浏览器的工作流回归；真实平台、真实宿主的端到端验收。前两类通过不能替代真实官网登录与下载验收。

本次建议的首个实施批次是第一阶段，并定义第二阶段 ToolResult 的兼容适配格式。教师全局策略编辑、插件市场、第三方任意代码安装、复杂多租户授权和重写原批改引擎均不属于这个批次。
