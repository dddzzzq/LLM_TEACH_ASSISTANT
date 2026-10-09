# 官网作业下载：人工协作 Agent

入口为「AI 教学助手」对话区，快捷指令提供「从教务系统下载作业并批改」和「查询学生成绩」。对话创建下载任务后，右侧自动弹出作业控制台并选中该任务；可收起控制台，之后从任务消息或「打开作业控制台」按钮重新进入。收起只停止前端浏览器连接与状态轮询，后台任务继续执行。目标固定为西电教学平台，课程、作业名称及可选学期由用户提供；每次导出必须确认班级范围。

## 模块职责

| 模块 | 实现与职责 |
| --- | --- |
| 对话工具 | `backend-go/internal/agent/tool_fetch_homework.go`：创建、查询、暂停、恢复、取消本人任务，不接收官网密码 |
| 浏览器 Tool | `ai_engine_python/app/rpa/tools.py`：代码定义动作及参数协议，模型提示与执行校验共用；控制器继续核验实时页面及业务范围 |
| 页面 Skill | `skills/fetch-homework/SKILL.md`：业务步骤、观察引用使用原则和请求人工的判断依据，由 Go/Eino 按需加载，下载任务固定版本快照 |
| 页面 Agent | `backend-go/internal/browseragent/agent.go`：通过 Eino 加载导航 Skill、调用模型并返回一个结构化动作 |
| 平台适配器 | `portal_adapter.py`：观察多页面与 iframe、验证目标、读回班级勾选、提取稳定导出记录；保留旧脚本的站点规则 |
| 任务控制器 | `task_controller.py`：独立浏览器生命周期、控制权切换、恢复检查、导出防重、下载调度 |
| 文件管理 | `download_manager.py`：临时文件、传输停滞检测、ZIP 非空及 CRC 校验、SHA-256、原子保存 |
| 浏览器 Worker | `worker.py`：独立进程，内部 HTTP 控制 API、任务快照和超时清理，不加载 Skill、不调用模型，也不加载 OCR 或批改模型 |
| Go 调度器 | `backend-go/internal/rpa/`：创建者隔离、数据库状态镜像、派发重试、附件导入、评分标准门控、独立批改任务 |
| Vue 控制台 | `RpaTaskPanel.vue`、`RpaBrowserPanel.vue`、`RpaClassSelector.vue`：任务状态、远程键鼠、班级选择、导出记录确认与批改进度 |

## 执行与人工接管

1. Go 保存 `RPAJob` 和异步任务。Kafka 只传任务 ID；数据库中的排队记录也由调度器重试，人工等待不会占用 Kafka 消费线程。
2. Worker 打开任务独立的 Chromium，进入 `AUTHENTICATE/HUMAN`。用户直接操作同一个浏览器完成完整登录与验证码。等待期间查询任务会刷新登录状态；检测成功后显示「已登录，等待继续」，仍需用户点击「交还自动化」才会继续执行。
3. Go/Eino 页面 Agent 依据可见控件选择点击、搜索、滚动、切换页面或请求人工。Go 经内部 HTTP 获取观察，持久保存动作后提交 Worker；观察 ID、revision 和 control_epoch 共同验证。每步重新观察；无效动作、重复动作、目标歧义或预算耗尽会暂停。认证挑战出现时重新交给用户。
4. 控制器核验目标，读取班级列表。用户确认具体班级或全部，控制器应用选择并读回核验；不默认选择全部。
5. 导出前持久化已有记录和提交意图。提交后只跟踪本次新增的记录，无法可靠关联时要求人工勾选；不会按行号猜测或自动重复导出。用户手动点击导出确认也会记录检查点。
6. 下载完成校验 ZIP，再匹配本地作业。班级无法判断时允许更正；题目或评分标准缺失时保留附件并等待。标准齐备后创建稳定的批改任务 ID，投递原有 Kafka 批改流水线。

自动动作期间可请求暂停。先关闭执行门，再等当前浏览器动作结束；模型调用的迟到结果通过 `control_epoch` 丢弃。人工输入校验控制权、页面及画面版本，防止旧画面事件在恢复后执行。文件传输可在暂停期间继续，后续页面动作等待用户恢复。

## 持久化与恢复

- Go 在 `rpa_jobs` 中保存导航 Skill/动作协议快照、待确认动作和决策租约。模型决策不持有数据库事务；任务间可并行，单任务每次只有一个决策持有者。Worker 持久保存动作摘要与回执，超时后重发同一动作 ID，不重复执行。
- MySQL 使用 `rpa_jobs`、`rpa_files`，由现有 GORM `AutoMigrate` 创建。阶段、已选范围、导出引用和事件保存在状态 JSON；事件不是独立事件表。
- Worker 在 `$RUNTIME_DIR/rpa-tasks/` 原子保存权限为600的任务快照；下载默认在 `downloads/<job_id>/`。Go/Python 需共享文件路径。
- 刷新前端会重新查询本人任务。Worker 重启后未结束任务变为 `INTERRUPTED`；浏览器和 Cookie 不持久化。点击重试后需要重新登录，保留已校验文件及导出检查点。数据库快照也可用于恢复丢失的 Worker 记录。
- 已提交导出的任务重试时先导航到原作业和下载中心，不重新导出。已校验文件跳过，失败文件可以重试。导出记录被平台清理等无法恢复的情形需要人工处理。
- 下载状态与批改状态分开；`SUCCEEDED` 表示下载校验完成，批改完成以附件行的 `SUCCESS` 为准。

账号密码、验证码、表单值、Cookie 和签名下载链接不写入任务日志，也不提供给页面模型。模型不接收截图，只接收经过筛选的页面控件。浏览器画面和人工输入经 Go 的 JWT、教师/管理员角色和创建者校验转发；内部 API 只绑定本机并验证共享令牌。

## 人工浏览器响应速度

任务浏览器通过 `/api/rpa/jobs/:id/stream` 建立 WebSocket。连接后第一条消息发送 JWT，令牌不会进入 URL；Go 校验 Origin、令牌有效期、教师/管理员角色及创建者，再通过内部共享令牌连接 Worker。连接不能延长 JWT 有效期。前端在重连前通过现有 HTTP 认证流程检查登录，必要时刷新令牌。

键鼠事件使用 12ms 合并窗口，每条消息最多包含 32 个按序执行的动作。连续文字和鼠标移动合并，按下、释放及 Tab 等离散事件保持顺序；下一条消息不等待上一条确认。Worker 使用独立接收和执行协程，按连接中的递增 ID 返回确认。双方限制队列与传输积压，失败后丢弃未确认输入，不自动重放。切换任务、控制权、隐藏页面或卸载组件会关闭连接并清理待发送输入；断线会释放该连接按下的鼠标，关闭其他查看窗口不会影响正在拖动的连接。

失焦、指针取消和断线清理会先将鼠标移出画面再释放，避免将取消动作变成按钮点击；正常释放仍在用户指定的位置执行。

Worker 为每个任务共享一个 Chromium CDP screencast，推送 JPEG 画面（质量 55、1440×1000，发送上限约 30 帧/秒）。拖动时继续更新画面；慢连接只保留最新待发送帧。静态页面约每秒补一帧，保持点击元数据有效。页面或控制权变化后重新建立对应的画面流，捕获和输入不共用长时间锁。保留最近 128 帧的元数据，仅接受 3 秒内且页面、URL、控制 epoch 一致的按下事件。实际帧率和操作延迟取决于网络、官网与 Chromium，不承诺固定公网延迟。原 HTTP 画面和输入接口仍兼容。

开发环境 Vite 已启用该路径的 WebSocket 转发，并保留 Host 用于 Origin 校验。生产反向代理也需要转发 WebSocket Upgrade；分离部署前端时，将允许的完整 Origin（含协议和端口）加入 Go 环境变量 `RPA_BROWSER_ALLOWED_ORIGINS`，以逗号分隔。

## 运行与验证

安装 `ai_engine_python/requirements.txt`（含 `websockets==15.0.1`）和 Playwright Chromium 后，在 Python 项目目录运行 `python -m app.rpa.worker`；现有 `scripts/start.sh` 已加入独立 Worker，并检查 `browser.stream.v1` 能力，防止复用缺少流式接口的旧服务。默认端口8765，配置代理后需重启 Worker。更新中的 Go 后端需重新编译并重启，前端需重新构建或由 Vite 热更新。

```bash
cd ai_engine_python
PLAYWRIGHT_BROWSERS_PATH=/root/autodl-tmp/.playwright .venv/bin/python -m unittest discover -s tests -p 'test_rpa_worker.py' -v

cd ../backend-go
go test ./...

cd ../vue-grading-frontend
node --test src/services/browserInputQueue.test.js src/services/browserStream.test.js
npm run build
```

Python 行为测试在真实 Chromium 上拦截模拟站点，覆盖登录交接、班级选择、手动导出防重、真实 ZIP 下载、控制权竞争、密钥不落盘、文件失败和恢复。Go 数据库集成测试仅接受显式提供的 `RPA_TEST_DSN`，数据库名必须以 `rpa_test_` 开头，覆盖创建者隔离、状态版本、重复导入和评分标准门控。

真实教师账号登录后的课程列表、班级控件和下载中心仍需人工验收。平台改版超出已验证规则时会请求接管，不能保证无人干预完成任意页面。

## Tool 与 Skill 边界

Go Tool 创建和控制持久任务；页面 Tool 执行受限的浏览器动作；`fetch-homework` Skill 提供页面导航知识。修改 Skill 不会增加可执行动作或改变工具参数。工具管理使用 `/api/admin/tools`，旧 `/api/admin/skills` 继续兼容，均不用于编辑 Markdown Skill。详见 [边界与迁移说明](tool-skill-boundaries.md)。
