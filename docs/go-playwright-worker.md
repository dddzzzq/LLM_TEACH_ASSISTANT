# Go Playwright 浏览器执行迁移

日期：2026-10-31。

主助手和页面 Agent 继续使用现有 Go/Eino 运行时。主助手注册业务 Tool，提交任务后返回 `job_id`；RPA 调度器继续通过 `DecidePage` 驱动页面 Agent，每次产生一个动作。浏览器执行由独立的 `backend-go/cmd/browser-worker` 进程承担，不将浏览器动作注册到主助手，也不新增 Eino 父子 Agent 运行树。

## 执行边界

- `internal/browserworker` 固定 `github.com/mxschmitt/playwright-go v0.6201.2-0.20260914222347-775648cd1805`，对应 Playwright 1.62.1。该上游提交修复了 CDP 下载事件的 `guid` 被误识别为内部对象、进而关闭整个连接的问题，不能回退到有该问题的 v0.6201.1。使用 Chromium 的新无头模式（`channel=chromium`）。Go 包、Node 驱动和浏览器需配套安装。
- `go.mod` 使用 `third_party/playwright-go` 的本地替换，保留上游许可及版本来源。仅修复新标签页初始化与首个请求之间的 `frame.page` 并发访问：通过原子指针在完整初始化后发布页面。修补说明见 `third_party/playwright-go/LOCAL_PATCH.md`，上游修复且真实浏览器竞争检测通过后可移除替换。
- HTTP 控制端口仍默认 `127.0.0.1:8765`。保留 `browser.v1`、`page.snapshot.v2`、`browser.stream.v1` 和原任务、观察、动作、人工接管接口。健康检查增加 `execution_driver=go-playwright`。
- Worker 不调用模型、不读取导航 Skill。页面 Agent 继续从 `/capabilities` 获取动作协议。Go Worker 内的动作协议、DOM 快照、课程与作业提取脚本来自现有执行协议；JavaScript 仅在受控页面中执行。
- 保留独立浏览器会话、配置代次、凭据版本 fencing 和 `.lease` 锁。凭据仍通过已有 Unix socket 的一次性票据获取，不进入模型、任务 JSON 或动作参数。
- 保存动作 UNKNOWN 回执后才操作浏览器；重复动作 ID 返回原回执。导出意图在确认前保存，下载只绑定唯一且经过新旧记录与班级范围核验的记录；ZIP 逐文件 CRC 和 SHA-256 校验通过后才标记完成。
- 画面由 CDP screencast 产生，通过现有 WebSocket 转发；人工输入继续校验控制代次、页面和画面版本，断开连接时释放该连接的拖动状态。
- 门户首页的公开课程链接和“个人空间”导航不作为已登录证据。可见登录入口优先进入统一认证；门户需存在账号或退出登录控件，超星课程页则使用其课程控件核验。认证状态读取不切换当前操作页，旧门户标签的登录按钮也不会打断另一已认证课程页。
- Python 计算服务继续承担 OCR、模型计算和评分。滑块图像识别恢复调用原 Python/OpenCV 算法，Go 保持浏览器和鼠标操作的所有权。旧 Python 浏览器文件及其测试暂保留，用于历史兼容和行为对照，不再由默认启动脚本启动；旧的计算 RPC 抓取入口属于历史兼容接口。

## 安装与启动

在已加载项目 Go/Node 环境的 Linux/WSL 中：

```bash
source scripts/wsl-env.sh
cd backend-go
go run ./cmd/browser-worker --install
go build -o "$RUNTIME_DIR/browser-worker" ./cmd/browser-worker
```

`scripts/setup-wsl-app.sh` 已加入上述依赖安装和编译。`scripts/start.sh` 会编译并启动 Go Worker，校验执行驱动，防止误复用旧 Python Worker。`service_process.py` 同时识别新旧 Worker，仍按已核实的进程身份停止服务。

部署时需先结束或安排恢复现有浏览器任务，再停止旧 Worker 并启动新 Worker；不能让两个进程同时使用同一运行目录或浏览器配置。迁移代码本身不重启正在处理教师任务的服务。浏览器活对象不会跨进程恢复；重启后任务保留导出、文件和动作回执检查点，状态转为 `INTERRUPTED`，需重新启动任务并核验登录。

## 验证

```bash
source scripts/wsl-env.sh
cd backend-go
RPA_GO_BROWSER_TESTS=1 go test ./internal/browserworker -count=1 -timeout=180s
RPA_GO_BROWSER_TESTS=1 go test -race ./internal/browserworker ./internal/browseragent ./internal/rpa -count=1 -timeout=180s
go test ./...
(cd third_party/playwright-go && go test -race ./...)
```

2026-10-31 已通过 WSL 全量 Go 测试、真实 Chromium 回归、Worker/页面 Agent/调度器竞争检测、本地依赖修补竞争检测，以及独立运行目录下的 Worker 编译、鉴权接口、启动停止和锁释放检查。启动脚本 Bash 语法检查及服务进程测试也已通过。

浏览器回归使用真实 Go Playwright/Chromium、本地拦截的页面夹具和模拟模型；未使用真实教师账号或调用在线模型。覆盖登录和动态码、滑块、同名课程、逐份题目资源、五种导出页面、下载校验、重复动作、人工导出立即恢复、启动中接管、过期输入和流连接断开。真实账号联调尚未执行，数据库集成测试仍由原来的显式测试数据库环境变量控制。

## Python 行为兼容核对

本轮以工作区现有 Python 代码及其回归场景为依据，保留主助手、页面 Agent、后台调度和前端接口。下面记录需要延续的行为及 Go 中补齐的位置；不沿用已确认有误的“公开课程/个人空间即已登录”判断。

| 范围 | Python 依据 | Go 实现及核对结果 |
| --- | --- | --- |
| 启动与网络恢复 | `browser_startup.py` | `startup.go`：每轮 4 次、最多 2 轮，单次 25 秒、总预算 420 秒；退避 5/15/30 秒，轮间冷却 60 秒。仅瞬时网络错误、超时、408/429/5xx 重试；普通 4xx、证书错误等交还人工。等待延迟入口及登录表单，已就绪页面不重复加载，详情课程入口失败后回退门户。 |
| 启动期间接管 | `BrowserStartup`、`Task.pause/resume` | 导航 RPC 不持有任务操作锁，暂停、画面及关闭仍可处理；控制代次变化后停止后续导航，保留教师接管原因。启动失败保留浏览器，恢复使用同一任务和新预算。错误只记录阶段、次数、HTTP 状态、网络错误码等受控信息。 |
| 凭据与身份 | `portal_auth.py`、`portal_sessions.py` | 使用原 Unix socket 一次性票据；HTTPS 登录来源和表单 action 均须获授权，规范化默认 443 端口。等待账号失焦验证检查再提交，不重复填写账号密码。身份读取限教学平台，人工确认不能覆盖明确的账号不匹配。 |
| 动态码与设备信任 | `PortalAuth.request_code/verify_code/trust_prompt` | 等待延迟出现的 `getDynamicCode`，优先官网对应控件；保留倒计时、禁用、发送结果未确认及提交失败原因。验证码仅在 Worker 内读取，提交后保留输入并观察异步结果，同码不重复提交，更换被拒绝的码后可以继续。沿用保存的设备信任偏好，不更改认证渠道。 |
| 交还与学期确认 | `Task.resume`、`target_confirmation.py` | 仅 HUMAN 模式可交还，校验空验证码、未完成手动登录、待确认候选及范围。人工切换学期后只读当前选项，保存原目标和确认历史；学期变化拒绝旧候选。 |
| 目标与详情导航 | `PortalAdapter.reconcile/record_navigation`、详情导航协调 | 排除隐藏及列表行标题，作业标题还需同页导出入口；核对官网课程/班级/作业 ID。详情只绑定一个页面，延迟弹窗期间不派发重复点击；同页重复切换及连续无进展切换有边界。 |
| 目录采集 | `portal_catalog.py`、`catalog_viewport.py`、`homework_dom.py` | 保留 DOM/脚本入口真实 ID、同名课程、iframe/内部滚动和分页覆盖；保留累计列表覆盖及班级并集。未完整读取时继续采集，提前完成不会消耗连续失败预算。返回原列表后重新核验，包含 URL 未变但内部内容已切走的情况。保留 500 门课程、每门 2000 个作业/500 个班级、总计 10000 个作业及 6 MiB 目录限制。 |
| 题目资源 | `homework_detail.py`、`Task.read_detail_resources` | 保留原文/图片/附件，排除学生答案和答案解析，核验所选 ID 及标题。空或加载中详情最多重试 3 次，恢复列表时等待原作业重新出现。单资源 20 MiB、总量 60 MiB、最多 40 项；逐跳校验来源，拒绝登录网页；附件等待有界，暂停不覆盖人工状态。 |
| 导出范围 | `portal_download.py`、`PortalAdapter.apply_scope`、`Task.prepare_export` | 默认全部班级，保留历史明确选择；选项删除或重排后按唯一名称恢复，不猜测重名。保留主复选框联动、命名及附件格式。未提交设置属于错误班级时先取消、切换官网班级，再重新打开和核验，提交前保存意图。 |
| 记录与下载 | `refreshed_export_rows/reconcile_exports`、`download_manager.py` | 绑定稳定身份而非行号/完成时间，全部班级完整后关联。已就绪记录先下载，再处理暂时缺失的其他记录。点击绑定 DOM 保留官网 handler；传输失败检查、非空 ZIP、CRC、SHA-256 后原子保存。保留 120 秒停滞和 1800 秒总传输时限、已成功文件。 |
| 滑块 | `slider_captcha.py`、`Task.execute` | 2026-11-01 按用户要求撤回 Go 图像算法，`captcha_opencv.go` 经本地 stdin/stdout 调用原 `app.rpa.slider_captcha.captcha_distance`。浏览器、画面缩放及 iframe 顶层坐标仍由 Go 控制，拖动恢复原有十步及轻微纵向偏移。原 Python 合成 PNG 向量覆盖透明块、透明噪点和 RGB 块。保留最多三次/时间预算、阶段动作限制及暂停释放鼠标；滑块消失后仍核验登录。 |
| 画面与输入 | `browser_stream.py`、`Task.frame/browser_input` | 保留共享 CDP 流、逐连接有序输入及确认、批量输入、控制代次、页面 URL、3 秒画面和拖动所有者校验。截图在操作锁外执行，控制权变化后拒绝发布；断线释放拖动，文本与验证码不入任务记录。 |
| 重启和回执 | `worker.Manager`、`Task.apply_action` | 保留任务/导出/文件检查点，重启转 INTERRUPTED，重置新会话的旧认证及启动展示状态。兼容 Python 排序 JSON 默认分隔符的摘要，返回原 UNKNOWN/完成回执，不重放动作。已有导出检查点时仍允许页面 Agent 导航回目标，核验归属后继续下载。 |
| 生命周期与协议 | `worker.py`、`portal_sessions.py`、`tools.py` | 保留单实例锁、配置代次/凭据 fencing、连接租约锁及锁 inode、任务配额和空闲期限，内部接口继续鉴权。动作协议保持兼容；导出相关 DOM 脚本按下述中英文规则调整，其他导航和作业提取仍沿用原协议。 |

补充回归位于 `internal/browserworker/startup_test.go`、`compatibility_test.go`、`captcha_test.go` 及既有浏览器测试。`testdata/captcha_python_vectors.json` 由合成图案和原 Python 算法生成，不包含真实账号或真实验证码。真实 Chromium 配合本地页面夹具验证行为边界，不能替代当前网络下的真实账号完整认证。

OpenCV 工具使用启动脚本已有的 `PYTHON` 环境，以及 `ROOT/ai_engine_python` 作为模块目录。只将两张挑战画布通过管道传入本地图像子进程，不传账号、密码、控制令牌或浏览器对象，不落盘图片。单次识别限 5 秒，整轮仍限 20 秒；子进程错误不输出原始异常或输入。`last_captcha_result` 只记录 `python-opencv` 驱动、尝试次数、结果及失败阶段/代码，区分图像读取、OpenCV 依赖、匹配和官网未接受拖动。

从“重新登录并重试”建立新的浏览器会话时，为本轮登录重新建立三次验证预算，不继承已关闭浏览器的失败计数；原浏览器中暂停后“交还自动化”仍保留已用次数。页面 Agent 没有重启任务的动作，不会通过自动重启循环增加尝试次数。

## 中英文导出菜单兼容

2026-11-01 的官网页面显示 `More → Export homework`。Go 导出工具兼容这组英文名称及中文入口；读取观察不点击菜单，只有核验作业后的 `open_export_settings` 才展开唯一的 More。已展开时直接继续，班级切换导致菜单关闭或重建时重新定位；旧页面没有 More 控件时保留原 `ul.morePop` 站内动作兼容。当前操作页优先，父菜单与子链接合并为同一入口，同页两个独立导出入口仍拒绝猜测。失败记录 `last_export_navigation` 的入口和 More 数量，不保存页面正文或签名 URL。

导出设置的范围、附件格式、命名、确认/取消，以及下载中心的入口、空状态、导出状态和下载按钮兼容中英文。无法识别附件格式时停止，不沿用可能导出完整答题记录的默认项。单班级仍按平台 ID 切换和核验；弹窗范围不能扩大为全部班级，提交前保存意图，重复调用不再次提交，下载仍执行原 ZIP/CRC/SHA-256 校验。

`export_navigation_test.go` 覆盖菜单已开/关闭、菜单延迟创建、切班后重建、默认全班范围纠正、嵌套控件去重、独立入口歧义、未知格式、其他作业 ID 和防重复提交。`TestGoEinoBrowserDownloadsVerifiedArchive/english-export` 使用真实 Chromium、本地拦截页和模拟模型验证英文导出到 ZIP 校验；不调用真实教务导出或在线模型。

后续重试已切到目标班级，但仍在 `open_export_settings` 遇到页面操作异常。补回最早 Python 脚本的菜单原生激活：More 与导出项均先核验当前课程/作业、控件名称、连接状态及禁用状态，再执行站内 `element.click()`，不移动鼠标去点击可能随指针移出而隐藏的菜单。隐藏入口仅允许已核验且所在课程视图仍活动的旧 `ul.morePop`，不能激活其他隐藏页面或已改名为导出成绩的节点。菜单无法显示时保留这条受限的旧页面兼容路径。

`last_export_step` 和错误证据区分下载中心、菜单、班级切换、触发设置、附件格式、范围核验和确认步骤，并仅记录超时、遮挡、脱离 DOM、严格匹配等分类，不保存原始 Playwright 调用日志、HTML 或签名 URL。内部只读 `export-diagnostics` 同时返回这些信息及当前候选数量。`TestExportMenuThatHidesWhenPointerLeavesMore` 在修复前复现普通点击超时，修复后通过；`english-export-hover` 进一步验证该菜单到 ZIP 校验的完整链路。

11:37 的实际英文弹窗使用 `Export commit attachments`，本班级和全部班级分别使用 `Export the class work attachment` / `Export all class work attachments`。这些翻译与中文选项一并兼容；Go 格式核验和浏览器内分组共用 `assets/constants.json` 的规则，避免两处识别不一致。附件格式按选项开头识别，完整答题记录说明中的 “submission of attachments” 不能成为选择依据。格式缺失或歧义时保存至多 10 个、每个至多 180 字符的选项名称及选中状态，供定位翻译差异。

`TestPortalAttachmentFormatsAndSingleClassTranslations` 使用门户 `.grade_check` 控件验证中文、真实英文及缺少分组类名的页面，要求仅选中提交附件、保持命名设置、只导出目标班级。英文菜单切班测试和英文完整下载测试也使用上述实际文案，覆盖默认完整答题记录到提交附件的切换。

本轮全量 Go 测试和真实 Chromium 的 Worker/Agent/调度器竞争检测通过。部署后从原下载检查点重试，11:48 官网班级 `124CS512601` 的提交附件成功下载：ZIP 为 49,923,942 字节，状态 `VERIFIED`，解压识别 19 名学生、0 项材料异常，随后原工作流继续批改；这次真实验证未扩大到其他班级。
