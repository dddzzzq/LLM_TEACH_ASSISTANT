# 教务自动化实现与部署

更新：2026-10-28。本版本复用原 RPA Agent、任务控制器和浏览器界面；整体设计见 [复用方案](teacher-portal-automation-design.md)。

## 当前实现

- 门户启动每轮最多 4 次导航，退避间隔 5/15/30 秒；首轮耗尽后冷却 60 秒自动再试一轮，总计最多 8 次、420 秒。仅网络瞬断、超时、408/429/5xx 等暂时错误重试；证书错误、域名拦截等直接保留页面交还人工。等待期间持续检查已有页面，页面就绪即继续，暂停与取消立即停止后续尝试。
- 教师凭据加密保存。账号初始化、课程作业获取、作业下载均为普通 RPA 任务，使用原 Go/Eino 页面 Agent。
- 设置页嵌入 `RpaTaskPanel`，继续使用原 `RpaBrowserPanel`、WebSocket 输入、暂停接管和交还自动化。独立认证浏览器组件与独立 Python 操作控制器已移除。
- `PortalAuth` 只提供凭据填写、验证码提交与设备信任动作，不运行自己的导航循环。密码及验证码通过 Worker 内部读取，不进入模型参数。
- 新 `sync-teacher-portal` Skill 统一说明首次认证与本学期课程作业更新；首次认证和随后的课程采集在同一任务、同一浏览器内完成；不会在认证成功后关闭浏览器再登录。独立更新目录仍使用 CATALOG_SYNC。`collect-homework` 保留兼容。
- 同步在每门课程先核验完整清单，`manifest_ready=true` 后按 `homework_manifest` 中的 `next_homework_id` 逐份 select_homework_detail，再点击作业名称并复用 read_homework_detail 获取正文、图片和题目附件。已读条目标记 READ，返回详情后不重置清单完成标记；未完成时执行器保留当前课程，不提前返回再重复进入。页面 Agent 只获得当前 next_step 允许的业务工具，避免在课程首页读取列表。后台复用原解析器并在同一事务保存目录与要求，不再追加逐作业的浏览器任务。作业页仍支持单独“重新同步要求”；旧 Worker 的无详情目录继续由原队列补全。
- Agent 逐门进入课程与作业页面、翻页和读取。明确“暂无作业”也完成该课程，不要求滚到底部。每门读完调用无参数 `return_to_course_list`：课程在新标签打开时切回原页，同页跳转时恢复上次成功读取的课程列表地址，并核验原卡片来源，避免恢复到错误学期。该地址仅在执行器内存中保存，模型不能传入任意地址。
- `PortalCatalog` / `CatalogCollection` 只提取、核验和累计数据；返回动作在原任务控制器执行，不建立新的课程遍历器。观察提供当前课程、下一门未完成课程、已完成/总课程数，所有课程读完才发布；重复读取列表保留已完成记录和已读取的分页覆盖状态。
- 采集任务的原 `scroll` 动作由 `CatalogViewport` 定位实际课程/作业所在 frame 与内部滚动容器，不依赖鼠标上次停留的位置。观察中的 `catalog.viewport` 提供滚动位置、剩余距离、下一页和加载状态；滚动后等待懒加载稳定，再由 Agent 继续读取。末尾判定使用同一个列表区域，不把外层门户或侧栏的滚动状态算入目录完整性。原下载任务的滚动行为保持不变。
- `finish_catalog` 在提交前强制返回并重新核验原课程列表，包含最后一门课。若作业已读完但课程清单覆盖尚未核验到末尾，任务保持运行并提示继续滚动/翻页读取，不连续报结束失败，也不发布不完整目录。
- 课程卡片只有名称与脚本入口时，先保存本次采集引用，进入实际课程/作业页面后核验官网 ID；临时引用不作为官网 ID 发布。直达 URL 可缺省，页面导航仍通过 Agent 点击真实入口。
- 作业读取覆盖原下载实现使用的 `li / h2.list_li_tit / a.piyueBtn` 列表，识别显式作业属性、隐藏作业 ID、带前缀的记录 ID、脚本里的字面量链接，以及参数明确名为 `workId/homeworkId/taskId` 的具名函数入口。仅读取 DOM 和函数签名，不点击批阅或执行处理函数，不按未知函数的参数顺序猜 ID；同一行 ID 冲突时拒绝发布。
- 官网选择目标和原名称式下载都由原 `fetch-homework` 页面导航。默认自动导出全部班级，不依赖目录中的班级 ID；附件导出、导出关联和 ZIP 校验继续使用。
- 官网作业、上传 ZIP、指定目录三种入口保留，共用材料整理与评分链路。

## 教师操作

1. 打开“教务账号”，填写账号密码和设备信任选项，点击“保存并连接”。
2. 原任务浏览器自动打开，Agent 自动填写已保存凭据，并依据页面处理下一步。
3. 首次出现 WeChat Work / 企业微信动态码页面时，直接等待教师输入，不查找“获取验证码”按钮，不切换到短信或扫码方式。打开企业微信查看当前动态验证码，在浏览器面板上方的“企业微信动态验证码”输入框填写后，点击“提交验证码并继续登录”。验证码通过现有认证 WebSocket 直接填写到当前官网输入框，Worker 确认填写后才交还自动化；不进入 AI 聊天、日志或本地存储。也可直接在官网填写后，点击“验证码已在官网填写，继续登录”。未填写时不能继续登录。兼容已有的其他验证码页面，但只有存在真实发送按钮时才执行发送。
4. Agent 点击官网登录按钮，等待官网读取验证码，再按设置选择中文或英文的“信任此设备 / Trust this device”或“仅本次登录 / Not now”；身份核验成功后复用当前浏览器继续采集课程与作业。按钮缺失、不可用或验证码被拒绝时保留具体提示，不盲目连续重试。
5. 同步完成后进入“作业批改工作台”，选择课程、作业并配置评分要求，默认批改全部班级，开始批改。
6. “更新课程”让 Agent 依次读取页面当前已选学期的所有课程，遇到无作业的课程也返回继续。不自动切换学期，不下载附件或触发评分。

后续登录复用同一教师的专用浏览器配置。有效会话直接使用；需要登录时自动填写已保存凭据。设备信任是否继续有效由学校网站决定，再次出现验证要求时仍在原任务浏览器中处理。

没有设置教务账号时，原下载任务保持人工登录；设置并认证成功后，原名称式任务也会自动复用本人凭据。

## 部署

适用于 Go 与单个 Python Worker 同在一台 WSL 主机的当前部署方式。先备份数据库，在没有真实任务运行时重新编译并启动相关服务。

```bash
source /mnt/d/dzq/LLM_TEACH_ASSISTANT/scripts/wsl-env.sh
cd /mnt/d/dzq/LLM_TEACH_ASSISTANT
"$PYTHON" scripts/init-portal-key.py
```

密钥脚本只在不存在时创建 `$RUNTIME_DIR/.portal-credential-key`，不覆盖已有密钥、不输出密钥内容，文件权限为 0600。密钥应与数据库分开备份。Go 与 Worker 必须加载相同的运行目录配置。

新 Worker 健康标识为 `portal_automation=teacher.portal.agent.v2`。`scripts/start.sh` 和 Go 的凭据任务启动都会检查该标识，避免新后端连接到旧独立流程 Worker。旧进程占用端口时需按原进程管理方式重启；启动脚本不会强制替换正在运行的服务。

作业列表适配版本可通过受保护的 Worker `/health` 中 `catalog_reader=homework.dom.v2` 核验。本次仅修改 Python 提取与诊断实现，部署时只需在任务空闲后重启 RPA Worker。

验证码采集支持可通过 `portal_code_input=teacher.code.browser.v1` 核验。更新前端并在任务空闲后重启 RPA Worker；`vue-grading-frontend/tests/portal_authentication_smoke.py` 使用模拟 API 和浏览器流验证验证码采集、填写确认后才继续登录、输入清理与移动端布局，不发送真实验证码。

企业微信动态码识别通过 `portal_wechat_code=wechat.dynamic-code.v1` 核验。识别依据是当前可见的企业微信动态码说明及验证码输入框；隐藏的替代页签、缺少发送按钮本身均不能作为判据。该流程直接进入 `WAITING_OTP`，不声称验证码已请求或已发送。

列表滚动与结束返回修复通过 `catalog_navigation=catalog.viewport.v1` 核验。采集 Skill 更新为 `sync-teacher-portal` 1.1.0，原工具参数协议不变；新建“更新课程”任务即可加载新版 Skill。

已有独立认证流程的未结束操作在升级后标记为中断，需从设置页重新认证或更新课程；已保存的凭据、专用浏览器配置、成功目录和评分结果保留。

### 配置

| 环境变量 | 默认值或作用 |
| --- | --- |
| `RUNTIME_DIR` | 沿用现有运行目录，Go 与 Worker 一致 |
| `PORTAL_CREDENTIAL_KEY_FILE` | `$RUNTIME_DIR/.portal-credential-key` |
| `PORTAL_CREDENTIAL_KEY_BASE64` | 可选，受控部署环境注入，不能写入仓库 |
| `PORTAL_CREDENTIAL_SOCKET` | `$RUNTIME_DIR/portal-credentials.sock`，本机 Unix socket |
| `PORTAL_LOGIN_ORIGINS` | `https://ids.xidian.edu.cn`，精确的 HTTPS 登录来源 |
| `PORTAL_USERNAME_SELECTOR`、`PORTAL_LOGIN_SUBMIT_SELECTOR` | 必要时适配学校实际登录控件 |
| `PORTAL_IDENTITY_SELECTOR` | 稳定教师账号的页面控件，不能只核对姓名 |
| `PORTAL_COURSE_SELECTOR`、`PORTAL_HOMEWORK_SELECTOR`、`PORTAL_CLASS_SELECTOR` | 当前列表数据的只读提取规则 |

浏览器页面 Agent 继续使用原模型配置。采集课程列表需要模型调用；测试使用模型替身，真实网站仍需首次联调。学校 DOM 缺少可核验标识时需补充数据提取适配，不能由模型编造 ID。

## 源码阅读路径

```text
vue-grading-frontend/src/views/PortalSettingsView.vue
  → components/RpaTaskPanel.vue
  → components/RpaBrowserPanel.vue

backend-go/internal/portal/service.go、runner.go
  → 原 internal/rpa/service.go、navigation.go
  → 原 internal/browseragent/agent.go
  → skills/fetch-homework 或 skills/sync-teacher-portal

ai_engine_python/app/rpa/worker.py
  → 原 task_controller.py
  → portal_auth.py（凭据及认证动作）
  → portal_catalog.py（读取当前列表）
  → 原 portal_adapter.py、download_manager.py（导航核验与下载）
```

`PortalOperation` 只关联业务操作状态，ID 与它对应的普通 RPA 任务一致。页面观察、动作执行、回执、暂停恢复和浏览器流都使用原 `/api/rpa/jobs/:id` 与内部 `/tasks/:id`。原新增的 `/api/portal/auth-runs/:id/:action` 返回 410，提示改用任务浏览器。

课程和作业选择仍使用 `/api/portal/courses` 与课程作业接口；账号保存、重新认证和更新课程接口保持不变。主助手的六个教务业务工具继续可用，它们不直接接收或读取凭据。

## 数据与权限

同步会读取班级筛选菜单中的具体班级，按官网班级 ID 去重并累计保存；“全部班级”不计为一个班级。作业批改面板显示官网班级数量，默认选择全部班级，也可切换当前班级或自选班级。`POST /api/portal/assignments/:id/grade` 的 `all_classes:true` 由后端按当前目录展开为逐班任务，不能同时提交非空 `class_ids`；每班材料和成绩保持独立。

导出先选择“导出提交附件”，再核验班级范围。若官网弹窗没有班级范围控件，仅在批阅页的当前官网班级 ID 与任务目标一致时沿用该班级，导出记录仍绑定该具体班级；不能据此推定导出了全部班级。

账号密码整体使用 AES-256-GCM 加密，绑定用户、平台、连接及凭据版本。Worker 通过本机模式 0600 的 socket，用短期一次性票据读取凭据；票据不写入任务、聊天或 Kafka。认证输入和浏览器异常信息不会记录原始凭据。

教师连接和浏览器配置隔离，配置由文件锁独占。换账号或断开会撤销旧版本，阻止旧任务读取新账号凭据。已提交的官网目标保留稳定 ID，Agent 结合当前页面标识定位，同名记录不能替代。

历史作业 `owner_id=0` 作为待分配数据保留，管理员在设置页依据实际记录明确分配教师；不会把全部历史作业归给某一位教师。已有成绩保留。

## 验证范围

测试覆盖原页面 Agent 加载新 Skill、普通 RPA 任务复用、凭据隔离、首次验证码与设备信任、会话复用、逐页目录获取、同名 ID 核验，以及原班级导出和附件下载。设置页浏览器输入和三种批改入口通过实际 Vue/Chromium 的模拟接口测试。

浏览器测试使用真实 Chromium，但学校页面、账号、验证码和附件均为拦截的测试数据，模型使用替身；不代表已完成真实教师账号或真实评分联调。

目录任务在原单轮 60 次动作预算基础上，每完成一门不同课程增加 12 次预算，每读取一条新的已核验作业增加 1 次预算（作业部分最多增加 400 次），支持长列表逐段滚动和读取。反复读取已有作业不增加预算；原无进展检测和任务 1000 次动作总上限保留。下载任务预算不变。

2026-10-24 验证结果：Go 全量测试、`go vet` 及相关包并发检查通过；Python 126 项测试、前端 13 项测试与构建、五组界面回归通过。原 Go/Eino→Python→Chromium 附件下载的五种页面用例也通过，部署辅助脚本 8 项测试通过。

课程卡片适配修复追加验证：教务测试 13 项、原页面与学期导航回归 18 项通过，覆盖 iframe/srcdoc 中仅有名称与脚本入口的课程卡片、进入课程后的 ID 核验及原下载链路。此类修复需更新并重启 Go 后端和 Python Worker，再从设置页“更新课程”发起新任务；已有可信浏览器配置保留。

首次认证与整学期采集 Skill 追加验证：Python 全量回归 131 项，以及随后追加的课程分页覆盖测试 1 项通过。覆盖同页与新标签下三门课程按“无作业→有作业→无作业”依次读取并返回、加载中不判为空、返回后保留已读分页、原学期核验，以及完成课程后获得额外动作预算。Go 全量测试、`go vet`、相关包并发检查及原 Go/Eino→Python→Chromium 五种附件下载页面回归通过。运行中的真实服务未为测试重启，实际官网效果需部署后验证。

作业提取修复：最近任务 `2a070e9b-0e70-4918-bf75-2c34c6c92fd6` 已成功读取第一门空作业课程并返回，在第二门课程连续三次触发 `HOMEWORK_IDS_UNAVAILABLE` 的“无记录”分支；课程归属核验已通过。旧日志没有保存匹配数量和 DOM 属性来源，无法据此确定真实页面的具体 ID 存放方式。现已补充失败证据（脱敏 frame 路径、候选数量、元素结构和处理函数名称）及成功日志（当前页/累计作业数、空状态与完整性），不记录完整 HTML、表单值或带票据的 URL。

该修复验证了 8 类作业 ID 入口、隐藏记录过滤、未知 ID 与冲突拒绝、分页按钮不作为作业，以及不执行批阅处理函数。Python 全量 134 项、最终教务回归 18 项及原 Go/Eino→Python→Chromium 五种附件下载页面用例通过。确认无活动或排队浏览器任务后，已备份任务记录并只重启 RPA Worker，健康检查确认加载 `homework.dom.v2`。未重新发起真实教师目录采集，需从设置页“更新课程”重试确认。

滚动与最后返回修复依据：任务 `108f7c23-5fff-4fdd-867c-e2ac0369f114` 的四门课程均标记已读取（作业数为 0、7、0、1），但 `courses_complete=false`，最后停留在课程内连续调用 `finish_catalog`。旧实现按所有 frame 的整页滚动位置判末尾，且滚动动作未定位列表区域；现已统一到真实列表的 viewport，并在结束动作中验证返回。新增回归覆盖课程清单未核验时的继续采集、外层长侧栏不影响 iframe 列表、最后一门漏调返回，以及 iframe 整页/内部列表的两轮异步加载与自动读取。

该次修复的教务回归 21 项通过，旧版无 DOCTYPE 页面、异步滚动及最终返回的追加验证 3 项通过；Go 页面 Agent/RPA 测试与原五种附件下载页面回归通过，源码格式检查通过。旧任务随后因等待超时结束；确认无活动或排队任务后，已备份记录并更新 RPA Worker，健康检查确认 `catalog_navigation=catalog.viewport.v1`。未自动创建新的真实采集任务，请从设置页“更新课程”重新发起，以加载新版 Skill。
