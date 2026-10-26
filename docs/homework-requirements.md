# 官网作业要求读取

在“作业批改”点击“更新课程与作业”，目录同步后会直接创建本地作业，并在后台按顺序读取、整理官网要求。每份官网作业只关联一份本地作业，所有班级共享题目和可编辑的 100 分评分模板；重新同步复用该作业，保留教师修改的评分标准。要求未完整就绪时不能开始官网班级批改。

进入作业的“材料与批改”页，先保存评分标准，再选择班级开始批改。已有材料和评分结果保留在同一作业下，可按稳定班级 ID 筛选、导出；上传压缩包和指定目录也可选择材料所属班级。同步本身不触发学生作业批改。

`portal/import_assignments.go` 负责创建/复用作业、旧目录补建和自动读取队列；`portal/workflows.go` 复用当前作业并为本次班级批改保存题目、评分标准快照。课程和作业名称用于界面查找，执行仍使用官网 ID，避免同名记录误匹配。原 `read-homework-detail` 与 `fetch-homework` 页面 Agent 和工具继续使用，指定班级导出不能自动扩大到全部班级。

升级时已有目录会补建本地作业，并采用仍有效的完整要求缓存；其余要求可点击“继续补全要求”或“更新课程与作业”。首次认证或主动更新得到的新目录会自动补全，无需另行创建作业。暂停后续同步会保留当前正在读取的作业和剩余队列；重启后仍可恢复。

## 执行路径

1. `PortalAssignmentPicker.vue` 提供目录中的真实作业 ID 与快照版本，`PortalHomeworkRequirements.vue` 按顺序读取，避免同一教师浏览器并发启动。
2. `POST /api/portal/homeworks/:id/detail-reads` 创建 `HOMEWORK_DETAIL` 操作。仍由原 `portal.Runner` 和 `rpa` 调度任务，并使用原教师持久浏览器、自动认证、暂停/接管和取消机制。
3. 详情任务先在原教师浏览器尝试目录中已保存的课程入口；现有登录会话可用就继续，入口失效则回到原认证流程。页面 Agent 加载 `skills/read-homework-detail/SKILL.md`，沿“课程 → 作业 → 点击作业名称 → 作业详情”导航。右侧“查重”和“批阅”不属于本流程。
4. 观察中的 `homework_detail.ready=true` 后，调用无参数 `read_homework_detail`。执行器核验课程 ID、作业 ID、标题及详情页来源，读取题目区域的正文、图片和附件。旧版列表的已发布作业 ID 与作业库预览 ID 是不同标识：点击名称前复用目录 DOM 解析器核验已发布 ID，再绑定实际打开的页与地址；库预览只在该绑定仍有效且课程、标题一致时读取。地址改变或缺少入口证据不能沿用绑定。学生答案、答案解析、评分和导航区域不会作为题目提取。
5. 资源读取是异步操作，避免超过页面 Agent 单次请求时限。只跟随已支持平台域名的 HTTPS 资源和跳转，使用当前登录上下文；图片获取失败可截取该题目图片。
6. 后端复用现有文档解析器，把原始题目图片、DOCX 内嵌图及 PDF 图像页传给当前配置的模型，并结合正文整理完整作业要求；不是仅对 OCR 文本做摘要。保留资源哈希、私有原件及图片转写。`PortalHomeworkDetail` 缓存独立于课程目录，绑定教师、浏览器代次、目录快照与本次操作。
7. 批改请求设置 `use_portal_requirements=true` 时，服务端分别读取所选作业的完整要求，写入各自的 `Assignment.question` 与流程项。客户端不能以“已读取”的标记替代服务端校验。

## 文件处理与限制

| 内容 | 处理 |
|---|---|
| 网页文字、列表、表格 | 读取题目区域 DOM，保留换行及文本顺序 |
| PNG/JPEG/WebP/GIF 等题目图片 | 原图以 Base64 传给当前模型，使用原图精度；保留原图及模型转写 |
| DOCX | 原有正文解析，内嵌图片交给模型理解；原 Word 文件保留 |
| DOC | 原有 antiword 提取正文；明确提示内嵌图片未核验，不能直接作为完整要求自动批改 |
| PDF | 逐页提取文字；扫描页和包含图片的页面渲染后交给模型理解 |
| TXT/Markdown | 复用现有文本编码处理 |

PDF 逐页解析依赖 `poppler-utils`（`pdfinfo`、`pdftotext`、`pdfimages`、`pdftoppm`），已加入 WSL 初始化依赖。原 OCR 解析入口继续保留，其他调用方不变；官网要求改用文档正文与原图理解。

## 作业要求整理

`backend-go/internal/portal/requirement_organization.go` 复用 `modelclient.NewDeepSeekClient`，沿用 `DEEPSEEK_API_KEY`、`AGENT_MODEL_URL`、`AGENT_MODEL`，不引入新供应商或另一套浏览器。当前 `deepseek-chat` 配置已用真实作业原图验证可用，服务实际返回 `deepseek-flash`。更换模型时仍需支持图片消息与 JSON 输出；模型不支持或调用失败时保留材料并标记未完整整理，不悄悄降级为“完整”。

一次模型请求同时提供来源清单、图片清单、正文及原图。输出完整 Markdown 和逐图转写，保留命令、数值、单位、输入输出示例和提交要求，移除网页操作及空答案区，不生成新的评分标准。服务端校验 JSON、结束原因、每个来源和图片的覆盖情况；模型指出模糊、缺失或冲突时展示提示并阻止自动采用。覆盖校验不能证明模型语义绝对正确，原件始终可下载核对。

缓存的 `question` 为整理后的要求，`raw_question` 为原始正文与图片转写，`sources` 保留逐来源内容，`processing` 记录版本、响应模型及是否使用原图。缓存读取不重复调用模型；“重新读取”会重新获取、整理官网内容。界面用经过过滤的 Markdown 展示要求，并提供原始文本入口。自动批改沿用每份作业自己的 `question`。

单次最多 40 个来源、单资源 20 MiB、资源合计 60 MiB、PDF 30 页、合并正文 100 KB；DOCX 解压内容及图片也有上限。模型单次最多 40 张图、原图合计 30 MiB，避免 Base64 超过请求大小限制。超过限制、资源失败、图片不可读、模型结果截断或核验不通过等情况保留材料，并标记 `complete=false`，不会悄悄用于自动批改。没有题目区域时不回退为整页正文，不把错误提示当作题目。

## 接口和 Agent 工具

- `GET /api/portal/homeworks/:id/detail`：获取本人当前目录的题目缓存及最近读取状态，不启动浏览器。
- `POST /api/portal/homeworks/:id/detail-reads`：请求体为 `{"snapshot_id":"目录版本"}`，返回原 RPA 任务对应的操作 ID。
- `POST /api/portal/operations/:id/cancel`：取消作业详情读取；取消后的迟到结果不能发布。
- `GET /api/portal/homeworks/:id/detail/sources/:source`：校验所有者后下载原件；公共详情响应不包含磁盘路径、Cookie 或资源签名地址。
- AI 助手工具 `read_portal_homework_detail` 与 `get_portal_homework_detail`：分别创建只读任务和查询提取结果，不触发学生附件下载、查重或评分。
- `GET /api/portal/assignments/:id`：查询当前本地作业对应的官网作业、班级目录与要求同步状态。
- `PUT /api/portal/assignments/:id/rubric`：保存 `{"rubric":{...}}`，下次同步不会覆盖已编辑标准。
- `POST /api/portal/assignments/:id/grade`：发送 `{"class_ids":["目录班级ID"]}`，携带 `Idempotency-Key`，复用当前作业批改所选班级。
- `PUT /api/portal/imports`：`{"paused":true/false}` 暂停或恢复后续要求读取。
- 作业结果、Excel 导出和材料列表支持 `class_id` 筛选；官网批次持有独立 `portal_class_id`，不同班级不会按相同显示名称混合。

目录更新或更换账号后，旧快照/浏览器代次的缓存不会自动应用于新批改。官网关联作业的题目随完整同步结果更新；手动创建的本地作业不参与官网同步。已经提交的班级批改使用当次保存的配置快照。
