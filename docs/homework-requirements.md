# 官网作业要求读取

在“作业批改 → 导入与自动批改 → 选择官网作业”中，把作业加入待批改列表后，默认自动读取官网要求。每份作业分别保存并展示题目，支持重新读取、取消以及下载原始图片和附件。取消“自动读取官网要求”可沿用手动填写方式；上传压缩包和指定目录的原流程保持不变。

## 执行路径

1. `PortalAssignmentPicker.vue` 提供目录中的真实作业 ID 与快照版本，`PortalHomeworkRequirements.vue` 按顺序读取，避免同一教师浏览器并发启动。
2. `POST /api/portal/homeworks/:id/detail-reads` 创建 `HOMEWORK_DETAIL` 操作。仍由原 `portal.Runner` 和 `rpa` 调度任务，并使用原教师持久浏览器、自动认证、暂停/接管和取消机制。
3. 详情任务先在原教师浏览器尝试目录中已保存的课程入口；现有登录会话可用就继续，入口失效则回到原认证流程。页面 Agent 加载 `skills/read-homework-detail/SKILL.md`，沿“课程 → 作业 → 点击作业名称 → 作业详情”导航。右侧“查重”和“批阅”不属于本流程。
4. 观察中的 `homework_detail.ready=true` 后，调用无参数 `read_homework_detail`。执行器核验课程 ID、作业 ID、标题及详情页来源，读取题目区域的正文、图片和附件。学生答案、答案解析、评分和导航区域不会作为题目提取。
5. 资源读取是异步操作，避免超过页面 Agent 单次请求时限。只跟随已支持平台域名的 HTTPS 资源和跳转，使用当前登录上下文；图片获取失败可截取该题目图片。
6. 后端复用现有文档解析器和 OCR RPC，按题目中的资源顺序整理正文；保留资源哈希和私有原件。`PortalHomeworkDetail` 缓存独立于课程目录，绑定教师、浏览器代次、目录快照与本次操作。
7. 批改请求设置 `use_portal_requirements=true` 时，服务端分别读取所选作业的完整要求，写入各自的 `Assignment.question` 与流程项。客户端不能以“已读取”的标记替代服务端校验。

## 文件处理与限制

| 内容 | 处理 |
|---|---|
| 网页文字、列表、表格 | 读取题目区域 DOM，保留换行及文本顺序 |
| PNG/JPEG/WebP/GIF 等题目图片 | 复用 OCR 识别，保留原图 |
| DOCX | 原有正文解析，补充内嵌图片 OCR；原 Word 文件保留 |
| DOC | 原有 antiword 提取正文；明确提示内嵌图片未核验，不能直接作为完整要求自动批改 |
| PDF | 逐页提取文字；扫描页和包含图片的页面补充渲染及 OCR |
| TXT/Markdown | 复用现有文本编码处理 |

PDF 逐页解析依赖 `poppler-utils`（`pdfinfo`、`pdftotext`、`pdfimages`、`pdftoppm`），已加入 WSL 初始化依赖。OCR 继续使用原 AI gRPC 服务。OCR 主要识别文字，复杂图形关系、公式排版的语义理解仍需核对原件。

单次最多 40 个来源、单资源 20 MiB、资源合计 60 MiB、PDF 30 页、合并正文 100 KB；DOCX 解压内容及图片也有上限。超过限制、资源失败、OCR 无有效正文等情况保留已读取内容，并标记 `complete=false`，不会悄悄用于自动批改。没有题目区域时不回退为整页正文，不把错误提示当作题目。

## 接口和 Agent 工具

- `GET /api/portal/homeworks/:id/detail`：获取本人当前目录的题目缓存及最近读取状态，不启动浏览器。
- `POST /api/portal/homeworks/:id/detail-reads`：请求体为 `{"snapshot_id":"目录版本"}`，返回原 RPA 任务对应的操作 ID。
- `POST /api/portal/operations/:id/cancel`：取消作业详情读取；取消后的迟到结果不能发布。
- `GET /api/portal/homeworks/:id/detail/sources/:source`：校验所有者后下载原件；公共详情响应不包含磁盘路径、Cookie 或资源签名地址。
- AI 助手工具 `read_portal_homework_detail` 与 `get_portal_homework_detail`：分别创建只读任务和查询提取结果，不触发学生附件下载、查重或评分。

目录更新或更换账号后，旧快照/浏览器代次的缓存不会自动应用于新批改。已有本地作业的题目不会被后台读取静默覆盖。
