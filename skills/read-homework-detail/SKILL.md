---
name: read-homework-detail
description: 复用当前页面 Agent 与教师浏览器，打开指定作业详情，只读获取题目文字、图片和附件。
---

本任务只读取选定作业的要求，不下载学生提交，不查重，不批阅，不评分，不修改官网内容。

本 Skill 也用于 AUTH_BOOTSTRAP / CATALOG_SYNC 同步任务中已经选中作业的详情子阶段。此时只按当前 target 定位并读取该份作业；不要因为任务类型仍是同步任务就重新读取作业列表或继续采集其他课程。read_homework_detail 完成后，由执行器返回原作业列表并恢复同步 Skill；独立 HOMEWORK_DETAIL 任务仍按原流程结束。

复用原导航：门户→个人空间→本学期课程→作业。依据 target.portal 的课程和作业 ID 及真实页面名称核验目标；复用已经打开的课程页，不重新登录或构造 URL。账号、验证码、设备信任沿当前 authentication 状态及原认证工具处理；未出现验证码就直接继续。

本任务采用单详情页流程。观察中的 homework_detail.open 表示执行器仍持有存活的目标详情页，active 表示当前操作页就是该详情页，page_id 是该详情页的页面 ID；这三个字段与 ready 共同决定下一步，不能只看页面标题或阶段名称。

1. 仅在 open 为 false 时进入所选课程的“作业”列表，必要时搜索、滚动或翻页查找目标作业。点击准确的作业名称打开“作业详情”，不要点击右侧“查重”“批阅”或“批改”。新标签打开后沿该详情页继续。
2. open=true、active=false 时，只对指定的 page_id 调用一次 switch_page，然后重新观察。不得返回作业列表、再次点击作业名称或打开另一个详情标签。
3. open=true、active=true、ready=false 时，在当前详情页核对原因。页面仍在加载可重新观察；持续不可读、身份不符或缺少入口证据时调用 request_human，由教师核对当前页面。不得把返回列表重新点击当作恢复方法，也不要反复 switch_page 到 active_page_id。
4. 当前观察 ready=true 时，调用无参数 read_homework_detail。工具核验当前课程、作业身份和题目区域，获取正文及图片、附件并结束浏览器任务；后台自动完成 OCR 和文档解析。不要自行改写、概括或补全题目。
5. 只有原详情页被关闭、open 恢复为 false 后，才允许从准确的作业名称重新进入一次。reason=DETAIL_PAGE_NOT_ACTIVE 只表示应切回已绑定的 page_id，不表示入口失效；active=true 时的 DETAIL_ENTRY_UNVERIFIED 应请求人工核对，不能重新创建详情页。

旧版官网的“已发布作业 ID”与“作业库详情 ID”可能不同。执行器会在点击名称前复用目录解析器核验已发布 ID，再绑定这次真实打开的详情页，不要求两个不同业务标识相等。读取工具报题目区域未定位时，不把作业列表、学生答案或查重窗口当成作业要求，也不要点击“重新编辑”。

禁止调用 open_export_settings、read_course_list、read_homework_list、finish_catalog。不要因为已核验作业而进入原来的附件导出流程。只有页面无法继续或真实目标存在歧义时请求人工处理。
