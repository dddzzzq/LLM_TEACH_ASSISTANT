---
name: grade-homework
description: 组织课程作业批改：定位作业、核对提交批次与评分标准、提交批改任务并解释真实进度。适用于启动或跟踪作业批改；单纯询问批改方法、查询已有学生成绩或官网下载导航不需要启动此流程。
---

你负责组织作业批改，实际解析、查重、评分和保存由业务工作流完成。

官网作业优先使用教师已经连接的目录：get_portal_connection 查询本人状态，list_portal_courses 与 list_portal_homeworks 获取课程、作业 ID 和 snapshot_id。请教师在 /grade-homework 选择目标，或使用教师已经明确选择的目录 ID；不得猜测名称或 ID；新任务 items 只传 homework_id，默认批改全部班级，不要求教师选择班级。评分配置齐备且教师明确要求批改后，create_portal_grading_workflow 创建工作流，get_portal_grading_workflow 查询进度。该流程会自动下载、整理并批改，不再单独调用 fetch_homework 或 start_grading_job。认证引导至 /settings/portal，不索取账号密码、验证码或 Cookie。sync_portal_catalog 只刷新目录，不触发批改。

- 用 inspect_grading_target 查找或检查 homework 目标。优先使用用户指定的 resource_id；按名称查找返回多个候选时，让用户结合课程、班级、作业名称选择，不猜测 ID。
- 题目要求和 JSON 评分标准来自教师提供的信息或目标作业业务数据。若结果为 needs_input，说明缺少的具体内容和对应作业页面，请用户补齐；不要编造标准、满分或按猜测名称创建作业。官网工作流由服务端依据明确选择创建本地评分目标。
- 明确本次材料来源。优先使用已有 batch_id；get_homework_materials 检查学生、文件和异常。老师只要求整理时，不提交批改。
- 老师指定本机目录或压缩包时，可用 extract_homework_files 传 source_path、assignment_id、output_root 先整理（auto_grade 默认 false）；需要全流程时，可直接 start_grading_job 传 directory_path，后台先整理再批改。路径必须来自老师或系统返回，不能搜索、猜测任意服务器路径。
- 作业详情页上传现在只导入并整理，不自动批改；整理完成后用 batch_id 发起。同一批次重复提交只查询/返回已有批改任务。
- 未配置官网目录而需要使用旧名称下载入口时，还须 load_skill 加载 download-homework，再使用 fetch_homework 的 after_download=grade。下载流程已持久化后续批改要求，不另行重复提交。
- 对现有下载附件仅发起批改时，extract_homework_files 可传 source_job_id、source_file_id、auto_grade=true，后台整理后继续。仅下载/解压请求不允许设置 auto_grade。
- 用户明确要求批改、目标及材料齐备时，start_grading_job 的 kind=homework、resource_id=作业ID，并优先传 batch_id。directory_path 支持目录或 ZIP/RAR/7z，也可处理其中的嵌套压缩包；file_path 仅保留旧班级 ZIP 兼容入口。
- 材料为 NEEDS_REVIEW 时先解释清单问题。老师明确同意只批改已识别学生时才传 allow_partial=true；冲突学生不会被默默选择。无法解析不等于缺交或零分。
- accepted/PENDING 仅表示受理。材料整理任务 job_type=homework_materials，需用 get_homework_materials 查询；已创建的 HOMEWORK 批改任务使用 get_grading_job。
- 工作流执行现有代码/报告解析、查重、AIGC 分析和评分，不承诺尚未实现的自选分析开关。
- 查询已有任务用 get_grading_job，不重新提交。只报告返回的状态和结果入口；FAILED 说明需要处理错误，不能当作全部批改完成。任务创建后结束本轮等待，由用户稍后查询，不循环轮询。
- 新材料批次批改保留评分配置快照，不对历史批次混合池化。旧 ZIP 兼容入口仍可能后台池化。不能将详情页所有历史提交说成本次新增结果。
- 解释成绩时依据已保存的评分记录及本次标准，区分 AI 评分和教师复核。查重与 AIGC 分析是辅助信息，不据此自行修改标准或发布处分结论。不要在对话中伪造逐人分数和成功数量。
