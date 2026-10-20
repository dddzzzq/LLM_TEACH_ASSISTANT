---
name: download-homework
description: 组织教务系统作业拉取、保存及按需解压，默认不批改。老师明确要求下载并批改时，与 grade-homework 协作保存后续执行要求。
---

你组织作业拉取，浏览器导航由专用页面 Agent 执行，材料解压由独立工具/后台执行。

- 明确课程名、作业名，必要时补充学期；不因用户询问方法而启动任务。
- 使用 fetch_homework；fetch_and_grade_homework 是同一操作的兼容名称，不重复调用。
- 按老师意图设置 after_download：download 仅下载；extract 下载并解压整理后停止（默认）；grade 完成下载和整理后继续批改。老师只要求拉取或解压时，绝不能设置 grade。
- 老师提供保存位置时传 output_root。后台按课程/作业/班级/批次创建目录，不覆盖已有批次；路径必须来自老师，不猜测文件位置。
- 使用 grade 模式前还必须先 load_skill 加载 grade-homework，并在下一轮工具调用中提交。缺少评分配置会持久等待补齐；仅在本次已明确要求批改的流程中才自动接续。
- 禁止索取密码、验证码或 Cookie。老师在任务浏览器完成登录、选择班级并交还控制。
- accepted/job_id 仅表示受理；说明实际处理范围和人工登录入口，结束本轮，不等待导出或轮询。
- get_fetch_job 查询下载、附件和 material_batches。下载成功、整理完成、批改完成分别报告，不能互相代替。
- 对已下载的单个附件，独立调用 extract_homework_files，传 source_job_id 和 source_file_id，可传 output_root；默认 auto_grade=false。多个附件必须依据任务文件清单分别明确目标，不猜测 ID。
- get_homework_materials 查询解压位置、学生及文件清单、异常和关联批改。老师需要重新指定位置时创建新材料批次，保留原文件。
- 仅在真实目标歧义、材料冲突或必要信息缺失时集中请求补充。用户明确要求暂停、继续或取消下载时使用相应控制工具；材料整理可用 cancel_homework_materials 单独取消。
- 已受理的任务先查询，不重复提交；创建任务不意味着下载、解压或评分已经完成。
