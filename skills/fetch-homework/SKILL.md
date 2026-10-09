---
name: fetch-homework
description: Guide the browser Agent through locating a requested Xidian course and assignment for attachment export after human login. Use within an existing download task, not for general questions about downloading homework.
---

此 Skill 为页面导航 Agent 提供西电教学平台的业务操作知识。输入为目标课程、作业名称、可选学期、当前阶段和页面观察。它不创建后台任务、不注册工具，也不定义动作参数协议。

用户负责完整登录和验证码；任务控制器负责班级确认、导出提交、导出关联及文件传输。依据运行时提供的工具协议选择动作。

沿用业务流程：门户→个人空间→课程→作业菜单→指定作业批阅入口→更多菜单→导出作业附件。
LOCATE_COURSE：定位目标课程。多个同名课程必须根据学期等上下文消歧；无法确定则请求人工选择。
LOCATE_ASSIGNMENT：确认课程，进入作业列表并定位精确名称的作业，点击该作业对应的批阅入口。
EXPORT：打开更多菜单和“导出作业附件”设置，打开后由任务控制器处理。

element_ref 只对当前观察有效。优先选择与目标名称精确对应的控件；观察中的 context 是控件周围的业务信息。操作失败后重新观察，不能重复猜测引用。如用户已手动进入后续页面，应从当前位置继续。
进入导出设置即完成当前导航阶段；导出必须由任务控制器按已确认班级范围提交。不要声称文件已下载，完成状态以执行器核验为准。没有明确进展、遇到歧义或认证挑战时请求人工操作。
