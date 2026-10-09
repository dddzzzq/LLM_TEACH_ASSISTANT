---
name: download-homework
description: Create and track a teaching-portal homework download task when a teacher explicitly requests downloading course assignment attachments. Use the task browser for human login and class selection.
---

先确认课程名称、作业名称，必要时补充学期。用户询问使用方法不等于要求启动任务。

通过 fetch_homework 创建下载任务；fetch_and_grade_homework 是兼容入口，二者是同一业务操作，不应同时调用。
禁止索取官网密码、验证码或 Cookie。用户在任务浏览器内完成登录，明确选择班级，并交还自动化。

收到 accepted 和 job_id 后，说明任务已创建及人工操作入口，结束本轮。等待人工或导出打包期间不反复调用工具轮询。
用户需要进度时调用 get_fetch_job；用户明确要求暂停、恢复或取消时使用相应控制工具。
恢复失败时解释缺少的操作，不猜测已经完成登录或班级确认。下载与批改分别报告；只有已校验的附件才算下载完成。

当前教学系统兼容下载后按配置自动批改：缺少题目或评分标准时会等待补齐。创建任务不能表述为下载或评分已完成。
