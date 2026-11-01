---
name: collect-homework
description: 复用现有页面 Agent 和任务浏览器，打开课程与作业页面，只读获取教师课程、班级和作业目录。
---

本任务是获取课程作业目录，不下载附件、不评分、不修改官网内容。复用 fetch-homework 已有的“门户→个人空间→课程→作业”导航方式。先整体理解当前观察，从当前页面继续，复用已打开的标签页。

认证依照 authentication 状态完成：用 fill_portal_credentials 填入已保存凭据；需要验证码时用 request_login_code 获取，由教师在当前任务浏览器输入后交还自动化，再用 submit_login_code 提交；出现可信设备弹窗用 apply_device_trust。不要向教师索取聊天中的密码或验证码。滑块沿用已有工具与次数限制。

1. 进入实际课程列表，在页面就绪后调用 read_course_list 保存当前页的真实课程标识与名称。工具只读取，不点击导航。需要翻页或滚动时由你选择当前观察中的分页/滚动动作，再重新读取。catalog.courses_complete 为 true 才表示当前课程列表已读取到末尾。只查询当前已选学期，不自动切换学期。
   旧课程卡片可能只有名称与脚本点击入口，没有直达链接。identity_pending 为 true 时，course_id 是本次采集的临时引用，不是官网 ID：按该记录名称、学期和页面上下文点击进入课程，再以这个临时引用调用 read_homework_list。执行器将从实际课程/作业页面核验官网 ID 后保存，不能因为卡片上没有 ID 就反复读取同一列表或声称尚未进入课程页面。
2. 根据 catalog.courses 列表逐一进入尚未 complete 的课程。结合 platform_ids、学期、班级和控件上下文识别课程，同名课程须以官网 ID 区分。只能点击当前观察中的真实引用；不构造 URL、不按行号猜测。
3. 进入该课程的“作业”页面，调用 read_homework_list，course_id 必须使用刚才读取到的 course_id。核对班级筛选菜单内所有具体班级及 catalog.courses 的 classes、class_count；“全部班级”是汇总项，不计为具体班级。菜单尚未加载时先展开真实菜单并等待加载，再重新读取，不能把当前一个班级当作完整清单或编造班级。工具会核对页面归属并读取班级与作业；catalog.viewport.has_next 为 true 时继续翻页，needs_scroll 为 true 时调用 scroll（如 delta_y=800），它会滚动作业所在 iframe/内部列表并等待加载，之后重新读取，直到该课程 complete。不要在需要滚动时反复读取原页。空列表也必须有网站明确的“暂无作业”等提示。
4. 先依据 catalog.next_step 核验整门课程的完整清单：COLLECT_HOMEWORKS 阶段读取、滚动和翻页，manifest_ready=true 后才进入 READ_REQUIREMENTS。homework_manifest 提供清单及 READ/PENDING 状态，按 next_homework_id 逐份定位并 select_homework_detail，再点击 target 中的作业名称。homework_detail.open 为 true 时保持单详情页，active 为 false 只切到其 page_id；ready 为 true 时调用 read_homework_detail。执行器读完返回原作业列表并更新清单进度，已读条目不重读，清单完成标记不随滚动位置重置。仅 next_step=RETURN_TO_COURSE_LIST 时调用 return_to_course_list，再继续下一门未完成课程。工具报页面未就绪时，先观察和继续导航，不能把失败解释成空列表。
5. 最后一门课也必须返回课程列表。所有课程 complete、requirements_complete 且 courses_complete 为 true 时调用 finish_catalog；courses_complete 为 false 时，根据课程列表的 viewport 滚动或翻页并重新 read_course_list，不能反复结束。完成由执行器校验返回及目录与详情完整性，未读取完整时保留上次已发布目录。

只有缺少可核验 ID、页面无法继续或真正有歧义时 request_human。不要打开“导出作业附件”、点击下载或提交按钮。本 Skill 不改变原附件下载流程。
