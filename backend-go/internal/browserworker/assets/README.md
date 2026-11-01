# 浏览器执行资源

这些文件由 Go Worker 通过 `go:embed` 打包，在浏览器页面中执行；运行时不导入或启动 Python。

初始脚本沿用迁移时 `ai_engine_python/app/rpa/` 的语义提取和受控 DOM 操作：

- `page_snapshot.js`：`page_snapshot.py` 的页面观察，排除表单机密和原始 HTML。
- `catalog_extract.js`、`homework_extract.js`、`catalog_viewport.js`：课程、作业真实标识，以及 iframe 和内部滚动容器的读取。
- `question_*.js`：`homework_detail.py` 的题目正文、图片和附件提取，排除学生答案与答案解析。
- `auth_*.js`、`slider_canvases.js`：认证控件、表单来源、教师身份和挑战图像的局部读取。
- `export_*.js`、`hidden_export_entry.js`、`class_selection_ack.js`：导出选项、记录身份及已绑定控件的操作。
- `manual_export_listener.js`：检测教师在同一浏览器手动提交导出。
- `verified_headings.js`：沿用原适配器的标题核验范围，排除隐藏内容及列表行，防止列表标题被当成已进入目标。
- `actions.json`：Go 执行器发布并校验的 `browser.v1` 动作协议。
- `constants.json`：上述页面读取使用的 CSS 选择器和固定平台入口。

新执行行为在此包中维护，并通过真实 Chromium 的 Go 回归测试验证。旧 Python 文件保留作兼容与行为对照，不会自动改变 Go 执行器或注册额外工具。
