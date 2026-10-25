"""Real App/router assignment workspace, using fixture APIs and no grading services."""

import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright, expect


async def main():
    base = os.environ.get("RPA_UI_BASE_URL", "http://127.0.0.1:6017")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + "/src/views/PortalSettingsView.vue") as response:
        source = response.read().decode()
    vue = next(
        path for path in re.findall(r'from "([^"]+)"', source) if "/vue.js" in path
    )
    diagnostics = Path("/mnt/d/dzq/LLM_TEACH_ASSISTANT/.runtime/diagnostics")
    rubric = {
        "功能实现": {"score": 60, "description": "核心功能完整，异常处理正确"},
        "代码质量": {"score": 20, "description": "结构清晰，命名规范"},
        "文档报告": {"score": 20, "description": "说明设计与测试方法"},
    }
    items = [
        dict(
            id=i,
            course_name=course,
            class_name="全部班级",
            task_name=title,
            question=question,
            rubric=rubric,
            summary=dict(
                status=status,
                graded_count=count,
                reviewed_count=0,
                batch_count=2 if count else 0,
                issue_count=0,
                failed_count=0,
                updated_at="2026-10-25T10:32:00+08:00" if count else None,
            ),
        )
        for i, course, title, question, status, count in [
            (
                11,
                "移动互联网导论",
                "编程小作业 1",
                "实现一个移动端待办事项应用。\n\n1. 支持任务新增、修改与删除。\n2. 使用本地存储保存任务状态。\n3. 提交源代码、设计说明及测试截图。",
                "GRADED",
                108,
            ),
            (
                12,
                "自主可控嵌入式系统",
                "ARMv8 汇编实验",
                "完成 ARMv8 汇编程序，说明寄存器与内存的变化。",
                "GRADING",
                19,
            ),
            (
                13,
                "移动互联网导论",
                "编程小作业 2",
                "实现带用户登录的网络请求模块。",
                "READY",
                0,
            ),
            (
                14,
                "软件工程",
                "需求分析与建模",
                "提交需求规格说明书与用例图。",
                "ATTENTION",
                36,
            ),
            (15, "软件工程", "综合实践报告", "提交小组实践总结。", "EMPTY", 0),
        ]
    ]
    submissions = [
        dict(
            id=1,
            student_id="20260001",
            student_name="张三",
            score=92,
            feedback="功能完整，代码组织清楚。",
            plagiarism_reports=[],
            is_human_reviewed=False,
        )
    ]
    errors, mutations, uploads = [], [], []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(viewport={"width": 1536, "height": 1080})

        async def api(route):
            req = route.request
            path = urllib.parse.urlsplit(req.url).path
            if path == "/api/profile":
                data = dict(
                    id=7, username="teacher-demo", name="陈老师", role="teacher"
                )
            elif path.endswith("/attention"):
                data = []
            elif path == "/api/materials":
                data = {
                    "items": [
                        {
                            "id": "11111111-1111-4111-8111-111111111111",
                            "assignment_id": 11,
                            "class_name": "全部班级",
                            "created_at": "2026-10-25T10:30:00+08:00",
                            "status": "READY",
                            "student_count": 108,
                            "issue_count": 0,
                            "grading_job_id": "grade-fixture",
                            "output_dir": "fixture/materials",
                        }
                    ],
                    "grading_jobs": [{"id": "grade-fixture", "status": "SUCCESS"}],
                }
            elif path == "/assignments/":
                if req.method == "POST":
                    body = req.post_data_json
                    mutations.append(body)
                    item = dict(id=16, **body)
                    items.insert(0, item)
                    data = item
                else:
                    data = items
            elif path.endswith("/submit"):
                uploads.append(req.post_data_buffer.decode("utf-8", errors="replace"))
                data = {"message": "材料已受理"}
            elif path.endswith("/results"):
                data = submissions
            elif re.fullmatch(r"/assignments/\d+", path):
                data = dict(
                    next(
                        item
                        for item in items
                        if item["id"] == int(path.rsplit("/", 1)[1])
                    ),
                    submissions=submissions,
                )
            else:
                data = []
            await route.fulfill(
                content_type="application/json",
                body=json.dumps(data, ensure_ascii=False),
            )

        harness = f"""<html><head><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app"></div><script type="module">
          import {{createApp}} from '{vue}';import App from '/src/App.vue';import router from '/src/router/index.js';
          localStorage.setItem('access_token','fixture');localStorage.setItem('user_info',JSON.stringify({{user_id:7,username:'teacher-demo',role:'teacher'}}));
          window.fixtureRouter=router;await router.replace('/');createApp(App).use(router).mount('#app');
        </script></body></html>"""
        page = await context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.route("**/api/**", api)
        await page.route("**/assignments/**", api)
        await page.route(
            base + "/assignment-smoke",
            lambda route: route.fulfill(
                content_type="text/html; charset=utf-8", body=harness
            ),
        )
        await page.goto(base + "/assignment-smoke")
        await expect(page.locator(".assignment-table tbody tr")).to_have_count(5)
        assert not mutations and not uploads, (
            "Opening the workspace must not start grading"
        )
        await expect(
            page.get_by_role("complementary", name="作业预览").get_by_text(
                "核心功能完整，异常处理正确"
            )
        ).to_be_visible()
        await page.screenshot(
            path=str(diagnostics / "assignment-workspace-desktop.png"), full_page=True
        )
        await page.get_by_label("搜索作业").fill("ARMv8")
        await expect(page.locator(".assignment-table tbody tr")).to_have_count(1)
        await expect(page.locator(".assignment-preview h2")).to_have_text(
            "ARMv8 汇编实验"
        )
        await page.get_by_role("button", name="清除筛选").click()
        await page.get_by_label("筛选课程").select_option("移动互联网导论")
        await page.get_by_label("筛选状态").select_option("READY")
        await expect(page.locator(".assignment-table tbody tr")).to_have_count(1)
        await expect(page.locator(".assignment-preview h2")).to_have_text(
            "编程小作业 2"
        )
        await page.get_by_role("button", name="清除筛选").click()
        await page.get_by_role("button", name="编程小作业 1", exact=True).click()
        await page.get_by_role("link", name="查看完整作业", exact=False).click()
        await expect(page.locator(".full-question")).to_contain_text(
            "支持任务新增、修改与删除"
        )
        await expect(
            page.get_by_role("heading", name="导入学生作业", exact=True)
        ).not_to_be_visible()
        await page.screenshot(
            path=str(diagnostics / "assignment-content-desktop.png"), full_page=True
        )
        await page.get_by_role("button", name="材料与批改", exact=True).click()
        await expect(
            page.get_by_role("checkbox", name="材料整理完成后自动批改")
        ).not_to_be_checked()
        await page.locator("#file-upload").set_input_files(
            {"name": "fixture.zip", "mimeType": "application/zip", "buffer": b"fixture"}
        )
        await page.get_by_role("button", name="上传并整理材料", exact=True).click()
        await expect(page.get_by_text("材料已受理", exact=True)).to_be_visible()
        assert 'name="auto_grade"\r\n\r\nfalse' in uploads[-1]
        await page.get_by_role("button", name="查看本批成绩", exact=True).click()
        await expect(page.get_by_text("20260001 - 张三", exact=True)).to_be_visible()
        await expect(
            page.get_by_role("button", name="导出为Excel", exact=True)
        ).to_be_enabled()
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.screenshot(
            path=str(diagnostics / "assignment-results-mobile.png"), full_page=True
        )
        assert await page.evaluate(
            "document.documentElement.scrollWidth <= innerWidth + 1"
        )
        await page.evaluate("window.fixtureRouter.push('/assignments')")
        await expect(page.locator(".assignment-table tbody tr")).to_have_count(5)
        await page.screenshot(
            path=str(diagnostics / "assignment-workspace-mobile.png"), full_page=True
        )
        assert await page.evaluate(
            "document.documentElement.scrollWidth <= innerWidth + 1"
        )
        await page.get_by_role("link", name="新建作业", exact=True).click()
        await page.get_by_label("课程名称", exact=True).fill("新课程")
        await page.get_by_label("作业名称", exact=True).fill("新作业")
        await page.get_by_label("题目要求", exact=True).fill("完成一个队列。")
        await page.get_by_role("button", name="保存并查看作业", exact=True).click()
        await expect(page.locator(".full-question")).to_have_text("完成一个队列。")
        assert mutations[0]["class_name"] == "全部班级"
        assert mutations[0]["rubric"]["功能实现"]["score"] == 60
        await browser.close()
    assert not errors, errors
    print(
        "PASS: default assignment index, real content preview, combined filters, independent content/materials/results tabs, safe upload default, results actions, creation and mobile overflow."
    )


asyncio.run(main())
