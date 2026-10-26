"""Real DOM fixtures for scoped question reading; no school accounts/network."""

import base64
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from playwright.async_api import async_playwright

from app.rpa.errors import RpaIssue
from app.rpa.browser_startup import BrowserStartup
from app.rpa.homework_detail import HomeworkDetailReader, resource_url
from app.rpa.task_controller import Task
from app.rpa.tools import validate_action


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aU1sAAAAASUVORK5CYII="
)


class HomeworkDetailTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=True, args=["--no-sandbox"])
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        self.state = {
            "job_id": "fixture",
            "target": {
                "course_name": "课程",
                "assignment_name": "编程作业",
                "portal": {"course_id": "11:22", "homework_id": "33"},
            },
        }
        self.active = True
        self.events = []
        self.task = SimpleNamespace(
            page=self.page,
            context=self.context,
            state=self.state,
            manager=SimpleNamespace(download_root=Path(self.temp.name)),
            adapter=SimpleNamespace(course_confirmed=True),
            automatic=lambda: self.active,
            update=lambda **kw: self.state.update(kw),
            event=self.events.append,
        )
        self.reader = HomeworkDetailReader(self.task)

    async def asyncTearDown(self):
        await self.browser.close()
        await self.pw.stop()
        self.temp.cleanup()

    async def load(self, body, query="courseid=11&workid=33"):
        html = '<meta charset="utf-8"><h2>编程作业</h2>' + body
        await self.page.route(
            "**/*", lambda route: route.fulfill(body=html, content_type="text/html")
        )
        await self.page.goto("https://mooc2-ans.chaoxing.com/work/library/view-work?" + query)

    async def test_mixed_sources_preserve_order_and_exclude_answers(self):
        image = "data:image/png;base64," + base64.b64encode(PNG).decode()
        await self.load(
            f'<nav>其他课程</nav><div class="mark_item"><p>使用 TypeScript 完成程序</p><img src="{image}"/><p>报告要求如下</p><a href="/download/123">实验要求.docx</a><div class="mark_answer">学生答案不可作为题目</div></div>'
        )
        self.reader.fetch = AsyncMock(
            return_value=(
                b"fixture-docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        )
        self.assertTrue((await self.reader.observation())["ready"])
        await self.reader.read()
        detail = self.state["homework_detail"]
        self.assertEqual(
            [s["kind"] for s in detail["sources"]], ["text", "image", "text", "attachment"]
        )
        self.assertNotIn("学生答案", str(detail))
        self.assertNotIn("其他课程", str(detail))
        self.assertTrue(detail["complete"])
        self.assertTrue(Path(detail["sources"][1]["path"]).is_file())
        self.assertNotIn("url", str(detail))

    async def test_wrong_work_or_course_cannot_be_read(self):
        for query in ("courseid=99&workid=33", "courseid=11&workid=99"):
            await self.load('<div class="mark_item">题目</div>', query)
            self.assertFalse((await self.reader.observation())["ready"])
            with self.assertRaises(RpaIssue):
                await self.reader.read()

    async def test_hidden_id_conflict_and_list_headings_rejected(self):
        await self.load(
            '<input type="hidden" name="workId" value="99"><div class="mark_item">题目</div>'
        )
        self.assertIsNone(await self.reader.frame())
        await self.load('<div class="mark_item">题目</div>', "courseid=11")
        self.assertIsNone(await self.reader.frame())
        self.task.detail_page = self.page
        self.assertIsNotNone(await self.reader.frame())

    async def test_page_body_is_never_a_question_fallback(self):
        await self.load("<p>整个页面还有学生答案，不能当成题目</p>")
        with self.assertRaises(RpaIssue):
            await self.reader.read()
        self.assertNotIn("homework_detail", self.state)

    async def test_detail_title_and_loading_placeholder_are_not_requirements(self):
        await self.load("<main><h2>编程作业</h2><p>加载中...</p></main>")
        with self.assertRaises(RpaIssue):
            await self.reader.read()
        self.assertNotIn("homework_detail", self.state)

    async def test_failed_attachment_is_explicitly_incomplete(self):
        await self.load(
            '<main><p>请按附件完成</p><a href="https://example.org/task.pdf">题目.pdf</a></main>'
        )
        await self.reader.read()
        self.assertFalse(self.state["homework_detail"]["complete"])
        self.assertTrue(self.state["homework_detail"]["warnings"])

    async def test_download_endpoint_uses_attachment_name_for_binary_mime(self):
        await self.load('<main><a href="/download/123">题目.pdf</a></main>')
        self.reader.fetch = AsyncMock(return_value=(b"%PDF-fixture", "application/octet-stream"))
        await self.reader.read()
        source = self.state["homework_detail"]["sources"][0]
        self.assertTrue(source["path"].endswith(".pdf"))
        self.assertTrue(self.state["homework_detail"]["complete"])

    async def test_site_owned_attachment_download_control(self):
        await self.load(
            """<main><a href="javascript:void(0)" onclick="location.href='/generated-file'">实验要求.pdf</a></main>"""
        )
        await self.page.route(
            "**/generated-file",
            lambda route: route.fulfill(
                body=b"%PDF-fixture",
                content_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=question.pdf"},
            ),
        )
        await self.reader.read()
        source = self.state["homework_detail"]["sources"][0]
        self.assertEqual(source["name"], "question.pdf")
        self.assertIn("path", source, self.events)
        self.assertEqual(Path(source["path"]).read_bytes(), b"%PDF-fixture")
        self.assertTrue(self.state["homework_detail"]["complete"])

    async def test_attachment_login_html_and_external_redirect_are_rejected(self):
        for status, headers in (
            (200, {"content-type": "text/html"}),
            (302, {"location": "https://example.org/secret.pdf"}),
        ):
            response = SimpleNamespace(
                status=status,
                headers=headers,
                dispose=AsyncMock(),
                body=AsyncMock(return_value=b"login html"),
            )
            request = SimpleNamespace(get=AsyncMock(return_value=response))
            self.task.context = SimpleNamespace(request=request)
            with self.assertRaises(ValueError):
                await self.reader.fetch("https://mooc2-ans.chaoxing.com/task.pdf")
            self.assertEqual(request.get.await_count, 1)
            response.dispose.assert_awaited_once()
            response.body.assert_not_awaited()

    async def test_pause_during_download_does_not_publish_success(self):
        await self.load('<main><a href="/task.pdf">题目.pdf</a></main>')

        async def fetch(url):
            self.active = False
            return b"%PDF-fixture", "application/pdf"

        self.reader.fetch = fetch
        with self.assertRaises(RpaIssue):
            await self.reader.read()
        self.assertNotIn("homework_detail", self.state)

    async def test_tool_is_narrow_and_exports_are_blocked(self):
        self.assertEqual(
            validate_action({"tool": "read_homework_detail"}), {"tool": "read_homework_detail"}
        )
        with self.assertRaises(ValueError):
            validate_action({"tool": "read_homework_detail", "url": "https://example.org"})
        task = Task(None, {"task_type": "HOMEWORK_DETAIL"})
        with self.assertRaises(ValueError):
            await task.execute({"tool": "open_export_settings"})

    async def test_resource_reads_do_not_hold_page_action_and_can_pause(self):
        task = Task(
            SimpleNamespace(persist=lambda state: None),
            {
                "job_id": "fixture",
                "task_type": "HOMEWORK_DETAIL",
                "status": "RUNNING",
                "control_mode": "AUTO",
                "control_epoch": 1,
            },
        )
        task.adapter = object()
        task.gate.set()
        task.portal_auth = object()
        task.authentication_view = AsyncMock(return_value={"authenticated": True})

        async def slow_read():
            await asyncio.sleep(30)

        task.detail_reader.read = slow_read
        # An action acknowledges immediately, even when downloading would exceed
        # the backend's 15-second request timeout. Pausing cancels that read.
        await asyncio.wait_for(task.execute({"tool": "read_homework_detail"}), 1)
        self.assertIsNotNone(task.detail_operation)
        await asyncio.wait_for(task.pause(), 1)
        self.assertEqual(task.state["status"], "WAITING_USER")
        self.assertNotIn("homework_detail", task.state)

    async def test_detail_startup_reuses_verified_course_entry(self):
        entry = "https://mooc2-ans.chaoxing.com/course?courseid=11&clazzid=22"
        self.state.update(
            task_type="HOMEWORK_DETAIL", status="RUNNING", control_mode="AUTO", control_epoch=1
        )
        self.state["target"]["portal"]["course_url"] = entry
        startup = BrowserStartup(self.task)
        startup.probe = AsyncMock(side_effect=[None, "AUTHENTICATED"])
        self.task.page = SimpleNamespace(goto=AsyncMock(return_value=SimpleNamespace(status=200)))
        self.assertEqual(await startup.navigate(), "AUTHENTICATED")
        self.task.page.goto.assert_awaited_once_with(
            entry, wait_until="domcontentloaded", timeout=10000
        )

    def test_resource_urls_reject_credentials_nonplatform_and_nonhttps(self):
        for url in (
            "http://mooc2-ans.chaoxing.com/a",
            "https://chaoxing.com.evil.test/a",
            "https://user:pass@chaoxing.com/a",
            "file:///etc/passwd",
            "https://127.0.0.1/a",
        ):
            self.assertFalse(resource_url(url))
        self.assertTrue(resource_url("https://p.ananas.chaoxing.com/question.png"))


if __name__ == "__main__":
    unittest.main()
