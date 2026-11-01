"""A catalog sync reads each question in its original browser, without a new job."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

from playwright.async_api import async_playwright

from app.rpa.errors import RpaIssue
from app.rpa.portal_adapter import PortalAdapter
from app.rpa.task_controller import Task


class CatalogHomeworkDetailTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_reads_multiple_details_and_restores_same_tab_popup_and_iframe_lists(self):
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                for popup, iframe, delayed_third, long_manifest in (
                    (False, False, False, False),
                    (True, False, False, False),
                    (False, True, False, False),
                    (True, True, False, False),
                    (True, True, True, False),
                    (True, True, False, True),
                ):
                    work_ids = (
                        tuple(range(33, 40))
                        if long_manifest
                        else (33, 34, 35, 36)
                        if delayed_third
                        else (33, 34)
                    )
                    with (
                        self.subTest(
                            popup=popup,
                            iframe=iframe,
                            delayed_third=delayed_third,
                            long_manifest=long_manifest,
                        ),
                        tempfile.TemporaryDirectory() as folder,
                    ):
                        context = await browser.new_context()
                        course_visits = []
                        try:

                            async def route(request):
                                parsed = urlsplit(request.request.url)
                                if parsed.path == "/course":
                                    course_visits.append(parse_qs(parsed.query)["courseid"][0])
                                if parsed.path == "/courses":
                                    body = '<a href="/course?courseid=11&clazzid=22">课程</a>'
                                    if long_manifest:
                                        body += '<a href="/course?courseid=12&clazzid=23">下一门课程</a>'
                                elif parsed.path == "/course" and parse_qs(parsed.query).get(
                                    "courseid"
                                ) == ["12"]:
                                    body = "<p>暂无作业</p>"
                                elif parsed.path == "/course" and iframe:
                                    body = '<iframe height="800" src="/works?courseid=11&clazzid=22"></iframe>'
                                elif parsed.path in {"/course", "/works"}:
                                    target = ' target="_blank"' if popup else ""

                                    def link(i):
                                        if delayed_third and i == 35:
                                            return '<li data-workid="35"><a href="javascript:void(0)" onclick="showWork(35)">作业35</a></li>'
                                        style = ' style="height:110px"' if long_manifest else ""
                                        return f'<li{style}><a{target} href="/detail?courseid=11&workid={i}">作业{i}</a></li>'

                                    body = (
                                        '<script>function showWork(workId){setTimeout(()=>window.open("/detail?courseid=11&workid="+workId),650)}</script><ul>'
                                        + "".join(link(i) for i in work_ids)
                                        + "</ul>"
                                    )
                                    if long_manifest:
                                        body = body.replace(
                                            "<ul>",
                                            '<ul style="height:220px;overflow-y:auto;margin:0">',
                                        )
                                else:
                                    work = parse_qs(parsed.query)["workid"][0]
                                    body = f'<h2>作业{work}</h2><div class="mark_item">要求{work}<div class="mark_answer">学生答案</div></div>'
                                await request.fulfill(
                                    content_type="text/html; charset=utf-8", body=body
                                )

                            await context.route("**/*", route)
                            page = await context.new_page()
                            await page.goto("https://mooc1.chaoxing.com/courses")
                            original = {"course_name": "目录", "assignment_name": "同步"}
                            task = Task(
                                SimpleNamespace(
                                    download_root=Path(folder), persist=lambda state: None
                                ),
                                {
                                    "job_id": "combined",
                                    "task_type": "CATALOG_SYNC",
                                    "target": original,
                                    "status": "RUNNING",
                                    "control_mode": "AUTO",
                                    "control_epoch": 1,
                                },
                            )
                            task.context, task.page = context, page
                            task.adapter = PortalAdapter(context, original)
                            task.authentication_view = AsyncMock(
                                return_value={"authenticated": True}
                            )
                            task.prepare_captcha = AsyncMock(return_value=False)
                            task.gate.set()

                            async def click(name):
                                task.adapter.active_page = task.page
                                view = await task.adapter.observe()
                                entries = [e for e in view["elements"] if e["name"] == name]
                                entries.sort(key=lambda e: e.get("role") not in {"link", "button"})
                                entry = entries[0]
                                await task.execute({"tool": "click", "element_ref": entry["ref"]})

                            await task.execute({"tool": "read_course_list"})
                            await click("课程")
                            await task.execute({"tool": "read_homework_list", "course_id": "11:22"})
                            collection = task.catalog_collection
                            if long_manifest:
                                manifest_view = (await task.navigation_view())["catalog"]
                                self.assertEqual(manifest_view["phase"], "COLLECT_HOMEWORKS")
                                self.assertFalse(manifest_view["manifest_ready"])
                                self.assertEqual(manifest_view["next_step"], "SCROLL_HOMEWORK_LIST")
                                with self.assertRaises(RpaIssue):
                                    await task.execute(
                                        {"tool": "select_homework_detail", "homework_id": "33"}
                                    )
                                source = task.page
                                await task.execute({"tool": "return_to_course_list"})
                                self.assertIs(task.page, source)
                                await task.execute({"tool": "finish_catalog"})
                                self.assertIs(task.page, source)
                                self.assertEqual(task.state["status"], "RUNNING")
                                await task.execute({"tool": "scroll", "delta_y": 800})
                                await task.execute(
                                    {"tool": "read_homework_list", "course_id": "11:22"}
                                )
                                manifest_view = (await task.navigation_view())["catalog"]
                                self.assertTrue(manifest_view["manifest_ready"])
                                self.assertEqual(len(manifest_view["homework_manifest"]), 7)
                                works_frame = next(f for f in task.page.frames if "/works" in f.url)
                                await works_frame.locator("ul").evaluate("e => e.scrollTop = 0")
                                await task.execute(
                                    {"tool": "read_homework_list", "course_id": "11:22"}
                                )
                                manifest_view = (await task.navigation_view())["catalog"]
                                self.assertTrue(manifest_view["viewport"]["needs_scroll"])
                                self.assertTrue(manifest_view["manifest_ready"])
                                self.assertEqual(manifest_view["phase"], "READ_REQUIREMENTS")
                                self.assertEqual(
                                    manifest_view["next_step"], "SELECT_HOMEWORK_DETAIL"
                                )
                            self.assertFalse(collection.observation()["return_required"])
                            with self.assertRaises(ValueError):
                                collection.finish()
                            with self.assertRaises(RpaIssue):
                                await task.execute(
                                    {"tool": "select_homework_detail", "homework_id": "99"}
                                )
                            for work in map(str, work_ids):
                                await task.execute(
                                    {"tool": "select_homework_detail", "homework_id": work}
                                )
                                selected_view = await task.navigation_view()
                                self.assertEqual(selected_view["task_type"], "CATALOG_SYNC")
                                self.assertIn("homework_detail", selected_view)
                                self.assertNotIn("catalog", selected_view)
                                self.assertFalse(selected_view["homework_detail"]["open"])
                                with self.assertRaises(RpaIssue):
                                    await task.execute({"tool": "return_to_course_list"})
                                await click("作业" + work)
                                detail_view = await task.detail_observation()
                                self.assertTrue(
                                    detail_view["ready"],
                                    (detail_view, task.page.url, task.adapter.assignment_confirmed),
                                )
                                await task.execute({"tool": "read_homework_detail"})
                                operation = task.detail_operation
                                await asyncio.wait_for(operation, 10)
                                self.assertIsNone(task.detail_reader, task.state)
                                self.assertEqual(task.state["status"], "RUNNING")
                                self.assertEqual(task.state["target"], original)
                                self.assertEqual(len(context.pages), 1)
                                resumed_view = await task.navigation_view()
                                self.assertIn("catalog", resumed_view)
                                self.assertNotIn("homework_detail", resumed_view)
                                manifest = resumed_view["catalog"]["homework_manifest"]
                                self.assertEqual(
                                    sum(row["status"] == "READ" for row in manifest),
                                    list(map(str, work_ids)).index(work) + 1,
                                )
                            self.assertTrue(collection.observation()["return_required"])
                            await task.execute({"tool": "read_homework_list", "course_id": "11:22"})
                            self.assertEqual(await collection.pending_homeworks(), [])
                            if long_manifest:
                                await task.page.set_content("<h1>课程首页</h1>")
                                finished_view = (await task.navigation_view())["catalog"]
                                self.assertFalse(finished_view["homework_list_ready"])
                                self.assertEqual(
                                    finished_view["next_step"], "RETURN_TO_COURSE_LIST"
                                )
                                await task.execute({"tool": "return_to_course_list"})
                                self.assertEqual(
                                    collection.observation()["next_course_id"], "12:23"
                                )
                                self.assertEqual(course_visits.count("11"), 1)
                                await click("下一门课程")
                                await task.execute(
                                    {"tool": "read_homework_list", "course_id": "12:23"}
                                )
                                await task.execute({"tool": "return_to_course_list"})
                            await task.execute({"tool": "finish_catalog"})
                            result = task.portal_result()["catalog"]
                            self.assertTrue(result["details_included"])
                            self.assertEqual(task.state["status"], "SUCCEEDED")
                            details = [w["detail"] for w in result["courses"][0]["homeworks"]]
                            self.assertEqual(
                                [d["homework_id"] for d in details], list(map(str, work_ids))
                            )
                            self.assertEqual(
                                [d["sources"][0]["text"] for d in details],
                                [f"要求{i}" for i in work_ids],
                            )
                            self.assertNotIn("学生答案", str(details))
                            self.assertNotIn("homework_detail", task.state)
                            self.assertNotIn("catalog_progress", task.state)
                        finally:
                            await context.close()
            finally:
                await browser.close()
