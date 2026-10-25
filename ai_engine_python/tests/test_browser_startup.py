"""Real Chromium startup failures and recovery; no real portal/accounts/jobs."""

import asyncio
import json
import time
import unittest
from unittest.mock import patch

from app.rpa.browser_startup import BrowserStartup
from app.rpa.task_controller import Task
from tests import test_rpa_worker as fixtures


class BrowserStartupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await fixtures.BrowserTaskTests.asyncSetUp(self)
        self.policy = [
            patch.object(BrowserStartup, "attempts", 3),
            patch.object(BrowserStartup, "retry_delays", (0.05, 0.05)),
            patch.object(BrowserStartup, "cooldown_delays", ()),
        ]
        for item in self.policy:
            item.start()

    async def asyncTearDown(self):
        try:
            await fixtures.BrowserTaskTests.asyncTearDown(self)
        finally:
            for item in reversed(self.policy):
                item.stop()

    bare_task = fixtures.BrowserTaskTests.bare_task

    async def until(self, predicate):
        async with asyncio.timeout(8):
            while not predicate():
                await asyncio.sleep(0.02)

    def begin(self, task):
        task.startup_pending = True
        task.update(stage="OPEN_PORTAL")
        task.runner = asyncio.create_task(task.open_login())

    async def test_transient_failure_retries_then_waits_for_delayed_login(self):
        task = await self.bare_task()
        calls = []

        async def route(r):
            calls.append(r.request.url)
            if len(calls) == 1:
                await r.abort("connectionreset")
            elif "/portal" in r.request.url:
                await r.fulfill(
                    content_type="text/html",
                    body="""<script>setTimeout(()=>document.body.innerHTML=
                    '<a class=denglu href="https://ids.xidian.edu.cn/login">Login</a>',150)</script>""",
                )
            else:
                await r.fulfill(
                    content_type="text/html",
                    body="""<script>setTimeout(()=>document.body.innerHTML=
                    '<input type=password>',250)</script>""",
                )

        await task.context.route("**/*", route)
        with patch.object(BrowserStartup, "retry_delays", (0.01, 0.01)):
            self.begin(task)
            await asyncio.wait_for(task.runner, 8)
        self.assertEqual(task.state["startup_attempt"], 2)
        self.assertEqual(task.state["waiting_reason"], "LOGIN")
        self.assertTrue(await task.page.locator("input").is_visible())
        self.assertIsNone(task.state["last_error"])
        self.assertFalse(task.gate.is_set())
        self.assertFalse(task.startup_pending)
        self.assertEqual(len(calls), 3)

    async def test_exhausted_retry_keeps_browser_and_resume_reuses_same_task(self):
        task = await self.bare_task()
        healthy, count = False, 0

        async def route(r):
            nonlocal count
            count += 1
            if healthy:
                await r.fulfill(content_type="text/html", body="<input type=password>")
            else:
                await r.abort("connectionreset")

        await task.context.route("**/*", route)
        with patch.object(BrowserStartup, "retry_delays", (0.01, 0.01)):
            self.begin(task)
            await self.until(lambda: task.state.get("waiting_reason") == "BROWSER_STARTUP")
            browser, page, job = task.browser, task.page, task.state["job_id"]
            self.assertEqual(count, 3)
            self.assertFalse(page.is_closed())
            self.assertEqual(task.state["status"], "WAITING_USER")
            self.assertFalse((await task.observe())["ready"])
            self.assertEqual(
                task.state["last_error"]["evidence"]["network_codes"], ["ERR_CONNECTION_RESET"]
            )
            healthy = True
            await task.resume()
            with self.assertRaises(ValueError):
                await task.resume()  # Duplicate resume cannot start another worker.
            await asyncio.wait_for(task.runner, 5)
        self.assertIs(task.browser, browser)
        self.assertIs(task.page, page)
        self.assertEqual(task.state["job_id"], job)
        self.assertEqual(task.state["waiting_reason"], "LOGIN")
        self.assertIsNone(task.state["last_error"])
        self.assertEqual(count, 4)

    async def test_existing_login_is_not_reloaded(self):
        task = await self.bare_task()
        await task.page.set_content("<input type=password>")
        with patch.object(
            task.page, "goto", side_effect=AssertionError("must not reload a usable login")
        ):
            self.begin(task)
            await asyncio.wait_for(task.runner, 3)
        self.assertEqual(task.state["waiting_reason"], "LOGIN")

    async def test_timeout_after_login_ready_does_not_repeat_navigation(self):
        task = await self.bare_task()

        async def timeout_after_commit(*args, **kwargs):
            await task.page.set_content("<input type=password>")
            raise TimeoutError("navigation timed out")

        with patch.object(task.page, "goto", side_effect=timeout_after_commit) as goto:
            self.begin(task)
            await asyncio.wait_for(task.runner, 3)
        self.assertEqual(goto.call_count, 1)
        self.assertEqual(task.state["waiting_reason"], "LOGIN")
        self.assertIsNone(task.state["last_error"])

    async def test_login_popup_becomes_active_page(self):
        task = await self.bare_task()

        async def route(r):
            body = (
                '<a class=denglu target=_blank href="https://ids.xidian.edu.cn/login">Login</a>'
                if "/portal" in r.request.url
                else "<input type=password>"
            )
            await r.fulfill(content_type="text/html", body=body)

        await task.context.route("**/*", route)
        self.begin(task)
        await asyncio.wait_for(task.runner, 5)
        self.assertEqual(len(task.context.pages), 2)
        self.assertIs(task.page, task.context.pages[-1])
        self.assertEqual(task.state["waiting_reason"], "LOGIN")

    async def test_server_error_retries_but_client_error_stops(self):
        task = await self.bare_task()
        calls = []

        async def route(r):
            calls.append(r.request.url)
            await r.fulfill(status=503 if len(calls) == 1 else 403, body="unavailable")

        await task.context.route("**/*", route)
        with patch.object(BrowserStartup, "retry_delays", (0.01, 0.01)):
            self.begin(task)
            await self.until(lambda: task.state.get("waiting_reason") == "BROWSER_STARTUP")
        self.assertEqual(len(calls), 2)
        self.assertEqual(task.state["last_error"]["evidence"]["http_status"], 403)
        self.assertFalse(task.state["last_error"]["evidence"]["retryable"])
        await task.cancel()

    async def test_blank_http_success_is_not_login_ready_and_is_bounded(self):
        task = await self.bare_task()
        await task.context.route(
            "**/*", lambda r: r.fulfill(content_type="text/html", body="<body></body>")
        )
        with (
            patch.object(BrowserStartup, "attempt_timeout", 0.3),
            patch.object(BrowserStartup, "retry_delays", (0.01, 0.01)),
        ):
            self.begin(task)
            await self.until(lambda: task.state.get("waiting_reason") == "BROWSER_STARTUP")
        self.assertEqual(task.state["startup_attempt"], 3)
        self.assertEqual(task.state["stage"], "OPEN_PORTAL")
        self.assertFalse(task.page.is_closed())
        await task.cancel()
        self.assertTrue(task.runner.done())

    async def test_pause_interrupts_loading_without_blocking_frames_or_restarting(self):
        task = await self.bare_task()
        entered = asyncio.Event()

        async def loading(*args, **kwargs):
            entered.set()
            await asyncio.sleep(30)

        with patch.object(task.page, "goto", side_effect=loading):
            self.begin(task)
            await asyncio.wait_for(entered.wait(), 3)
            image, _, _ = await asyncio.wait_for(task.frame(), 3)
            self.assertTrue(image)
            await asyncio.wait_for(task.pause(), 2)
            self.assertEqual(task.state["waiting_reason"], "USER_TAKEOVER")
            self.assertIsNone(task.startup_operation)
            await task.page.set_content("<input type=password>")
            await task.resume()
            await asyncio.wait_for(task.runner, 3)
        self.assertEqual(task.state["waiting_reason"], "LOGIN")

    async def test_nontransient_error_is_sanitized_and_not_retried(self):
        task = await self.bare_task()
        from playwright.async_api import Error

        private = "net::ERR_CERT_AUTHORITY_INVALID https://secret:password@example.com/?token=private-value"
        with patch.object(task.page, "goto", side_effect=Error(private)) as goto:
            self.begin(task)
            await self.until(lambda: task.state.get("waiting_reason") == "BROWSER_STARTUP")
        self.assertEqual(goto.call_count, 1)
        state = json.dumps(task.state)
        for secret in ("private-value", "password@example", "secret:"):
            self.assertNotIn(secret, state)
        self.assertEqual(
            task.state["last_error"]["evidence"]["network_codes"], ["ERR_CERT_AUTHORITY_INVALID"]
        )
        await task.cancel()

    async def test_total_deadline_stops_retry_and_cancel_stops_backoff(self):
        task = await self.bare_task()
        await task.context.route("**/*", lambda r: r.fulfill(content_type="text/html", body=""))
        started = time.monotonic()
        with (
            patch.object(BrowserStartup, "total_timeout", 0.35),
            patch.object(BrowserStartup, "attempt_timeout", 1),
        ):
            self.begin(task)
            await self.until(lambda: task.state.get("waiting_reason") == "BROWSER_STARTUP")
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(task.state["startup_attempt"], 1)
        await task.cancel()

    async def test_cancel_during_backoff_does_not_retry(self):
        task = await self.bare_task()
        calls = []

        async def route(r):
            calls.append(r.request.url)
            await r.abort("connectionreset")

        await task.context.route("**/*", route)
        self.begin(task)
        await self.until(lambda: "正在准备" in task.state.get("message", ""))
        await task.cancel()
        await asyncio.sleep(0.1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(task.state["status"], "CANCELLED")

    async def test_domain_block_keeps_human_reason_without_retry(self):
        task = await self.bare_task()
        calls = []

        async def route(r):
            calls.append(r.request.url)
            task.wait_human("DOMAIN", "目标页面不在已配置的教学平台域名范围内")
            await r.abort()

        await task.context.route("**/*", route)
        self.begin(task)
        await self.until(lambda: task.state.get("waiting_reason") == "DOMAIN")
        await asyncio.sleep(0.1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(task.state["waiting_reason"], "DOMAIN")
        await task.cancel()

    async def test_cooldown_recovers_in_same_browser_and_bounded_rounds(self):
        task = await self.bare_task()
        calls = []
        browser, page = task.browser, task.page

        async def route(r):
            calls.append(r.request.url)
            if len(calls) <= 2:
                await r.abort("connectionreset")
            else:
                await r.fulfill(content_type="text/html", body="<input type=password>")

        await task.context.route("**/*", route)
        with (
            patch.object(BrowserStartup, "attempts", 2),
            patch.object(BrowserStartup, "cooldown_delays", (0.1,)),
        ):
            self.begin(task)
            await self.until(lambda: task.state.get("startup_retry_kind") == "cooldown")
            self.assertEqual(len(calls), 2)
            self.assertFalse((await task.observe())["ready"])
            await asyncio.wait_for(task.runner, 5)
        self.assertEqual(task.state["startup_round"], 2)
        self.assertEqual(task.state["startup_result"]["attempts"], 3)
        self.assertEqual(task.state["waiting_reason"], "LOGIN")
        self.assertIs(task.page, page)
        self.assertIs(task.browser, browser)
        self.assertIsNone(task.state["startup_retry_at"])

    async def test_cooldown_exhaustion_stops_and_manual_resume_gets_new_budget(self):
        task = await self.bare_task()
        calls = []

        async def route(r):
            calls.append(r.request.url)
            await r.abort("connectionreset")

        await task.context.route("**/*", route)
        with (
            patch.object(BrowserStartup, "attempts", 2),
            patch.object(BrowserStartup, "cooldown_delays", (0.05,)),
        ):
            self.begin(task)
            await self.until(lambda: task.state.get("waiting_reason") == "BROWSER_STARTUP")
            self.assertEqual(len(calls), 4)
            await asyncio.sleep(0.1)
            self.assertEqual(len(calls), 4)
            await task.page.set_content("<input type=password>")
            await task.resume()
            await asyncio.wait_for(task.runner, 3)
        self.assertEqual(len(calls), 4)
        self.assertEqual(task.state["waiting_reason"], "LOGIN")

    async def test_cancel_and_pause_interrupt_cooldown_without_more_requests(self):
        for action in ("cancel", "pause"):
            with self.subTest(action=action):
                task = await self.bare_task()
                calls = []

                async def route(r):
                    calls.append(r.request.url)
                    await r.abort("connectionreset")

                await task.context.route("**/*", route)
                with (
                    patch.object(BrowserStartup, "attempts", 1),
                    patch.object(BrowserStartup, "cooldown_delays", (30,)),
                ):
                    self.begin(task)
                    await self.until(lambda: task.state.get("startup_retry_kind") == "cooldown")
                    await asyncio.wait_for(getattr(task, action)(), 2)
                await asyncio.sleep(0.1)
                self.assertEqual(len(calls), 1)
                self.assertIsNone(task.state.get("startup_retry_at"))
                if action == "pause":
                    await task.cancel()

    async def test_page_becomes_ready_during_cooldown_without_navigation(self):
        task = await self.bare_task()
        calls = []

        async def route(r):
            calls.append(r.request.url)
            await r.abort("connectionreset")

        await task.context.route("**/*", route)
        with (
            patch.object(BrowserStartup, "attempts", 1),
            patch.object(BrowserStartup, "cooldown_delays", (10,)),
        ):
            self.begin(task)
            await self.until(lambda: task.state.get("startup_retry_kind") == "cooldown")
            await task.page.set_content("<input type=password>")
            await asyncio.wait_for(task.runner, 3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(task.state["waiting_reason"], "LOGIN")

    async def test_existing_course_list_frame_is_reused_without_portal_navigation(self):
        task = await self.bare_task()
        await task.context.route(
            "**/*",
            lambda r: r.fulfill(
                content_type="text/html", body='<div data-courseid="11">课程一</div>'
            ),
        )
        await task.page.goto("https://mooc1.chaoxing.com/courses")
        with patch.object(
            task.page, "goto", side_effect=AssertionError("must reuse the course list")
        ):
            self.begin(task)
            await asyncio.wait_for(task.runner, 3)
        self.assertEqual(task.state["waiting_reason"], "LOGIN_COMPLETE")
        self.assertEqual(task.state["startup_result"]["auth_status"], "AUTHENTICATED")

    async def test_existing_portal_uses_its_login_link_without_reloading(self):
        task = await self.bare_task()

        async def route(r):
            await r.fulfill(
                content_type="text/html",
                body='<a class=denglu href="https://ids.xidian.edu.cn/login">登录</a>'
                if "/portal" in r.request.url
                else "<input type=password>",
            )

        await task.context.route("**/*", route)
        await task.page.goto("https://learning.xidian.edu.cn/portal")
        with patch.object(
            task.page, "goto", side_effect=AssertionError("must reuse the portal login link")
        ):
            self.begin(task)
            await asyncio.wait_for(task.runner, 3)
        self.assertEqual(task.state["waiting_reason"], "LOGIN")

    async def test_active_course_page_is_preferred_over_an_old_portal_tab(self):
        task = await self.bare_task()

        async def route(r):
            await r.fulfill(
                content_type="text/html",
                body="<p>个人空间</p>"
                if "learning." in r.request.url
                else '<div data-courseid="11">课程一</div>',
            )

        await task.context.route("**/*", route)
        await task.page.goto("https://learning.xidian.edu.cn/portal")
        course = await task.context.new_page()
        await course.goto("https://mooc1.chaoxing.com/courses")
        task.page = course
        self.begin(task)
        await asyncio.wait_for(task.runner, 3)
        self.assertIs(task.page, course)
