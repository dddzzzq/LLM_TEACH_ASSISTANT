"""Real Chromium, intercepted school pages and synthetic credentials only."""

import asyncio
import json
import tempfile
import unittest
import zipfile
import io
import html
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
from unittest.mock import patch, AsyncMock
import httpx

from playwright.async_api import async_playwright

from app.rpa.errors import RpaIssue
from app.rpa.portal_auth import PortalAuth
from app.rpa.portal_catalog import PortalCatalog, clean_url, normalize_record
from app.rpa.portal_download import selected_scope
from app.rpa.portal_sessions import PortalSessions
from app.rpa.task_controller import Task
from app.rpa.worker import Manager, app


class Owner:
    def __init__(self, context, page):
        self.context, self.page = context, page
        self.lock = asyncio.Lock()
        self.state = {}

    def update(self, **fields):
        self.state.update(fields)


async def wait_until(predicate, timeout=10):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.05)


class PortalBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_class_dropdown_is_read_and_switched_before_export(self):
        from app.rpa.portal_adapter import PortalAdapter

        browser = await self.playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        try:
            page = await context.new_page()
            body = """<script>window.currentClass=0;function selectClass(e,id){document.querySelectorAll('.classli').forEach(n=>n.classList.remove('active'));e.classList.add('active');window.currentClass=id}</script>
            <ul style="display:none"><li class="classli active" onclick="selectClass(this,0)">全部班级</li><li class="classli" onclick="selectClass(this,22)">班级A</li><li class="classli" onclick="selectClass(this,23)">班级B</li></ul>
            <ul class=morePop><li><a onclick="window.exportedClass=window.currentClass">导出作业附件</a></li></ul><div id=downloadcenter></div>"""
            await page.route(
                "**/*",
                lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=body),
            )
            await page.goto("https://mooc1.chaoxing.com/work?courseid=11&clazzid=22")
            classes = await PortalCatalog(context, lambda **kw: None).classes(page, {})
            self.assertEqual(
                {c["platform_id"]: c["name"] for c in classes}, {"22": "班级A", "23": "班级B"}
            )
            adapter = PortalAdapter(
                context,
                {"course_name": "课", "assignment_name": "作业", "portal": {"class_id": "23"}},
            )
            self.assertEqual(await adapter.current_class_id(page), "0")
            await adapter.select_target_class(page, "22")
            self.assertEqual(await page.evaluate("window.currentClass"), 22)
            adapter.assignment_confirmed = True
            await adapter.open_export()
            self.assertEqual(await adapter.current_class_id(page), "23")
            self.assertEqual(await page.evaluate("window.exportedClass"), 23)
            with self.assertRaises(RpaIssue):
                await adapter.select_target_class(page, "999")
            self.assertEqual(await page.evaluate("window.currentClass"), 23)
        finally:
            await browser.close()

    async def test_catalog_reads_native_and_scripted_classes_without_all_option(self):
        browser = await self.playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        try:
            page = await context.new_page()
            await page.route(
                "**/*",
                lambda route: route.fulfill(
                    content_type="text/html; charset=utf-8",
                    body="""<select id="clazzid"><option value="0">全部班级</option><option value="22">A班</option><option value="23">A班</option></select><ul><li onclick="changeClass('24')">B班</li></ul>""",
                ),
            )
            await page.goto("https://learning.xidian.edu.cn/work?courseid=11")
            result = await PortalCatalog(context, lambda **kw: None).classes(page, {})
            self.assertEqual(
                {r["platform_id"]: r["name"] for r in result},
                {"22": "A班", "23": "A班", "24": "B班"},
            )
        finally:
            await browser.close()

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.playwright = await async_playwright().start()
        self.manager = SimpleNamespace(playwright=self.playwright, proxy=None, tasks={})
        self.sessions = PortalSessions(self.manager, self.root / "profiles")
        self.session = {
            "connection_id": str(uuid4()),
            "profile_generation": 1,
            "credential_version": 1,
            "trust_device": True,
        }
        self.credentials = {"username": "teacher-test", "password": "synthetic-password"}
        self.sent_codes = 0
        self.documents = []
        self.tasks = []

        async def credentials(ticket):
            return dict(self.credentials)

        self.patches = [
            patch.object(Task, "route", lambda task, route: self.fixture(route)),
            patch("app.rpa.portal_sessions.take_credentials", new=credentials),
        ]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        for task in self.tasks:
            await task.cancel()
            await asyncio.gather(task.runner, return_exceptions=True)
        for item in self.patches:
            item.stop()
        await self.playwright.stop()
        self.temp.cleanup()

    async def fixture(self, route):
        p = urlsplit(route.request.url)
        if route.request.resource_type == "document":
            self.documents.append((p.hostname, p.path))
        query = parse_qs(p.query)
        if p.hostname == "learning.xidian.edu.cn":
            trusted = "fixture_trusted=yes" in route.request.headers.get("cookie", "")
            if not trusted:
                await route.fulfill(
                    content_type="text/html",
                    body="<script>location.replace('https://ids.xidian.edu.cn/login')</script>",
                )
                return
            body = """<p>个人空间</p><span data-teacher-account="teacher-test">teacher-test</span>
            <ul><li data-term="2025-2026第一学期"><a title="同名课程" data-class-name="A班" href="https://mooc1.chaoxing.com/course?courseid=11&clazzid=22">同名课程</a></li>
            <li data-term="2026-2027第一学期"><a title="同名课程" data-class-name="B班" href="https://mooc1.chaoxing.com/course?courseid=12&clazzid=23">同名课程</a></li></ul>"""
        elif p.hostname == "ids.xidian.edu.cn" and p.path == "/login":
            body = """<form action="https://ids.xidian.edu.cn/login" onsubmit="event.preventDefault(); if(username.value==='teacher-test' && password.value==='synthetic-password') location.href='/mfa'; else document.querySelector('#error').textContent='密码错误';">
            <input id="username" name="username"><input id="password" type="password"><button id="login_submit" type="submit">登录</button></form><p id="error"></p>"""
        elif p.hostname == "ids.xidian.edu.cn" and p.path == "/mfa":
            body = """<button onclick="window.codes=(window.codes||0)+1">获取验证码</button><input name="otp" placeholder="请输入">
            <button onclick="if(document.querySelector('input').value==='123456') document.querySelector('#trust').hidden=false">登录</button>
            <div id="trust" role="dialog" hidden><h2>可信浏览器提示</h2>
            <button onclick="document.cookie='fixture_trusted=yes;Domain=xidian.edu.cn;Path=/;Max-Age=3600;Secure;SameSite=Lax';location.href='https://learning.xidian.edu.cn/portal'">信任此设备</button>
            <button>仅本次登录</button></div>"""
        elif p.hostname == "mooc1.chaoxing.com" and p.path == "/course":
            course = query["courseid"][0]
            clazz, name = ("22", "A班") if course == "11" else ("23", "B班")
            body = f'<h1>同名课程</h1><select name="clazzId"><option value="{clazz}">{name}</option></select><a href="/works?courseid={course}">作业</a>'
        elif p.hostname == "mooc1.chaoxing.com" and p.path == "/works":
            course = query["courseid"][0]
            work = "33" if course == "11" else "34"
            body = f'<ul><li><a href="/work?workid={work}&courseid={course}" title="第一次作业">第一次作业</a></li></ul><a href="https://learning.xidian.edu.cn/portal">返回课程列表</a>'
        elif p.hostname == "mooc1.chaoxing.com" and p.path == "/work":
            body = '<h1>第一次作业</h1><a title="作业">作业</a>'
        else:
            raise AssertionError("Unexpected external request: " + p.hostname + p.path)
        await route.fulfill(content_type="text/html; charset=utf-8", body=body)

    def start_task(self, kind="AUTH_BOOTSTRAP", target=None, session=None):
        manager = Manager(self.root / "tasks")
        manager.playwright, manager.proxy, manager.portal_sessions = (
            self.playwright,
            None,
            self.sessions,
        )
        manager.download_root = self.root / "downloaded"
        task = manager.start(
            str(uuid4()),
            target or {"course_name": "教务账号", "assignment_name": "首次认证"},
            portal_session={**(session or self.session), "credential_ticket": "a" * 64},
            task_type=kind,
        )
        self.tasks.append(task)
        return task

    async def ready(self, task):
        async with asyncio.timeout(10):
            while True:
                obs = await task.observe()
                if obs["ready"]:
                    return obs
                self.assertNotIn(task.state["status"], {"FAILED", "INTERRUPTED"}, task.state)
                await asyncio.sleep(0.05)

    async def action(self, task, tool, **fields):
        obs = await self.ready(task)
        result = await task.apply_action(
            {
                "action_id": str(uuid4()),
                "observation_id": obs["observation_id"],
                "revision": obs["revision"],
                "epoch": obs["epoch"],
                "action": {"tool": tool, **fields},
                "skill_version": "fixture-page-agent",
            }
        )
        self.assertEqual(result["outcome"], "completed", task.state)

    async def wait_auth(self, task, field):
        async with asyncio.timeout(10):
            while not (await self.ready(task))["observation"]["authentication"][field]:
                await asyncio.sleep(0.05)

    async def click(self, task, name, course_id=None):
        obs = await self.ready(task)
        items = [item for item in obs["observation"]["elements"] if item["name"] == name]
        if course_id:
            items = [
                item for item in items if item.get("platform_ids", {}).get("course_id") == course_id
            ]
        items.sort(key=lambda item: item.get("role") not in {"link", "button"})
        self.assertTrue(items, (name, obs))
        await self.action(task, "click", element_ref=items[0]["ref"])

    async def trusted(self):
        async with self.sessions.open(self.session, self.root / "prelogin") as context:
            await context.add_cookies(
                [
                    {
                        "name": "fixture_trusted",
                        "value": "yes",
                        "domain": ".xidian.edu.cn",
                        "path": "/",
                        "secure": True,
                        "expires": 2000000000,
                    }
                ]
            )

    async def test_login_waits_for_username_blur_verification_check_before_submit(self):
        for need_captcha in (True, False):
            with self.subTest(need_captcha=need_captcha):
                async with self.sessions.open(self.session, self.root / "preflight") as context:
                    page = context.pages[0]

                    async def route(request):
                        if urlsplit(request.request.url).path.endswith("checkNeedCaptcha.htl"):
                            await asyncio.sleep(0.3)
                            await request.fulfill(
                                content_type="application/json",
                                body=json.dumps({"isNeed": need_captcha}),
                            )
                        else:
                            await request.fulfill(
                                content_type="text/html",
                                body="""<form onsubmit="event.preventDefault()">
                              <input id=username onblur="checkNeedCaptcha()"><input type=password>
                              <button id=login_submit type=button onclick="window.submissions=(window.submissions||0)+1;
                              if(!window.checked) document.querySelector('#error').textContent='Verification code error';
                              else if(needCaptcha) document.querySelector('#sliderDiv').hidden=false;
                              else window.accepted=true">Login</button></form><p id=error></p><div id=sliderDiv hidden>challenge</div>
                              <script>var needCaptcha=false;function checkNeedCaptcha(){fetch('/authserver/checkNeedCaptcha.htl').then(r=>r.json()).then(data=>{needCaptcha=data.isNeed;window.checked=true})}</script>""",
                            )

                    await page.route("**/*", route)
                    await page.goto("https://ids.xidian.edu.cn/authserver/login")
                    auth = PortalAuth(Owner(context, page), dict(self.credentials), True)
                    self.assertTrue(await auth.fill_credentials(await auth.login_frame()))
                    self.assertEqual(await page.locator("#error").inner_text(), "")
                    self.assertEqual(await page.evaluate("window.submissions"), 1)
                    self.assertEqual(await page.locator("#sliderDiv").is_visible(), need_captcha)
                    if not need_captcha:
                        self.assertTrue(await page.evaluate("window.accepted"))

    async def test_english_login_errors_are_reported_without_refilling_credentials(self):
        for message, code in [
            ("Verification code error", "LOGIN_VERIFICATION_FAILED"),
            ("Invalid password", "INVALID_CREDENTIALS"),
        ]:
            with self.subTest(message=message):
                async with self.sessions.open(self.session, self.root / "english-error") as context:
                    page = context.pages[0]
                    await page.route(
                        "**/*",
                        lambda route: route.fulfill(
                            content_type="text/html",
                            body="<input id=username><input type=password><p>" + message + "</p>",
                        ),
                    )
                    await page.goto("https://ids.xidian.edu.cn/authserver/login")
                    auth = PortalAuth(Owner(context, page), dict(self.credentials), True)
                    auth.filled = True
                    with self.assertRaises(RpaIssue) as error:
                        await auth.observation()
                    self.assertEqual(error.exception.code, code)

    async def test_login_does_not_submit_when_verification_check_fails(self):
        async with self.sessions.open(self.session, self.root / "preflight-failed") as context:
            page = context.pages[0]

            async def route(request):
                if urlsplit(request.request.url).path.endswith("checkNeedCaptcha.htl"):
                    await request.fulfill(status=503, content_type="application/json", body="{}")
                else:
                    await request.fulfill(
                        content_type="text/html",
                        body="""<input id=username onblur="checkNeedCaptcha()"><input type=password>
                      <button id=login_submit onclick="window.submitted=true">Login</button>
                      <script>var needCaptcha=false;function checkNeedCaptcha(){fetch('/authserver/checkNeedCaptcha.htl')}</script>""",
                    )

            await page.route("**/*", route)
            await page.goto("https://ids.xidian.edu.cn/authserver/login")
            auth = PortalAuth(Owner(context, page), dict(self.credentials), True)
            with self.assertRaises(RpaIssue) as error:
                await auth.fill_credentials(await auth.login_frame())
            self.assertEqual(error.exception.code, "LOGIN_CHECK_FAILED")
            self.assertFalse(auth.filled)
            self.assertIsNone(await page.evaluate("window.submitted"))

    async def finish_fixture_catalog(self, task):
        await self.action(task, "read_course_list")
        for course in ("11:22", "12:23"):
            await self.click(task, "同名课程", course.split(":")[0])
            await self.click(task, "作业")
            await self.action(task, "read_homework_list", course_id=course)
            await self.action(task, "return_to_course_list")
        await self.action(task, "finish_catalog")
        await asyncio.wait_for(task.runner, 10)

    async def test_same_task_handles_credentials_mfa_trust_and_profile_reuse(self):
        task = self.start_task()
        obs = await self.ready(task)
        context, page = task.context, task.page
        self.assertTrue(obs["observation"]["authentication"]["login_form"])
        await self.action(task, "fill_portal_credentials")
        await self.wait_auth(task, "otp_required")
        await self.action(task, "request_login_code")
        self.assertEqual(task.state["control_mode"], "HUMAN")
        self.assertEqual(task.state["waiting_reason"], "LOGIN_CODE")
        await task.page.locator("input[name=otp]").fill(
            "123456"
        )  # Teacher input in the existing browser.
        await task.resume()
        obs = await self.ready(task)
        self.assertTrue(obs["observation"]["authentication"]["otp_entered"])
        self.assertNotIn("123456", json.dumps(obs))
        await self.action(task, "submit_login_code")
        await self.wait_auth(task, "trust_prompt")
        await self.action(task, "apply_device_trust")
        obs = await self.ready(task)
        self.assertEqual(obs["stage"], "READING_CATALOG")
        self.assertEqual(task.state["status"], "RUNNING")
        self.assertIs(task.context, context)
        self.assertIs(task.page, page)
        login_count = self.documents.count(("ids.xidian.edu.cn", "/login"))
        self.assertEqual(login_count, 1)
        await self.finish_fixture_catalog(task)
        self.assertEqual(task.state["status"], "SUCCEEDED", task.state)
        self.assertEqual(task.portal_result()["verified_subject"], "teacher-test")
        self.assertTrue(task.portal_result()["catalog"]["complete"])
        self.assertEqual(self.documents.count(("ids.xidian.edu.cn", "/login")), login_count)
        journal = (task.manager.root / (task.state["job_id"] + ".json")).read_text()
        for secret in ["teacher-test", "synthetic-password", "123456", "a" * 64]:
            self.assertNotIn(secret, journal)
        task2 = self.start_task()
        obs = await self.ready(task2)
        self.assertEqual(obs["stage"], "READING_CATALOG")
        await self.finish_fixture_catalog(task2)
        self.assertEqual(task2.state["status"], "SUCCEEDED", task2.state)
        self.assertFalse(any(r["tool"] == "fill_portal_credentials" for r in task2.recent))
        with self.assertRaises(ValueError):
            async with self.sessions.open(
                {**self.session, "profile_generation": 0}, self.root / "invalid"
            ):
                self.fail("invalid profile opened")
        async with self.sessions.open(
            {**self.session, "connection_id": str(uuid4())}, self.root / "other"
        ) as context:
            self.assertFalse(any(c["name"] == "fixture_trusted" for c in await context.cookies()))

    async def test_catalog_uses_page_agent_actions_and_keeps_same_name_ids_separate(self):
        await self.trusted()
        task = self.start_task("CATALOG_SYNC")
        await self.action(task, "read_course_list")
        self.assertEqual(len(task.catalog_collection.observation()["courses"]), 2)
        with self.assertRaises(ValueError):
            task.catalog_collection.finish()
        for index, course in enumerate(["11:22", "12:23"]):
            await self.click(task, "同名课程", course.split(":")[0])
            await self.click(task, "作业")
            await self.action(task, "read_homework_list", course_id=course)
            if index == 0:
                with self.assertRaises(RpaIssue):
                    await task.catalog_collection.read_homeworks("12:23")
                await self.click(task, "返回课程列表")
        returned = {}
        original_return = task.return_to_course_list

        async def verify_final_return():
            await original_return()
            returned["original_page"] = task.page is task.catalog_collection.list_page
            returned["verified_list"] = await task.catalog_collection.is_course_list(task.page)

        with patch.object(task, "return_to_course_list", new=verify_final_return):
            await self.action(task, "finish_catalog")
        await asyncio.wait_for(task.runner, 10)
        result = task.portal_result()["catalog"]
        self.assertTrue(result["complete"])
        self.assertEqual([c["platform_id"] for c in result["courses"]], ["11:22", "12:23"])
        self.assertEqual(
            [c["homeworks"][0]["platform_id"] for c in result["courses"]], ["33", "34"]
        )
        self.assertEqual(result["courses"][0]["classes"], [{"platform_id": "22", "name": "A班"}])
        self.assertFalse(task.state.get("export_attempted"))

        # The last course intentionally omitted return_to_course_list above.
        # finish_catalog must restore it before publishing or closing the task.
        self.assertEqual(returned, {"original_page": True, "verified_list": True})
        self.assertIsNone(task.catalog_collection.observation()["current_course_id"])
        self.assertTrue(any("已返回课程列表" in e["message"] for e in task.state["events"]))

    async def test_finish_restores_list_and_requests_missing_list_coverage_without_failing(self):
        await self.trusted()
        original = self.fixture

        async def fixture(route):
            if urlsplit(route.request.url).hostname != "learning.xidian.edu.cn":
                return await original(route)
            body = """<!doctype html><p>个人空间</p><span data-teacher-account="teacher-test">teacher-test</span>
            <ul id="courses" style="height:150px;overflow-y:auto;margin:0">
            <li style="height:250px"><a href="https://mooc1.chaoxing.com/course?courseid=11&clazzid=22">课程11</a></li>
            <li style="height:250px"><a href="https://mooc1.chaoxing.com/course?courseid=12&clazzid=23">课程12</a></li></ul>"""
            await route.fulfill(content_type="text/html; charset=utf-8", body=body)

        with patch.object(self, "fixture", new=fixture):
            task = self.start_task("CATALOG_SYNC")
            await self.action(task, "read_course_list")
            self.assertFalse(task.catalog_collection.progress["courses_complete"])
            for index, course in enumerate(("11:22", "12:23")):
                await self.click(task, f"课程{index + 11}")
                await self.click(task, "作业")
                await self.action(task, "read_homework_list", course_id=course)
                if index == 0:
                    await self.action(task, "return_to_course_list")
            await self.action(task, "finish_catalog")
            self.assertEqual(task.state["status"], "RUNNING")
            self.assertEqual(task.state["navigation_failures"], 0)
            self.assertNotIn("portal_catalog", task.state)
            obs = (await self.ready(task))["observation"]["catalog"]
            self.assertIsNone(obs["current_course_id"])
            self.assertTrue(obs["viewport"]["needs_scroll"])
            await self.action(task, "scroll", delta_y=800)
            await self.action(task, "read_course_list")
            self.assertTrue(task.catalog_collection.progress["courses_complete"])
            await self.action(task, "finish_catalog")
            await asyncio.wait_for(task.runner, 10)
            self.assertEqual(task.state["status"], "SUCCEEDED")
            self.assertEqual(len(task.portal_result()["catalog"]["courses"]), 2)

    async def test_course_coverage_uses_iframe_list_instead_of_outer_sidebar(self):
        await self.trusted()
        task = self.start_task("CATALOG_SYNC")
        await self.ready(task)
        cards = '<ul><li><a href="https://mooc1.chaoxing.com/course?courseid=11&clazzid=22">课程11</a></li><li><a href="https://mooc1.chaoxing.com/course?courseid=12&clazzid=23">课程12</a></li></ul>'
        await task.page.set_content(
            '<p>个人空间</p><span data-teacher-account="teacher-test">teacher-test</span><aside style="height:3000px"><button>下一页</button></aside><iframe style="position:fixed;top:100px;width:900px;height:500px" srcdoc="'
            + html.escape(cards, quote=True)
            + '"></iframe>'
        )
        await self.action(task, "read_course_list")
        catalog = (await self.ready(task))["observation"]["catalog"]
        self.assertTrue(catalog["courses_complete"])
        self.assertEqual(catalog["viewport"]["record_count"], 2)
        self.assertFalse(catalog["viewport"]["has_next"])

    async def test_scroll_reads_lazy_homeworks_in_iframe_document_and_inner_container(self):
        await self.trusted()
        for inner in (False, True):
            with self.subTest(inner=inner):
                task = self.start_task("CATALOG_SYNC")
                await self.action(task, "read_course_list")
                await self.click(task, "同名课程", "11")
                await self.click(task, "作业")
                items = "".join(
                    f'<li style="height:180px"><a href="https://mooc1.chaoxing.com/work?workid={i}&courseid=11">作业{i}</a></li>'
                    for i in (31, 32, 33)
                )
                style = "height:200px;overflow-y:auto" if inner else ""
                doctype = "<!doctype html>" if inner else ""
                child = f"""{doctype}<body style="margin:0"><ul id="works" style="margin:0;{style}">{items}</ul>
                <script>const panel=document.querySelector('#works');const inner={str(inner).lower()};
                const scroller=inner?panel:document.scrollingElement;let batch=0,id=34;
                (inner?panel:window).addEventListener('scroll',()=>{{
                  if((inner?scroller.scrollTop:scrollY)+(inner?scroller.clientHeight:innerHeight)<scroller.scrollHeight-4||batch>=2||panel.getAttribute('aria-busy')==='true')return;
                  batch++;panel.setAttribute('aria-busy','true');setTimeout(()=>{{
                    for(let n=0;n<2;n++){{const li=document.createElement('li');li.style.height='180px';li.innerHTML='<a href="https://mooc1.chaoxing.com/work?workid='+id+'&courseid=11">作业'+id+'</a>';panel.append(li);id++;}}
                    panel.removeAttribute('aria-busy');
                  }},350);
                }});</script></body>"""
                await task.page.set_content(
                    '<aside style="height:3000px"><button>下一页</button></aside><iframe style="position:fixed;top:0;left:100px;width:900px;height:320px" srcdoc="'
                    + html.escape(child, quote=True)
                    + '"></iframe>'
                )
                task.update(auto_actions=59)
                await self.action(task, "read_homework_list", course_id="11:22")
                course = task.catalog_collection.progress["courses"]["11:22"]
                self.assertFalse(course["complete"])
                viewport = (await self.ready(task))["observation"]["catalog"]["viewport"]
                self.assertEqual(viewport["container"], "inner" if inner else "document")
                self.assertTrue(viewport["needs_scroll"])
                for _ in range(8):
                    await self.action(task, "scroll", delta_y=800)
                    await self.action(task, "read_homework_list", course_id="11:22")
                    if course["complete"]:
                        break
                self.assertTrue(course["complete"], (await self.ready(task))["observation"])
                self.assertEqual(set(course["homeworks"]), {str(i) for i in range(31, 38)})
                self.assertEqual(await task.page.evaluate("scrollY"), 0)
                self.assertEqual(task.state["control_mode"], "AUTO")
                await task.cancel()
                await asyncio.gather(task.runner, return_exceptions=True)

    async def test_semester_walk_returns_after_empty_courses_in_same_tab_and_popups(self):
        await self.trusted()
        original = self.fixture
        semester = "2026-2027第一学期"
        for popup in (False, True):

            async def fixture(route):
                p = urlsplit(route.request.url)
                query = parse_qs(p.query)
                if p.hostname == "learning.xidian.edu.cn":
                    target = ' target="_blank"' if popup else ""
                    cards = "".join(
                        f'<li><a{target} href="https://mooc1.chaoxing.com/course?courseid={i}&clazzid={i + 11}">课程{i}</a></li>'
                        for i in (11, 12, 13)
                    )
                    body = f'<p>个人空间</p><span data-teacher-account="teacher-test">teacher-test</span><select><option>{semester}</option></select><ul>{cards}</ul>'
                elif p.path == "/course":
                    course = query["courseid"][0]
                    body = f'<h1>课程{course}</h1><a href="/works?courseid={course}">作业</a>'
                elif p.path == "/works":
                    course = query["courseid"][0]
                    # Explicit empty state is complete even on a long page;
                    # neither course has a visible link back to the course list.
                    body = (
                        '<p>暂未发布作业</p><div style="height:2000px"></div>'
                        if course in {"11", "13"}
                        else '<a href="/work?workid=34&courseid=12">作业一</a>'
                    )
                else:
                    return await original(route)
                await route.fulfill(content_type="text/html; charset=utf-8", body=body)

            with self.subTest(popup=popup), patch.object(self, "fixture", new=fixture):
                task = self.start_task("CATALOG_SYNC")
                await self.action(task, "read_course_list")
                list_page = task.page
                with self.assertRaises(RpaIssue):
                    await task.catalog_collection.read_homeworks("11:22")
                for index, course in enumerate(("11:22", "12:23", "13:24")):
                    self.assertEqual(
                        task.catalog_collection.observation()["next_course_id"], course
                    )
                    await self.click(task, f"课程{index + 11}")
                    await self.click(task, "作业")
                    if index == 0:
                        task.update(auto_actions=59)
                    await self.action(task, "read_homework_list", course_id=course)
                    progress = task.catalog_collection.observation()
                    self.assertEqual(progress["current_course_id"], course)
                    self.assertTrue(progress["return_required"])
                    self.assertEqual(progress["completed_courses"], index + 1)
                    self.assertEqual(progress["courses"][index]["empty"], index != 1)
                    self.assertEqual(task.state["control_mode"], "AUTO")
                    if index < 2:
                        with self.assertRaises(ValueError):
                            task.catalog_collection.finish()
                    await self.action(task, "return_to_course_list")
                    self.assertIs(task.page, list_page)
                    self.assertIsNone(task.catalog_collection.observation()["current_course_id"])
                    # Re-reading the list must retain visited/empty courses.
                    await self.action(task, "read_course_list")
                    self.assertEqual(
                        task.catalog_collection.observation()["completed_courses"], index + 1
                    )
                await self.action(task, "finish_catalog")
                await asyncio.wait_for(task.runner, 10)
                courses = task.portal_result()["catalog"]["courses"]
                self.assertEqual([c["term"] for c in courses], [semester] * 3)
                self.assertEqual([len(c["homeworks"]) for c in courses], [0, 1, 0])
                self.assertFalse(task.state.get("export_attempted"))

    async def test_return_to_course_list_requires_successful_list_bookmark(self):
        await self.trusted()
        task = self.start_task("CATALOG_SYNC")
        await self.ready(task)
        with self.assertRaises(RpaIssue):
            await task.return_to_course_list()
        await self.action(task, "read_course_list")
        await self.click(task, "同名课程", "11")
        await self.click(task, "作业")
        await task.page.set_content("<h1>作业</h1><p>加载中</p>")
        with self.assertRaises(RpaIssue):
            await task.catalog_collection.read_homeworks("11:22")
        self.assertFalse(task.catalog_collection.progress["courses"]["11:22"]["complete"])
        self.assertFalse(task.catalog_collection.observation()["return_required"])

    async def test_course_page_coverage_survives_returning_to_first_page(self):
        await self.trusted()
        original = self.fixture

        async def fixture(route):
            p = urlsplit(route.request.url)
            if p.hostname != "learning.xidian.edu.cn":
                return await original(route)
            second = parse_qs(p.query).get("page") == ["2"]
            course, clazz = (12, 23) if second else (11, 22)
            nav = (
                '<a href="/portal">上一页</a>'
                if second
                else '<a rel="next" href="/portal?page=2">下一页</a>'
            )
            body = f'<p>个人空间</p><span data-teacher-account="teacher-test">teacher-test</span><ul><li data-term="2026-2027第一学期"><a href="https://mooc1.chaoxing.com/course?courseid={course}&clazzid={clazz}">课程{course}</a></li></ul>{nav}'
            await route.fulfill(content_type="text/html; charset=utf-8", body=body)

        with patch.object(self, "fixture", new=fixture):
            task = self.start_task("CATALOG_SYNC")
            await self.action(task, "read_course_list")
            self.assertFalse(task.catalog_collection.observation()["courses_complete"])
            await self.click(task, "下一页")
            await self.action(task, "read_course_list")
            self.assertTrue(task.catalog_collection.observation()["courses_complete"])
            await self.click(task, "上一页")
            await self.action(task, "read_course_list")
            self.assertTrue(task.catalog_collection.observation()["courses_complete"])
            self.assertEqual(task.catalog_collection.observation()["total_courses"], 2)
            await self.click(task, "课程11")
            await self.click(task, "作业")
            await self.action(task, "read_homework_list", course_id="11:22")
            await self.action(task, "return_to_course_list")
            self.assertTrue(task.catalog_collection.observation()["courses_complete"])
            self.assertEqual(task.catalog_collection.observation()["next_course_id"], "12:23")
            # Same route with a different semester must fail list verification.
            await task.page.locator("li").evaluate(
                "e=>e.setAttribute('data-term','2026-2027第二学期')"
            )
            self.assertFalse(await task.catalog_collection.is_course_list(task.page))

    async def test_catalog_pagination_is_navigated_by_agent_not_reader(self):
        await self.trusted()
        task = self.start_task("CATALOG_SYNC")
        await self.action(task, "read_course_list")
        await self.click(task, "同名课程", "11")
        await self.click(task, "作业")
        await task.page.set_content(
            """<div id="rows"><a href="/work?workid=33&courseid=11">作业一</a></div><a rel="next" href="#" onclick="event.preventDefault();setTimeout(()=>{let a=document.createElement('a');a.href='/work?workid=34&courseid=11';a.textContent='作业二';document.querySelector('#rows').replaceChildren(a);this.hidden=true},300)">下一页</a>"""
        )
        await self.action(task, "read_homework_list", course_id="11:22")
        course = task.catalog_collection.progress["courses"]["11:22"]
        self.assertFalse(course["complete"])
        self.assertEqual(list(course["homeworks"]), ["33"])
        await self.click(task, "下一页")
        await task.page.get_by_text("作业二", exact=True).wait_for()
        await self.action(task, "read_homework_list", course_id="11:22")
        self.assertTrue(course["complete"])
        self.assertEqual(list(course["homeworks"]), ["33", "34"])

    async def test_legacy_work_rows_read_dom_ids_and_script_links_without_clicks(self):
        await self.trusted()
        task = self.start_task("CATALOG_SYNC")
        await self.action(task, "read_course_list")
        await self.click(task, "同名课程", "11")
        await self.click(task, "作业")
        rows = [
            '<li data-id="41"><h2 class="list_li_tit">课后作业7</h2><a class="piyueBtn" onclick="review()">批阅</a></li>',
            '<li id="work_42"><h2 class="list_li_tit">ARMv8 汇编实验</h2><a class="piyueBtn" onclick="review()">批阅</a></li>',
            '<li><h2 class="list_li_tit">课后作业5</h2><input type="hidden" name="workId" value="43"><a class="piyueBtn" onclick="review()">批阅</a></li>',
            '<li><h2 class="list_li_tit">课后作业4</h2><a class="piyueBtn" href="javascript:;" onclick="review(\'/work?workId=44&amp;courseid=11\')">批阅</a></li>',
            '<li><h2 class="list_li_tit">课后作业3</h2><button onclick="review({workId:\'45\'})">查看</button></li>',
            '<li data-homework-id="46"><h2 class="list_li_tit">课后作业2</h2><a class="piyueBtn">批阅</a></li>',
            '<li><h2 class="list_li_tit">课后作业1</h2><a href="/work?WORKID=47&amp;courseid=11">查看</a></li>',
            '<li><h2 class="list_li_tit">具名函数入口作业</h2><a class="piyueBtn" onclick="reviewById(\'11\', \'48\', this)">批阅</a></li>',
            '<li hidden data-workid="999"><h2>已隐藏的旧作业</h2></li>',
        ]
        await task.page.set_content(
            "<script>window.clicks=0;function review(){window.clicks++} function reviewById(courseId,workId,element){window.clicks++}</script><ul>"
            + "".join(rows)
            + "</ul>"
        )
        await self.action(task, "read_homework_list", course_id="11:22")
        course = task.catalog_collection.progress["courses"]["11:22"]
        self.assertEqual(list(course["homeworks"]), [str(i) for i in range(41, 49)])
        self.assertEqual(
            [v["title"] for v in course["homeworks"].values()],
            [
                "课后作业7",
                "ARMv8 汇编实验",
                "课后作业5",
                "课后作业4",
                "课后作业3",
                "课后作业2",
                "课后作业1",
                "具名函数入口作业",
            ],
        )
        self.assertEqual(await task.page.evaluate("window.clicks"), 0)
        self.assertEqual(task.state["last_catalog_read"]["rows"], 8)
        self.assertTrue(course["complete"])

    async def test_work_row_missing_or_conflicting_ids_reports_sanitized_evidence(self):
        await self.trusted()
        task = self.start_task("CATALOG_SYNC")
        await self.action(task, "read_course_list")
        await self.click(task, "同名课程", "11")
        await self.click(task, "作业")
        for identity, code in [
            ("", "HOMEWORK_IDS_UNAVAILABLE"),
            ('data-workid="41"', "CATALOG_AMBIGUOUS"),
        ]:
            await task.page.set_content(
                f'<ul><li {identity}><h2 class="list_li_tit">真实作业标题</h2><a class="piyueBtn" onclick="review(\'opaque-id\')" href="/work?workId={"42" if identity else ""}&amp;token=synthetic-private-token">批阅</a></li></ul>'
            )
            with self.assertRaises(RpaIssue) as caught:
                await task.catalog_collection.read_homeworks("11:22")
            self.assertEqual(caught.exception.code, code)
            evidence = caught.exception.evidence
            self.assertGreater(evidence["frames"][0]["candidate_count"], 0)
            self.assertEqual(evidence["frames"][0]["legacy_work_rows"], 1)
            self.assertNotIn("synthetic-private-token", json.dumps(evidence))
            self.assertNotIn("opaque-id", json.dumps(evidence))
            self.assertEqual(task.catalog_collection.progress["courses"]["11:22"]["homeworks"], {})

    async def test_script_course_cards_in_srcdoc_are_read_before_ids_are_visible(self):
        await self.trusted()
        names = [
            "嵌入式操作系统内核原理与分析",
            "自主可控嵌入式系统",
            "新生研讨与学科导论(I)",
            "移动互联网导论",
        ]
        cards = """<base href="https://learning.xidian.edu.cn/portal?ticket=synthetic-routing-secret"><select><option selected>2026-2027第一学期</option></select>
        <script>function enterCourse(id){window.open('https://mooc1.chaoxing.com/course?courseid='+id+'&clazzid=22')}</script>"""
        for index, name in enumerate(names):
            cards += f'<div class="myde_course_item" cname="{name}"><a href="javascript:;" onclick="enterCourse(\'{11 + index}\')"><img alt="课程封面"><span>{name}</span></a></div>'

        async def fixture(task, route):
            if urlsplit(route.request.url).hostname == "learning.xidian.edu.cn":
                await route.fulfill(
                    content_type="text/html; charset=utf-8",
                    body='<p>个人空间</p><span data-teacher-account="teacher-test">teacher-test</span><iframe width="1200" height="800" srcdoc="'
                    + html.escape(cards, quote=True)
                    + '"></iframe>',
                )
            else:
                await self.fixture(route)

        with patch.object(Task, "route", fixture):
            task = self.start_task("CATALOG_SYNC")
            await self.action(task, "read_course_list")
            entries = task.catalog_collection.observation()["courses"]
            self.assertEqual([entry["title"] for entry in entries], names)
            self.assertEqual(len({entry["course_id"] for entry in entries}), 4)
            self.assertTrue(all(entry["identity_pending"] for entry in entries))
            self.assertTrue(all(entry["term"] == "2026-2027第一学期" for entry in entries))
            self.assertNotIn("synthetic-routing-secret", json.dumps(task.state))
            with self.assertRaises(ValueError):
                task.catalog_collection.finish()
            reference = entries[0]["course_id"]
            with self.assertRaises(RpaIssue):
                await task.catalog_collection.read_homeworks(reference)
            await self.click(task, names[0])
            await self.click(task, "作业")
            await self.action(task, "read_homework_list", course_id=reference)
            resolved = task.catalog_collection.progress["courses"][reference]
            self.assertEqual(resolved["platform_id"], "11:22")
            self.assertEqual(list(resolved["homeworks"]), ["33"])
            self.assertEqual(resolved["classes"], [{"platform_id": "22", "name": "A班"}])
            self.assertFalse(
                task.catalog_collection.observation()["courses"][0]["identity_pending"]
            )

    async def test_card_ids_do_not_require_a_direct_url_or_query_whitelist_match(self):
        async with self.sessions.open(self.session, self.root / "cards") as context:
            await context.route(
                "**/*", lambda route: route.fulfill(content_type="text/html", body="<p>fixture</p>")
            )
            page = context.pages[0]
            await page.goto("https://learning.xidian.edu.cn/portal")
            await page.set_content("""<div cname="同名课程" courseid="11" clazzid="22"><a href="javascript:;" onclick="enterCourse()">进入</a></div>
            <div cname="同名课程" data-course-id="12"><a href="https://mooc1.chaoxing.com/course?courseid=12&clazzid=23&ticket=synthetic-link-secret">进入</a></div>""")
            rows = await PortalCatalog(context, lambda **kw: None).records(
                "course", allow_unresolved=True
            )
            self.assertEqual([row["platform_id"] for row in rows], ["11:22", "12:23"])
            self.assertTrue(all(row["url"] == "" for row in rows))
            self.assertNotIn("synthetic-link-secret", json.dumps(rows))

    async def test_structured_download_verifies_ids_after_clicking_a_script_card(self):
        await self.trusted()

        async def fixture(task, route):
            if urlsplit(route.request.url).hostname == "learning.xidian.edu.cn":
                await route.fulfill(
                    content_type="text/html; charset=utf-8",
                    body="""<p>个人空间</p><span data-teacher-account="teacher-test">teacher-test</span>
                <script>function enterCourse(){window.open('https://mooc1.chaoxing.com/course?courseid=11&clazzid=22')}</script>
                <div class="myde_course_item" cname="同名课程"><a href="javascript:;" onclick="enterCourse()">同名课程</a></div>""",
                )
            else:
                await self.fixture(route)

        with patch.object(Task, "route", fixture):
            task = self.start_task(
                "",
                target={
                    "course_name": "同名课程",
                    "assignment_name": "第一次作业",
                    "portal": {
                        "course_id": "11:22",
                        "homework_id": "33",
                        "class_id": "22",
                        "class_name": "A班",
                    },
                },
            )
            await self.ready(task)
            self.assertFalse(task.adapter.course_confirmed)
            await self.click(task, "同名课程")
            await self.ready(task)
            self.assertTrue(task.adapter.course_confirmed)
            await self.click(task, "作业")
            await self.click(task, "第一次作业")
            await self.ready(task)
            self.assertTrue(task.adapter.assignment_confirmed)
            self.assertFalse(task.state.get("export_attempted"))

    async def test_worker_api_uses_regular_tasks_and_private_result(self):
        await self.trusted()
        manager = Manager(self.root / "api-tasks")
        manager.playwright, manager.proxy, manager.portal_sessions = (
            self.playwright,
            None,
            self.sessions,
        )
        manager.download_root = self.root / "downloaded"
        task_id = str(uuid4())
        with (
            patch.object(app.state, "manager", manager, create=True),
            patch.object(app.state, "token", "portal-test-token", create=True),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://worker",
                headers={"X-RPA-Token": "portal-test-token"},
            ) as client:
                response = await client.post(
                    "/tasks/" + task_id,
                    json={
                        "target": {"course_name": "教务账号", "assignment_name": "首次认证"},
                        "task_type": "AUTH_BOOTSTRAP",
                        "portal_session": {**self.session, "credential_ticket": "a" * 64},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                task = manager.tasks[task_id]
                self.tasks.append(task)
                await self.finish_fixture_catalog(task)
                public = (await client.get("/tasks/" + task_id)).json()
                self.assertNotIn("verified_subject", public)
                result = (await client.get("/tasks/" + task_id + "/portal-result")).json()
                self.assertEqual(result["verified_subject"], "teacher-test")
                self.assertNotIn("teacher-test", (manager.root / (task_id + ".json")).read_text())

    async def test_selected_download_uses_agent_navigation_and_existing_export_controller(self):
        await self.download_flow(False)

    async def test_all_classes_download_without_catalog_class_ids(self):
        await self.download_flow(True)

    async def test_reexport_updates_existing_rows_and_downloads_each_class(self):
        await self.download_flow(True, replace_rows=True)

    async def download_flow(self, all_classes, replace_rows=False):
        await self.trusted()
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("20260001_张三/report.txt", "student answer")
        exports = 0
        html = """<h1 class=courseName>同名课程</h1><div id=grade><h2>第一次作业</h2><div><ul class=morePop><a onclick="document.querySelector('[role=dialog]').hidden=false">导出作业附件</a></ul></div></div>
        <div role=dialog hidden>导出设置<label class=export-range><input type=checkbox>A班</label><label class=export-range><input type=checkbox>B班</label>
        <a class=confirmDown onclick="fetch('/export');this.closest('[role=dialog]').hidden=true;document.querySelector('#downloadcenter').innerHTML='<table><tbody><tr data-id=task-a><td>第一次作业 A班 导出成功 <a class=download_ic href=/download>下载</a></td></tr></tbody></table>'">确定</a></div><div id=downloadcenter></div>"""

        if all_classes:
            html = html.replace(
                "<div id=downloadcenter></div>",
                "<div id=downloadcenter><p class=empty>暂无下载记录</p></div>",
            )
            html = html.replace("第一次作业 A班 导出成功", "第一次作业 全部班级 导出成功")
        if replace_rows:
            from datetime import datetime, timedelta, timezone

            now = datetime.now(timezone(timedelta(hours=8)))
            old = (now - timedelta(days=4)).strftime("%m-%d %H:%M")
            current = now.strftime("%m-%d %H:%M")
            old_rows = "".join(
                f"<tr data-id=old-{name}><td>班级{name}-第一次作业(附件) {old} 导出成功</td></tr>"
                for name in ["A班", "B班"]
            )
            new_rows = "".join(
                f"<tr data-id=old-{name}><td>班级{name}-第一次作业(附件) {current} 导出成功 <a class=download_ic href=/download>下载</a></td></tr>"
                for name in ["A班", "B班"]
            )
            html = html.replace(
                "<div id=downloadcenter><p class=empty>暂无下载记录</p></div>",
                "<div id=downloadcenter><table><tbody>" + old_rows + "</tbody></table></div>",
            )
            html = html.replace(
                "<tr data-id=task-a><td>第一次作业 全部班级 导出成功 <a class=download_ic href=/download>下载</a></td></tr>",
                new_rows,
            )

        async def fixture(task, route):
            nonlocal exports
            path = urlsplit(route.request.url).path
            if path == "/work":
                await route.fulfill(content_type="text/html; charset=utf-8", body=html)
            elif path == "/export":
                exports += 1
                await route.fulfill(body="ok")
            elif path == "/download":
                await route.fulfill(
                    body=archive.getvalue(),
                    content_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="homework.zip"'},
                )
            else:
                await self.fixture(route)

        with patch.object(Task, "route", fixture):
            task = self.start_task(
                "",
                target={
                    "course_name": "同名课程",
                    "assignment_name": "第一次作业",
                    "portal": {
                        "course_id": "11:22",
                        "homework_id": "33",
                        "class_id": "" if all_classes else "22",
                        "class_name": "全部班级" if all_classes else "A班",
                    },
                },
            )
            obs = await self.ready(task)
            wrong = next(
                e
                for e in obs["observation"]["elements"]
                if e["role"] == "link" and e["platform_ids"]["course_id"] == "12"
            )
            with self.assertRaises(RpaIssue):
                await task.execute({"tool": "click", "element_ref": wrong["ref"]})
            self.assertFalse(task.adapter.course_confirmed)
            await self.click(task, "同名课程", "11")
            await self.click(task, "作业")
            await self.click(task, "第一次作业")
            await self.action(task, "open_export_settings")
            await asyncio.wait_for(task.runner, 20)
            self.assertEqual(task.state["status"], "SUCCEEDED", task.state)
            self.assertEqual(exports, 1)
            self.assertEqual(
                task.state["class_scope"]["mode"], "all" if all_classes else "selected"
            )
            self.assertNotEqual(task.state.get("waiting_reason"), "CLASS_SCOPE")
            if not all_classes:
                self.assertEqual(task.state["files"][0]["class_name"], "A班")
            self.assertEqual(task.state["files"][0]["status"], "VERIFIED")
            if replace_rows:
                self.assertEqual(len(task.state["files"]), 2)
                self.assertEqual({f["class_name"] for f in task.state["files"]}, {"A班", "B班"})

    async def test_credentials_not_filled_into_unconfigured_origin_or_mismatched_identity(self):
        async with self.sessions.open(self.session, self.root / "downloads") as context:
            await context.route(
                "**/*",
                lambda route: route.fulfill(
                    content_type="text/html", body='<input id="username"><input type="password">'
                ),
            )
            page = context.pages[0]
            await page.goto("https://untrusted.xidian.edu.cn/login")
            owner = Owner(context, page)
            auth = PortalAuth(owner, dict(self.credentials), True)
            self.assertIsNone(await auth.login_frame())
            self.assertEqual(await page.locator("#username").input_value(), "")
            await page.goto("https://learning.xidian.edu.cn/portal")
            await page.set_content('<span data-teacher-account="other-teacher">other</span>')
            auth.identity_confirmed = True
            with self.assertRaises(RpaIssue):
                await auth.identity()

    async def test_old_generation_cannot_reopen_or_delete_newer_trust_state(self):
        await self.sessions.fence(
            {**self.session, "profile_generation": 2, "credential_version": 2}
        )
        with self.assertRaises(RpaIssue):
            async with self.sessions.open(self.session, self.root / "stale"):
                self.fail("stale profile reopened")
        restored = PortalSessions(self.manager, self.sessions.root)
        with self.assertRaises(RpaIssue):
            await restored.fence(self.session)


class CatalogValidationTests(unittest.TestCase):
    def test_current_class_alias_requires_matching_platform_evidence(self):
        options = [
            {"id": "current", "name": "导出本班级作业附件", "all": False},
            {"id": "all", "name": "导出全部班级作业附件", "all": True},
        ]
        target = {"class_id": "22", "class_name": "A班"}
        self.assertEqual(selected_scope(target, options, "22")["class_ids"], ["current"])
        with self.assertRaises(RpaIssue):
            selected_scope(target, options, "23")
        identified = [
            {"id": "a", "name": "同名班级", "all": False, "platform_id": "22"},
            {"id": "b", "name": "同名班级", "all": False, "platform_id": "23"},
        ]
        self.assertEqual(selected_scope(target, identified)["class_ids"], ["a"])

    def test_unsafe_links_and_ambiguous_class_are_rejected(self):
        for url in [
            "https://evilchaoxing.com/work?workid=1",
            "http://mooc1.chaoxing.com/work?workid=1",
            "https://mooc1.chaoxing.com/work?token=secret",
            "https://user:pass@mooc1.chaoxing.com/work",
        ]:
            self.assertEqual(clean_url(url), "")
        self.assertIsNone(
            normalize_record(
                {"href": "/work?workid=1", "title": ""}, "homework", "https://mooc1.chaoxing.com/"
            )
        )
        with self.assertRaises(RpaIssue):
            selected_scope(
                {"class_name": "A班"},
                [
                    {"id": "1", "name": "A班", "all": False},
                    {"id": "2", "name": "A班", "all": False},
                ],
            )

    def test_restart_marks_operation_interrupted_and_does_not_store_ticket(self):
        with tempfile.TemporaryDirectory() as folder:
            operation_id = str(uuid4())
            manager = SimpleNamespace(tasks={})
            path = Path(folder) / (operation_id + ".json")
            path.write_text(
                json.dumps(
                    {
                        "job_id": operation_id,
                        "target": {"course_name": "fixture", "assignment_name": "fixture"},
                        "status": "RUNNING",
                        "type": "AUTH_BOOTSTRAP",
                        "state_version": 1,
                    }
                )
            )
            operations = Manager(folder)
            self.assertEqual(operations.tasks[operation_id].state["status"], "INTERRUPTED")
            self.assertNotIn("credential_ticket", path.read_text())


if __name__ == "__main__":
    unittest.main()
