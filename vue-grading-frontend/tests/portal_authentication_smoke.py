"""Teacher OTP collection with delayed Worker acknowledgements; fixture traffic only."""

import asyncio
import json
import os
import re
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright, expect


async def main():
    base = os.environ.get("RPA_UI_BASE_URL", "http://127.0.0.1:6017")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + "/src/components/RpaTaskPanel.vue", timeout=10) as response:
        source = response.read().decode()
    vue = next(p for p in re.findall(r'from "([^"]+)"', source) if "/vue.js" in p)
    with opener.open(base + "/src/router/index.js", timeout=10) as response:
        router_source = response.read().decode()
    router = next(p for p in re.findall(r'from "([^"]+)"', router_source) if "vue-router" in p)
    job = dict(job_id="auth-fixture", task_type="AUTH_BOOTSTRAP", status="WAITING_USER",
               stage="AUTHENTICATE", control_mode="HUMAN", control_epoch=3,
               waiting_reason="LOGIN_CODE", login_code_method="wechat_work", message="请在企业微信中查看本人动态验证码，在下方输入并提交",
               target=dict(course_name="教务账号", assignment_name="首次认证"),
               grading_files=[], material_batches=[], events=[])
    received, acknowledge = asyncio.Event(), asyncio.Event()
    resumes, errors = [], []
    filled = False

    async def api(route):
        nonlocal filled
        path = route.request.url.split("?", 1)[0]
        if route.request.method == "POST":
            assert path.endswith("/auth-fixture/resume"), path
            assert filled, "Login resumed before the Worker confirmed code entry"
            assert not route.request.post_data, "Code must not enter a REST control body"
            resumes.append(path)
            job.update(control_mode="AUTO", status="RUNNING", waiting_reason="",
                       message="正在登录并处理可信设备提示", control_epoch=4)
            data = job
        elif path.endswith("/api/profile"):
            data = dict(id=77, username="fixture", role="teacher")
        elif path.endswith("/api/rpa/jobs"):
            data = [job]
        elif path.endswith("/api/rpa/jobs/auth-fixture"):
            data = job
        elif "/api/materials" in path:
            data = dict(items=[], grading_jobs=[])
        else:
            raise AssertionError("Unexpected fixture API: " + path)
        await route.fulfill(content_type="application/json", body=json.dumps(data, ensure_ascii=False))

    async def socket(ws):
        async def message(raw):
            nonlocal filled
            value = json.loads(raw)
            if value["type"] == "auth":
                ws.send(json.dumps(dict(type="ready")))
                ws.send(json.dumps(dict(type="frame", epoch=3, revision=1, page_id="0",
                                        width=1440, height=1000, pages=[],
                                        image="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aC1kAAAAASUVORK5CYII=")))
            elif value["type"] == "input":
                action = value["action"]
                assert action["kind"] == "login_code" and action["text"] == "123456"
                assert action["epoch"] == 3 and action["page_id"] == "0"
                received.set()
                await acknowledge.wait()
                filled = True
                ws.send(json.dumps(dict(type="input_ack", id=value["id"])))
        ws.on_message(lambda raw: asyncio.create_task(message(raw)))

    harness = f'''<html><head><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app" style="max-width:900px;margin:auto"></div><script type="module">
    import {{createApp,h}} from '{vue}';import {{createRouter,createMemoryHistory}} from '{router}';
    import Panel from '/src/components/RpaTaskPanel.vue';
    localStorage.setItem('access_token','fixture-token');localStorage.setItem('user_info',JSON.stringify({{user_id:77,role:'teacher'}}));
    const router=createRouter({{history:createMemoryHistory(),routes:[{{path:'/',component:{{render:()=>null}}}}]}});
    createApp({{render:()=>h(Panel,{{taskId:'auth-fixture',embedded:true}})}}).use(router).mount('#app');</script></body></html>'''
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport=dict(width=1100, height=1000))
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.route("**/api/**", api)
        await page.route_web_socket("**/api/rpa/jobs/*/stream", socket)
        await page.route(base + "/auth-fixture", lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=harness))
        await page.goto(base + "/auth-fixture")
        field = page.get_by_label("企业微信动态验证码", exact=True)
        await expect(field).to_be_visible()
        await expect(page.get_by_text("请输入企业微信动态验证码", exact=True)).to_be_visible()
        assert await page.get_by_text("验证码已请求", exact=False).count() == 0
        await expect(page.get_by_text("实时连接", exact=True)).to_be_visible()
        await field.fill("12x")
        await page.get_by_role("button", name="提交验证码并继续登录", exact=True).click()
        await expect(page.get_by_role("alert")).to_contain_text("4 至 12 位数字")
        assert not received.is_set() and not resumes
        await page.set_viewport_size(dict(width=390, height=844))
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
        assert (await field.bounding_box())["y"] < 700, "Teacher code input is hidden below the browser canvas"
        output = Path(__file__).resolve().parents[2] / ".runtime/diagnostics"
        output.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(output / "portal-code-input-mobile.png"), full_page=True)
        await field.fill("123456")
        await page.get_by_role("button", name="提交验证码并继续登录", exact=True).click()
        await asyncio.wait_for(received.wait(), 5)
        await expect(page.get_by_role("button", name="正在提交…", exact=True)).to_be_disabled()
        assert not resumes
        assert await field.input_value() == ""
        acknowledge.set()
        await expect(page.get_by_text("正在登录并处理可信设备提示", exact=True)).to_be_visible()
        assert len(resumes) == 1
        assert not await page.evaluate("JSON.stringify(localStorage).includes('123456')")
        assert not errors, errors
        await browser.close()
    print("PASS: teacher code entry, format validation, delayed Worker acknowledgement before login, secret clearing, and mobile layout; fixture APIs only.")


asyncio.run(main())
