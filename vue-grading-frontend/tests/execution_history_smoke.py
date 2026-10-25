"""Read-only execution history UI checks against intercepted fixture APIs."""
import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright, expect


async def main():
    base = os.environ.get("RPA_UI_BASE_URL", "http://127.0.0.1:6006")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + "/src/components/AgentChatWidget.vue") as response:
        source = response.read().decode()
    vue = next(p for p in re.findall(r'from "([^"]+)"', source) if "/vue.js" in p)
    with opener.open(base + "/src/router/index.js") as response:
        source = response.read().decode()
    router_module = next(p for p in re.findall(r'from "([^"]+)"', source) if "vue-router" in p)
    harness = f'''<html><head><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app"></div><script type="module">
      import {{createApp,h}} from '{vue}';import {{createRouter,createMemoryHistory}} from '{router_module}';
      import Widget from '/src/components/AgentChatWidget.vue';
      const router=createRouter({{history:createMemoryHistory(),routes:[{{path:'/',component:{{render:()=>h(Widget)}}}}]}});
      await router.push('/');createApp({{render:()=>h(Widget)}}).use(router).mount('#app');
      </script></body></html>'''
    run = dict(id="run-fixture", turn=2, status="INTERRUPTED", created_at="2026-10-22T12:00:00Z", last_seq=2)
    canary = '<img src=x onerror="globalThis.EXECUTION_XSS=true">'
    calls, errors = [], []
    state = {"denied": False}

    async def api(route):
        request = route.request
        parsed = urllib.parse.urlparse(request.url)
        path = parsed.path
        assert request.method == "GET", "Viewing/rebuilding dispatched a mutation: " + path
        if "/runs" in path:
            calls.append(request.url)
        if path.endswith("/history"):
            data = {"messages": []}
        elif state["denied"] and "/runs" in path:
            await route.fulfill(status=404, content_type="application/json", body=json.dumps({"error": "会话或执行记录不存在，或无权查看"}))
            return
        elif path.endswith("/runs"):
            data = {"runs": [run]}
        elif path.endswith("/context"):
            data = {"run": run, "projection": {"messages": [{"role": "user", "content": canary}, {"role": "tool", "tool_call_id": "pending", "content": '{"outcome":"unknown"}'}], "unresolved_tool_calls": ["pending"]}}
        elif path.endswith("/run-fixture"):
            after = int(urllib.parse.parse_qs(parsed.query).get("after_seq", [0])[0])
            row = {"id": f"event-{after + 1}", "seq": after + 1, "type": "user.message" if after == 0 else "model.message", "payload": {"content": canary if after == 0 else "工具调用等待核对"}}
            data = {"run": run, "events": [row], "next_seq": after + 1, "has_more": after == 0}
        else:
            data = []
        await route.fulfill(content_type="application/json", body=json.dumps(data, ensure_ascii=False))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        await context.add_init_script('localStorage.setItem("access_token","execution-fixture");localStorage.setItem("user_info",JSON.stringify({user_id:1,username:"fixture",role:"teacher"}));localStorage.setItem("current_session_id","session-fixture");')
        page = await context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.route("**/api/**", api)
        await page.route(base + "/execution-history-smoke", lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=harness))
        await page.goto(base + "/execution-history-smoke")
        await expect(page.get_by_role("button", name="执行记录", exact=True)).to_be_enabled()
        assert not calls, "Opening the chat should not fetch private execution payloads"
        await page.get_by_role("button", name="执行记录", exact=True).click()
        dialog = page.get_by_role("dialog", name="会话执行记录")
        await expect(dialog).to_be_visible()
        await dialog.get_by_role("button", name=re.compile("第 2 轮")).click()
        await dialog.get_by_text("1. 用户输入", exact=True).click()
        assert json.loads(await dialog.locator("pre").first.inner_text())["content"] == canary
        assert await dialog.locator("img").count() == 0
        assert not await page.evaluate("Boolean(globalThis.EXECUTION_XSS)")
        await dialog.get_by_role("button", name="加载更多事件", exact=True).click()
        await expect(dialog.get_by_text("2. 模型回复", exact=True)).to_be_visible()
        await expect(dialog.get_by_role("button", name="加载更多事件", exact=True)).to_have_count(0)
        await dialog.get_by_role("button", name="重建上下文", exact=True).click()
        await expect(dialog.get_by_role("status")).to_contain_text("1 次工具调用缺少结果")
        await expect(dialog.get_by_text(re.compile("本操作只读取记录"))).to_be_visible()
        assert len([url for url in calls if url.endswith("/context")]) == 1
        await page.set_viewport_size({"width": 390, "height": 844})
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), "Mobile page overflow"
        box = await dialog.bounding_box()
        assert box["width"] <= 390 and box["x"] >= 0
        out = Path(__file__).resolve().parents[2] / ".runtime/diagnostics"
        out.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(out / "execution-history-mobile.png"))
        state["denied"] = True
        await dialog.get_by_role("button", name="刷新记录", exact=True).click()
        await expect(dialog.get_by_role("alert")).to_have_text("会话或执行记录不存在，或无权查看")
        await page.keyboard.press("Escape")
        await expect(dialog).to_have_count(0)
        assert not errors, errors
        await browser.close()
    print("PASS: execution entry, ordered events, cursor pagination, read-only reconstruction, unresolved result warning, escaped payloads, access errors and mobile layout; fixture APIs only.")


asyncio.run(main())
