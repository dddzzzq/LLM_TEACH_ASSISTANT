"""Real Chromium, local intercepted challenges; no school accounts or live login."""

import asyncio
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from playwright.async_api import async_playwright

from app.rpa.slider_captcha import captcha_distance, solve_slider_captcha
from app.rpa.task_controller import Task, allowed_url
from app.rpa.portal_adapter import PortalAdapter
from app.rpa.tools import validate_action


CHALLENGE = """<style>
#sliderDiv {width:220px;height:110px;position:relative}
canvas {position:absolute;top:0;left:0;width:220px;height:80px}
.slider {position:absolute;top:85px;width:20px;height:20px;background:blue}
</style><input id=username value=private-account><input type=password value=private-secret>
<div id=sliderDiv><canvas width=220 height=80></canvas>
<canvas class=block width=220 height=80></canvas><div class=slider></div></div>
<script>
const canvases = document.querySelectorAll('canvas');
function pattern(ctx, x) {
  ctx.fillStyle='white';ctx.fillRect(x,20,25,25);
  ctx.fillStyle='black';ctx.fillRect(x+4,24,7,13);ctx.fillRect(x+14,32,7,8);
}
const bg=canvases[0].getContext('2d');bg.fillStyle='#111';bg.fillRect(0,0,220,80);
pattern(bg,100);pattern(canvases[1].getContext('2d'),5);
let start=null;window.releases=0;window.complete=true;
document.addEventListener('mousedown', e => {start=e.clientX});
document.addEventListener('mouseup', e => {
  window.releases++;
  if(start!==null && window.complete && Math.abs(e.clientX-start-95)<4) {
    document.querySelector('#sliderDiv').remove();
    document.querySelectorAll('input').forEach(e=>e.remove());
    const a=document.createElement('a');a.textContent='个人空间';document.body.append(a);
  }
  start=null;
});
</script>"""


class DistanceTests(unittest.TestCase):
    def test_matches_alpha_crop_and_rejects_invalid_images(self):
        bg = np.zeros((80, 220), np.uint8)
        block = np.zeros((80, 220, 4), np.uint8)
        patch = np.full((25, 25), 255, np.uint8)
        patch[4:17, 4:11] = 0
        patch[12:20, 14:21] = 0
        bg[20:45, 100:125] = patch
        block[20:45, 5:30, :3] = patch[:, :, None]
        block[20:45, 5:30, 3] = 255

        def png(value):
            return cv2.imencode(".png", value)[1].tobytes()

        self.assertEqual(captcha_distance(png(bg), png(block)), 95)
        with self.assertRaises(ValueError):
            captcha_distance(b"broken", png(block))


class MemoryManager:
    def persist(self, state):
        self.saved = copy.deepcopy(state)


class SliderBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=True, args=["--no-sandbox"])
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        await self.page.route(
            "**/*",
            lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=CHALLENGE),
        )
        await self.page.goto("https://learning.xidian.edu.cn/portal")

    async def asyncTearDown(self):
        await self.browser.close()
        await self.pw.stop()

    def task(self):
        task = Task(
            MemoryManager(),
            {
                "job_id": "slider-test",
                "target": {"course_name": "C", "assignment_name": "A"},
                "status": "WAITING_USER",
                "stage": "AUTHENTICATE",
                "control_mode": "HUMAN",
                "control_epoch": 1,
                "waiting_reason": "LOGIN",
            },
        )
        task.browser, task.context, task.page = self.browser, self.context, self.page
        task.adapter = PortalAdapter(self.context, task.state["target"])
        return task

    async def test_tool_protocol_resume_observe_execute_and_reconcile_receipt(self):
        task = self.task()
        self.assertEqual(
            validate_action({"tool": "solve_slider_captcha"}), {"tool": "solve_slider_captcha"}
        )
        with self.assertRaises(ValueError):
            validate_action({"tool": "solve_slider_captcha", "max_attempts": 100})
        self.assertFalse((await task.observe())["ready"])
        self.assertEqual(await self.page.evaluate("window.releases"), 0)
        await task.resume()
        self.assertEqual(task.state["auth_status"], "CAPTCHA_REQUIRED")
        observation = await task.observe()
        self.assertTrue(observation["ready"])
        self.assertEqual(observation["stage"], "SOLVE_CAPTCHA")
        self.assertNotIn("private-", str(observation))
        payload = {
            "action_id": observation["observation_id"],
            "observation_id": observation["observation_id"],
            "revision": observation["revision"],
            "epoch": observation["epoch"],
            "action": {"tool": "solve_slider_captcha"},
        }
        receipt = await task.apply_action(payload)
        self.assertEqual(receipt["outcome"], "completed")
        self.assertEqual(task.state["auth_status"], "AUTHENTICATED")
        self.assertEqual(task.state["stage"], "LOCATE_COURSE")
        self.assertEqual(await self.page.evaluate("window.releases"), 1)
        await task.apply_action(payload)
        self.assertEqual(await self.page.evaluate("window.releases"), 1)

    async def test_failed_challenge_consumes_budget_and_cannot_resume_it_again(self):
        task = self.task()
        await task.resume()
        # An unsupported challenge fails quickly without guessing a drag offset.
        await self.page.locator("canvas.block").evaluate("e=>e.remove()")
        await task.execute({"tool": "solve_slider_captcha"})
        self.assertEqual(task.state["captcha_attempts"], 3)
        self.assertEqual(task.state["control_mode"], "HUMAN")
        with self.assertRaises(ValueError):
            await task.resume()
        self.assertEqual(await self.page.evaluate("window.releases"), 0)

    async def test_iframe_challenge_uses_top_level_mouse_coordinates(self):
        await self.page.unroute("**/*")
        await self.page.route(
            "**/*",
            lambda route: route.fulfill(
                content_type="text/html",
                body=CHALLENGE
                if "/challenge" in route.request.url
                else '<iframe style="margin:60px" src="/challenge"></iframe>',
            ),
        )
        await self.page.goto("https://learning.xidian.edu.cn/portal")
        result = await solve_slider_captcha(self.page)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(await self.page.frames[1].evaluate("window.releases"), 1)

    async def test_no_challenge_does_not_claim_solved(self):
        await self.page.locator("#sliderDiv").evaluate("e=>e.remove()")
        self.assertEqual(
            await solve_slider_captcha(self.page), {"status": "not_present", "attempts": 0}
        )

    async def test_rejected_drags_are_bounded_to_three_attempts(self):
        await self.page.evaluate("window.complete=false")
        result = await solve_slider_captcha(self.page, max_attempts=100)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(await self.page.evaluate("window.releases"), 3)

    async def test_other_actions_cannot_run_in_captcha_stage(self):
        task = self.task()
        await task.resume()
        with self.assertRaises(ValueError):
            await task.execute({"tool": "scroll", "delta_y": 100})
        self.assertEqual(await self.page.evaluate("window.releases"), 0)

    async def test_cancelled_drag_releases_mouse(self):
        movement = asyncio.Event()
        original_move = self.page.mouse.move

        async def move(*args, **kwargs):
            await original_move(*args, **kwargs)
            if await self.page.evaluate("window.releases===0") and args[0] > 25:
                movement.set()
                await asyncio.sleep(30)

        with patch.object(self.page.mouse, "move", side_effect=move):
            operation = asyncio.create_task(solve_slider_captcha(self.page))
            await asyncio.wait_for(movement.wait(), 5)
            operation.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await operation
        self.assertEqual(await self.page.evaluate("window.releases"), 1)

    async def test_pause_interrupts_tool_and_releases_pointer(self):
        task = self.task()
        await task.resume()
        attempts = asyncio.Event()
        original_move = self.page.mouse.move

        async def move(*args, **kwargs):
            await original_move(*args, **kwargs)
            if args[0] > 25:
                attempts.set()
                await asyncio.sleep(0.05)

        with patch.object(self.page.mouse, "move", side_effect=move):

            async def execute():
                async with task.lock:
                    await task.execute({"tool": "solve_slider_captcha"})

            operation = asyncio.create_task(execute())
            await asyncio.wait_for(attempts.wait(), 5)
            await task.pause()
            await operation
        self.assertEqual(task.state["control_mode"], "HUMAN")
        self.assertEqual(await self.page.evaluate("window.releases"), 1)

    async def test_disappearing_slider_without_authenticated_portal_stays_manual(self):
        task = self.task()
        await task.resume()
        # The slider succeeds, but a password form still remains afterward.
        await self.page.evaluate("""document.addEventListener('mouseup',()=>{
          const input=document.createElement('input'); input.type='password';document.body.append(input);
        })""")
        await task.execute({"tool": "solve_slider_captcha"})
        self.assertEqual(task.state["control_mode"], "HUMAN")
        self.assertNotEqual(task.state["auth_status"], "AUTHENTICATED")

    async def test_untrusted_challenge_cannot_be_executed(self):
        await self.page.goto("https://attacker.example/login")
        task = self.task()
        with self.assertRaises(ValueError):
            await task.resume()
        self.assertFalse(allowed_url(self.page.url))
        self.assertEqual(await self.page.evaluate("window.releases"), 0)
