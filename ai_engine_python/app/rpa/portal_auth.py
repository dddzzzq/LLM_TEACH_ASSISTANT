"""Secret-bearing login actions for the existing page Agent; no navigation loop."""

import os
import re
import time
from urllib.parse import urlsplit

from .errors import RpaIssue
from playwright.async_api import TimeoutError as BrowserTimeout


PASSWORD = "input[type=password]:visible"
USERNAME = "#username:visible,#account:visible,input[name=username]:visible,input[autocomplete=username]:visible"
OTP = "input[autocomplete=one-time-code]:visible,#dynamicCode:visible,input[name=otp]:visible,input[placeholder*=验证码]:visible,input[placeholder=请输入]:visible"
SENSITIVE = "input[type=password],input:not([type=checkbox]):not([type=radio]),textarea"


def origin(value):
    p = urlsplit(value)
    if p.scheme != "https" or not p.hostname or p.username or p.password:
        return ""
    return f"https://{p.hostname}" + (f":{p.port}" if p.port and p.port != 443 else "")


def login_origins():
    values = os.environ.get("PORTAL_LOGIN_ORIGINS", "https://ids.xidian.edu.cn").split(",")
    return {origin(v.strip()) for v in values if origin(v.strip())}


async def visible_one(locator):
    if await locator.count() == 1 and await locator.is_visible():
        return locator
    return None


class PortalAuth:
    def __init__(self, owner, credentials, trust_device):
        self.owner = owner
        self.credentials = credentials
        self.trust_device = trust_device
        self.filled = False
        self.identity_confirmed = False
        self.last_code_request = 0
        self.otp_submitted = 0

    async def trust_available(self):
        for page in self.owner.context.pages:
            if page.is_closed() or origin(page.url) not in login_origins():
                continue
            for frame in page.frames:
                if origin(frame.url) not in login_origins():
                    continue
                title = frame.get_by_text("可信浏览器提示", exact=True)
                if await title.count() == 1 and await title.is_visible():
                    self.owner.page = page
                    return True
        return False

    async def observation(self):
        """Only booleans reach the model, never form values or the account."""
        login = await self.login_frame()
        otp = await self.otp_frame()
        if login and self.filled:
            wrong = login.get_by_text(
                re.compile(
                    "密码错误|用户名或密码错误|账号或密码错误|账号被锁定|incorrect (?:username|password)|invalid (?:username|password)|account.{0,20}locked",
                    re.I,
                )
            )
            if await wrong.count() and await wrong.first.is_visible():
                raise RpaIssue(
                    "INVALID_CREDENTIALS", "官网提示账号或密码错误，请修改设置后重新连接"
                )
            invalid_code = login.get_by_text(
                re.compile(
                    "验证码错误|验证码无效|验证码已过期|verification\\s+code\\s+(?:error|invalid|expired)|(?:invalid|incorrect|expired)\\s+(?:verification\\s+code|captcha)",
                    re.I,
                )
            )
            if await invalid_code.count() and await invalid_code.first.is_visible():
                raise RpaIssue(
                    "LOGIN_VERIFICATION_FAILED",
                    "官网提示验证码校验失败，已停止重复填写账号密码；请检查当前登录验证",
                )
        return {
            "credentials_saved": True,
            "credentials_filled": self.filled,
            "login_form": login is not None,
            "otp_required": otp is not None,
            "otp_entered": bool(await otp.locator(OTP).input_value()) if otp else False,
            "code_requested": bool(self.last_code_request),
            "trust_prompt": await self.trust_available(),
            "trust_device": self.trust_device,
        }

    async def login_frame(self):
        for page in self.owner.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames:
                if (
                    origin(page.url) not in login_origins()
                    or origin(frame.url) not in login_origins()
                ):
                    continue
                if await frame.locator(PASSWORD).count():
                    self.owner.page = page
                    return frame
        return None

    async def fill_credentials(self, frame):
        username = await visible_one(
            frame.locator(os.environ.get("PORTAL_USERNAME_SELECTOR", USERNAME))
        )
        password = await visible_one(frame.locator(PASSWORD))
        if not username or not password:
            return False
        action = await password.evaluate(
            "e => e.form ? new URL(e.form.action || location.href, location.href).origin : location.origin"
        )
        if action not in login_origins():
            raise RpaIssue("LOGIN_ORIGIN_REJECTED", "登录表单来源未获配置授权，已停止填写")
        self.owner.update(stage="FILLING_CREDENTIALS", message="正在填写已保存的教务账号")
        # The school's username blur handler checks verification asynchronously.
        # Observe its normal UI request; never invoke or skip the school's check.
        preflight = await frame.evaluate("typeof window.checkNeedCaptcha === 'function'")
        if preflight:
            try:
                async with frame.page.expect_response(
                    lambda response: origin(response.url) == origin(frame.url)
                    and urlsplit(response.url).path == "/authserver/checkNeedCaptcha.htl",
                    timeout=5000,
                ) as pending:
                    await username.fill(self.credentials["username"])
                    await password.fill(self.credentials["password"])
                response = await pending.value
                result = await response.json()
                if (
                    not response.ok
                    or not isinstance(result, dict)
                    or not isinstance(result.get("isNeed"), bool)
                ):
                    raise RpaIssue("LOGIN_CHECK_FAILED", "官网登录验证检查未正常完成，已停止提交")
                await frame.wait_for_function(
                    "expected => typeof window.needCaptcha === 'boolean' && window.needCaptcha === expected",
                    arg=result["isNeed"],
                    timeout=3000,
                )
            except BrowserTimeout as exc:
                raise RpaIssue(
                    "LOGIN_CHECK_TIMEOUT", "官网登录验证检查超时，尚未提交账号密码，请稍后恢复"
                ) from exc
        else:
            await username.fill(self.credentials["username"])
            await password.fill(self.credentials["password"])
        # A single submission avoids repeated bad-password attempts and account lockout.
        self.filled = True
        self.identity_confirmed = True
        if await frame.locator("#sliderDiv:visible").count():
            return True  # Continue through the existing challenge handling.
        submit = frame.locator(
            os.environ.get(
                "PORTAL_LOGIN_SUBMIT_SELECTOR",
                "#login_submit:visible,#login_submit_btn:visible,button[type=submit]:visible,input[type=submit]:visible",
            )
        )
        if await submit.count() == 1:
            await submit.click(timeout=5000)
        else:
            button = frame.get_by_role("button", name="登录", exact=True)
            if await button.count() == 1:
                await button.click(timeout=5000)
        return True

    async def otp_frame(self):
        for page in self.owner.context.pages:
            if page.is_closed() or origin(page.url) not in login_origins():
                continue
            for frame in page.frames:
                if (
                    origin(frame.url) not in login_origins()
                    or await frame.locator(PASSWORD).count()
                ):
                    continue
                if await frame.locator(OTP).count() == 1:
                    self.owner.page = page
                    return frame
        return None

    async def trust_prompt(self):
        for page in self.owner.context.pages:
            if page.is_closed() or origin(page.url) not in login_origins():
                continue
            for frame in page.frames:
                if origin(frame.url) not in login_origins():
                    continue
                title = frame.get_by_text("可信浏览器提示", exact=True)
                if not await title.count() or not await title.first.is_visible():
                    continue
                label = "信任此设备" if self.trust_device else "仅本次登录"
                button = frame.get_by_role("button", name=label, exact=True)
                if await button.count() != 1:
                    button = frame.get_by_text(label, exact=True)
                if await button.count() == 1 and await button.is_visible():
                    self.owner.update(stage="TRUST_DECISION", message="正在应用教师的可信设备设置")
                    await button.click(timeout=5000)
                    return True
        return False

    async def identity(self):
        selector = os.environ.get(
            "PORTAL_IDENTITY_SELECTOR",
            "[data-teacher-account],[data-login-account],#teacherAccount",
        )
        for page in self.owner.context.pages:
            if page.is_closed() or not origin(page.url):
                continue
            host = urlsplit(page.url).hostname or ""
            if host != "learning.xidian.edu.cn" and not host.endswith(".chaoxing.com"):
                continue
            for frame in page.frames:
                host = urlsplit(frame.url).hostname or ""
                if host != "learning.xidian.edu.cn" and not host.endswith(".chaoxing.com"):
                    continue
                controls = frame.locator(selector)
                if await controls.count() == 1:
                    value = await controls.evaluate(
                        "e => (e.getAttribute('data-teacher-account') || e.getAttribute('data-login-account') || e.value || e.textContent || '').trim()"
                    )
                    if value:
                        if value != self.credentials["username"]:
                            raise RpaIssue(
                                "IDENTITY_MISMATCH",
                                "当前官网账号与绑定账号不一致，请断开后重新连接",
                            )
                        return True
        return self.identity_confirmed

    async def request_code(self):
        if time.monotonic() - self.last_code_request < 60:
            raise ValueError("请等待官网验证码倒计时结束")
        frame = await self.otp_frame()
        if not frame:
            raise ValueError("当前页面不需要验证码")
        button = frame.get_by_text("获取验证码", exact=True)
        if await button.count() != 1 or not await button.is_enabled():
            raise ValueError("官网暂不允许发送验证码")
        self.last_code_request = time.monotonic()
        await button.click(timeout=5000)

    async def verify_code(self, code):
        if not isinstance(code, str) or not re.fullmatch(r"[0-9]{4,12}", code):
            raise ValueError("验证码格式无效")
        frame = await self.otp_frame()
        if not frame or time.monotonic() - self.otp_submitted < 3:
            raise ValueError("请等待当前验证完成")
        button = frame.get_by_role("button", name="登录", exact=True)
        if await button.count() != 1:
            button = frame.get_by_text("登录", exact=True)
        if await button.count() != 1:
            raise ValueError("请在认证浏览器中提交验证码")
        await frame.locator(OTP).fill(code)
        self.otp_submitted = time.monotonic()
        await button.click(timeout=5000)
        if await frame.locator(OTP).count() == 1:
            await frame.locator(OTP).fill("")

    async def submit_entered_code(self):
        frame = await self.otp_frame()
        if not frame:
            raise ValueError("当前页面不需要验证码")
        # Read inside the Worker only; the Agent calls a parameterless action.
        await self.verify_code(await frame.locator(OTP).input_value())
