"""Secret-bearing login actions for the existing page Agent; no navigation loop."""

import asyncio
import hashlib
import os
import re
import time
from urllib.parse import urlsplit

from .errors import RpaIssue
from playwright.async_api import TimeoutError as BrowserTimeout


PASSWORD = "input[type=password]:visible"
USERNAME = "#username:visible,#account:visible,input[name=username]:visible,input[autocomplete=username]:visible"
OTP = "input[autocomplete=one-time-code]:visible,#dynamicCode:visible,input[name=otp]:visible,input[placeholder*=验证码]:visible,input[placeholder=请输入]:visible,input[placeholder*='verification code' i]:visible,input[placeholder*='security code' i]:visible"
SENSITIVE = "input[type=password],input:not([type=checkbox]):not([type=radio]),textarea"
CODE_REQUEST = re.compile(
    r"^(?:(?:重新)?(?:获取|发送)(?:短信)?验证码|重新发送|获取动态码|发送动态码|"
    r"(?:get|send|request|resend)(?:\s+(?:a|the))?\s+(?:(?:sms|verification|security)\s+)?code|resend)$",
    re.I,
)
CODE_COUNTDOWN = re.compile(
    r"^(?:(?:重新获取|重新发送|获取验证码|发送验证码|resend(?:\s+code)?(?:\s+in)?)\s*)?"
    r"[（(]?\s*\d+\s*(?:秒|s|seconds?)\s*[）)]?\s*(?:后(?:重新)?(?:获取|发送))?$",
    re.I,
)
TRUST_TITLE = re.compile(
    r"可信(?:浏览器|设备)|trust(?:ed)?\s+(?:(?:this|your)\s+)?(?:browser|device)", re.I
)
TRUST_YES = re.compile(
    r"^(?:信任此设备|信任此浏览器|trust\s+(?:this\s+)?(?:device|browser))\s*[?.！!。]?$", re.I
)
TRUST_NO = re.compile(
    r"^(?:仅本次登录|暂不信任|不信任|(?:only\s+)?(?:this\s+time|log\s*in\s+once)|do\s+not\s+trust|don['’]t\s+trust(?:\s+this\s+(?:device|browser))?|not\s+now|skip)\s*[.!。]?$",
    re.I,
)
INVALID_CODE = re.compile(
    r"验证码错误|验证码无效|验证码已过期|verification\s+code\s+(?:error|invalid|expired)|"
    r"(?:invalid|incorrect|expired)\s+(?:(?:verification|security)\s+code|captcha)",
    re.I,
)


async def named_controls(frame, pattern):
    """Prefer visible controls; hidden copies and unrelated containers are not actions."""
    controls = frame.locator(
        "button:visible,a:visible,input[type=button]:visible,input[type=submit]:visible,"
        "[role=button]:visible,[onclick]:visible"
    )
    matches = []
    for i in range(await controls.count()):
        control = controls.nth(i)
        label = await control.evaluate(
            "e => (e.getAttribute('aria-label') || e.innerText || e.value || '').trim().replace(/\\s+/g, ' ')"
        )
        if pattern.fullmatch(label):
            matches.append(control)
    if not matches:
        text = frame.get_by_text(pattern).filter(visible=True)
        matches = [text.nth(i) for i in range(await text.count())]
    return matches


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
        self.submitted_code_digest = None

    async def trust_frame(self):
        for page in self.owner.context.pages:
            if page.is_closed() or origin(page.url) not in login_origins():
                continue
            for frame in page.frames:
                if origin(frame.url) not in login_origins():
                    continue
                title = frame.get_by_text(TRUST_TITLE).filter(visible=True)
                if await title.count():
                    self.owner.page = page
                    return frame
        return None

    async def trust_available(self):
        return await self.trust_frame() is not None

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
        for frame in [login, otp]:
            if frame is None:
                continue
            invalid_code = frame.get_by_text(INVALID_CODE).filter(visible=True)
            if await invalid_code.count():
                if frame == otp and self.submitted_code_digest is not None:
                    entered = await frame.locator(OTP).input_value()
                    if (
                        entered
                        and hashlib.sha256(entered.encode()).digest() != self.submitted_code_digest
                    ):
                        continue  # A teacher has replaced the rejected code; allow a new submission.
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
            "code_requested": bool(self.last_code_request)
            or bool(otp and await named_controls(otp, CODE_COUNTDOWN)),
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
        frame = await self.trust_frame()
        if frame is None:
            return False
        buttons = await named_controls(frame, TRUST_YES if self.trust_device else TRUST_NO)
        if len(buttons) != 1 or not await buttons[0].is_enabled():
            raise RpaIssue(
                "DEVICE_TRUST_UNAVAILABLE", "官网可信设备选项暂不可用，请按已保存的设备设置核对弹窗"
            )
        self.owner.update(stage="TRUST_DECISION", message="正在应用教师的可信设备设置")
        await buttons[0].click(timeout=5000)
        return True

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
        frame = await self.otp_frame()
        if not frame:
            raise RpaIssue("LOGIN_CODE_PAGE_MISSING", "当前未找到官网验证码输入页，请核对认证页面")
        # A countdown is evidence of an existing request, including codes sent by
        # the school or teacher. Hand over for entry instead of clicking again.
        if (
            self.last_code_request and time.monotonic() - self.last_code_request < 60
        ) or await named_controls(frame, CODE_COUNTDOWN):
            await frame.locator(OTP).focus()
            return
        buttons = await named_controls(frame, CODE_REQUEST)
        if len(buttons) != 1:
            raise RpaIssue(
                "LOGIN_CODE_BUTTON_UNAVAILABLE",
                "未找到唯一可见的获取验证码按钮，请核对官网验证页面",
                {"visible_matches": len(buttons)},
            )
        if not await buttons[0].is_enabled():
            raise RpaIssue(
                "LOGIN_CODE_SEND_DISABLED",
                "官网获取验证码按钮暂不可用，请检查页面提示后继续",
                {"visible_matches": 1, "enabled": False},
            )
        self.owner.update(message="正在点击官网获取验证码按钮")
        self.last_code_request = time.monotonic()
        try:
            await buttons[0].click(timeout=5000)
        except BrowserTimeout as exc:
            raise RpaIssue(
                "LOGIN_CODE_REQUEST_UNCONFIRMED",
                "获取验证码操作未确认，请检查官网是否已发送，避免重复获取",
            ) from exc
        await frame.locator(OTP).focus()

    async def fill_code(self, code, page=None):
        if not isinstance(code, str) or not re.fullmatch(r"[0-9]{4,12}", code):
            raise RpaIssue("LOGIN_CODE_FORMAT", "请输入 4 至 12 位数字验证码")
        frame = await self.otp_frame()
        if not frame or (page is not None and frame.page is not page):
            raise RpaIssue("LOGIN_CODE_PAGE_MISSING", "验证码页面已变化，请检查当前认证页面")
        await frame.locator(OTP).fill(code)
        return frame

    async def verify_code(self, code):
        if not isinstance(code, str) or not re.fullmatch(r"[0-9]{4,12}", code):
            raise RpaIssue("LOGIN_CODE_FORMAT", "请输入 4 至 12 位数字验证码")
        frame = await self.otp_frame()
        if not frame or time.monotonic() - self.otp_submitted < 3:
            raise RpaIssue("LOGIN_CODE_PENDING", "请等待官网完成当前验证码校验")
        digest = hashlib.sha256(code.encode()).digest()
        if self.submitted_code_digest == digest:
            raise RpaIssue("LOGIN_CODE_PENDING", "已提交该验证码，请核对官网校验结果，避免重复登录")
        buttons = await named_controls(
            frame,
            re.compile(
                r"^(?:登录|验证并登录|log\s*in|sign\s*in|verify(?:\s+and\s+(?:log|sign)\s*in)?)$",
                re.I,
            ),
        )
        if len(buttons) != 1:
            raise RpaIssue(
                "LOGIN_CODE_SUBMIT_UNAVAILABLE", "未找到唯一可见的登录按钮，请在认证浏览器中核对"
            )
        await frame.locator(OTP).fill(code)
        self.otp_submitted = time.monotonic()
        self.submitted_code_digest = digest
        await buttons[0].click(timeout=5000)
        # Some login handlers read the field asynchronously. Keep it in the
        # masked browser until the school consumes it; never clear it on click.
        for _ in range(30):
            if await self.trust_available() or await self.otp_frame() is None:
                return
            if await frame.get_by_text(INVALID_CODE).filter(visible=True).count():
                raise RpaIssue(
                    "LOGIN_VERIFICATION_FAILED", "官网提示验证码无效或已过期，请重新核对验证码"
                )
            await asyncio.sleep(0.1)

    async def submit_entered_code(self):
        frame = await self.otp_frame()
        if not frame:
            raise RpaIssue("LOGIN_CODE_PAGE_MISSING", "当前未找到官网验证码输入页")
        # Read inside the Worker only; the Agent calls a parameterless action.
        await self.verify_code(await frame.locator(OTP).input_value())
