"""Bounded recovery for public portal/login navigation, never export or grading."""
import asyncio
import re
import time
from urllib.parse import urlsplit

from playwright.async_api import Error

from .errors import RpaIssue
from .portal_adapter import TARGET_URL


class StartupPaused(Exception):
    pass


class BrowserStartup:
    attempts = 3
    attempt_timeout = 25
    total_timeout = 90
    retry_delays = (2, 4)
    ready_probe_timeout = 2
    transient = {'ERR_EMPTY_RESPONSE', 'ERR_CONNECTION_RESET', 'ERR_CONNECTION_CLOSED',
                 'ERR_CONNECTION_TIMED_OUT', 'ERR_TIMED_OUT', 'ERR_NAME_NOT_RESOLVED',
                 'ERR_NETWORK_CHANGED', 'ERR_INTERNET_DISCONNECTED', 'ERR_CONNECTION_REFUSED',
                 'ERR_PROXY_CONNECTION_FAILED', 'ERR_TUNNEL_CONNECTION_FAILED', 'ERR_ABORTED'}

    def __init__(self, task):
        self.task = task
        self.epoch = task.state['control_epoch']
        self.phase = 'portal_load'
        self.http_status = None

    def check_owner(self):
        s = self.task.state
        if s['status'] != 'RUNNING' or s['control_mode'] != 'AUTO' or s['control_epoch'] != self.epoch:
            raise StartupPaused()

    async def ready(self):
        self.check_owner()
        task = self.task
        # Query actual controls, not HTTP success or an empty document's readyState.
        for page in task.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames:
                try:
                    if await frame.locator('input[type=password]:visible,#sliderDiv:visible,input#captcha:visible').count():
                        self.check_owner()
                        task.page = page
                        return 'REQUIRED'
                except Error:
                    continue  # Frame may be replaced by the login redirect.
        if await task.adapter.authenticated():
            self.check_owner()
            return 'AUTHENTICATED'
        return None

    async def probe(self):
        try:
            return await asyncio.wait_for(self.ready(), self.ready_probe_timeout)
        except (TimeoutError, Error):
            return None

    async def navigate(self):
        ready = await self.probe()
        if ready:
            return ready
        self.check_owner()
        self.phase = 'portal_load'
        self.task.event('正在加载教学门户')
        response = await self.task.page.goto(TARGET_URL, wait_until='domcontentloaded', timeout=20000)
        self.check_owner()
        self.http_status = response.status if response else None
        if self.http_status and self.http_status >= 400:
            raise RpaIssue('PORTAL_HTTP_ERROR', '教学门户暂未返回正常页面', {'http_status': self.http_status})
        clicked = False
        while True:
            self.check_owner()
            ready = await self.probe()
            if ready:
                return ready
            if not clicked:
                login = self.task.page.locator('.denglu')
                if await login.count() == 1 and await login.is_visible():
                    self.check_owner()
                    self.phase = 'login_redirect'
                    self.task.event('正在打开并核验官网登录页')
                    await login.click(timeout=10000)
                    clicked = True
            await asyncio.sleep(.2)

    async def open_login(self):
        task = self.task
        started = time.monotonic()
        deadline = started + self.total_timeout
        for attempt in range(1, self.attempts + 1):
            self.check_owner()
            self.phase, self.http_status = 'portal_load', None
            task.update(startup_attempt=attempt)
            attempt_started = time.monotonic()
            try:
                async with asyncio.timeout(min(self.attempt_timeout, max(.01, deadline - time.monotonic()))):
                    ready = await self.navigate()
                return self.succeeded(ready, attempt, started)
            except (StartupPaused, asyncio.CancelledError):
                raise
            except Exception as exc:
                self.check_owner()
                # A navigation can time out after the usable login form has appeared.
                remaining = deadline - time.monotonic()
                ready = await asyncio.wait_for(self.probe(), remaining) if remaining > self.ready_probe_timeout else None
                if ready:
                    return self.succeeded(ready, attempt, started)
                codes = sorted(set(re.findall(r'net::(ERR_[A-Z_]+)', str(exc))))
                retryable = (type(exc).__name__ == 'TimeoutError' or bool(set(codes) & self.transient)
                             or self.http_status in {408, 429} or bool(self.http_status and self.http_status >= 500))
                evidence = {'phase': self.phase, 'attempt': attempt,
                            'elapsed_ms': round((time.monotonic()-attempt_started)*1000),
                            'host': urlsplit(TARGET_URL if self.phase == 'portal_load' else task.page.url).hostname,
                            'network_codes': codes, 'exception_type': type(exc).__name__,
                            'http_status': self.http_status, 'retryable': retryable}
                task.record_failure(RpaIssue('BROWSER_STARTUP_FAILED', '官网登录页面暂未加载完成', evidence),
                                    'startup:' + self.phase)
                if not retryable or attempt == self.attempts or time.monotonic() >= deadline:
                    break
                delay = self.retry_delays[min(attempt-1, len(self.retry_delays)-1)]
                if time.monotonic() + delay >= deadline:
                    break
                task.event(f'官网连接暂未完成，正在准备第 {attempt+1}/{self.attempts} 次加载')
                await asyncio.sleep(delay)
        self.check_owner()
        task.wait_human('BROWSER_STARTUP', '官网登录页加载失败，浏览器已保留。请点击“重新加载登录页”重试。')
        return None

    def succeeded(self, ready, attempt, started):
        self.check_owner()
        elapsed = round(time.monotonic() - started, 2)
        self.task.update(startup_result={'attempts': attempt, 'elapsed_seconds': elapsed, 'auth_status': ready})
        self.task.event(f'官网页面已就绪，本轮加载 {attempt} 次，耗时 {elapsed} 秒')
        return ready
