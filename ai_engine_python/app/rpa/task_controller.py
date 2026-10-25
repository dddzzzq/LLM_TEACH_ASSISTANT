import asyncio
import hashlib
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

from .tools import validate_action
from .slider_captcha import find_slider_challenge, solve_slider_captcha
from .errors import RpaIssue
from .target_confirmation import TargetConfirmation, SEMESTER
from .download_manager import save_download
from .browser_startup import BrowserStartup, StartupPaused
from .portal_adapter import (
    PortalAdapter,
    TARGET_URL,
    FORBIDDEN,
    EXPORT_CONFIRM,
    match_new_exports,
    reconcile_exports,
)


TERMINAL = {"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED"}
ALLOWED_HOSTS = ("xidian.edu.cn", "chaoxing.com", "chaoxing.com.cn")
logger = logging.getLogger("uvicorn.error")


def allowed_url(url):
    parsed = urlsplit(url)
    return url == "about:blank" or (
        parsed.scheme in ("http", "https")
        and any(
            parsed.hostname == h or (parsed.hostname or "").endswith("." + h) for h in ALLOWED_HOSTS
        )
    )


class Task:
    def __init__(self, manager, state):
        self.manager, self.state = manager, state
        self.browser = self.context = self.adapter = self.page = None
        self.lock = asyncio.Lock()
        self.frame_lock = asyncio.Lock()
        self.gate = asyncio.Event()
        self.runner = None
        self.scope_request = state.get("requested_scope")
        self.last_activity = time.monotonic()
        self.last_progress = time.monotonic()
        self.recent = []
        self.frame_revision = 0
        self.frame_meta = None
        self.frame_history = {}
        self.manual_export_evidence = {}
        self.download_guid = None
        self.download_bytes = {}
        self.download_urls = {}
        self.observation = None
        self.observation_revision = 0
        self.browser_frames = None
        self.pointer_owner = self.pointer_page = None
        self.navigation_marker = None
        self.stagnant_actions = 0
        self.target_confirmation = TargetConfirmation(self)
        self.startup_pending = False
        self.startup_wake = asyncio.Event()
        self.startup_operation = None
        self.portal_ticket = ""
        self.portal_profile = None
        self.portal_auth = None
        self.verified_subject = ""
        self.catalog_course_pages = {}
        self.catalog_collection = None
        if state.get("task_type") == "CATALOG_SYNC":
            from .portal_catalog import CatalogCollection

            self.catalog_collection = CatalogCollection(self)

    def update(self, **fields):
        self.state.update(
            fields, updated_at=time.time(), state_version=self.state.get("state_version", 0) + 1
        )
        self.manager.persist(self.state)

    def event(self, message):
        events = self.state.setdefault("events", [])
        events.append({"time": time.time(), "message": message})
        self.state["events"] = events[-60:]
        self.update(message=message)
        logger.info(
            "RPA event job=%s stage=%s reason=%s message=%s",
            self.state["job_id"],
            self.state.get("stage"),
            self.state.get("waiting_reason", ""),
            message,
        )

    def record_failure(self, exc, operation, details=None):
        # Browser/network exception strings can contain signed URLs, credentials,
        # or page text. Only application-owned messages and stack locations leave
        # this boundary; no raw exception strings, HTML, or form values.
        known = isinstance(exc, RpaIssue)
        report = {
            "code": exc.code
            if known
            else "BROWSER_TIMEOUT"
            if type(exc).__name__ == "TimeoutError"
            else "PAGE_OPERATION_FAILED",
            "type": type(exc).__name__,
            "operation": operation,
            "stage": self.state.get("stage"),
            "message": exc.public_message
            if known
            else "页面操作暂未完成（" + type(exc).__name__ + "）",
            "locations": [
                f"{Path(f.filename).name}:{f.lineno}:{f.name}"
                for f in traceback.extract_tb(exc.__traceback__)[-5:]
            ],
        }
        if details:
            report["controls"] = details
        if known and exc.evidence:
            report["evidence"] = exc.evidence
        self.update(last_error=report)
        logger.warning(
            "RPA failure job=%s diagnostic=%s",
            self.state["job_id"],
            json.dumps(report, ensure_ascii=False),
        )
        self.event(report["message"])
        return report

    def wait_human(self, reason, message):
        self.gate.clear()
        self.last_activity = time.monotonic()
        self.update(
            status="WAITING_USER",
            control_mode="HUMAN",
            waiting_reason=reason,
            control_epoch=self.state["control_epoch"] + 1,
            **({"auth_status": "REQUIRED"} if reason == "LOGIN" else {}),
        )
        self.event(message)

    async def captcha_target(self):
        """Find a live challenge only on configured portal/login domains."""
        if not self.context:
            return None
        for page in self.context.pages:
            if page.is_closed() or page.url == "about:blank" or not allowed_url(page.url):
                continue
            frame = await find_slider_challenge(
                page, lambda url: url != "about:blank" and allowed_url(url)
            )
            if frame:
                return page, frame
        return None

    async def prepare_captcha(self):
        if self.state.get("captcha_attempts", 0) >= 3:
            return False
        target = await self.captcha_target()
        if not target:
            return False
        self.page = target[0]
        if self.state.get("stage") != "SOLVE_CAPTCHA":
            self.update(stage="SOLVE_CAPTCHA", auth_status="CAPTCHA_REQUIRED")
        return True

    def captcha_view(self):
        return {
            "captcha": {
                "type": "slider",
                "visible": True,
                "attempts_remaining": 3 - self.state.get("captcha_attempts", 0),
            },
            "pages": [
                {"page_id": str(self.context.pages.index(self.page)), "title": "登录滑块验证"}
            ],
            "frames": [],
            "elements": [],
        }

    async def refresh_login_state(self):
        """Observe login while waiting, without returning browser control to the Agent."""
        if (
            self.state["status"] in TERMINAL
            or self.state["control_mode"] != "HUMAN"
            or self.state.get("waiting_reason") not in {"LOGIN", "LOGIN_COMPLETE"}
            or not self.adapter
        ):
            return
        async with self.lock:
            # Pause/resume may have changed control while waiting for the action lock.
            if (
                self.state["status"] in TERMINAL
                or self.state["control_mode"] != "HUMAN"
                or self.state.get("waiting_reason") not in {"LOGIN", "LOGIN_COMPLETE"}
            ):
                return
            try:
                authenticated = await self.adapter.authenticated()
                required = not authenticated and await self.adapter.login_required()
            except Exception:
                # A redirect or detached frame is not evidence that login failed.
                return
            auth_status = (
                "AUTHENTICATED" if authenticated else "REQUIRED" if required else "UNKNOWN"
            )
            if auth_status == self.state.get("auth_status"):
                return
            self.update(
                auth_status=auth_status,
                waiting_reason="LOGIN_COMPLETE" if authenticated else "LOGIN",
                message="官网登录已完成。点击“交还自动化”后，Agent 将继续定位课程和作业。"
                if authenticated
                else "请填写账号密码并提交登录；出现滑块后可点击“交还自动化”尝试验证。"
                if required
                else "正在核验官网登录状态。如已完成登录，请点击“交还自动化”继续。",
            )

    async def run(self):
        startup_phase = "browser_launch"
        self.startup_pending = True
        try:
            proxy = self.manager.proxy
            browser_downloads = self.manager.download_root / self.state["job_id"] / ".browser"
            browser_downloads.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.state.get("portal_session"):
                self.portal_profile = self.manager.portal_sessions.open(
                    self.state["portal_session"], browser_downloads
                )
                self.context = await self.portal_profile.__aenter__()
                self.browser = self.context.browser
            else:
                self.browser = await self.manager.playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                    proxy={"server": proxy} if proxy else None,
                    downloads_path=str(browser_downloads),
                )
                startup_phase = "browser_context"
                self.context = await self.browser.new_context(
                    viewport={"width": 1440, "height": 1000}, accept_downloads=True
                )
            await self.context.route("**/*", self.route)
            await self.context.expose_binding("__rpa_manual_export", self.manual_export_submitted)
            await self.context.add_init_script("""document.addEventListener('click', event => {
              const button=event.target.closest('a.confirmDown,button');
              const modal=button?.closest('.popDiv.centerPop,[role=dialog]');
              if(modal && /导出/.test(modal.innerText) && /确定|确认/.test(button.innerText))
                window.__rpa_manual_export().catch(()=>{});
            }, true);""")
            self.page = (
                self.context.pages[-1]
                if self.portal_profile and self.context.pages
                else await self.context.new_page()
            )
            page_cdp = await self.context.new_cdp_session(self.page)
            target_info = await page_cdp.send("Target.getTargetInfo")
            browser_cdp = await self.browser.new_browser_cdp_session() if self.browser else page_cdp
            browser_cdp.on("Browser.downloadWillBegin", self.download_begin)
            browser_cdp.on("Browser.downloadProgress", self.download_progress)
            download_options = {
                "behavior": "allowAndName",
                "downloadPath": str(browser_downloads),
                "eventsEnabled": True,
            }
            # Persistent profiles use Chromium's default context. Its target ID
            # cannot be passed as an incognito browserContextId to this command.
            if not self.portal_profile and target_info["targetInfo"].get("browserContextId"):
                download_options["browserContextId"] = target_info["targetInfo"]["browserContextId"]
            await browser_cdp.send("Browser.setDownloadBehavior", download_options)
            self.adapter = PortalAdapter(self.context, self.state["target"])
            startup_phase = "login_navigation"
            await self.open_login()
            errors = 0
            while True:
                await self.gate.wait()
                await asyncio.sleep(0.2)
                if self.state["status"] in TERMINAL:
                    return
                epoch = self.state["control_epoch"]
                try:
                    if (
                        self.state.get("export_attempted")
                        and self.adapter.assignment_confirmed
                        and self.state.get("stage") != "SOLVE_CAPTCHA"
                    ):
                        await self.process_exports()
                        if self.state["status"] in TERMINAL:
                            return
                        await asyncio.sleep(2)
                        continue
                    async with self.lock:
                        if not self.gate.is_set() or epoch != self.state["control_epoch"]:
                            continue
                        if self.portal_auth:
                            auth = await self.authentication_view()
                            if not auth["authenticated"]:
                                continue
                            if self.state.get("task_type") == "AUTH_BOOTSTRAP":
                                self.update(
                                    status="SUCCEEDED",
                                    stage="DONE",
                                    message="教务登录完成，正在准备目录同步",
                                )
                                return
                            if self.catalog_collection:
                                continue  # The same page Agent navigates and reads each list.
                        if await self.prepare_captcha():
                            continue  # Go's page Agent chooses the bounded slider tool.
                        if await self.adapter.login_required():
                            self.wait_human("LOGIN", "官网需要重新验证身份，请完成登录")
                            continue
                        await self.adapter.reconcile()
                        area = await self.adapter.export_area()
                        if area:
                            if self.state.get("export_attempted"):
                                self.wait_human(
                                    "EXPORT_ASSOCIATION",
                                    "已有导出检查点，请关闭设置并打开下载中心，系统不会重新提交",
                                )
                                continue
                            await self.prepare_export(area)
                            continue
                        # Navigation is driven exclusively by Go/Eino through observe/action.
                        # This loop owns only deterministic export and download mechanics.
                        if self.state.get("stage") not in {
                            "LOCATE_COURSE",
                            "LOCATE_ASSIGNMENT",
                            "EXPORT",
                        }:
                            self.update(stage="LOCATE_COURSE")
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    errors += 1
                    report = self.record_failure(exc, "export_loop")
                    if errors >= 3 or isinstance(exc, ValueError):
                        self.wait_human(
                            "RECOVERY", report["message"] + "；请检查当前页面后交还自动化"
                        )
                        errors = 0
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.record_failure(
                exc, "startup:" + startup_phase if self.startup_pending else "task_run"
            )
            self.update(status="FAILED", control_mode="HUMAN", error_code=type(exc).__name__)
            self.event("浏览器任务启动或执行失败，请检查网络及 Worker 配置")
        finally:
            if self.portal_auth:
                self.portal_auth.credentials.clear()
                self.portal_auth = None
            if self.startup_operation:
                self.startup_operation.cancel()
                await asyncio.gather(self.startup_operation, return_exceptions=True)
            if self.portal_profile:
                await self.portal_profile.__aexit__(None, None, None)
                self.portal_profile = None
            elif self.browser:
                await self.browser.close()
            self.browser = self.context = self.page = None

    async def open_login(self):
        if self.state.get("portal_session"):
            from .portal_auth import PortalAuth
            from .portal_sessions import take_credentials

            value = await take_credentials(self.portal_ticket)
            self.portal_ticket = ""
            self.portal_auth = PortalAuth(
                self, value, self.state["portal_session"].get("trust_device", False)
            )
            self.portal_auth.identity_confirmed = bool(
                self.state["portal_session"].get("identity_verified")
            )
        # Navigation has its own cancellable operation. Do not hold the frame/input
        # lock over network waits; pause cancels it before handing control over.
        while self.startup_pending:
            self.startup_wake.clear()
            ready = None
            self.startup_operation = asyncio.create_task(BrowserStartup(self).open_login())
            try:
                ready = await self.startup_operation
            except StartupPaused:
                pass
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                if self.state["control_mode"] == "AUTO":
                    raise
            finally:
                self.startup_operation = None
            if ready:
                self.startup_pending = False
                self.update(
                    stage="AUTHENTICATE", error_code=None, last_error=None, auth_status=ready
                )
                if self.portal_auth:
                    self.update(status="RUNNING", control_mode="AUTO", waiting_reason="")
                    self.gate.set()
                    return
                self.wait_human(
                    "LOGIN_COMPLETE" if ready == "AUTHENTICATED" else "LOGIN",
                    "官网登录已完成，请交还自动化继续"
                    if ready == "AUTHENTICATED"
                    else "登录页已就绪，请填写账号密码并提交登录；出现滑块后可交还自动化尝试验证",
                )
                return
            await self.startup_wake.wait()

    def automatic(self):
        return (
            self.state["status"] == "RUNNING"
            and self.state["control_mode"] == "AUTO"
            and self.gate.is_set()
            and self.adapter is not None
            and not self.startup_pending
        )

    async def authentication_view(self):
        auth = await self.portal_auth.observation()
        authenticated = (
            not auth["login_form"]
            and not auth["otp_required"]
            and not auth["trust_prompt"]
            and (bool(self.verified_subject) or await self.adapter.authenticated())
        )
        if authenticated:
            if not await self.portal_auth.identity():
                if self.state.get("waiting_reason") != "LOGIN_IDENTITY":
                    self.wait_human(
                        "LOGIN_IDENTITY", "请核对官网教师身份，确认与设置账号一致后交还自动化"
                    )
                authenticated = False
            else:
                self.verified_subject = self.portal_auth.credentials["username"]
                self.portal_auth.identity_confirmed = True
        auth["authenticated"] = authenticated
        return auth

    async def navigation_view(self):
        auth = await self.authentication_view() if self.portal_auth else None
        captcha = (
            not (auth and auth["login_form"] and not auth["credentials_filled"])
            and await self.prepare_captcha()
        )
        if not captcha and not self.portal_auth and await self.adapter.login_required():
            self.wait_human("LOGIN", "官网需要重新验证身份，请完成登录")
            return None
        self.adapter.active_page = self.page
        if not self.catalog_collection and (not auth or auth["authenticated"]):
            await self.adapter.reconcile()
            if not captcha and await self.adapter.export_area():
                return None
        stage = (
            "SOLVE_CAPTCHA"
            if captcha
            else "AUTHENTICATE"
            if auth and not auth["authenticated"]
            else "READING_CATALOG"
            if self.catalog_collection
            else "EXPORT"
            if self.adapter.assignment_confirmed
            else "LOCATE_ASSIGNMENT"
            if self.adapter.course_confirmed
            else "LOCATE_COURSE"
        )
        if self.state.get("stage") != stage:
            self.update(stage=stage)
        view = self.captcha_view() if captcha else await self.adapter.observe()
        view["task_type"] = self.state.get("task_type") or "DOWNLOAD"
        if auth:
            view["authentication"] = auth
        if self.catalog_collection:
            view["catalog"] = self.catalog_collection.observation()
            if not captcha and (not auth or auth["authenticated"]):
                view["catalog"]["viewport"] = await self.catalog_collection.viewport().observation()
        return view

    def portal_result(self):
        status = self.state["status"]
        if status == "WAITING_USER":
            status = (
                "WAITING_OTP"
                if self.state.get("waiting_reason") == "LOGIN_CODE"
                else "WAITING_AUTH"
            )
        return {
            "status": status,
            "stage": self.state.get("stage", ""),
            "message": self.state.get("message", ""),
            "error_code": self.state.get("error_code") or "",
            "state_version": self.state.get("state_version", 0),
            **({"verified_subject": self.verified_subject} if self.verified_subject else {}),
            **({"catalog": self.state["portal_catalog"]} if "portal_catalog" in self.state else {}),
        }

    async def observation_fingerprint(self, view):
        # Full URLs stay inside the Worker; query strings may contain authentication tokens.
        identity = {"view": view, "urls": [p.url for p in self.context.pages if not p.is_closed()]}
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    async def observe(self):
        async with self.lock:
            if (
                not self.automatic()
                or self.state.get("export_attempted")
                and self.adapter.assignment_confirmed
                and self.state.get("stage") != "SOLVE_CAPTCHA"
            ):
                return {"ready": False, "state": self.state}
            view = await self.navigation_view()
            if view is None or not self.automatic():
                return {"ready": False, "state": self.state}
            stage = self.state["stage"]
            fingerprint = await self.observation_fingerprint(view)
            epoch = self.state["control_epoch"]
            if (
                not self.observation
                or self.observation["fingerprint"] != fingerprint
                or self.observation["epoch"] != epoch
            ):
                self.observation_revision += 1
                self.observation = {
                    "id": str(uuid4()),
                    "revision": self.observation_revision,
                    "epoch": epoch,
                    "fingerprint": fingerprint,
                }
            return {
                "ready": True,
                "observation_id": self.observation["id"],
                "revision": self.observation["revision"],
                "epoch": epoch,
                "target": self.state["target"],
                "stage": stage,
                "observation": view,
                "recent_actions": self.recent[-5:],
                "state": self.state,
            }

    async def apply_action(self, payload):
        action = validate_action(payload.get("action"))
        action_id = payload.get("action_id")
        if not isinstance(action_id, str) or not 1 <= len(action_id) <= 128:
            raise ValueError("动作 ID 无效")
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        async with self.lock:
            receipts = self.state.setdefault("action_receipts", {})
            prior = receipts.get(action_id)
            if prior:
                if prior["digest"] != digest:
                    raise ValueError("动作 ID 已用于不同请求")
                # Never replay an action whose external effect may already have happened.
                return {"outcome": prior["outcome"], "action_id": action_id, "state": self.state}
            current = self.observation
            if (
                not self.automatic()
                or not current
                or payload.get("epoch") != self.state["control_epoch"]
                or payload.get("observation_id") != current["id"]
                or payload.get("revision") != current["revision"]
            ):
                raise ValueError("观察或控制权已变化，请重新观察")
            view = await self.navigation_view()
            if view is None or not self.automatic():
                raise ValueError("Observation is no longer actionable")
            if await self.observation_fingerprint(view) != current["fingerprint"]:
                self.observation = None
                raise ValueError("页面已变化，请重新观察")
            if len(receipts) >= 1000:
                self.wait_human("ACTION_BUDGET", "任务动作总预算已到，请检查并结束任务")
                raise ValueError("任务动作预算已到")
            receipts[action_id] = {
                "digest": digest,
                "outcome": "unknown",
                "tool": action["tool"],
                "skill_version": str(payload.get("skill_version", ""))[:128],
            }
            self.observation = None
            self.update(action_receipts=receipts)
            try:
                await self.execute(action)
                receipts[action_id]["outcome"] = "completed"
                count = self.state.get("auto_actions", 0) + 1
                self.update(
                    action_receipts=receipts,
                    auto_actions=count,
                    navigation_failures=0,
                    decision_failures=0,
                    last_error=None,
                )
                marker = (current["fingerprint"], {k: v for k, v in action.items() if k != "value"})
                self.stagnant_actions = (
                    self.stagnant_actions + 1 if marker == self.navigation_marker else 1
                )
                self.navigation_marker = marker
                if self.stagnant_actions >= 3 and self.automatic():
                    self.wait_human(
                        "NO_PROGRESS", "连续三次相同操作后页面仍没有变化，请协助检查页面"
                    )
                # A semester may require more actions than a single download.
                # Extra budget is earned only by verified new works/courses;
                # the existing 1000-receipt ceiling still applies.
                completed = (
                    self.catalog_collection.observation()["completed_courses"]
                    if self.catalog_collection
                    else 0
                )
                works = (
                    sum(
                        len(c["homeworks"])
                        for c in self.catalog_collection.progress["courses"].values()
                    )
                    if self.catalog_collection
                    else 0
                )
                if count >= min(1000, 60 + 12 * completed + min(400, works)) and self.automatic():
                    self.wait_human(
                        "ACTION_BUDGET", "自动页面操作已达到本轮预算，请检查页面后交还自动化"
                    )
            except asyncio.CancelledError:
                # Durable UNKNOWN receipt survives Worker shutdown; do not execute again.
                raise
            except Exception as exc:
                report = self.record_failure(exc, "navigation:" + action["tool"])
                self.recent.append(
                    {
                        "tool": action["tool"],
                        "outcome": "unknown",
                        "error_code": report["code"],
                        "message": report["message"],
                    }
                )
                if self.automatic():
                    failures = self.state.get("navigation_failures", 0) + 1
                    self.update(navigation_failures=failures)
                    if isinstance(exc, RpaIssue) and exc.code == "TARGET_AMBIGUOUS":
                        self.wait_human("TARGET_AMBIGUOUS", report["message"])
                    elif failures >= 3:
                        self.wait_human(
                            "NAVIGATION",
                            "连续三次页面操作未完成：" + report["message"] + "；请协助检查页面",
                        )
                    else:
                        self.event(
                            "页面操作暂未完成，正在重新观察并尝试恢复（" + str(failures) + "/3）"
                        )
            return {
                "outcome": receipts[action_id]["outcome"],
                "action_id": action_id,
                "state": self.state,
            }

    async def decision_failed(self, payload):
        async with self.lock:
            if self.automatic() and payload.get("epoch") == self.state["control_epoch"]:
                self.observation = None
                if payload.get("reason") == "AGENT_CONFIGURATION":
                    self.wait_human(
                        "AGENT_CONFIGURATION",
                        "导航 Skill 或浏览器协议不兼容，请检查服务版本后恢复任务",
                    )
                else:
                    failures = self.state.get("decision_failures", 0) + 1
                    self.update(decision_failures=failures)
                    if failures >= 3:
                        self.wait_human(
                            "AGENT_UNAVAILABLE", "页面决策连续三次失败，请稍后交还自动化或手动操作"
                        )
                    else:
                        self.event("页面决策暂时失败，正在自动重试（" + str(failures) + "/3）")
            return self.state

    async def route(self, route):
        request = route.request
        if request.is_navigation_request() and not allowed_url(request.url):
            await route.abort()
            self.wait_human("DOMAIN", "目标页面不在已配置的教学平台域名范围内")
        else:
            await route.continue_()

    def manual_export_submitted(self, source):
        if self.state["control_mode"] != "HUMAN" or self.state.get("export_attempted"):
            return
        evidence = self.manual_export_evidence
        # A manual submit is an irreversible fact even if its scope cannot yet be established.
        self.update(
            export_attempted=True,
            export_started_at=time.time(),
            stage="WAIT_EXPORT",
            waiting_reason="EXPORT_ASSOCIATION",
            export_baseline=evidence.get("baseline", []),
            class_scope=evidence.get("scope", {"mode": "manual", "classes": []}),
        )
        self.event("检测到用户手动提交导出；恢复后核对导出记录，不重复提交")

    def download_begin(self, event):
        self.download_guid = event["guid"]
        self.download_urls[event["url"]] = event["guid"]
        self.download_bytes[event["guid"]] = 0

    def download_progress(self, event):
        self.download_bytes[event["guid"]] = event["receivedBytes"]
        self.state["download_progress"] = {
            "received_bytes": event["receivedBytes"],
            "total_bytes": event["totalBytes"],
            "state": event["state"],
        }

    async def execute(self, decision):
        validate_action(decision)
        tool = decision["tool"]
        if tool in {
            "fill_portal_credentials",
            "request_login_code",
            "submit_login_code",
            "apply_device_trust",
        }:
            if not self.portal_auth or not self.automatic():
                raise ValueError("当前任务没有可用的已保存教务凭据")
            if self.state.get("stage") != "AUTHENTICATE":
                raise ValueError("当前页面不处于登录认证阶段")
            if tool == "fill_portal_credentials":
                frame = await self.portal_auth.login_frame()
                if self.portal_auth.filled:
                    raise RpaIssue(
                        "LOGIN_ALREADY_SUBMITTED",
                        "已填写并提交账号密码，请根据当前页面继续登录验证，不要重复填写",
                    )
                if not frame or not await self.portal_auth.fill_credentials(frame):
                    raise ValueError("当前页面不可填写凭据，请核对登录页")
            elif tool == "request_login_code":
                await self.portal_auth.request_code()
                self.wait_human(
                    "LOGIN_CODE", "请在当前任务浏览器填写本人收到的验证码，然后点击交还自动化"
                )
            elif tool == "submit_login_code":
                await self.portal_auth.submit_entered_code()
            elif not await self.portal_auth.trust_prompt():
                raise ValueError("当前未出现可信设备弹窗")
            self.recent.append({"tool": tool})
            return
        if tool in {
            "read_course_list",
            "read_homework_list",
            "finish_catalog",
            "return_to_course_list",
        }:
            if (
                not self.catalog_collection
                or not (await self.authentication_view())["authenticated"]
            ):
                raise ValueError("当前任务不处于已认证的目录采集阶段")
            if tool == "read_course_list":
                await self.catalog_collection.read_courses()
            elif tool == "read_homework_list":
                await self.catalog_collection.read_homeworks(decision["course_id"])
            elif tool == "return_to_course_list":
                await self.return_to_course_list()
            else:
                # Finishing must restore and verify the original course list,
                # even when the Agent skips its final explicit return action.
                await self.return_to_course_list()
                await self.catalog_collection.read_courses()
                if not self.catalog_collection.progress["courses_complete"]:
                    self.event("已返回课程列表，尚需滚动或翻页核验课程清单末尾，再完成同步")
                    self.recent.append({"tool": tool, "result": "course_list_incomplete"})
                    return
                catalog = self.catalog_collection.finish()
                if len(json.dumps(catalog).encode()) > 6 * 1024 * 1024:
                    raise RpaIssue("CATALOG_LIMIT", "目录超过本次读取上限")
                self.update(
                    portal_catalog=catalog,
                    status="SUCCEEDED",
                    stage="DONE",
                    message="课程与作业目录读取完成，已返回课程列表",
                )
            self.recent.append({"tool": tool})
            return
        if tool == "solve_slider_captcha":
            if not self.automatic() or self.state.get("stage") != "SOLVE_CAPTCHA":
                raise ValueError("滑块工具仅可在已交还控制权的验证阶段执行")
            target = await self.captcha_target()
            if not target or self.state.get("captcha_attempts", 0) >= 3:
                self.wait_human("LOGIN", "未发现可处理的滑块或尝试次数已到，请手动完成登录")
                return
            epoch = self.state["control_epoch"]
            self.page, frame = target

            def record_attempt():
                self.update(captcha_attempts=self.state.get("captcha_attempts", 0) + 1)

            try:
                result = await asyncio.wait_for(
                    solve_slider_captcha(
                        self.page,
                        frame=frame,
                        max_attempts=3 - self.state.get("captcha_attempts", 0),
                        can_continue=lambda: (
                            self.automatic() and self.state["control_epoch"] == epoch
                        ),
                        on_attempt=record_attempt,
                    ),
                    20,
                )
            except TimeoutError:
                result = {"status": "failed", "message": "滑块处理超时，请手动完成验证"}
            self.recent.append({"tool": tool, **result})
            if not self.automatic() or self.state["control_epoch"] != epoch:
                return
            if result["status"] in {"solved", "not_present"}:
                if self.portal_auth:
                    self.event("校验工具已返回，页面 Agent 将观察当前认证页面继续处理")
                    return
                for _ in range(20):
                    if await self.adapter.authenticated():
                        if not self.automatic() or self.state["control_epoch"] != epoch:
                            return
                        self.update(
                            auth_status="AUTHENTICATED",
                            stage="WAIT_EXPORT"
                            if self.state.get("export_attempted")
                            else "LOCATE_COURSE",
                            captcha_attempts=0,
                        )
                        self.event("滑块验证完成，已核验官网登录，继续定位课程和作业")
                        return
                    if not self.automatic() or self.state["control_epoch"] != epoch:
                        return
                    await asyncio.sleep(0.2)
            self.wait_human("LOGIN", result.get("message", "滑块已处理，请检查并完成后续登录验证"))
            return
        if self.state.get("stage") == "SOLVE_CAPTCHA" and tool != "request_human":
            raise ValueError("滑块验证阶段仅允许验证工具或请求人工处理")
        if tool == "request_target_confirmation":
            await self.target_confirmation.propose(decision)
            self.recent.append(dict(decision))
            return
        if tool == "request_human":
            reason = str(decision.get("reason", "请检查当前页面"))[:180]
            waiting_reason = (
                "COURSE_OR_TERM"
                if self.state.get("stage") == "LOCATE_COURSE"
                and "课程" in reason
                and "学期" in reason
                else "AGENT_REQUEST"
            )
            self.wait_human(waiting_reason, reason)
            return
        if tool == "open_export_settings":
            if self.catalog_collection or self.state.get("task_type") == "AUTH_BOOTSTRAP":
                raise ValueError("账号连接和目录采集任务不能导出附件")
            self.page = await self.adapter.open_export()
        elif tool == "switch_page":
            idx = int(decision["page_id"])
            self.page = self.context.pages[idx]
        elif tool == "scroll":
            delta = max(-1000, min(1000, int(decision.get("delta_y", 600))))
            if self.catalog_collection:
                await self.catalog_collection.viewport().scroll(delta)
            else:
                await self.page.mouse.wheel(0, delta)
        else:
            ref = decision.get("element_ref")
            if ref not in self.adapter.refs:
                raise ValueError("观察引用已失效")
            page, loc, item = self.adapter.refs[ref]
            self.page = page
            if item.get("submits_form") and not item.get("search_submit"):
                raise ValueError("表单提交必须由任务控制器核验")
            if not allowed_url(page.url) or FORBIDDEN.search(item["name"]):
                raise ValueError("该操作不属于附件下载流程")
            if item["name"].strip() == "下载":
                raise ValueError("文件下载必须使用已关联的导出记录")
            # A read-only review entry can be named "查看提交" in a redesigned
            # portal. It is not a submit button; actual confirmations remain
            # controller-owned and cannot be clicked by the page Agent.
            review_entry = re.fullmatch(
                r"(查看|浏览)(?:学生)?提交(?:记录|详情|列表|情况|附件)?", item["name"].strip()
            )
            if EXPORT_CONFIRM.search(item["name"]) and not review_entry:
                raise ValueError("提交动作必须由任务控制器核验")
            if self.catalog_collection and re.search("导出|下载", item["name"]):
                raise ValueError("目录采集只允许读取，不导出或下载")
            if (
                item.get("semester_kind")
                or SEMESTER.fullmatch(item["name"].strip())
                or re.fullmatch(r"(?:选择|切换|搜索)?学期", item["name"].strip())
            ):
                self.wait_human(
                    "COURSE_OR_TERM",
                    f"请确认课程名“{self.state['target']['course_name']}”或学期。"
                    "如需切换学期，请在任务浏览器中手动操作，完成后点击“交还自动化”继续。",
                )
                return
            if tool == "fill":
                if not any(
                    x in (item["name"] + item["context"]).lower()
                    for x in ("搜索", "search", "查找")
                ):
                    raise ValueError("自动填写只允许搜索字段")
                await loc.fill(str(decision.get("value", ""))[:120], timeout=8000)
            else:
                old_course, old_assignment = (
                    self.adapter.course_confirmed,
                    self.adapter.assignment_confirmed,
                )
                catalog_course = None
                if self.catalog_collection:
                    catalog_course = await self.catalog_collection.record_navigation(loc, page)
                else:
                    await self.adapter.record_navigation(loc, item, page)
                old_pages = set(self.context.pages)
                popup = asyncio.get_running_loop().create_future()

                def opened(new_page):
                    if not popup.done():
                        popup.set_result(new_page)

                self.context.on("page", opened)
                try:
                    await loc.click(timeout=8000)
                    if item.get("opens_new_page"):
                        try:
                            await asyncio.wait_for(asyncio.shield(popup), 2.5)
                        except asyncio.TimeoutError:
                            pass
                    else:
                        await asyncio.sleep(0.2)
                except Exception:
                    self.adapter.course_confirmed, self.adapter.assignment_confirmed = (
                        old_course,
                        old_assignment,
                    )
                    raise
                finally:
                    self.context.remove_listener("page", opened)
                    if not popup.done():
                        popup.cancel()
                new_pages = [p for p in self.context.pages if p not in old_pages]
                if new_pages:
                    self.page = new_pages[-1]
                    self.event(
                        "已进入新打开的页面 page=" + str(self.context.pages.index(self.page))
                    )
                if catalog_course:
                    self.catalog_course_pages[self.page] = catalog_course
        self.recent.append({k: v for k, v in decision.items() if k != "value"})

    async def return_to_course_list(self):
        collection = self.catalog_collection
        page = collection.list_page
        if not page or page.is_closed():
            raise RpaIssue(
                "CATALOG_LIST_UNAVAILABLE", "原课程列表页不可用，请返回本学期课程页并重新读取列表"
            )
        self.page = page
        if not await collection.is_course_list(page):
            # Restore only the URL captured by a verified live list read. The
            # Agent cannot supply a URL, script or a different semester here.
            await page.goto(collection.list_url, wait_until="domcontentloaded", timeout=15000)
            for _ in range(20):
                if await collection.is_course_list(page):
                    break
                await asyncio.sleep(0.2)
            else:
                raise RpaIssue(
                    "CATALOG_LIST_UNAVAILABLE",
                    "返回后未核验到原课程列表或原学期，请核查页面后重新读取",
                )
        self.catalog_course_pages.pop(page, None)
        self.event("已返回课程列表，核验课程清单并继续未完成课程")

    async def prepare_export(self, area):
        page, frame, modal = area
        self.page = page
        if not self.adapter.course_confirmed or not self.adapter.assignment_confirmed:
            self.wait_human(
                "TARGET_CONFIRMATION", "请先进入目标课程与作业，当前导出页面归属尚未核验"
            )
            return
        options = await self.adapter.classes(modal)
        target = self.state["target"].get("portal")
        if target and target.get("class_id"):
            from .portal_download import selected_scope
            from .portal_catalog import query_id

            current_ids = {query_id(frame.url, ("clazzid", "classid")) for frame in page.frames}
            current_ids.discard("")
            current_id = next(iter(current_ids)) if len(current_ids) == 1 else ""
            self.scope_request = selected_scope(self.state["target"]["portal"], options, current_id)
            self.update(requested_scope=self.scope_request)
        elif target or not self.scope_request:
            from .portal_download import all_scope

            self.scope_request = all_scope(options)
            self.update(requested_scope=self.scope_request)
        self.update(stage="SELECT_CLASSES", classes=options)
        if self.scope_request:
            self.scope_request = self.reconcile_scope(self.scope_request, options)
            if self.scope_request != self.state.get("requested_scope"):
                self.update(requested_scope=self.scope_request)
        if not self.scope_request:
            self.wait_human(
                "SCOPE_SELECTION", "当前导出范围与页面可用班级不一致，请检查作业导出设置"
            )
            return
        try:
            scope = await self.adapter.apply_scope(modal, self.scope_request, options)
            if target and target.get("class_id"):
                scope["classes"] = [
                    {**entry, "name": target["class_name"], "platform_id": target["class_id"]}
                    for entry in scope.get("classes", [])
                ]
        except RpaIssue as exc:
            report = self.record_failure(
                exc, "apply_scope", await self.adapter.scope_diagnostics(modal)
            )
            self.wait_human(
                "EXPORT_FORMAT" if exc.code.startswith("ATTACHMENT_") else "SCOPE_SELECTION",
                report["message"],
            )
            return
        baseline = getattr(self.adapter, "pre_export_baseline", None)
        before, observed = (
            baseline if baseline and baseline[1] else await self.adapter.export_baseline()
        )
        scope["baseline_observed"] = observed
        self.update(
            class_scope=scope,
            export_baseline=[r["ref"] for r in before],
            export_attempted=True,
            stage="EXPORT",
            export_started_at=time.time(),
            last_error=None,
        )
        self.event("班级范围和附件导出选项已核验，正在提交导出")
        # Persist intent BEFORE clicking; an uncertain result can never blindly submit again.
        confirm = modal.locator("a.confirmDown").filter(has_text="确定")
        if not await confirm.count():
            confirm = modal.get_by_role("button", name="确定", exact=True)
        await confirm.click(timeout=8000)
        self.update(stage="WAIT_EXPORT")
        self.event("已提交导出，正在关联下载中心任务")

    @staticmethod
    def reconcile_scope(request, options):
        """Retain explicit intent after excluding an unrelated option or reordering."""
        names = {option["id"]: option["name"] for option in options}
        if request.get("mode") == "all":
            # All classes is the default intent for the verified assignment.
            return {**request, "class_ids": [], "option_names": names} if options else None
        if request.get("mode") != "selected" or not request.get("class_ids"):
            return None
        normalize = lambda value: re.sub(r"\s+", " ", value).strip()
        selected = []
        for ref in request["class_ids"]:
            old_name = normalize(request.get("option_names", {}).get(ref, ""))
            matches = [
                o["id"] for o in options if not o["all"] and normalize(o["name"]) == old_name
            ]
            if not old_name or len(matches) != 1:
                return None
            selected.append(matches[0])
        return {**request, "class_ids": selected, "option_names": names}

    async def process_exports(self):
        async with self.lock:
            if not self.gate.is_set():
                return
            if await self.adapter.login_required():
                self.wait_human("LOGIN", "等待导出期间会话过期，请重新登录")
                return
            rows = await self.adapter.export_rows()
            if not rows:
                for page in self.context.pages:
                    center = page.get_by_text("下载中心", exact=True)
                    if await center.count() and await center.first.is_visible():
                        await center.first.click(timeout=5000)
                        break
                if time.time() - self.state["export_started_at"] > 30:
                    self.wait_human(
                        "EXPORT_ASSOCIATION", "请打开下载中心；导出可能已提交，请勿重复导出"
                    )
                return
            exports = self.state.get("exports", [])
            if not exports:
                exports = match_new_exports(
                    rows,
                    self.state["export_baseline"],
                    self.state["target"]["assignment_name"],
                    self.state["class_scope"],
                )
                if not exports:
                    candidates = [r for r in rows if r["ref"] not in self.state["export_baseline"]]
                    self.update(export_candidates=candidates)
                    elapsed = time.time() - self.state["export_started_at"]
                    collecting_all = self.state["class_scope"].get("mode") == "all"
                    if (candidates and not collecting_all) or elapsed > 30:
                        self.wait_human(
                            "EXPORT_ASSOCIATION",
                            "请核对并勾选本次导出的全部记录；系统不会按行号猜测归属",
                        )
                    return
                self.update(exports=exports)
            exports = reconcile_exports(exports, rows, self.state.get("export_baseline", []))
            if exports != self.state.get("exports"):
                self.update(exports=exports)
            files = self.state.get("files", [])
            completed = {f["export_ref"] for f in files if f["status"] == "VERIFIED"}
            failed = {f["export_ref"] for f in files if f["status"] == "FAILED"}
            ready, missing, pending = [], [], []
            for export in exports:
                ref = export["ref"]
                if ref in completed or ref in failed:
                    continue
                if not export["available"]:
                    missing.append(export)
                elif export["status"] == "FAILED":
                    files.append(
                        {"export_ref": ref, "status": "FAILED", "error": "平台导出失败或已过期"}
                    )
                    self.update(files=files)
                elif export["status"] == "READY":
                    ready.append(export)
                else:
                    pending.append(export)
            if ready:
                export = ready[0]
                ref = export["ref"]
                page, _ = self.adapter.export_locators[export["row_ref"]]
                self.page = page
                self.update(stage="DOWNLOAD", download_progress=None, last_error=None)
                self.event(
                    f"正在下载已导出的附件（已校验 {len(completed)}/{len(exports)}，仍在打包 {len(pending)}）"
                )
                async with page.expect_download(timeout=30000) as info:
                    await self.adapter.click_export_download(export["row_ref"])
                download = await info.value
            else:
                if missing:
                    self.wait_human(
                        "EXPORT_ASSOCIATION",
                        "已关联的导出记录不可见或不唯一，请检查下载中心；已成功附件已保留",
                    )
                    return
                if pending:
                    if time.time() - self.state["export_started_at"] > 900:
                        self.wait_human("EXPORT_TIMEOUT", "平台打包等待超过15分钟，请检查导出状态")
                    else:
                        if self.state.get("stage") != "WAIT_EXPORT":
                            self.update(stage="WAIT_EXPORT")
                        message = f"等待平台打包 {len(pending)} 个附件，已下载校验 {len(completed)}/{len(exports)} 个"
                        if self.state.get("message") != message:
                            self.event(message)
                    return
                verified = sum(f["status"] == "VERIFIED" for f in files)
                status = (
                    "SUCCEEDED"
                    if verified == len(exports)
                    else "PARTIAL_SUCCESS"
                    if verified
                    else "FAILED"
                )
                self.update(status=status, stage="DONE", files=files)
                self.event(f"下载结束，已校验 {verified}/{len(exports)} 个附件")
                return
        # Transfer proceeds without owning the browser action lock. Human takeover stays responsive.
        try:
            result = await save_download(
                download,
                self.manager.download_root / self.state["job_id"],
                ref,
                progress=lambda: self.download_bytes.get(self.download_urls.get(download.url), 0),
                on_verifying=lambda: self.verifying_download(len(completed), len(exports)),
            )
            result["class_name"] = next(
                (
                    c["name"]
                    for c in self.state["class_scope"].get("classes", [])
                    if not c.get("all") and c["name"] in export["text"]
                ),
                "",
            )
            files.append(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.record_failure(exc, "download")
            files.append({"export_ref": ref, "status": "FAILED", "error": "下载或压缩包校验失败"})
        self.update(files=files, stage="WAIT_EXPORT")
        if self.state["control_mode"] == "AUTO":
            verified = sum(f["status"] == "VERIFIED" for f in files)
            self.event(f"附件处理进度：已下载校验 {verified}/{len(exports)} 个")

    def verifying_download(self, completed, total):
        self.update(stage="VERIFY")
        if self.state["control_mode"] == "AUTO":
            self.event(f"附件传输完成，正在校验压缩包（此前已校验 {completed}/{total} 个）")

    async def pause(self):
        if self.state["status"] in TERMINAL:
            raise ValueError("任务已结束")
        self.gate.clear()
        self.update(control_mode="PAUSING", control_epoch=self.state["control_epoch"] + 1)
        if self.startup_operation:
            self.startup_operation.cancel()
            await asyncio.gather(self.startup_operation, return_exceptions=True)
        async with self.lock:
            await self._release_pointer_locked()
            self.wait_human("USER_TAKEOVER", "已暂停，用户可以操作浏览器")

    async def resume(self):
        if self.state["status"] in TERMINAL or self.state["control_mode"] != "HUMAN":
            raise ValueError("当前任务不能交还自动化")
        async with self.lock:
            if self.state["status"] in TERMINAL or self.state["control_mode"] != "HUMAN":
                raise ValueError("当前任务不能交还自动化")
            if self.startup_pending:
                await self._release_pointer_locked()
                self.update(
                    control_mode="AUTO",
                    status="RUNNING",
                    waiting_reason="",
                    control_epoch=self.state["control_epoch"] + 1,
                )
                self.event("正在重新核验并加载官网登录页")
                self.startup_wake.set()
                return
            if self.state.get("target_resolution"):
                raise ValueError("请先在弹窗确认候选项或修改名称，确认后会自动继续")
            authenticated = self.adapter and await self.adapter.authenticated()
            captcha = not authenticated and self.adapter and await self.prepare_captcha()
            if (
                self.portal_auth
                and self.state.get("waiting_reason") == "LOGIN_IDENTITY"
                and authenticated
            ):
                self.portal_auth.identity_confirmed = True
                await self.portal_auth.identity()
            if not authenticated and not captcha and not self.portal_auth:
                raise ValueError("请先在浏览器中完成登录并返回教学平台")
            if self.state.get("waiting_reason") == "CLASS_SCOPE" and not self.scope_request:
                raise ValueError("请先确认班级范围")
            if self.state.get("waiting_reason") == "EXPORT_ASSOCIATION" and not self.state.get(
                "exports"
            ):
                # Opening the download center may be all that was needed; observe again first.
                rows = await self.adapter.export_rows()
                if not rows:
                    raise ValueError("请先打开下载中心并确认本次导出记录")
            if self.state.get("waiting_reason") == "COURSE_OR_TERM" and not self.state.get(
                "export_attempted"
            ):
                # Returning control confirms the teacher's current page. Read
                # its semester only; never open a menu or select an option.
                self.adapter.active_page = self.page
                view = await self.adapter.observe()
                terms = {
                    e.get("selected_option") or e["name"]
                    for e in view["elements"]
                    if SEMESTER.fullmatch(e.get("selected_option", ""))
                    or (e.get("semester_kind") == "trigger" and SEMESTER.fullmatch(e["name"]))
                }
                if len(terms) == 1 and self.state["target"].get("term") != next(iter(terms)):
                    self.state.setdefault("requested_target", dict(self.state["target"]))
                    target = {**self.state["target"], "term": next(iter(terms))}
                    self.update(target=target)
                    self.adapter.target = target
                    self.event("教师已确认当前页面学期：" + target["term"])
            await self._release_pointer_locked()
            self.update(
                control_mode="AUTO",
                status="RUNNING",
                waiting_reason="",
                auth_status="CAPTCHA_REQUIRED" if captcha else "AUTHENTICATED",
                captcha_attempts=self.state.get("captcha_attempts", 0) if captcha else 0,
                stage="SOLVE_CAPTCHA"
                if captcha
                else "LOCATE_COURSE"
                if self.state["stage"] in {"AUTHENTICATE", "SOLVE_CAPTCHA"}
                else self.state["stage"],
                control_epoch=self.state["control_epoch"] + 1,
                auto_actions=0,
                navigation_failures=0,
                decision_failures=0,
            )
            self.navigation_marker = None
            self.stagnant_actions = 0
            self.event("正在重新核验页面并继续自动化")
            self.gate.set()

    async def cancel(self, status="CANCELLED"):
        if self.state["status"] in TERMINAL and self.state["status"] != "INTERRUPTED":
            return
        self.gate.clear()
        self.update(status=status, control_epoch=self.state["control_epoch"] + 1)
        if self.runner:
            self.runner.cancel()
            try:
                await self.runner
            except asyncio.CancelledError:
                pass

    async def frame(self):
        # Screenshot compression/transport must not hold the browser action lock.
        # Only one capture per task runs at a time, even with multiple viewers.
        async with self.frame_lock:
            async with self.lock:
                page, context, epoch = self.page, self.context, self.state["control_epoch"]
                if not page or page.is_closed():
                    raise ValueError("浏览器当前不可用")
                url = page.url
            if self.state.get("portal_session"):
                from .portal_auth import SENSITIVE

                data = await page.screenshot(
                    type="jpeg",
                    quality=55,
                    timeout=8000,
                    mask=[frame.locator(SENSITIVE) for frame in page.frames],
                )
            else:
                data = await page.screenshot(type="jpeg", quality=55, timeout=8000)
            pages = [
                {"id": str(i), "title": (await p.title())[:100]}
                for i, p in enumerate(context.pages)
                if not p.is_closed()
            ]
            async with self.lock:
                if (
                    page != self.page
                    or page.is_closed()
                    or url != page.url
                    or epoch != self.state["control_epoch"]
                ):
                    raise ValueError("页面或控制权已变化，请刷新画面")
                return data, self.record_frame(page, url, epoch), pages

    def record_frame(self, page, url, epoch):
        """Called with the action lock held, for HTTP captures and pushed frames."""
        self.frame_revision += 1
        self.frame_meta = {
            "revision": self.frame_revision,
            "page_id": str(self.context.pages.index(page)),
            "url": url,
            "epoch": epoch,
        }
        self.frame_history[self.frame_revision] = {
            **self.frame_meta,
            "captured_at": time.monotonic(),
        }
        # Retain at least three seconds of frames at the streaming rate.
        self.frame_history = dict(list(self.frame_history.items())[-128:])
        return self.frame_meta

    async def release_pointer(self, owner):
        async with self.lock:
            await self._release_pointer_locked(owner)

    async def _release_pointer_locked(self, owner=None):
        if self.pointer_page and (owner is None or self.pointer_owner is owner):
            page = self.pointer_page
            self.pointer_page = self.pointer_owner = None
            if not page.is_closed():
                # Releasing on the original button could accidentally submit a form.
                await page.mouse.move(-1, -1)
                await page.mouse.up()

    async def browser_input(self, payload, owner=None):
        if not isinstance(payload, dict):
            raise ValueError("操作参数无效")
        actions = payload.get("actions") if payload.get("kind") == "batch" else [payload]
        if (
            not isinstance(actions, list)
            or not 1 <= len(actions) <= 32
            or any(
                not isinstance(action, dict) or action.get("kind") == "batch" for action in actions
            )
        ):
            raise ValueError("批量操作参数无效")
        async with self.lock:
            for action in actions:
                await self._browser_input_locked(action, owner)

    async def _browser_input_locked(self, payload, owner=None):
        if (
            self.state["status"] in TERMINAL
            or self.state["control_mode"] != "HUMAN"
            or not self.context
        ):
            raise ValueError("请先暂停并接管浏览器")
        if int(payload.get("epoch", -1)) != self.state["control_epoch"]:
            raise ValueError("浏览器控制权已经变化，请刷新画面")
        kind = payload.get("kind")
        if kind == "switch_page":
            index = int(payload["page_id"])
            if index < 0 or index >= len(self.context.pages):
                raise ValueError("页面不存在")
            if self.context.pages[index].is_closed():
                raise ValueError("页面已关闭")
            await self._release_pointer_locked()
            self.page = self.context.pages[index]
            self.frame_meta = None
            return
        meta = self.frame_meta
        if (
            not meta
            or str(payload.get("page_id")) != meta["page_id"]
            or meta["url"] != self.page.url
            or meta["epoch"] != self.state["control_epoch"]
        ):
            raise ValueError("页面已变化，请刷新画面")
        if not allowed_url(self.page.url):
            raise ValueError("页面域名不在允许范围内")
        if self.pointer_page and self.pointer_owner is not owner:
            raise ValueError("另一连接正在拖动，请稍后重试")
        if not self.state.get("export_attempted") and (
            kind == "pointer_up" or (kind == "key" and payload.get("key") == "Enter")
        ):
            area = await self.adapter.export_area()
            if area:
                options = await self.adapter.classes(area[2])
                selected = [
                    o
                    for o in options
                    if await self.adapter.checked(self.adapter.class_locators[o["id"]])
                ]
                rows = await self.adapter.export_rows()
                self.manual_export_evidence = {
                    "baseline": [r["ref"] for r in rows],
                    "scope": {
                        "mode": "all"
                        if any(o["all"] for o in selected)
                        else "selected"
                        if selected
                        else "manual",
                        "classes": selected,
                    },
                }
        x, y = float(payload.get("x", 0)), float(payload.get("y", 0))
        if not 0 <= x <= 1440 or not 0 <= y <= 1000:
            raise ValueError("坐标越界")
        if kind == "pointer_down":
            displayed = self.frame_history.get(int(payload.get("revision", -1)))
            if (
                not displayed
                or time.monotonic() - displayed["captured_at"] > 3
                or any(displayed[key] != meta[key] for key in ("page_id", "url", "epoch"))
            ):
                raise ValueError("画面已刷新，请重试")
            await self.page.mouse.move(x, y)
            self.pointer_page, self.pointer_owner = self.page, owner
            await self.page.mouse.down()
        elif kind == "pointer_move":
            await self.page.mouse.move(x, y)
        elif kind == "pointer_up":
            await self.page.mouse.move(x, y)
            await self.page.mouse.up()
            self.pointer_page = self.pointer_owner = None
        elif kind == "pointer_cancel":
            await self._release_pointer_locked(owner)
        elif kind == "text":
            await self.page.keyboard.insert_text(str(payload.get("text", ""))[:2000])
        elif kind == "key":
            key = str(payload.get("key", ""))
            if key not in {
                "Enter",
                "Tab",
                "Backspace",
                "Delete",
                "Escape",
                "ArrowLeft",
                "ArrowRight",
                "ArrowUp",
                "ArrowDown",
                "Control+A",
            }:
                raise ValueError("不支持的按键")
            await self.page.keyboard.press(key)
        elif kind == "scroll":
            await self.page.mouse.wheel(0, max(-1000, min(1000, int(payload.get("delta_y", 0)))))
        else:
            raise ValueError("不支持的操作")
        self.last_activity = time.monotonic()
