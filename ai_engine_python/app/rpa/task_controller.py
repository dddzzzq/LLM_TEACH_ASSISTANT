import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

from .tools import validate_action
from .download_manager import save_download
from .portal_adapter import PortalAdapter, TARGET_URL, FORBIDDEN, EXPORT_CONFIRM, match_new_exports


TERMINAL = {'SUCCEEDED', 'PARTIAL_SUCCESS', 'FAILED', 'CANCELLED', 'INTERRUPTED'}
ALLOWED_HOSTS = ('xidian.edu.cn', 'chaoxing.com', 'chaoxing.com.cn')


def allowed_url(url):
    parsed = urlsplit(url)
    return url == 'about:blank' or (parsed.scheme in ('http', 'https') and
        any(parsed.hostname == h or (parsed.hostname or '').endswith('.' + h) for h in ALLOWED_HOSTS))


class Task:
    def __init__(self, manager, state):
        self.manager, self.state = manager, state
        self.browser = self.context = self.adapter = self.page = None
        self.lock = asyncio.Lock()
        self.frame_lock = asyncio.Lock()
        self.gate = asyncio.Event()
        self.runner = None
        self.scope_request = None
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

    def update(self, **fields):
        self.state.update(fields, updated_at=time.time(), state_version=self.state.get('state_version', 0) + 1)
        self.manager.persist(self.state)

    def event(self, message):
        events = self.state.setdefault('events', [])
        events.append({'time': time.time(), 'message': message})
        self.state['events'] = events[-60:]
        self.update(message=message)

    def wait_human(self, reason, message):
        self.gate.clear()
        self.last_activity = time.monotonic()
        self.update(status='WAITING_USER', control_mode='HUMAN', waiting_reason=reason,
                    control_epoch=self.state['control_epoch'] + 1,
                    **({'auth_status': 'REQUIRED'} if reason == 'LOGIN' else {}))
        self.event(message)

    async def refresh_login_state(self):
        """Observe login while waiting, without returning browser control to the Agent."""
        if (self.state['status'] in TERMINAL or self.state['control_mode'] != 'HUMAN'
                or self.state.get('waiting_reason') not in {'LOGIN', 'LOGIN_COMPLETE'} or not self.adapter):
            return
        async with self.lock:
            # Pause/resume may have changed control while waiting for the action lock.
            if (self.state['status'] in TERMINAL or self.state['control_mode'] != 'HUMAN'
                    or self.state.get('waiting_reason') not in {'LOGIN', 'LOGIN_COMPLETE'}):
                return
            try:
                authenticated = await self.adapter.authenticated()
                required = not authenticated and await self.adapter.login_required()
            except Exception:
                # A redirect or detached frame is not evidence that login failed.
                return
            auth_status = 'AUTHENTICATED' if authenticated else 'REQUIRED' if required else 'UNKNOWN'
            if auth_status == self.state.get('auth_status'):
                return
            self.update(auth_status=auth_status,
                        waiting_reason='LOGIN_COMPLETE' if authenticated else 'LOGIN',
                        message='官网登录已完成。点击“交还自动化”后，Agent 将继续定位课程和作业。' if authenticated
                        else '请完成官网登录，然后点击“交还自动化”继续。' if required
                        else '正在核验官网登录状态。如已完成登录，请点击“交还自动化”继续。')

    async def run(self):
        try:
            proxy = self.manager.proxy
            browser_downloads = self.manager.download_root / self.state['job_id'] / '.browser'
            browser_downloads.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.browser = await self.manager.playwright.chromium.launch(headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage'], proxy={'server': proxy} if proxy else None,
                downloads_path=str(browser_downloads))
            self.context = await self.browser.new_context(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
            await self.context.route('**/*', self.route)
            await self.context.expose_binding('__rpa_manual_export', self.manual_export_submitted)
            await self.context.add_init_script('''document.addEventListener('click', event => {
              const button=event.target.closest('a.confirmDown,button');
              const modal=button?.closest('.popDiv.centerPop,[role=dialog]');
              if(modal && /导出/.test(modal.innerText) && /确定|确认/.test(button.innerText))
                window.__rpa_manual_export().catch(()=>{});
            }, true);''')
            self.page = await self.context.new_page()
            page_cdp = await self.context.new_cdp_session(self.page)
            target_info = await page_cdp.send('Target.getTargetInfo')
            browser_cdp = await self.browser.new_browser_cdp_session()
            browser_cdp.on('Browser.downloadWillBegin', self.download_begin)
            browser_cdp.on('Browser.downloadProgress', self.download_progress)
            await browser_cdp.send('Browser.setDownloadBehavior', {
                'behavior':'allowAndName', 'browserContextId':target_info['targetInfo']['browserContextId'],
                'downloadPath':str(browser_downloads), 'eventsEnabled':True})
            self.adapter = PortalAdapter(self.context, self.state['target'])
            async with self.lock:
                await self.page.goto(TARGET_URL, wait_until='domcontentloaded', timeout=30000)
                login = self.page.locator('.denglu')
                if await login.count() and await login.is_visible():
                    await login.click(timeout=10000)
            self.update(stage='AUTHENTICATE')
            self.wait_human('LOGIN', '请在任务浏览器中完成账号、密码及验证码，然后交还自动化')
            errors = 0
            while True:
                await self.gate.wait()
                if self.state['status'] in TERMINAL:
                    return
                epoch = self.state['control_epoch']
                try:
                    if self.state.get('export_attempted') and self.adapter.assignment_confirmed:
                        await self.process_exports()
                        if self.state['status'] in TERMINAL:
                            return
                        await asyncio.sleep(2)
                        continue
                    async with self.lock:
                        if not self.gate.is_set() or epoch != self.state['control_epoch']:
                            continue
                        if await self.adapter.login_required():
                            self.wait_human('LOGIN', '官网需要重新验证身份，请完成登录')
                            continue
                        await self.adapter.reconcile()
                        area = await self.adapter.export_area()
                        if area:
                            if self.state.get('export_attempted'):
                                self.wait_human('EXPORT_ASSOCIATION', '已有导出检查点，请关闭设置并打开下载中心，系统不会重新提交')
                                continue
                            await self.prepare_export(area)
                            continue
                        # Navigation is driven exclusively by Go/Eino through observe/action.
                        # This loop owns only deterministic export and download mechanics.
                        if self.state.get('stage') not in {'LOCATE_COURSE', 'LOCATE_ASSIGNMENT', 'EXPORT'}:
                            self.update(stage='LOCATE_COURSE')
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    errors += 1
                    # Error messages from network/browser libraries may contain URLs or tokens.
                    self.event('页面操作未完成：' + type(exc).__name__)
                    if errors >= 3 or isinstance(exc, ValueError):
                        self.wait_human('RECOVERY', '自动化需要协助，请检查当前页面后交还自动化')
                        errors = 0
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.update(status='FAILED', control_mode='HUMAN', error_code=type(exc).__name__)
            self.event('浏览器任务启动或执行失败，请检查网络及 Worker 配置')
        finally:
            if self.browser:
                await self.browser.close()
            self.browser = self.context = self.page = None

    def automatic(self):
        return (self.state['status'] == 'RUNNING' and self.state['control_mode'] == 'AUTO'
                and self.gate.is_set() and self.adapter is not None)

    async def observation_fingerprint(self, view):
        # Full URLs stay inside the Worker; query strings may contain authentication tokens.
        identity = {'view': view, 'urls': [p.url for p in self.context.pages if not p.is_closed()]}
        return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    async def observe(self):
        async with self.lock:
            if not self.automatic() or self.state.get('export_attempted') and self.adapter.assignment_confirmed:
                return {'ready': False, 'state': self.state}
            if await self.adapter.login_required():
                self.wait_human('LOGIN', '官网需要重新验证身份，请完成登录')
                return {'ready': False, 'state': self.state}
            await self.adapter.reconcile()
            if await self.adapter.export_area():
                return {'ready': False, 'state': self.state}
            stage = ('EXPORT' if self.adapter.assignment_confirmed else
                     'LOCATE_ASSIGNMENT' if self.adapter.course_confirmed else 'LOCATE_COURSE')
            if self.state.get('stage') != stage:
                self.update(stage=stage)
            view = await self.adapter.observe()
            fingerprint = await self.observation_fingerprint(view)
            epoch = self.state['control_epoch']
            if (not self.observation or self.observation['fingerprint'] != fingerprint
                    or self.observation['epoch'] != epoch):
                self.observation_revision += 1
                self.observation = {'id': str(uuid4()), 'revision': self.observation_revision,
                                    'epoch': epoch, 'fingerprint': fingerprint}
            return {'ready': True, 'observation_id': self.observation['id'],
                    'revision': self.observation['revision'], 'epoch': epoch,
                    'target': self.state['target'], 'stage': stage,
                    'observation': view, 'recent_actions': self.recent[-5:], 'state': self.state}

    async def apply_action(self, payload):
        action = validate_action(payload.get('action'))
        action_id = payload.get('action_id')
        if not isinstance(action_id, str) or not 1 <= len(action_id) <= 128:
            raise ValueError('动作 ID 无效')
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        async with self.lock:
            receipts = self.state.setdefault('action_receipts', {})
            prior = receipts.get(action_id)
            if prior:
                if prior['digest'] != digest:
                    raise ValueError('动作 ID 已用于不同请求')
                # Never replay an action whose external effect may already have happened.
                return {'outcome': prior['outcome'], 'action_id': action_id, 'state': self.state}
            current = self.observation
            if (not self.automatic() or not current
                    or payload.get('epoch') != self.state['control_epoch']
                    or payload.get('observation_id') != current['id']
                    or payload.get('revision') != current['revision']):
                raise ValueError('观察或控制权已变化，请重新观察')
            view = await self.adapter.observe()
            if await self.observation_fingerprint(view) != current['fingerprint']:
                self.observation = None
                raise ValueError('页面已变化，请重新观察')
            if len(receipts) >= 1000:
                self.wait_human('ACTION_BUDGET', '任务动作总预算已到，请检查并结束任务')
                raise ValueError('任务动作预算已到')
            receipts[action_id] = {'digest': digest, 'outcome': 'unknown', 'tool': action['tool'],
                                   'skill_version': str(payload.get('skill_version', ''))[:128]}
            self.observation = None
            self.update(action_receipts=receipts)
            try:
                await self.execute(action)
                receipts[action_id]['outcome'] = 'completed'
                count = self.state.get('auto_actions', 0) + 1
                self.update(action_receipts=receipts, auto_actions=count)
                if count >= 60 and self.automatic():
                    self.wait_human('ACTION_BUDGET', '自动页面操作已达到本轮预算，请检查页面后交还自动化')
            except asyncio.CancelledError:
                # Durable UNKNOWN receipt survives Worker shutdown; do not execute again.
                raise
            except Exception:
                if self.state['status'] not in TERMINAL:
                    self.wait_human('RECOVERY', '动作执行结果需要核对，请检查当前页面后交还自动化')
            return {'outcome': receipts[action_id]['outcome'], 'action_id': action_id, 'state': self.state}

    async def decision_failed(self, payload):
        async with self.lock:
            if self.automatic() and payload.get('epoch') == self.state['control_epoch']:
                self.observation = None
                if payload.get('reason') == 'AGENT_CONFIGURATION':
                    self.wait_human('AGENT_CONFIGURATION', '导航 Skill 或浏览器协议不兼容，请检查服务版本后恢复任务')
                else:
                    self.wait_human('AGENT_UNAVAILABLE', '页面决策服务暂不可用，请稍后交还自动化或手动操作')
            return self.state

    async def route(self, route):
        request = route.request
        if request.is_navigation_request() and not allowed_url(request.url):
            await route.abort()
            self.wait_human('DOMAIN', '目标页面不在已配置的教学平台域名范围内')
        else:
            await route.continue_()

    def manual_export_submitted(self, source):
        if self.state['control_mode'] != 'HUMAN' or self.state.get('export_attempted'):
            return
        evidence = self.manual_export_evidence
        # A manual submit is an irreversible fact even if its scope cannot yet be established.
        self.update(export_attempted=True, export_started_at=time.time(), stage='WAIT_EXPORT',
                    waiting_reason='EXPORT_ASSOCIATION',
                    export_baseline=evidence.get('baseline', []),
                    class_scope=evidence.get('scope', {'mode':'manual','classes':[]}))
        self.event('检测到用户手动提交导出；恢复后核对导出记录，不重复提交')

    def download_begin(self, event):
        self.download_guid = event['guid']
        self.download_urls[event['url']] = event['guid']
        self.download_bytes[event['guid']] = 0

    def download_progress(self, event):
        self.download_bytes[event['guid']] = event['receivedBytes']
        self.state['download_progress'] = {'received_bytes':event['receivedBytes'],
            'total_bytes':event['totalBytes'], 'state':event['state']}

    async def execute(self, decision):
        validate_action(decision)
        tool = decision['tool']
        if tool == 'request_human':
            self.wait_human('AGENT_REQUEST', str(decision.get('reason', '请检查当前页面'))[:180])
            return
        if tool == 'open_export_settings':
            self.page = await self.adapter.open_export()
        elif tool == 'switch_page':
            idx = int(decision['page_id'])
            self.page = self.context.pages[idx]
        elif tool == 'scroll':
            await self.page.mouse.wheel(0, max(-1000, min(1000, int(decision.get('delta_y', 600)))))
        else:
            ref = decision.get('element_ref')
            if ref not in self.adapter.refs:
                raise ValueError('观察引用已失效')
            page, loc, item = self.adapter.refs[ref]
            self.page = page
            if not allowed_url(page.url) or FORBIDDEN.search(item['name']):
                raise ValueError('该操作不属于附件下载流程')
            if item['name'].strip() == '下载':
                raise ValueError('文件下载必须使用已关联的导出记录')
            if EXPORT_CONFIRM.search(item['name']):
                raise ValueError('提交动作必须由任务控制器核验')
            if tool == 'fill':
                if not any(x in (item['name'] + item['context']).lower() for x in ('搜索', 'search', '查找')):
                    raise ValueError('自动填写只允许搜索字段')
                await loc.fill(str(decision.get('value', ''))[:120], timeout=8000)
            else:
                old_course, old_assignment = self.adapter.course_confirmed, self.adapter.assignment_confirmed
                await self.adapter.record_navigation(loc, item, page)
                old_pages = set(self.context.pages)
                try:
                    await loc.click(timeout=8000)
                except Exception:
                    self.adapter.course_confirmed, self.adapter.assignment_confirmed = old_course, old_assignment
                    raise
                await asyncio.sleep(0.2)
                new_pages = [p for p in self.context.pages if p not in old_pages]
                if new_pages:
                    self.page = new_pages[-1]
        self.recent.append({k: v for k, v in decision.items() if k != 'value'})
        if len(self.recent) >= 3 and self.recent[-1] == self.recent[-2] == self.recent[-3]:
            self.wait_human('NO_PROGRESS', '相同操作没有明确进展，请协助检查页面')

    async def prepare_export(self, area):
        page, frame, modal = area
        self.page = page
        if not self.adapter.course_confirmed or not self.adapter.assignment_confirmed:
            self.wait_human('TARGET_CONFIRMATION', '请先进入目标课程与作业，当前导出页面归属尚未核验')
            return
        options = await self.adapter.classes(modal)
        self.update(stage='SELECT_CLASSES', classes=options)
        if self.scope_request and self.scope_request.get('option_names') != {o['id']: o['name'] for o in options}:
            self.scope_request = None
        if not self.scope_request:
            self.wait_human('CLASS_SCOPE', '请在面板中明确选择要导出的班级，或选择全部班级')
            return
        scope = await self.adapter.apply_scope(modal, self.scope_request, options)
        before = await self.adapter.export_rows()
        self.update(class_scope=scope, export_baseline=[r['ref'] for r in before],
                    export_attempted=True, stage='EXPORT', export_started_at=time.time())
        # Persist intent BEFORE clicking; an uncertain result can never blindly submit again.
        confirm = modal.locator('a.confirmDown').filter(has_text='确定')
        if not await confirm.count():
            confirm = modal.get_by_role('button', name='确定', exact=True)
        await confirm.click(timeout=8000)
        self.update(stage='WAIT_EXPORT')
        self.event('已提交导出，正在关联下载中心任务')

    async def process_exports(self):
        async with self.lock:
            if not self.gate.is_set():
                return
            if await self.adapter.login_required():
                self.wait_human('LOGIN', '等待导出期间会话过期，请重新登录')
                return
            rows = await self.adapter.export_rows()
            if not rows:
                for page in self.context.pages:
                    center = page.get_by_text('下载中心', exact=True)
                    if await center.count() and await center.first.is_visible():
                        await center.first.click(timeout=5000)
                        break
                if time.time() - self.state['export_started_at'] > 30:
                    self.wait_human('EXPORT_ASSOCIATION', '请打开下载中心；导出可能已提交，请勿重复导出')
                return
            exports = self.state.get('exports', [])
            if not exports:
                exports = match_new_exports(rows, self.state['export_baseline'],
                    self.state['target']['assignment_name'], self.state['class_scope'])
                if not exports:
                    candidates = [r for r in rows if r['ref'] not in self.state['export_baseline']]
                    self.update(export_candidates=candidates)
                    if candidates or time.time() - self.state['export_started_at'] > 30:
                        self.wait_human('EXPORT_ASSOCIATION', '请核对并勾选本次导出的全部记录；系统不会按行号猜测归属')
                    return
                self.update(exports=exports)
            current = {r['ref']: r for r in rows}
            files = self.state.get('files', [])
            completed = {f['export_ref'] for f in files if f['status'] == 'VERIFIED'}
            failed = {f['export_ref'] for f in files if f['status'] == 'FAILED'}
            for export in exports:
                ref = export['ref']
                if ref in completed or ref in failed:
                    continue
                row = current.get(ref)
                if not row:
                    self.wait_human('EXPORT_ASSOCIATION', '已关联的导出记录暂不可见，请检查下载中心')
                    return
                if row['status'] == 'FAILED':
                    files.append({'export_ref': ref, 'status': 'FAILED', 'error': '平台导出失败'})
                    self.update(files=files)
                    continue
                if row['status'] != 'READY':
                    if time.time() - self.state['export_started_at'] > 900:
                        self.wait_human('EXPORT_TIMEOUT', '平台打包等待超过15分钟，请检查导出状态')
                    return
                page, loc = self.adapter.export_locators[ref]
                self.page = page
                button = loc.locator('a.download_ic').first
                if not await button.count():
                    button = loc.get_by_text('下载', exact=True).first
                self.update(stage='DOWNLOAD')
                async with page.expect_download(timeout=30000) as info:
                    await button.click(timeout=8000)
                download = await info.value
                break
            else:
                verified = sum(f['status'] == 'VERIFIED' for f in files)
                status = 'SUCCEEDED' if verified == len(exports) else 'PARTIAL_SUCCESS' if verified else 'FAILED'
                self.update(status=status, stage='DONE', files=files)
                self.event(f'下载结束，已校验 {verified}/{len(exports)} 个附件')
                return
        # Transfer proceeds without owning the browser action lock. Human takeover stays responsive.
        try:
            result = await save_download(download, self.manager.download_root / self.state['job_id'], ref,
                progress=lambda: self.download_bytes.get(self.download_urls.get(download.url), 0))
            result['class_name'] = next((c['name'] for c in self.state['class_scope'].get('classes', [])
                if not c.get('all') and c['name'] in export['text']), '')
            files.append(result)
        except asyncio.CancelledError:
            raise
        except Exception:
            files.append({'export_ref': ref, 'status': 'FAILED', 'error': '下载或压缩包校验失败'})
        self.update(files=files, stage='WAIT_EXPORT')

    async def pause(self):
        if self.state['status'] in TERMINAL:
            raise ValueError('任务已结束')
        self.gate.clear()
        self.update(control_mode='PAUSING', control_epoch=self.state['control_epoch'] + 1)
        async with self.lock:
            await self._release_pointer_locked()
            self.wait_human('USER_TAKEOVER', '已暂停，用户可以操作浏览器')

    async def resume(self):
        if self.state['status'] in TERMINAL or self.state['control_mode'] != 'HUMAN':
            raise ValueError('当前任务不能交还自动化')
        async with self.lock:
            if not self.adapter or not await self.adapter.authenticated():
                raise ValueError('请先在浏览器中完成登录并返回教学平台')
            if self.state.get('waiting_reason') == 'CLASS_SCOPE' and not self.scope_request:
                raise ValueError('请先确认班级范围')
            if self.state.get('waiting_reason') == 'EXPORT_ASSOCIATION' and not self.state.get('exports'):
                # Opening the download center may be all that was needed; observe again first.
                rows = await self.adapter.export_rows()
                if not rows:
                    raise ValueError('请先打开下载中心并确认本次导出记录')
            await self._release_pointer_locked()
            self.update(control_mode='AUTO', status='RUNNING', waiting_reason='', auth_status='AUTHENTICATED',
                        stage='LOCATE_COURSE' if self.state['stage'] == 'AUTHENTICATE' else self.state['stage'],
                        control_epoch=self.state['control_epoch'] + 1, auto_actions=0)
            self.event('正在重新核验页面并继续自动化')
            self.gate.set()

    async def cancel(self, status='CANCELLED'):
        if self.state['status'] in TERMINAL and self.state['status'] != 'INTERRUPTED':
            return
        self.gate.clear()
        self.update(status=status, control_epoch=self.state['control_epoch'] + 1)
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
                page, context, epoch = self.page, self.context, self.state['control_epoch']
                if not page or page.is_closed():
                    raise ValueError('浏览器当前不可用')
                url = page.url
            data = await page.screenshot(type='jpeg', quality=55, timeout=8000)
            pages = [{'id': str(i), 'title': (await p.title())[:100]} for i, p in enumerate(context.pages) if not p.is_closed()]
            async with self.lock:
                if page != self.page or page.is_closed() or url != page.url or epoch != self.state['control_epoch']:
                    raise ValueError('页面或控制权已变化，请刷新画面')
                return data, self.record_frame(page, url, epoch), pages

    def record_frame(self, page, url, epoch):
        """Called with the action lock held, for HTTP captures and pushed frames."""
        self.frame_revision += 1
        self.frame_meta = {'revision': self.frame_revision, 'page_id': str(self.context.pages.index(page)),
                           'url': url, 'epoch': epoch}
        self.frame_history[self.frame_revision] = {**self.frame_meta, 'captured_at': time.monotonic()}
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
            raise ValueError('操作参数无效')
        actions = payload.get('actions') if payload.get('kind') == 'batch' else [payload]
        if (not isinstance(actions, list) or not 1 <= len(actions) <= 32
                or any(not isinstance(action, dict) or action.get('kind') == 'batch' for action in actions)):
            raise ValueError('批量操作参数无效')
        async with self.lock:
            for action in actions:
                await self._browser_input_locked(action, owner)

    async def _browser_input_locked(self, payload, owner=None):
        if self.state['status'] in TERMINAL or self.state['control_mode'] != 'HUMAN' or not self.context:
            raise ValueError('请先暂停并接管浏览器')
        if int(payload.get('epoch', -1)) != self.state['control_epoch']:
            raise ValueError('浏览器控制权已经变化，请刷新画面')
        kind = payload.get('kind')
        if kind == 'switch_page':
            index = int(payload['page_id'])
            if index < 0 or index >= len(self.context.pages):
                raise ValueError('页面不存在')
            if self.context.pages[index].is_closed():
                raise ValueError('页面已关闭')
            await self._release_pointer_locked()
            self.page = self.context.pages[index]
            self.frame_meta = None
            return
        meta = self.frame_meta
        if (not meta or str(payload.get('page_id')) != meta['page_id'] or meta['url'] != self.page.url
                or meta['epoch'] != self.state['control_epoch']):
            raise ValueError('页面已变化，请刷新画面')
        if not allowed_url(self.page.url):
            raise ValueError('页面域名不在允许范围内')
        if self.pointer_page and self.pointer_owner is not owner:
            raise ValueError('另一连接正在拖动，请稍后重试')
        if not self.state.get('export_attempted') and (kind == 'pointer_up' or
                (kind == 'key' and payload.get('key') == 'Enter')):
            area = await self.adapter.export_area()
            if area:
                options = await self.adapter.classes(area[2])
                selected = [o for o in options if await self.adapter.checked(self.adapter.class_locators[o['id']])]
                rows = await self.adapter.export_rows()
                self.manual_export_evidence = {'baseline':[r['ref'] for r in rows],
                    'scope':{'mode':'all' if any(o['all'] for o in selected) else 'selected' if selected else 'manual',
                             'classes':selected}}
        x, y = float(payload.get('x', 0)), float(payload.get('y', 0))
        if not 0 <= x <= 1440 or not 0 <= y <= 1000:
            raise ValueError('坐标越界')
        if kind == 'pointer_down':
            displayed = self.frame_history.get(int(payload.get('revision', -1)))
            if (not displayed or time.monotonic() - displayed['captured_at'] > 3
                    or any(displayed[key] != meta[key] for key in ('page_id', 'url', 'epoch'))):
                raise ValueError('画面已刷新，请重试')
            await self.page.mouse.move(x, y)
            self.pointer_page, self.pointer_owner = self.page, owner
            await self.page.mouse.down()
        elif kind == 'pointer_move':
            await self.page.mouse.move(x, y)
        elif kind == 'pointer_up':
            await self.page.mouse.move(x, y)
            await self.page.mouse.up()
            self.pointer_page = self.pointer_owner = None
        elif kind == 'pointer_cancel':
            await self._release_pointer_locked(owner)
        elif kind == 'text':
            await self.page.keyboard.insert_text(str(payload.get('text', ''))[:2000])
        elif kind == 'key':
            key = str(payload.get('key', ''))
            if key not in {'Enter', 'Tab', 'Backspace', 'Delete', 'Escape', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Control+A'}:
                raise ValueError('不支持的按键')
            await self.page.keyboard.press(key)
        elif kind == 'scroll':
            await self.page.mouse.wheel(0, max(-1000, min(1000, int(payload.get('delta_y', 0)))))
        else:
            raise ValueError('不支持的操作')
        self.last_activity = time.monotonic()
