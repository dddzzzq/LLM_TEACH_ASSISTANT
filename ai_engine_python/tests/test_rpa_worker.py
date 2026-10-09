"""Behavioral tests use real Chromium against intercepted portal pages, never real accounts."""
import asyncio
import contextlib
import json
import socket
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
import uvicorn
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus
from playwright.async_api import async_playwright

from app.rpa.download_manager import safe_filename, verify_zip, save_download
from app.rpa.portal_adapter import fingerprint, match_new_exports, PortalAdapter
from app.rpa.task_controller import Task, allowed_url
from app.rpa.worker import Manager, app


class FileAndEvidenceTests(unittest.TestCase):
    def test_completed_export_keeps_identity_when_download_link_appears(self):
        pending = {'text': '第一次作业 班级A 2026-10-07 10:00 导出中'}
        ready = {'text': '第一次作业 班级A 2026-10-07 10:00 导出成功 下载', 'href': '/signed-download?token=private'}
        self.assertEqual(fingerprint(pending), fingerprint(ready))

    def test_never_associate_foreign_or_ambiguous_exports(self):
        rows = [{'ref':'old','text':'第一次作业 A'}, {'ref':'new','text':'其他作业 A'},
                {'ref':'ambiguous','text':'第一次作业 A','ambiguous':True}]
        scope = {'mode':'selected', 'classes':[{'name':'A'}]}
        self.assertEqual(match_new_exports(rows, ['old'], '第一次作业', scope), [])
        rows.append({'ref':'ours','text':'第一次作业 A','status':'READY'})
        self.assertEqual(match_new_exports(rows, ['old'], '第一次作业', scope)[0]['ref'], 'ours')

    def test_download_rejects_error_page_empty_and_corrupt_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'homework.zip'
            path.write_text('<html>login</html>')
            with self.assertRaises(zipfile.BadZipFile): verify_zip(path)
            with zipfile.ZipFile(path, 'w'): pass
            with self.assertRaises(ValueError): verify_zip(path)
            with zipfile.ZipFile(path, 'w') as archive: archive.writestr('20260001-test/report.txt', 'hello')
            result = verify_zip(path)
            self.assertEqual(result['status'], 'VERIFIED')
            self.assertEqual(len(result['sha256']), 64)
        self.assertNotIn('/', safe_filename('../../escape.zip'))

    def test_navigation_domain_boundary(self):
        self.assertTrue(allowed_url('https://ids.xidian.edu.cn/authserver/login'))
        self.assertTrue(allowed_url('https://mooc1.chaoxing.com/course'))
        self.assertFalse(allowed_url('https://xidian.edu.cn.attacker.example/login'))
        self.assertFalse(allowed_url('file:///etc/passwd'))


class BrowserTaskTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = Manager(Path(self.temp.name) / 'state')
        self.manager.proxy = None
        self.manager.download_root = Path(self.temp.name) / 'downloads'
        self.pw = await async_playwright().start()
        self.manager.playwright = self.pw
        self.tasks = []
        self.drivers = []

    async def asyncTearDown(self):
        for driver in self.drivers:
            driver.cancel()
        await asyncio.gather(*self.drivers, return_exceptions=True)
        for task in self.tasks:
            if task.runner and not task.runner.done(): await task.cancel()
            elif task.browser: await task.browser.close()
        await self.pw.stop()
        self.temp.cleanup()

    async def bare_task(self):
        state = {'job_id':str(uuid4()), 'target':{'course_name':'操作系统','assignment_name':'第一次作业'},
                 'status':'RUNNING','stage':'LOCATE_COURSE','control_mode':'AUTO','control_epoch':1, 'files':[]}
        task = Task(self.manager, state)
        task.browser = await self.pw.chromium.launch(headless=True, args=['--no-sandbox'])
        task.context = await task.browser.new_context(viewport={'width':1440,'height':1000})
        task.context.set_default_timeout(5000)
        task.page = await task.context.new_page()
        task.adapter = PortalAdapter(task.context, state['target'])
        self.tasks.append(task)
        return task

    async def test_pause_waits_for_action_boundary_and_invalidates_old_decision(self):
        task = await self.bare_task()
        await task.lock.acquire()
        pause = asyncio.create_task(task.pause())
        await asyncio.sleep(.02)
        self.assertEqual(task.state['control_mode'], 'PAUSING')
        self.assertFalse(task.gate.is_set())
        self.assertGreater(task.state['control_epoch'], 1)
        self.assertFalse(pause.done())
        task.lock.release()
        await pause
        self.assertEqual(task.state['control_mode'], 'HUMAN')
        with self.assertRaises(ValueError): await task.browser_input({'kind':'text','text':'private','epoch':1})

    async def test_resume_does_not_accept_unfinished_login(self):
        task = await self.bare_task()
        await task.page.route('**/*', lambda route: route.fulfill(content_type='text/html; charset=utf-8', body='<input type=password>'))
        await task.page.goto('https://ids.xidian.edu.cn/authserver/login')
        task.wait_human('LOGIN','请登录')
        with self.assertRaises(ValueError): await task.resume()
        self.assertEqual(task.state['control_mode'], 'HUMAN')

    async def test_login_status_refresh_does_not_resume_or_disclose_identity(self):
        task = await self.bare_task()
        await task.page.route('**/*', lambda route: route.fulfill(content_type='text/html; charset=utf-8',
            body='<a class=denglu>登录</a><a>个人空间</a>'))
        await task.page.goto('https://learning.xidian.edu.cn/portal')
        task.update(stage='AUTHENTICATE')
        task.wait_human('LOGIN', '请登录')
        epoch = task.state['control_epoch']
        await task.refresh_login_state()
        self.assertNotEqual(task.state['auth_status'], 'AUTHENTICATED')
        await task.page.set_content('<a class=denglu hidden>登录</a><span>fixture-private-user</span><a>个人空间</a>')
        await task.refresh_login_state()
        self.assertEqual(task.state['auth_status'], 'AUTHENTICATED')
        self.assertEqual(task.state['waiting_reason'], 'LOGIN_COMPLETE')
        self.assertEqual(task.state['control_mode'], 'HUMAN')
        self.assertEqual(task.state['control_epoch'], epoch)
        self.assertFalse(task.gate.is_set())
        version = task.state['state_version']
        await task.refresh_login_state()
        self.assertEqual(task.state['state_version'], version)
        saved = (self.manager.root / (task.state['job_id'] + '.json')).read_text()
        self.assertNotIn('fixture-private-user', saved)
        await task.resume()
        self.assertEqual(task.state['stage'], 'LOCATE_COURSE')
        self.assertEqual(task.state['control_mode'], 'AUTO')

    async def test_login_status_refresh_detects_expired_login(self):
        task = await self.bare_task()
        await task.page.route('**/*', lambda route: route.fulfill(content_type='text/html; charset=utf-8', body='<a>个人空间</a>'))
        await task.page.goto('https://learning.xidian.edu.cn/portal')
        task.wait_human('LOGIN', '请登录')
        await task.refresh_login_state()
        self.assertEqual(task.state['waiting_reason'], 'LOGIN_COMPLETE')
        await task.page.set_content('<input type=password><input id=captcha>')
        await task.refresh_login_state()
        self.assertEqual(task.state['auth_status'], 'REQUIRED')
        self.assertEqual(task.state['waiting_reason'], 'LOGIN')
        with self.assertRaises(ValueError): await task.resume()

    async def test_input_only_in_human_mode_and_never_persisted(self):
        task = await self.bare_task()
        await task.page.route('**/*', lambda route: route.fulfill(content_type='text/html; charset=utf-8', body='<input id="password" type=password>'))
        await task.page.goto('https://ids.xidian.edu.cn/authserver/login')
        task.wait_human('LOGIN','请登录')
        _, meta, _ = await task.frame()
        await task.page.locator('input').focus()
        await task.browser_input({'kind':'text','text':'test-secret-only','epoch':meta['epoch'], 'page_id':meta['page_id']})
        self.assertEqual(await task.page.locator('input').input_value(), 'test-secret-only')
        saved = (self.manager.root / (task.state['job_id'] + '.json')).read_text()
        self.assertNotIn('test-secret-only', saved)
        task.update(control_mode='AUTO')
        with self.assertRaises(ValueError): await task.browser_input({'kind':'text','text':'x'})

    async def test_scope_is_exact_and_no_default_all_classes(self):
        task = await self.bare_task()
        await task.page.set_content('''<div role=dialog><label class=export-range><input type=checkbox>班级A</label>
          <label class=export-range><input type=checkbox>班级B</label>
          <label class="export-range all"><input type=checkbox>所有班级</label></div>''')
        modal = task.page.locator('[role=dialog]')
        options = await task.adapter.classes(modal)
        self.assertEqual(len(options), 3)
        scope = await task.adapter.apply_scope(modal, {'mode':'selected','class_ids':['class_0']}, options)
        self.assertEqual(scope['classes'][0]['name'], '班级A')
        self.assertEqual(await task.page.locator('input:checked').count(), 1)
        with self.assertRaises(ValueError): await task.adapter.apply_scope(modal, {'mode':'selected','class_ids':[]}, options)

    async def human_input_task(self):
        task = await self.bare_task()
        await task.page.route('**/*', lambda route: route.fulfill(content_type='text/html',
            body='<input id=first><input id=second><button onclick="window.hits=(window.hits||0)+1">click</button>'))
        await task.page.goto('https://learning.xidian.edu.cn/portal')
        task.wait_human('LOGIN', 'fixture')
        _, meta, _ = await task.frame()
        return task, {key: meta[key] for key in ('epoch', 'page_id', 'revision')}

    @contextlib.asynccontextmanager
    async def stream_server(self):
        previous_manager = getattr(app.state, 'manager', None)
        previous_token = getattr(app.state, 'token', None)
        app.state.manager, app.state.token = self.manager, 'stream-fixture-token'
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        config = uvicorn.Config(app, lifespan='off', log_level='error', ws_max_size=16 * 1024)
        server = uvicorn.Server(config)
        runner = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if runner.done(): runner.result()
                    await asyncio.sleep(.01)
            yield f'ws://127.0.0.1:{listener.getsockname()[1]}'
        finally:
            server.should_exit = True
            await asyncio.wait_for(runner, 5)
            listener.close()
            app.state.manager, app.state.token = previous_manager, previous_token

    async def stream_message(self, stream, kind, predicate=lambda value: True):
        async with asyncio.timeout(5):
            while True:
                message = json.loads(await stream.recv())
                if message['type'] == kind and predicate(message):
                    return message

    async def test_stream_drag_pushes_frames_before_release_and_disconnect_releases_mouse(self):
        task, _ = await self.human_input_task()
        self.manager.tasks[task.state['job_id']] = task
        await task.page.set_content('''<div id=slider style="position:absolute;left:10px;top:10px;width:80px;height:50px;background:blue"></div>
          <script>window.moves=[];window.released=false;
          document.addEventListener('mousemove', e=>{if(e.buttons===1){
            moves.push(e.clientX);slider.style.left=e.clientX+'px'}});
          document.addEventListener('mouseup',()=>window.released=true);</script>''')
        async with self.stream_server() as url:
            async with connect(f'{url}/tasks/{task.state["job_id"]}/stream',
                               additional_headers={'X-RPA-Token':'stream-fixture-token'}) as stream:
                frame = await self.stream_message(stream, 'frame')
                meta = {key:frame[key] for key in ('epoch','page_id','revision')}
                async def send(seq, kind, **values):
                    await stream.send(json.dumps({'type':'input','id':seq,'action':{**meta,'kind':kind,**values}}))
                await send(1, 'pointer_down', x=30, y=30)
                # Pipeline movement without waiting for the down acknowledgement.
                for seq, x in enumerate((80, 130, 180, 230), 2):
                    await send(seq, 'pointer_move', x=x, y=30)
                await self.stream_message(stream, 'input_ack', lambda m: m['id'] == 5)
                pushed = await self.stream_message(stream, 'frame', lambda m: m['revision'] > frame['revision'])
                self.assertNotEqual(pushed['image'], frame['image'])
                self.assertEqual(await task.page.evaluate('window.moves.at(-1)'), 230)
                self.assertFalse(await task.page.evaluate('window.released'))
            async with asyncio.timeout(3):
                while not await task.page.evaluate('window.released'): await asyncio.sleep(.02)
            self.assertIsNone(task.pointer_page)
        journal = (self.manager.root / (task.state['job_id'] + '.json')).read_text()
        self.assertNotIn('pointer_move', journal)

    async def test_stream_disconnect_or_pointer_cancel_does_not_click_button(self):
        task, _ = await self.human_input_task()
        await task.page.evaluate('window.hits=0')
        self.manager.tasks[task.state['job_id']] = task
        box = await task.page.locator('button').bounding_box()
        async with self.stream_server() as url:
            async with connect(f'{url}/tasks/{task.state["job_id"]}/stream',
                               additional_headers={'X-RPA-Token':'stream-fixture-token'}) as stream:
                frame = await self.stream_message(stream, 'frame')
                meta = {key:frame[key] for key in ('epoch','page_id','revision')}
                down = {**meta,'kind':'pointer_down','x':box['x']+box['width']/2,'y':box['y']+box['height']/2}
                await stream.send(json.dumps({'type':'input','id':1,'action':down}))
                await self.stream_message(stream, 'input_ack')
                await stream.send(json.dumps({'type':'input','id':2,'action':{**meta,'kind':'pointer_cancel'}}))
                await self.stream_message(stream, 'input_ack', lambda m: m['id']==2)
                self.assertEqual(await task.page.evaluate('window.hits'), 0)
                await stream.send(json.dumps({'type':'input','id':3,'action':down}))
                await self.stream_message(stream, 'input_ack', lambda m: m['id']==3)
            async with asyncio.timeout(3):
                while task.pointer_page or task.browser_frames.runner is not None: await asyncio.sleep(.02)
            self.assertEqual(await task.page.evaluate('window.hits'), 0)

    async def test_stream_auth_control_epoch_and_shared_viewers(self):
        task, _ = await self.human_input_task()
        self.manager.tasks[task.state['job_id']] = task
        async with self.stream_server() as url:
            endpoint = f'{url}/tasks/{task.state["job_id"]}/stream'
            with self.assertRaises(InvalidStatus) as rejected:
                async with connect(endpoint): pass
            self.assertEqual(rejected.exception.response.status_code, 403)
            headers = {'X-RPA-Token':'stream-fixture-token'}
            async with connect(endpoint, additional_headers=headers) as first:
                old = await self.stream_message(first, 'frame')
                producer = task.browser_frames.runner
                async with connect(endpoint, additional_headers=headers) as second:
                    await self.stream_message(second, 'frame')
                    self.assertIs(producer, task.browser_frames.runner)
                    await first.send(json.dumps({'type':'input','id':1,'action':{
                        'kind':'pointer_down','epoch':old['epoch'],'revision':old['revision'],
                        'page_id':old['page_id'],'x':20,'y':20}}))
                    await self.stream_message(first, 'input_ack')
                # Closing another viewer must not release this connection's pointer.
                self.assertIsNotNone(task.pointer_page)
                task.update(control_epoch=task.state['control_epoch']+1)
                await first.send(json.dumps({'type':'input','id':2,'action':{
                    'kind':'text','text':'never-enter-this-secret','epoch':old['epoch'],'page_id':old['page_id']}}))
                await self.stream_message(first, 'error')
            self.assertEqual(await task.page.locator('#first').input_value(), '')
        self.assertIsNone(task.browser_frames.runner)

    async def test_stream_page_switch_and_idle_frame_remain_clickable(self):
        task, _ = await self.human_input_task()
        self.manager.tasks[task.state['job_id']] = task
        second = await task.context.new_page()
        await second.route('**/*', lambda route: route.fulfill(body='<input>'))
        await second.goto('https://ids.xidian.edu.cn/authserver/login')
        async with self.stream_server() as url:
            async with connect(f'{url}/tasks/{task.state["job_id"]}/stream',
                               additional_headers={'X-RPA-Token':'stream-fixture-token'}) as stream:
                frame = await self.stream_message(stream, 'frame')
                await stream.send(json.dumps({'type':'input','id':1,'action':{
                    'kind':'switch_page','epoch':frame['epoch'],'page_id':'1'}}))
                switched = await self.stream_message(stream, 'frame', lambda m: m['page_id'] == '1')
                fresh = await self.stream_message(stream, 'frame', lambda m: m['revision'] > switched['revision'])
                self.assertLess(time.monotonic() - task.frame_history[fresh['revision']]['captured_at'], 3)
                self.assertIs(task.page, second)
                await stream.send(json.dumps({'type':'ping'}))
                await self.stream_message(stream, 'pong')

    async def test_batch_input_preserves_field_and_keyboard_order_without_persisting_values(self):
        task, meta = await self.human_input_task()
        await task.page.locator('#first').focus()
        await task.browser_input({'kind': 'batch', 'actions': [
            {**meta, 'kind': 'text', 'text': 'private-first'},
            {**meta, 'kind': 'key', 'key': 'Tab'},
            {**meta, 'kind': 'text', 'text': 'private-second'},
        ]})
        self.assertEqual(await task.page.locator('#first').input_value(), 'private-first')
        self.assertEqual(await task.page.locator('#second').input_value(), 'private-second')
        saved = (self.manager.root / (task.state['job_id'] + '.json')).read_text()
        self.assertNotIn('private-first', saved)
        self.assertNotIn('private-second', saved)
        with self.assertRaises(ValueError):
            await task.browser_input({'kind': 'batch', 'actions': [{**meta, 'kind': 'text'}] * 33})
        await task.pause()
        with self.assertRaises(ValueError):
            await task.browser_input({'kind': 'batch', 'actions': [{**meta, 'kind': 'text', 'text': 'old'}]})

    async def test_slow_screenshot_does_not_block_input(self):
        task, meta = await self.human_input_task()
        await task.page.locator('#first').focus()
        started, release = asyncio.Event(), asyncio.Event()
        screenshot = task.page.screenshot
        async def delayed_capture(**kwargs):
            started.set()
            await release.wait()
            return await screenshot(**kwargs)
        with patch.object(task.page, 'screenshot', delayed_capture):
            capture = asyncio.create_task(task.frame())
            try:
                await asyncio.wait_for(started.wait(), 2)
                await asyncio.wait_for(task.browser_input({**meta, 'kind': 'text', 'text': 'responsive'}), 1)
                self.assertEqual(await task.page.locator('#first').input_value(), 'responsive')
                self.assertFalse(capture.done())
            finally:
                release.set()
                await capture

    async def test_click_on_displayed_frame_survives_next_frame_but_rejects_expired_frame(self):
        task, displayed = await self.human_input_task()
        box = await task.page.locator('button').bounding_box()
        event = {**displayed, 'x': box['x'] + box['width'] / 2, 'y': box['y'] + box['height'] / 2}
        await task.frame()
        await task.browser_input({'kind': 'batch', 'actions': [
            {**event, 'kind': 'pointer_down'}, {**event, 'kind': 'pointer_up'}]})
        self.assertEqual(await task.page.evaluate('window.hits'), 1)
        task.frame_history[displayed['revision']]['captured_at'] -= 4
        with self.assertRaises(ValueError):
            await task.browser_input({**event, 'kind': 'pointer_down'})

    async def test_capture_cannot_publish_frame_after_control_changes(self):
        task, _ = await self.human_input_task()
        screenshot = task.page.screenshot
        async def change_control(**kwargs):
            data = await screenshot(**kwargs)
            await task.pause()
            return data
        with patch.object(task.page, 'screenshot', change_control):
            with self.assertRaises(ValueError): await task.frame()

    async def test_observation_omits_form_values_and_hidden_secrets(self):
        task = await self.bare_task()
        await task.page.set_content('<input type=password value=secret><input type=hidden value=token><input placeholder=搜索 value=private><button>作业</button>')
        observation = json.dumps(await task.adapter.observe())
        for private in ('secret','token','private'): self.assertNotIn(private, observation)

    async def test_restart_retains_files_and_marks_lost_browser_interrupted(self):
        task = await self.bare_task()
        task.update(files=[{'export_ref':'r1','status':'VERIFIED'}])
        restarted = Manager(self.manager.root)
        restored = restarted.tasks[task.state['job_id']]
        self.assertEqual(restored.state['status'], 'INTERRUPTED')
        self.assertEqual(restored.state['files'][0]['export_ref'], 'r1')

    async def test_agent_navigation_login_scope_export_and_real_download(self):
        await self.fixture_flow()

    async def test_manual_export_is_detected_and_never_submitted_twice(self):
        await self.fixture_flow(manual=True)

    async def observed_task(self):
        task = await self.bare_task()
        await task.page.route('**/*', lambda route: route.fulfill(content_type='text/html; charset=utf-8',
            body='<button onclick="window.hits=(window.hits||0)+1">个人空间</button>'))
        await task.page.goto('https://learning.xidian.edu.cn/portal')
        task.gate.set()
        obs = await task.observe()
        self.assertTrue(obs['ready'])
        payload = {'action_id': obs['observation_id'], 'observation_id': obs['observation_id'],
                   'epoch': obs['epoch'], 'revision': obs['revision'],
                   'action': {'tool': 'click', 'element_ref': obs['observation']['elements'][0]['ref']}}
        return task, payload

    async def test_action_retry_returns_receipt_without_clicking_twice(self):
        task, payload = await self.observed_task()
        first = await task.apply_action(payload)
        second = await task.apply_action(payload)
        self.assertEqual(first['outcome'], 'completed')
        self.assertEqual(second['outcome'], 'completed')
        self.assertEqual(await task.page.evaluate('window.hits'), 1)
        restored = Manager(self.manager.root).tasks[task.state['job_id']]
        self.assertEqual((await restored.apply_action(payload))['outcome'], 'completed')
        changed = {**payload, 'action': {'tool': 'request_human', 'reason': 'different'}}
        with self.assertRaises(ValueError): await task.apply_action(changed)

    async def test_observation_rejects_changed_dom_and_old_control_epoch(self):
        task, payload = await self.observed_task()
        await task.page.locator('button').evaluate('(e)=>e.innerText="另一目标"')
        with self.assertRaises(ValueError): await task.apply_action(payload)
        self.assertIsNone(await task.page.evaluate('window.hits'))
        task, payload = await self.observed_task()
        await task.pause()
        with self.assertRaises(ValueError): await task.apply_action(payload)
        self.assertIsNone(await task.page.evaluate('window.hits'))

    async def test_uncertain_action_is_persisted_and_never_replayed(self):
        task, payload = await self.observed_task()
        from unittest.mock import AsyncMock
        with patch.object(task, 'execute', AsyncMock(side_effect=TimeoutError('uncertain'))) as execute:
            self.assertEqual((await task.apply_action(payload))['outcome'], 'unknown')
            self.assertEqual((await task.apply_action(payload))['outcome'], 'unknown')
            self.assertEqual(execute.await_count, 1)
        self.assertEqual(task.state['control_mode'], 'HUMAN')
        journal = (self.manager.root / (task.state['job_id'] + '.json')).read_text()
        self.assertNotIn('uncertain', journal)

    async def test_stalled_transfer_is_cancelled(self):
        class StalledDownload:
            suggested_filename = 'work.zip'
            cancelled = False
            async def save_as(self, path): await asyncio.sleep(30)
            async def cancel(self): self.cancelled = True
        download = StalledDownload()
        with self.assertRaises(TimeoutError):
            await save_download(download, self.temp.name, 'stalled', progress=lambda: 0, stall_seconds=.05)
        self.assertTrue(download.cancelled)

    async def fixture_flow(self, manual=False):
        export_count = 0
        zip_path = Path(self.temp.name) / 'fixture.zip'
        with zipfile.ZipFile(zip_path, 'w') as archive: archive.writestr('20260001-test/report.txt', 'student answer')
        course_html = '''<h1 class=courseName>操作系统</h1><ul><li id=work><h2>第一次作业</h2>
          <a onclick="document.querySelector('#work').hidden=true;document.querySelector('#grade').hidden=false">批阅</a></li></ul>
          <div id=grade hidden><h2>第一次作业</h2><div><ul class=morePop><a onclick="document.querySelector('[role=dialog]').hidden=false">导出作业附件</a></ul></div></div>
          <div role=dialog hidden>导出设置<label class=export-range><input type=checkbox>班级A</label>
          <label class=export-range><input type=checkbox>班级B</label><label class="export-range all"><input type=checkbox>所有班级</label>
          <a class=confirmDown onclick="fetch('/export');this.closest('[role=dialog]').hidden=true;document.querySelector('#downloadcenter').innerHTML='<table><tbody><tr data-id=task-a><td>第一次作业 班级A 导出成功 <a class=download_ic href=/download>下载</a></td></tr></tbody></table>'">确定</a></div>
          <div id=downloadcenter></div>'''
        async def fixture_route(task, route):
            nonlocal export_count
            url = route.request.url
            if '/download' in url:
                await route.fulfill(body=zip_path.read_bytes(), content_type='application/zip', headers={'Content-Disposition':'attachment; filename="homework.zip"'})
            elif '/export' in url:
                export_count += 1; await route.fulfill(body='ok')
            elif '/authserver' in url:
                await route.fulfill(content_type='text/html; charset=utf-8', body='<input id=username><input type=password><button onclick="location.href=\'https://learning.xidian.edu.cn/portal?ok=1\'">登录</button>')
            elif '/course' in url:
                await route.fulfill(content_type='text/html; charset=utf-8', body=course_html)
            elif '?ok=1' in url:
                await route.fulfill(content_type='text/html; charset=utf-8', body='<a>个人空间</a><div cname="操作系统"><a target=_blank href="https://learning.xidian.edu.cn/course">操作系统</a></div>')
            else:
                await route.fulfill(content_type='text/html; charset=utf-8', body='<a class=denglu href="https://ids.xidian.edu.cn/authserver/login">登录</a>')
        class FixtureAgent:
            async def decide(self, target, stage, observation, recent):
                if observation['can_open_export']: return {'tool':'open_export_settings'}
                wanted = '操作系统' if stage == 'LOCATE_COURSE' else '批阅'
                for element in observation['elements']:
                    if element['name'] == wanted: return {'tool':'click','element_ref':element['ref']}
                return {'tool':'request_human','reason':'fixture target missing'}
        with patch.object(Task, 'route', fixture_route):
            task = self.manager.start(str(uuid4()), {'course_name':'操作系统','assignment_name':'第一次作业'})
            self.tasks.append(task)
            # Exercise the authenticated Worker protocol from an external decision driver.
            app.state.manager = self.manager
            app.state.token = 'fixture-driver-token'
            async def external_driver():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://worker',
                        headers={'X-RPA-Token': app.state.token}) as client:
                    while task.state['status'] not in ('SUCCEEDED', 'FAILED', 'PARTIAL_SUCCESS', 'CANCELLED'):
                        response = await client.get('/tasks/' + task.state['job_id'] + '/observation')
                        response.raise_for_status()
                        obs = response.json()
                        if obs['ready']:
                            action = await FixtureAgent().decide(obs['target'], obs['stage'], obs['observation'], obs['recent_actions'])
                            response = await client.post('/tasks/' + task.state['job_id'] + '/action', json={
                                'action_id': obs['observation_id'], 'observation_id': obs['observation_id'],
                                'revision': obs['revision'], 'epoch': obs['epoch'], 'action': action})
                            response.raise_for_status()
                        await asyncio.sleep(.05)
            driver = asyncio.create_task(external_driver())
            self.drivers.append(driver)
            async def wait_for(predicate, timeout=15):
                async with asyncio.timeout(timeout):
                    while not predicate():
                        if driver.done(): driver.result()
                        await asyncio.sleep(.05)
            await wait_for(lambda: task.state.get('waiting_reason') == 'LOGIN')
            await task.page.locator('button').click()
            await task.page.wait_for_url('**/portal?ok=1')
            await task.resume()
            await wait_for(lambda: task.state.get('waiting_reason') == 'CLASS_SCOPE')
            options = task.state['classes']
            if manual:
                await task.page.locator('[role=dialog] input').first.check()
                _, meta, _ = await task.frame()
                box = await task.page.locator('a.confirmDown').bounding_box()
                payload = {'epoch':meta['epoch'],'page_id':meta['page_id'],'revision':meta['revision'],
                           'x':box['x']+box['width']/2,'y':box['y']+box['height']/2}
                await task.browser_input({**payload,'kind':'pointer_down'})
                await task.browser_input({**payload,'kind':'pointer_up'})
                await wait_for(lambda: task.state.get('export_attempted'))
            else:
                task.scope_request = {'mode':'selected','class_ids':['class_0'],'option_names':{o['id']:o['name'] for o in options}}
            await task.resume()
            await wait_for(lambda: task.state['status'] in ('SUCCEEDED','FAILED','PARTIAL_SUCCESS'))
            self.assertEqual(task.state['status'], 'SUCCEEDED', task.state)
            self.assertEqual(export_count, 1)
            self.assertEqual(task.state['files'][0]['status'], 'VERIFIED')
            self.assertEqual(task.state['files'][0]['class_name'], '班级A')
            self.assertTrue(Path(task.state['files'][0]['path']).exists())
            await task.runner


class ControlAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_api_rejects_missing_token(self):
        app.state.token = 'test-internal-token'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://worker') as client:
            response = await client.get('/health')
            self.assertEqual(response.status_code, 401)
            self.assertNotIn('test-internal-token', response.text)
            response = await client.get('/health', headers={'X-RPA-Token':'test-internal-token'})
            self.assertEqual(response.status_code, 200)


if __name__ == '__main__': unittest.main()
