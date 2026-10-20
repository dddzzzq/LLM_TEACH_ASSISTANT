"""Behavioral tests use real Chromium against intercepted portal pages, never real accounts."""
import asyncio
import contextlib
import hashlib
import json
import re
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
from app.rpa.portal_adapter import fingerprint, match_new_exports, reconcile_exports, PortalAdapter
from app.rpa.task_controller import Task, allowed_url
from app.rpa.worker import Manager, app
from tests.fixtures.export_dialog import PORTAL_EXPORT_OPTIONS


class FileAndEvidenceTests(unittest.TestCase):
    def test_completed_export_keeps_identity_when_download_link_appears(self):
        pending = {'text': '第一次作业 班级A 2026-10-07 10:00 导出中'}
        ready = {'text': '第一次作业 班级A 2026-10-07 10:00 导出成功 下载', 'href': '/signed-download?token=private'}
        self.assertEqual(fingerprint(pending), fingerprint(ready))

    def test_export_completion_time_does_not_change_identity(self):
        pending = {'text': '班级2CS20510501-第一次作业(附件)\n\n10-15 17:55 导出中'}
        ready = {'text': '班级2CS20510501-第一次作业(附件)\n\n10-15 17:59 导出成功'}
        self.assertEqual(fingerprint(pending), fingerprint(ready))
        ready['text'] = ready['text'].replace('10-15 17:59', '2026-10-15 18:01:02')
        self.assertEqual(fingerprint(pending), fingerprint(ready))
        self.assertNotEqual(fingerprint(pending), fingerprint({
            'text': ready['text'].replace('2CS20510501', '2CS20510502')}))
        self.assertEqual(fingerprint({'platform_id': 'record-a', **pending}),
                         fingerprint({'platform_id': 'record-a', **ready}))

    def test_legacy_export_recovery_preserves_reference_and_rejects_uncertainty(self):
        bound = {'ref': 'legacy-bound-ref', 'text': '班级A-第一次作业(附件) 10-15 17:55 导出中'}
        ready = {'ref': 'current', 'text': '班级A-第一次作业(附件) 10-15 17:59 导出成功',
                 'status': 'READY'}
        resolved = reconcile_exports([bound], [ready], [])[0]
        self.assertEqual(resolved['ref'], bound['ref'])
        self.assertEqual(resolved['row_ref'], ready['ref'])
        self.assertEqual(resolved['status'], 'READY')
        for rows, baseline in [([ready], ['current']),
                               ([{**ready, 'legacy_ref': 'old-history'}], ['old-history']),
                               ([{**ready, 'ambiguous': True}], []),
                               ([ready, {**ready, 'ref': 'another'}], [])]:
            self.assertFalse(reconcile_exports([bound], rows, baseline)[0]['available'])

    def test_recovery_does_not_bind_two_exports_to_one_record(self):
        exports = [{'ref': ref, 'text': '班级A-第一次作业 10-15 17:55 导出中'} for ref in ('a', 'b')]
        rows = [{'ref': 'new', 'text': '班级A-第一次作业 10-15 17:59 导出成功', 'status': 'READY'}]
        resolved = reconcile_exports(exports, rows, [])
        self.assertFalse(any(e['available'] for e in resolved))

    def test_never_associate_foreign_or_ambiguous_exports(self):
        rows = [{'ref':'old','text':'第一次作业 A'}, {'ref':'new','text':'其他作业 A'},
                {'ref':'ambiguous','text':'第一次作业 A','ambiguous':True}]
        scope = {'mode':'selected', 'classes':[{'name':'A'}]}
        self.assertEqual(match_new_exports(rows, ['old'], '第一次作业', scope), [])
        rows.append({'ref':'ours','text':'第一次作业 A','status':'READY'})
        self.assertEqual(match_new_exports(rows, ['old'], '第一次作业', scope)[0]['ref'], 'ours')

    def test_all_class_export_associates_only_one_unique_new_matching_record(self):
        scope = {'mode':'all','baseline_observed':True}
        rows = [{'ref':'old','text':'第一次作业'},{'ref':'foreign','text':'第二次作业'},
                {'ref':'ours','text':'第一次作业 全部班级 导出中'}]
        self.assertEqual(match_new_exports(rows, ['old'], '第一次作业', scope), [rows[-1]])
        self.assertEqual(match_new_exports(rows, ['old'], '第一次作业', {'mode':'all'}), [])
        rows.append({'ref':'other-new','text':'第一次作业 导出中'})
        self.assertEqual(match_new_exports(rows, ['old'], '第一次作业', scope), [])
        self.assertEqual(match_new_exports([{'ref':'suffix','text':'第一次作业补交 导出中'}],
                                          [], '第一次作业', scope), [])
        self.assertEqual(match_new_exports([{'ref':'ambiguous','text':'第一次作业 导出中','ambiguous':True}],
                                          [], '第一次作业', scope), [])

    def test_confirmed_class_names_survive_reordering_but_invalid_choices_do_not(self):
        old = {'mode':'selected','class_ids':['class_2'],'option_names':{
            'class_0':'默认格式包含班级名称','class_1':'班级A','class_2':'班级B'}}
        options = [{'id':'class_0','name':'班级A','all':False},
                   {'id':'class_1','name':'班级B','all':False},
                   {'id':'class_2','name':'全部班级','all':True}]
        self.assertEqual(Task.reconcile_scope(old, options)['class_ids'], ['class_1'])
        self.assertIsNone(Task.reconcile_scope({**old,'class_ids':['class_0']}, options))
        self.assertIsNone(Task.reconcile_scope(old, [options[0], options[2]]))
        self.assertIsNone(Task.reconcile_scope({'mode':'all'}, options[:2]))

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

    async def test_scope_supports_custom_parent_state_and_attachment_radio(self):
        task = await self.bare_task()
        await task.page.set_content('''<div role=dialog>选择格式
          <div class="export-range checked"><span class=grade_check onclick="choose(this, '.export-range')"></span>导出本班级作业附件</div>
          <div class=export-range><span class=grade_check onclick="choose(this, '.export-range')"></span>导出作业下所有班级作业附件</div>
          <div class="out checked"><span class=grade_check onclick="choose(this, '.out')"></span>导出完整答题记录</div>
          <div class=out><span class=grade_check onclick="choose(this, '.out')"></span>导出提交附件</div></div>
          <style>.grade_check {display:inline-block;width:20px;height:20px}</style>
          <script>function choose(e, group) {document.querySelectorAll(group).forEach(x=>x.classList.remove('checked'));e.parentElement.classList.add('checked')}</script>''')
        modal = task.page.locator('[role=dialog]')
        options = await task.adapter.classes(modal)
        scope = await task.adapter.apply_scope(modal, {'mode':'all'}, options)
        self.assertEqual(scope['mode'], 'all')
        self.assertTrue(await task.adapter.checked(task.page.locator('.export-range .grade_check').nth(1)))
        self.assertTrue(await task.adapter.checked(task.page.locator('.out .grade_check').nth(1)))
        self.assertFalse(await task.adapter.checked(task.page.locator('.out .grade_check').first))

    async def test_portal_naming_description_is_not_a_class_and_format_is_attachments(self):
        task = await self.bare_task()
        await task.page.set_content('<div role=dialog>选择格式' + PORTAL_EXPORT_OPTIONS + '</div>')
        modal = task.page.locator('[role=dialog]')
        options = await task.adapter.classes(modal)
        self.assertEqual(len(options), 2, options)
        self.assertFalse(any('默认格式' in option['name'] for option in options))
        await task.adapter.apply_scope(modal, {'mode':'all'}, options)
        self.assertTrue(await task.adapter.checked(task.page.locator('.export-range.all .grade_check')))
        self.assertTrue(await task.adapter.checked(task.page.locator('#attachment-option .grade_check')))
        self.assertEqual(await task.page.locator('.out .grade_checked').count(), 1)
        self.assertTrue(await task.adapter.checked(task.page.locator('#default-naming .grade_check')))
        self.assertIsNone(await task.page.evaluate('window.namingClicks'))

    async def test_semantic_export_groups_keep_naming_and_format_separate_from_scope(self):
        task = await self.bare_task()
        await task.page.set_content('''<div role=dialog>
          <fieldset><legend>导出范围</legend><label><input type=radio name=scope checked>本班级</label>
          <label><input type=radio name=scope>全部班级</label></fieldset>
          <fieldset><legend>导出格式</legend><label><input type=radio name=content checked>导出完整答题记录</label>
          <label><input type=radio name=content>导出提交附件</label></fieldset>
          <fieldset><legend>命名格式</legend><label><input type=radio name=naming checked>班级-学号-姓名</label>
          <label><input type=radio name=naming>默认格式，包含班级信息</label></fieldset></div>''')
        modal = task.page.locator('[role=dialog]')
        options = await task.adapter.classes(modal)
        self.assertEqual(len(options), 2, options)
        await task.adapter.apply_scope(modal, {'mode':'all'}, options)
        self.assertTrue(await task.page.locator('input[name=content]').nth(1).is_checked())
        self.assertTrue(await task.page.locator('input[name=naming]').first.is_checked())

    async def test_previously_confirmed_all_scope_survives_removing_spurious_naming_option(self):
        task = await self.bare_task()
        await task.page.set_content('<div role=dialog>选择格式' + PORTAL_EXPORT_OPTIONS +
            """<a class=confirmDown onclick="window.exports=(window.exports||0)+1">确定</a></div>""")
        task.adapter.course_confirmed = task.adapter.assignment_confirmed = True
        task.scope_request = {'mode':'all','class_ids':[],'option_names':{
            'class_0':'导出本班级作业附件',
            'class_1':'导出作业下所有班级作业附件 若班级内没有学生提交，将自动跳过导出',
            'class_2':'默认格式 学生提交的文件将保留文件原名；答题记录按照“院系-专业-班级-学号-姓名”命名'}}
        task.update(requested_scope=task.scope_request)
        task.gate.set()
        await task.prepare_export((task.page, task.page.main_frame, task.page.locator('[role=dialog]')))
        self.assertEqual(task.state['stage'], 'WAIT_EXPORT', task.state)
        self.assertNotEqual(task.state.get('status'), 'WAITING_USER')
        self.assertEqual(task.state['requested_scope']['mode'], 'all')
        self.assertEqual(len(task.state['requested_scope']['option_names']), 2)
        self.assertEqual(await task.page.evaluate('window.exports'), 1)
        self.assertTrue(await task.adapter.checked(task.page.locator('#attachment-option .grade_check')))
        self.assertIsNone(await task.page.evaluate('window.namingClicks'))

    async def test_missing_attachment_choice_does_not_request_class_confirmation_again(self):
        task = await self.bare_task()
        await task.page.set_content('''<div role=dialog>导出设置
          <label class=export-range><input type=checkbox checked>全部班级</label>
          <div class=out><span class="grade_check grade_checked"></span>导出完整答题记录</div>
          <button onclick="window.exports=1">确定</button></div>
          <style>.grade_check{display:inline-block;width:20px;height:20px}</style>''')
        modal = task.page.locator('[role=dialog]')
        options = await task.adapter.classes(modal)
        task.adapter.course_confirmed = task.adapter.assignment_confirmed = True
        task.scope_request = {'mode':'all','class_ids':[],'option_names':{o['id']:o['name'] for o in options}}
        task.gate.set()
        await task.prepare_export((task.page, task.page.main_frame, modal))
        self.assertEqual(task.state['waiting_reason'], 'EXPORT_FORMAT')
        self.assertEqual(task.state['last_error']['code'], 'ATTACHMENT_MISSING')
        self.assertFalse(task.state.get('export_attempted'))
        self.assertIsNone(await task.page.evaluate('window.exports'))

    async def test_scope_supports_hidden_native_radio_and_ignores_format_choices(self):
        task = await self.bare_task()
        await task.page.set_content('''<div role=dialog>
          <label><input hidden type=radio name=scope checked>导出本班级作业附件</label>
          <label><input hidden type=radio name=scope>导出所有班级作业附件</label>
          <label><input type=radio name=format checked>Word标准格式</label>
          <label><input type=radio name=format>PDF格式</label>
          <label><input hidden type=radio name=content checked>导出完整答题记录</label>
          <label><input hidden type=radio name=content>导出提交附件</label></div>''')
        modal = task.page.locator('[role=dialog]')
        options = await task.adapter.classes(modal)
        self.assertEqual(len(options), 2)
        await task.adapter.apply_scope(modal, {'mode':'all'}, options)
        self.assertTrue(await task.page.locator('input[name=scope]').nth(1).is_checked())
        self.assertTrue(await task.page.locator('input[name=content]').nth(1).is_checked())
        self.assertTrue(await task.page.locator('input[name=format]').first.is_checked())

    async def test_scope_waits_for_site_state_and_preserves_confirmed_request(self):
        task = await self.bare_task()
        await task.page.set_content('''<div role=dialog>导出设置
          <div class=export-range><span class=grade_check onclick="setTimeout(()=>this.classList.add('checked'),100)"></span>全部班级</div></div>
          <style>.grade_check {display:inline-block;width:20px;height:20px}</style>''')
        modal = task.page.locator('[role=dialog]')
        options = await task.adapter.classes(modal)
        await task.adapter.apply_scope(modal, {'mode':'all'}, options)
        task.update(requested_scope={'mode':'all','option_names':{o['id']:o['name'] for o in options}})
        restored = Task(self.manager, dict(task.state))
        self.assertEqual(restored.scope_request['mode'], 'all')

    async def test_scope_failure_explains_selection_without_requesting_target_again(self):
        task = await self.bare_task()
        await task.page.set_content('''<div role=dialog>导出设置
          <div class=export-range><span class=grade_check></span>全部班级</div>
          <button onclick="window.exports=(window.exports||0)+1">确定</button></div>
          <style>.grade_check {display:inline-block;width:20px;height:20px}</style>''')
        modal = task.page.locator('[role=dialog]')
        options = await task.adapter.classes(modal)
        task.adapter.course_confirmed = task.adapter.assignment_confirmed = True
        task.scope_request = {'mode':'all','option_names':{o['id']:o['name'] for o in options}}
        task.gate.set()
        await task.prepare_export((task.page, task.page.main_frame, modal))
        self.assertEqual(task.state['waiting_reason'], 'SCOPE_SELECTION')
        self.assertIn('班级选择', task.state['message'])
        self.assertEqual(task.state['last_error']['code'], 'SCOPE_MISMATCH')
        self.assertFalse(task.state.get('export_attempted'))
        self.assertIsNone(await task.page.evaluate('window.exports'))

    async def test_scope_control_api_keeps_confirmation_and_rejects_unrelated_target_confirmation(self):
        task = await self.bare_task()
        task.wait_human('CLASS_SCOPE', '选择班级')
        task.update(classes=[{'id':'class_0','name':'全部班级','all':True}])
        self.manager.tasks[task.state['job_id']] = task
        previous_manager = getattr(app.state, 'manager', None)
        previous_token = getattr(app.state, 'token', None)
        app.state.manager, app.state.token = self.manager, 'scope-fixture-token'
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://worker',
                                        headers={'X-RPA-Token':'scope-fixture-token'}) as client:
                path = '/tasks/' + task.state['job_id']
                response = await client.post(path + '/class-scope', json={'mode':'all','class_ids':[]})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(task.state['requested_scope']['mode'], 'all')
                task.wait_human('RECOVERY', '普通异常')
                response = await client.post(path + '/confirm-target', json=task.state['target'])
                self.assertEqual(response.status_code, 409)
        finally:
            app.state.manager, app.state.token = previous_manager, previous_token

    async def test_restart_reuses_named_or_all_scope_but_not_generic_current_class(self):
        task = await self.bare_task()
        from unittest.mock import AsyncMock
        for mode, name, reusable in [('all','全部班级',True), ('selected','导出本班级作业附件',False),
                                     ('selected','班级A',True)]:
            task.update(status='INTERRUPTED',requested_scope={
                'mode':mode,'class_ids':[] if mode == 'all' else ['class_0'],
                'option_names':{'class_0':name}})
            self.manager.tasks[task.state['job_id']] = task
            with patch.object(Task, 'run', AsyncMock()):
                restarted = self.manager.start(task.state['job_id'], task.state['target'], restart=True)
                await restarted.runner
            self.assertEqual(restarted.scope_request is not None, reusable)

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

    async def test_export_rows_ignore_hidden_history_and_reject_same_named_duplicates(self):
        task = await self.bare_task()
        row = '<tr><td>班级A-第一次作业 10-15 17:55 导出中</td></tr>'
        await task.page.set_content(f'<div id=downloadcenter><table>{row}</table></div>'
                                    f'<div class=downloadCenter hidden><table>{row}</table></div>')
        rows = await task.adapter.export_rows()
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['ambiguous'])
        await task.page.locator('#downloadcenter tbody').evaluate(
            "e=>e.insertAdjacentHTML('beforeend', '<tr><td>班级A-第一次作业 10-15 17:59 导出成功</td></tr>')")
        rows = await task.adapter.export_rows()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['ambiguous'] for r in rows))
        self.assertEqual(task.adapter.export_locators, {})

    async def test_bound_exports_survive_time_changes_and_download_ready_class_first(self):
        task = await self.bare_task()
        zip_path = Path(self.temp.name) / 'export.zip'
        with zipfile.ZipFile(zip_path, 'w') as archive:
            archive.writestr('20260001-test/report.txt', 'student answer')
        downloads = []
        html = '''<div id=downloadcenter><table><tbody>
          <tr id=first><td>班级2CS20510501-第一次作业(附件)</td><td class=meta>10-15 17:55 导出中</td>
            <td><a class=download_ic href=/download/first>下载</a></td></tr>
          <tr id=second><td>班级2CS20510502-第一次作业(附件)</td><td class=meta>10-15 17:58 导出成功</td>
            <td><a class=download_ic href=/download/second>下载</a></td></tr>
          </tbody></table></div>'''
        # These IDs are only for fixture access, not platform record identities.
        html = html.replace('tr id=first', 'tr data-fixture=first').replace('tr id=second', 'tr data-fixture=second')
        async def route(request):
            if '/download/' in request.request.url:
                name = request.request.url.rsplit('/', 1)[-1]
                downloads.append(name)
                await request.fulfill(body=zip_path.read_bytes(), content_type='application/zip',
                                      headers={'Content-Disposition': f'attachment; filename={name}.zip'})
            else:
                await request.fulfill(body=html, content_type='text/html; charset=utf-8')
        await task.page.route('**/*', route)
        await task.page.goto('https://mooc1.chaoxing.com/course')
        exports = []
        for class_name in ('2CS20510501', '2CS20510502'):
            text = f'班级{class_name}-第一次作业(附件)\n\n10-15 17:55 导出中'
            old_identity = re.sub(r'重新下载|导出中|等待|导出成功|下载|失败|处理中|已完成|重新导出|\s+', '', text)
            exports.append({'ref': hashlib.sha256(old_identity.encode()).hexdigest()[:24],
                            'text': text, 'status': 'PENDING'})
        task.update(exports=exports, export_attempted=True, export_baseline=[],
                    export_started_at=time.time(), stage='WAIT_EXPORT',
                    class_scope={'mode': 'all', 'classes': []})
        task.gate.set()
        await task.process_exports()
        self.assertEqual(downloads, ['second'])
        self.assertEqual(task.state['files'][0]['export_ref'], exports[1]['ref'])
        self.assertEqual(task.state['files'][0]['status'], 'VERIFIED')
        self.assertIn('17:58', task.state['exports'][1]['text'])
        self.assertEqual(task.state['exports'][1]['status'], 'READY')
        self.assertTrue(any('正在校验压缩包' in e['message'] for e in task.state['events']))
        await task.process_exports()
        self.assertEqual(task.state['status'], 'RUNNING')
        self.assertEqual(task.state['control_mode'], 'AUTO')
        self.assertIn('等待平台打包 1 个附件，已下载校验 1/2 个', task.state['message'])
        await task.page.locator('[data-fixture=first] .meta').evaluate("e=>e.innerText='10-15 17:59 导出成功'")
        await task.process_exports()
        await task.process_exports()
        self.assertEqual(task.state['status'], 'SUCCEEDED', task.state)
        self.assertEqual(downloads, ['second', 'first'])
        self.assertEqual({f['export_ref'] for f in task.state['files']}, {e['ref'] for e in exports})
        for file in task.state['files']:
            self.assertEqual(verify_zip(file['path'])['status'], 'VERIFIED')

    async def test_agent_navigation_login_scope_export_and_real_download(self):
        await self.fixture_flow()

    async def test_custom_export_widgets_complete_without_repeated_confirmation(self):
        await self.fixture_flow(custom=True)

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

    async def test_readonly_label_cannot_disguise_a_form_submission(self):
        task = await self.bare_task()
        await task.page.route('**/*', lambda route: route.fulfill(content_type='text/html; charset=utf-8',
            body='<form method=post><h2>第一次作业</h2><button aria-label="查看提交" onclick="window.submitted=true">查看</button></form>'))
        await task.page.goto('https://learning.xidian.edu.cn/portal')
        task.gate.set()
        obs = await task.observe()
        entry = next(e for e in obs['observation']['elements'] if e['name']=='查看提交')
        payload = {'action_id':obs['observation_id'],'observation_id':obs['observation_id'],
                   'epoch':obs['epoch'],'revision':obs['revision'],
                   'action':{'tool':'click','element_ref':entry['ref']}}
        receipt = await task.apply_action(payload)
        self.assertEqual(receipt['outcome'], 'unknown')
        self.assertIsNone(await task.page.evaluate('window.submitted'))

    async def test_uncertain_action_is_persisted_and_never_replayed(self):
        task, payload = await self.observed_task()
        from unittest.mock import AsyncMock
        with patch.object(task, 'execute', AsyncMock(side_effect=TimeoutError('uncertain'))) as execute:
            self.assertEqual((await task.apply_action(payload))['outcome'], 'unknown')
            self.assertEqual((await task.apply_action(payload))['outcome'], 'unknown')
            self.assertEqual(execute.await_count, 1)
        self.assertEqual(task.state['control_mode'], 'AUTO')
        self.assertEqual(task.state['navigation_failures'], 1)
        self.assertEqual(task.state['last_error']['code'], 'BROWSER_TIMEOUT')
        journal = (self.manager.root / (task.state['job_id'] + '.json')).read_text()
        self.assertNotIn('uncertain', journal)

    async def test_navigation_recovery_is_bounded_and_logs_no_raw_exception(self):
        task, payload = await self.observed_task()
        from unittest.mock import AsyncMock
        with patch.object(task, 'execute', AsyncMock(side_effect=TimeoutError('https://portal/?token=private-secret'))) as execute:
            for i in range(3):
                if i:
                    obs = await task.observe()
                    payload = {'action_id':obs['observation_id'],'observation_id':obs['observation_id'],
                               'epoch':obs['epoch'],'revision':obs['revision'],'action':payload['action']}
                receipt = await task.apply_action(payload)
                self.assertEqual(receipt['outcome'], 'unknown')
                self.assertEqual(task.state['control_mode'], 'HUMAN' if i == 2 else 'AUTO')
            self.assertEqual(execute.await_count, 3)
        self.assertEqual(task.state['waiting_reason'], 'NAVIGATION')
        self.assertNotIn('private-secret', (self.manager.root / (task.state['job_id'] + '.json')).read_text())

    async def test_decision_service_failure_retries_before_human_handoff(self):
        task, _ = await self.observed_task()
        for i in range(3):
            await task.decision_failed({'epoch':task.state['control_epoch']})
            self.assertEqual(task.state['control_mode'], 'HUMAN' if i == 2 else 'AUTO')
        self.assertEqual(task.state['waiting_reason'], 'AGENT_UNAVAILABLE')

    async def test_same_named_navigation_with_page_progress_does_not_request_confirmation(self):
        task = await self.bare_task()
        await task.page.route('**/*', lambda route: route.fulfill(content_type='text/html; charset=utf-8',
            body="""<h1>第0页</h1><button onclick="window.n=(window.n||0)+1;document.querySelector('h1').innerText='第'+window.n+'页'">下一页</button>"""))
        await task.page.goto('https://learning.xidian.edu.cn/portal')
        task.gate.set()
        for _ in range(4):
            obs = await task.observe()
            ref = next(e['ref'] for e in obs['observation']['elements'] if e['name']=='下一页')
            await task.apply_action({'action_id':obs['observation_id'],'observation_id':obs['observation_id'],
                                    'revision':obs['revision'],'epoch':obs['epoch'],
                                    'action':{'tool':'click','element_ref':ref}})
        self.assertEqual(task.state['control_mode'], 'AUTO')
        self.assertEqual(await task.page.locator('h1').inner_text(), '第4页')

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

    async def fixture_flow(self, manual=False, custom=False):
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
        if custom:
            course_html = course_html.replace(
                '<label class=export-range><input type=checkbox>班级A</label>',
                """<div class=export-range><span class=grade_check onclick="setTimeout(()=>this.parentElement.classList.add('checked'),100)"></span>班级A</div>""")
            course_html = course_html.replace(
                '<label class=export-range><input type=checkbox>班级B</label>',
                '<div class=export-range><span class=grade_check></span>班级B</div>')
            course_html = course_html.replace(
                '<label class="export-range all"><input type=checkbox>所有班级</label>',
                '<div class="export-range all"><span class=grade_check></span>所有班级</div>'
                """<div class=out><span class=grade_check onclick="this.parentElement.classList.add('checked')"></span>导出提交附件</div>"""
                '<style>.grade_check{display:inline-block;width:20px;height:20px}</style>')
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
            self.assertEqual(sum(e['message'] == '请在面板中明确选择要导出的班级，或选择全部班级'
                                 for e in task.state['events']), 1)
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
