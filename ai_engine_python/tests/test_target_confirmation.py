"""Model-driven navigation keeps the original flow; worker only executes proposals."""
import json
import unittest
from unittest.mock import AsyncMock

import httpx

from app.rpa.errors import RpaIssue
from app.rpa.worker import app
from tests import test_rpa_worker as fixtures


class TargetConfirmationTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.BrowserTaskTests.asyncSetUp
    asyncTearDown = fixtures.BrowserTaskTests.asyncTearDown
    bare_task = fixtures.BrowserTaskTests.bare_task

    async def task_page(self, html, assignment=False):
        task = await self.bare_task()
        await task.page.route('**/*',lambda route:route.fulfill(content_type='text/html; charset=utf-8',body=html))
        await task.page.goto('https://learning.xidian.edu.cn/portal')
        task.adapter.course_confirmed = assignment
        task.adapter.authenticated = AsyncMock(return_value=True)
        task.gate.set()
        return task

    async def action(self, task, build):
        obs = await task.observe()
        self.assertTrue(obs['ready'])
        action = build(obs['observation'])
        result = await task.apply_action(dict(action_id=obs['observation_id'],observation_id=obs['observation_id'],
            revision=obs['revision'],epoch=obs['epoch'],action=action))
        self.assertEqual(result['outcome'],'completed',result['state'].get('last_error'))
        return result

    def click(self, name):
        return lambda view:dict(tool='click',element_ref=next(e['ref'] for e in view['elements'] if e['name']==name))

    def proposal(self, kind, name, entry, term=''):
        return lambda view:dict(tool='request_target_confirmation',kind=kind,actual_name=name,term=term,
            reason='大模型根据页面上下文判断可能为同一目标，请教师核对',
            element_ref=next(e['ref'] for e in view['elements'] if e['name']==entry))

    async def confirm(self, task):
        pending = task.state['target_resolution']
        async with task.lock:
            return await task.target_confirmation.resolve(dict(resolution_id=pending['id'],choice='confirm',
                                                               candidate_id=pending['candidates'][0]['id']))

    async def test_portal_personal_space_click_is_never_compared_to_course_name(self):
        task = await self.task_page('<article><h3>教学新闻</h3><a href="/space">个人空间</a></article>')
        await self.action(task,self.click('个人空间'))
        self.assertTrue(task.page.url.endswith('/space'))
        self.assertFalse(task.adapter.course_confirmed)
        self.assertFalse(task.state.get('target_resolution'))
        self.assertFalse(task.state.get('semester_lookup'))

    async def test_observation_does_not_search_change_semester_or_request_confirmation(self):
        task = await self.task_page('''<label>学期<select><option>2026-2027第一学期</option>
            <option>2025-2026第二学期</option></select></label>
            <article role=listitem><h3>操作系統</h3><button>进入课程</button></article>''')
        before = task.page.url
        for _ in range(5):
            obs = await task.observe()
            self.assertTrue(obs['ready'])
            self.assertEqual(task.state['control_mode'],'AUTO')
            self.assertFalse(task.state.get('target_resolution'))
            self.assertFalse(task.state.get('semester_lookup'))
        self.assertEqual(task.page.url,before)
        self.assertEqual(await task.page.locator('select').input_value(),'2026-2027第一学期')
        select = next(e for e in obs['observation']['elements'] if e['tag']=='select')
        self.assertEqual(select['options'],['2026-2027第一学期','2025-2026第二学期'])

    async def test_full_original_flow_with_model_proposals_for_course_and_assignment(self):
        task = await self.bare_task()
        task.state['target'].update(course_name='操作系統',assignment_name='作业1')
        pages = {
            '/portal':'<article><h3>教学新闻</h3><a href=/space>个人空间</a></article>',
            '/space':'<nav><a href=/courses>课程</a></nav>',
            '/courses':'<article role=listitem><h3>操作系统</h3><a href=/course>进入课程</a></article>',
            '/course':'<h1>操作系统</h1><a href=/work>作业</a>',
            '/work':'<h1>操作系统</h1><article role=listitem><h3>作业一</h3><a href=/review>批阅</a></article>',
            '/review':'''<h1>作业一</h1><button onclick="document.querySelector('aside').hidden=false">更多</button>
                <aside hidden><a onclick="document.querySelector('[role=dialog]').hidden=false">导出作业附件</a></aside>
                <section role=dialog hidden>导出设置<label><input type=checkbox>班级A</label></section>''',
        }
        async def route(r):
            from urllib.parse import urlsplit
            await r.fulfill(content_type='text/html; charset=utf-8',body=pages[urlsplit(r.request.url).path])
        await task.page.route('**/*',route)
        await task.page.goto('https://learning.xidian.edu.cn/portal')
        task.adapter.authenticated = AsyncMock(return_value=True)
        task.gate.set()
        for name in ['个人空间','课程']:
            await self.action(task,self.click(name))
        self.assertTrue(task.page.url.endswith('/courses'))
        await self.action(task,self.proposal('course','操作系统','进入课程'))
        self.assertTrue(task.page.url.endswith('/courses'))  # proposal never clicks the candidate
        with self.assertRaises(ValueError): await task.resume()
        await self.confirm(task)
        self.assertEqual(task.state['target']['course_name'],'操作系统')
        await self.action(task,self.click('进入课程'))
        await self.action(task,self.click('作业'))
        await self.action(task,self.proposal('assignment','作业一','批阅'))
        await self.confirm(task)
        self.assertEqual(task.state['target']['assignment_name'],'作业一')
        for name in ['批阅','更多']:
            await self.action(task,self.click(name))
        await self.action(task,lambda _:dict(tool='open_export_settings'))
        self.assertTrue(await task.page.locator('[role=dialog]').is_visible())
        self.assertFalse(task.state.get('export_attempted',False))
        self.assertEqual([h['name'] for h in task.state['target_resolutions']],['操作系统','作业一'])

    async def test_model_reports_missing_using_original_human_popup(self):
        task = await self.task_page('<nav><a>个人空间</a></nav><h2>课程列表</h2>')
        await self.action(task,lambda _:dict(tool='request_human',reason='当前页面未找到目标课程，请确认课程名或学期'))
        self.assertEqual(task.state['waiting_reason'],'COURSE_OR_TERM')
        self.assertIn('请确认课程名或学期',task.state['message'])

    async def test_candidate_must_be_observed_and_stale_or_exported_confirmation_fails(self):
        task = await self.task_page('<article role=listitem><h3>作业一</h3><button>批阅</button></article>',True)
        task.state['target']['assignment_name'] = '作业1'
        view = (await task.observe())['observation']
        with self.assertRaises(RpaIssue):
            await task.target_confirmation.propose(self.proposal('assignment','不存在的作业','批阅')(view))
        await self.action(task,self.proposal('assignment','作业一','批阅'))
        pending = task.state['target_resolution']
        payload = dict(resolution_id=pending['id'],choice='confirm',candidate_id=pending['candidates'][0]['id'])
        task.state['export_attempted'] = True
        with self.assertRaises(ValueError): await task.target_confirmation.resolve(payload)
        task.state['export_attempted'] = False
        await task.page.locator('h3').evaluate("e=>e.innerText='作业二'")
        with self.assertRaisesRegex(ValueError,'候选已变化'): await task.target_confirmation.resolve(payload)
        self.assertEqual(task.state['target']['assignment_name'],'作业1')

    async def test_control_api_auth_edit_and_one_step_resume(self):
        task = await self.task_page('<article role=listitem><h3>作业一</h3><button>批阅</button></article>',True)
        task.state['target']['assignment_name'] = '作业1'
        await self.action(task,self.proposal('assignment','作业一','批阅'))
        pending = task.state['target_resolution']
        self.manager.tasks[task.state['job_id']] = task
        old_manager,old_token = getattr(app.state,'manager',None),getattr(app.state,'token',None)
        app.state.manager,app.state.token = self.manager,'target-confirm-test'
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://worker') as client:
                url = '/tasks/'+task.state['job_id']+'/resolve-target'
                payload = dict(resolution_id=pending['id'],choice='edit',name='作业二')
                self.assertEqual((await client.post(url,json=payload)).status_code,401)
                client.headers['X-RPA-Token'] = 'target-confirm-test'
                self.assertEqual((await client.post(url,json={**payload,'resolution_id':'old'})).status_code,409)
                response = await client.post(url,json=payload)
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.json()['control_mode'],'AUTO')
                self.assertEqual(response.json()['target']['assignment_name'],'作业二')
                self.assertEqual((await client.post(url,json=payload)).status_code,409)
        finally:
            app.state.manager,app.state.token = old_manager,old_token
