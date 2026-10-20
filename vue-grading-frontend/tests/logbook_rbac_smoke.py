"""Real app routes and role-specific log entries; all APIs are fixtures."""
import asyncio
import json
import os
import re
import urllib.parse
from playwright.async_api import async_playwright, expect


async def main():
    base = os.environ.get('RPA_UI_BASE_URL', 'http://127.0.0.1:6006')
    batch = dict(id='fixture-batch', status='READY', message='材料已整理', student_count=1,
                 issue_count=0, auto_grade=False, grading_job_id='', grading_state='NOT_REQUESTED',
                 class_name='A', output_dir='D:/dzq/homework', created_at='2026-10-19T10:00:00+08:00')
    task = dict(job_id='fixture-task', course_name='课程', assignment_name='作业', status='SUCCEEDED',
                stage='DONE', control_mode='AUTO', message='下载结束', target=dict(after_download='extract'),
                material_batches=[batch], grading_files=[], files=[], events=[])
    errors = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=['--no-sandbox'])
        for role in ['admin', 'teacher', 'student']:
            log_requests, mutations = [], []
            context = await browser.new_context(viewport={'width':1440, 'height':1050})
            user = json.dumps(dict(user_id=1, username='fixture', role=role))
            await context.add_init_script(f'localStorage.setItem("access_token","fixture");localStorage.setItem("user_info",JSON.stringify({user}));localStorage.setItem("current_session_id","fixture-session");')
            page = await context.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            async def api(route):
                path = urllib.parse.urlparse(route.request.url).path
                if route.request.method != 'GET':
                    mutations.append(path)
                    raise AssertionError('Role navigation submitted an action: ' + path)
                if path.startswith('/api/logs/'):
                    log_requests.append(path)
                    assert role == 'admin', 'Non-admin page attempted log lookup'
                    if path.endswith('/sources'):
                        data = dict(sources=[], retention_days=30, task_scoped=False)
                    elif path.endswith('/search'):
                        data = dict(items=[], has_more=False, next_cursor='')
                    else:
                        data = []
                elif path == '/api/rpa/jobs/fixture-task':
                    data = task
                elif path == '/api/rpa/jobs':
                    data = [task]
                elif path == '/api/materials':
                    data = dict(items=[batch], grading_jobs=[])
                elif path.endswith('/history'):
                    data = dict(messages=[])
                else:
                    data = []
                await route.fulfill(content_type='application/json', body=json.dumps(data, ensure_ascii=False))
            await page.route('**/api/**', api)
            await page.goto(base + '/logs?task_id=fixture-task&period=7d')
            if role == 'admin':
                await expect(page.get_by_role('heading', name='日志查阅', exact=True)).to_be_visible()
                await expect(page.get_by_role('link', name='日志查阅', exact=True)).to_be_visible()
                assert log_requests
            else:
                await expect(page).to_have_url(re.compile('/home$' if role == 'teacher' else '/ai-assistant$'))
                await expect(page.locator('a[href^="/logs"]')).to_have_count(0)
                assert not log_requests
            await page.goto(base + '/ai-assistant?task=fixture-task')
            if role in ['admin','teacher']:
                await expect(page.get_by_role('dialog', name='作业拉取与材料')).to_be_visible()
                await expect(page.locator('.batch-card')).to_have_count(1)
                if role == 'admin':
                    await expect(page.get_by_role('link', name='查看任务日志', exact=True)).to_be_visible()
                    await expect(page.get_by_role('link', name='查看日志', exact=True)).to_be_visible()
                else:
                    await expect(page.locator('a[href^="/logs"]')).to_have_count(0)
                    await expect(page.get_by_role('button', name='开始批改', exact=True)).to_be_visible()
                    assert not log_requests
            else:
                await expect(page.locator('.command-options button')).to_have_count(1)
                await expect(page.get_by_role('button', name='查询我的成绩', exact=True)).to_be_visible()
                await expect(page.get_by_role('button', name='打开作业控制台', exact=True)).to_have_count(0)
                await expect(page.locator('a[href^="/logs"]')).to_have_count(0)
                assert not log_requests
            assert not mutations, mutations
            await context.close()
        await browser.close()
    assert not errors, errors
    print('PASS: admin log route/sidebar/task/material links; teacher route denial and teaching actions retained; student own-score entry only; no model or task actions.')


asyncio.run(main())
