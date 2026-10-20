"""Exercise download/preparation status with fixture APIs; no real task or model calls."""
import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
from playwright.async_api import async_playwright, expect


async def main():
    base = os.environ.get('RPA_UI_BASE_URL', 'http://127.0.0.1:6006')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + '/src/components/MaterialBatchPanel.vue') as response:
        source = response.read().decode()
    vue = next(p for p in re.findall(r'from "([^"]+)"', source) if '/vue.js' in p)
    with opener.open(base + '/src/router/index.js') as response:
        source = response.read().decode()
    router_module = next(p for p in re.findall(r'from "([^"]+)"', source) if 'vue-router' in p)
    batch = dict(id='material-fixture', status='FAILED', message='文件处理数量超过30000限制',
                 student_count=0, issue_count=1, auto_grade=False, grading_job_id='',
                 grading_state='NOT_REQUESTED', class_name='A', output_dir='D:/dzq/homework/fixture',
                 created_at='2026-10-18T18:00:00+08:00')
    task = dict(job_id='download-fixture', course_name='课程', assignment_name='作业',
                status='SUCCEEDED', stage='DONE', control_mode='AUTO', message='下载结束，已校验1/1个附件',
                target=dict(course_name='课程', assignment_name='作业', after_download='extract'),
                files=[], grading_files=[], events=[], material_batches=[batch])
    posts, errors = [], []

    async def api(route):
        path = urllib.parse.urlparse(route.request.url).path
        if route.request.method != 'GET':
            posts.append(path)
            raise AssertionError('Status display must not submit actions: ' + path)
        if path == '/api/rpa/jobs':
            data = [task]
        elif path == '/api/rpa/jobs/download-fixture':
            data = task
        elif path == '/api/materials':
            data = dict(items=[batch], grading_jobs=[])
        else:
            data = []
        await route.fulfill(content_type='application/json', body=json.dumps(data, ensure_ascii=False))

    harness = f'''<html><body><div id="app"></div><script type="module">
    import {{createApp,h}} from '{vue}';import {{createRouter,createMemoryHistory}} from '{router_module}';
    import Panel from '/src/components/RpaTaskPanel.vue';
    const router=createRouter({{history:createMemoryHistory(),routes:[{{path:'/',component:{{render:()=>h('div')}}}},{{path:'/logs',name:'logs',component:{{render:()=>h('div')}}}}]}});
    await router.push('/');createApp({{render:()=>h(Panel,{{taskId:'download-fixture'}})}}).use(router).mount('#app');
    </script></body></html>'''
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=['--no-sandbox'])
        page = await browser.new_page()
        await page.add_init_script('localStorage.setItem("access_token","fixture");localStorage.setItem("user_info",JSON.stringify({user_id:1,role:"teacher"}));')
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.route('**/api/**', api)
        await page.route(base + '/materials-recovery-fixture', lambda route: route.fulfill(content_type='text/html; charset=utf-8', body=harness))
        await page.goto(base + '/materials-recovery-fixture')
        await expect(page.get_by_text('状态：下载已完成，解压整理失败', exact=True)).to_be_visible()
        await expect(page.get_by_role('alert')).to_contain_text('无需重新拉取')
        await expect(page.get_by_role('button', name='重新解压（不批改）')).to_be_visible()
        for status, label in [('PROCESSING', '正在解压整理'), ('NEEDS_REVIEW', '部分材料需要核对'), ('READY', '下载与解压已完成')]:
            batch['status'] = status
            await expect(page.get_by_text(re.compile('状态：.*' + label))).to_be_visible(timeout=8000)
            retry = page.get_by_role('button', name='重新解压（不批改）')
            if status == 'NEEDS_REVIEW':
                await expect(retry).to_be_visible(timeout=8000)
            else:
                await expect(retry).to_have_count(0, timeout=8000)
        assert not posts, posts
        assert not errors, errors
        await browser.close()
    print('PASS: completed downloads display preparation failure, progress, review and readiness; recovery entry visible; fixture APIs only.')


asyncio.run(main())
