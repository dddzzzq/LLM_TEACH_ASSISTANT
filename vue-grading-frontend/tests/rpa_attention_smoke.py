"""Real Chromium UI regression; business APIs and remote browser are fully mocked.

Run with the project's uv Python environment while the Vite dev server is running.
"""
import asyncio
import json
import os
import re
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright, expect


async def main():
    base = os.environ.get('RPA_UI_BASE_URL', 'http://127.0.0.1:6006')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + '/src/components/RpaAttentionCenter.vue') as response:
        source = response.read().decode()
    imports = re.findall(r'from "([^"]+)"', source)
    vue = next(p for p in imports if '/vue.js' in p)
    router_module = next(p for p in imports if 'vue-router' in p)
    rubric_notice = dict(key='fixture:rubric:A', kind='rubric', job_id='fixture-task', file_id='fixture-file',
                         assignment_id=3, course_name='分布式计算', assignment_name='第一次作业',
                         class_name='2CS20510502', rubric_missing=True, message='等待评分标准')
    fixture = dict(job_id='fixture-task', status='WAITING_USER', stage='SELECT_CLASSES',
                   control_mode='HUMAN', control_epoch=8, waiting_reason='CLASS_SCOPE', message='请确认导出班级',
                   target=dict(course_name='分布式计算', assignment_name='第一次作业'),
                   classes=[dict(id='class_0', name='全部班级', all=True)],
                   requested_scope=dict(mode='all', class_ids=[]), files=[], events=[], grading_files=[])
    state = dict(notices=[rubric_notice.copy()], fail_save=True, fail_attention=False)
    posts, errors, attention_requests = [], [], []
    harness = f'''<html><head><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app"></div>
      <script type="module">
        import {{createApp,h}} from '{vue}';
        import {{createRouter,createMemoryHistory,RouterLink,RouterView}} from '{router_module}';
        import Attention from '/src/components/RpaAttentionCenter.vue';
        import Assistant from '/src/views/AIAssistantView.vue';
        const router=createRouter({{history:createMemoryHistory(),routes:[
          {{path:'/',component:{{render:()=>h('h1','测试主页')}}}},
          {{path:'/ai-assistant',name:'ai-assistant',component:Assistant}}]}});
        createApp({{render:()=>h('div',[h(RouterLink,{{to:'/'}},()=> '测试主页'),h(RouterView),h(Attention)])}}).use(router).mount('#app');
      </script></body></html>'''

    async def api(route):
        request = route.request
        if '/jobs/attention' in request.url:
            attention_requests.append(request.url)
            await route.fulfill(status=503 if state['fail_attention'] else 200, content_type='application/json',
                                body=json.dumps(state['notices']))
            return
        if '/requirements' in request.url:
            if request.method == 'POST':
                payload = request.post_data_json
                posts.append(payload)
                if state['fail_save']:
                    await route.fulfill(status=503, content_type='application/json', body='{"error":"测试保存失败，稍后重试"}')
                else:
                    state['notices'] = [n for n in state['notices'] if n.get('file_id') != payload['file_id']]
                    await route.fulfill(content_type='application/json', body='{"message":"已保存"}')
            else:
                await route.fulfill(content_type='application/json', body=json.dumps(dict(question='已有题目要求', rubric={})))
            return
        if request.method == 'POST':
            raise AssertionError('Closing reminders must not submit task control actions: ' + request.url)
        data = [dict(fixture, course_name='分布式计算', assignment_name='第一次作业')] if request.url.endswith('/jobs') else fixture
        await route.fulfill(content_type='application/json', body=json.dumps(data))

    async def setup(context, role='teacher'):
        await context.add_init_script('localStorage.setItem("access_token","isolated-ui-test");localStorage.setItem("user_info",'
                                      + json.dumps(json.dumps(dict(user_id=99, role=role))) + ');')
        page = await context.new_page()
        page.on('pageerror', lambda e: (errors.append(str(e)), print('Browser error:', str(e))))
        page.on('console', lambda msg: print('Browser console:',msg.text) if msg.type == 'error' else None)
        await page.route('**/api/**', api)
        for component in ('RpaBrowserPanel', 'AgentChatWidget'):
            await page.route('**/src/components/' + component + '.vue*', lambda route: route.fulfill(
                content_type='application/javascript', body='export default {render(){return null}}'))
        await page.route(base + '/rpa-attention-smoke', lambda route: route.fulfill(content_type='text/html; charset=utf-8', body=harness))
        await page.goto(base + '/rpa-attention-smoke')
        return page

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=['--no-sandbox'])
        context = await browser.new_context(viewport={'width':1365,'height':1000})
        page = await setup(context)
        dialog = page.get_by_role('dialog')
        try:
            await expect(dialog).to_be_visible()
        except AssertionError:
            print('Initial fixture state:', await page.locator('body').inner_text(), errors, attention_requests)
            raise
        await expect(page.get_by_label('题目要求', exact=True)).to_have_value('已有题目要求')
        await page.get_by_label('评分标准（JSON）', exact=True).fill('not json')
        await page.get_by_role('button', name='保存评分配置', exact=True).click()
        await expect(page.get_by_role('alert')).to_contain_text('不是有效 JSON')
        assert not posts
        await page.get_by_label('题目要求', exact=True).fill('补充后的题目要求')
        await page.get_by_label('评分标准（JSON）', exact=True).fill('{"正确性": 60, "代码质量": 40}')
        await page.get_by_role('button', name='保存评分配置', exact=True).click()
        await expect(page.get_by_role('alert')).to_have_text('测试保存失败，稍后重试')
        await page.wait_for_timeout(5500)
        await expect(page.get_by_label('题目要求', exact=True)).to_have_value('补充后的题目要求')
        await page.get_by_role('button', name='稍后处理', exact=True).click()
        await page.wait_for_timeout(5500)
        await expect(dialog).to_have_count(0)
        assert len(posts) == 1
        await page.get_by_role('button', name=re.compile('作业待处理')).click()
        await page.get_by_role('button', name='处理此项', exact=True).click()
        await expect(page.get_by_label('题目要求', exact=True)).to_have_value('补充后的题目要求')
        await page.keyboard.press('Escape')
        await expect(dialog).to_have_count(0)
        await page.reload()
        await expect(page.get_by_role('button', name=re.compile('作业待处理'))).to_be_visible()
        await expect(dialog).to_have_count(0)
        state['fail_attention'] = True
        await page.wait_for_timeout(5500)
        await expect(dialog).to_have_count(0)
        state['fail_attention'] = False
        state['notices'] = [dict(key='fixture:human:8', kind='human', job_id='fixture-task',
                                  course_name='分布式计算', assignment_name='第一次作业', reason='CLASS_SCOPE', message='请确认导出班级')]
        await expect(page.get_by_role('heading', name='作业抓取需要人工处理')).to_be_visible(timeout=8000)
        await page.get_by_role('button', name='打开控制台处理', exact=True).click()
        await expect(page.get_by_role('button', name='更新范围并继续')).to_be_visible()
        await expect(page.get_by_role('radio', name='全部班级', exact=True)).to_be_checked()
        await expect(dialog).to_have_count(1)  # Only the existing console drawer remains.
        await page.get_by_role('button', name='收起作业控制台').click()
        await page.get_by_role('link', name='测试主页', exact=True).click()
        state['notices'][0].update(key='fixture:human:9', reason='EXPORT_FORMAT', message='请核对导出提交附件')
        await expect(page.get_by_role('heading', name='作业抓取需要人工处理')).to_be_visible(timeout=8000)
        await expect(page.get_by_text('请核对导出提交附件', exact=True)).to_be_visible()
        assert len(posts) == 1
        await page.get_by_role('button', name='稍后处理', exact=True).click()
        class_notice = dict(key='fixture:class:B', kind='class', job_id='fixture-task', file_id='class-file',
                            course_name='分布式计算', assignment_name='第一次作业', message='请核对班级')
        state['notices'] = [class_notice, rubric_notice.copy()]
        await expect(page.get_by_role('heading', name='请核对附件班级')).to_be_visible(timeout=8000)
        await page.get_by_role('button', name='稍后处理', exact=True).click()
        await page.get_by_role('button', name=re.compile('作业待处理')).click()
        await expect(page.get_by_role('button', name='处理此项', exact=True)).to_have_count(2)
        await page.get_by_role('button', name='处理此项', exact=True).nth(1).click()
        await expect(page.get_by_label('题目要求', exact=True)).to_have_value('已有题目要求')
        await page.get_by_label('评分标准（JSON）', exact=True).fill('{}')
        await page.get_by_role('button', name='保存评分配置', exact=True).click()
        await expect(page.get_by_role('alert')).to_contain_text('非空 JSON 对象')
        assert len(posts) == 1
        await page.get_by_label('评分标准（JSON）', exact=True).fill('{"正确性":100}')
        out = Path(__file__).resolve().parents[2] / '.runtime/diagnostics'
        out.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(out/'rpa-attention-desktop.png'))
        await page.set_viewport_size({'width':390,'height':844})
        box = await page.locator('dialog.app-modal').bounding_box()
        assert box['x'] >= 0 and box['x'] + box['width'] <= 390
        for _ in range(8):
            await page.keyboard.press('Tab')
            assert await page.evaluate("!!document.activeElement.closest('dialog')")
        await page.screenshot(path=str(out/'rpa-attention-mobile.png'))
        state['fail_save'] = False
        await page.get_by_role('button', name='保存评分配置', exact=True).click()
        await expect(page.get_by_role('status')).to_have_text('题目与评分标准已保存；后续按本次已选择的处理范围执行。')
        assert len(posts) == 2 and posts[-1]['rubric'] == {'正确性':100}
        await expect(page.locator('dialog.app-modal')).to_have_count(0)
        await context.close()
        before = len(attention_requests)
        student_context = await browser.new_context()
        student = await setup(student_context, 'student')
        await student.wait_for_timeout(1000)
        assert len(attention_requests) == before
        await expect(student.locator('dialog.app-modal')).to_have_count(0)
        assert not errors, errors
        await browser.close()
    print('PASS: automatic rubric/human prompts, cross-page handling, multi-task inbox, polling/reload deduplication, draft preservation, validation, save retry, keyboard/mobile layout, and student isolation.')


if __name__ == '__main__':
    asyncio.run(main())
