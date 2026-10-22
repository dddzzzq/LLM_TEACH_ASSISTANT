"""Real browser target-confirmation UI; all business endpoints are intercepted."""
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
    with opener.open(base+'/src/components/RpaAttentionCenter.vue') as response:
        imports = re.findall(r'from "([^"]+)"', response.read().decode())
    vue = next(p for p in imports if '/vue.js' in p)
    router = next(p for p in imports if 'vue-router' in p)
    harness = f'''<html><head><link rel=stylesheet href=/src/assets/main.css></head><body><div id=app></div>
      <script type=module>
      import {{createApp,h}} from '{vue}';
      import {{createRouter,createMemoryHistory}} from '{router}';
      import Attention from '/src/components/RpaAttentionCenter.vue';
      createApp({{render:()=>h(Attention)}}).use(createRouter({{history:createMemoryHistory(),routes:[
        {{path:'/',component:{{render:()=>null}}}},{{path:'/ai-assistant',name:'ai-assistant',component:{{render:()=>null}}}}]}})).mount('#app');
      </script></body></html>'''

    def notice(kind='assignment', previous=False, missing=False, epoch=1):
        name = '移动互联网导论' if kind=='course' else '作业一'
        return dict(key=f'target:{epoch}', kind='target', reason='TARGET_NOT_FOUND' if missing else 'TARGET_SUGGESTION',
            job_id='test-task',course_name='移动互联网导论',assignment_name='作业1',
            target_resolution=dict(id=f'resolution-{epoch}',kind=kind,requested_name='移动互连网导论' if kind=='course' else '作业1',
              term='2025-2026第三学期' if previous else '2026-2027第一学期', previous_term=False, semester_fallback=previous,
              message='当前页面未找到目标课程，请确认课程名或学期' if missing else '当前页面找到近似课程，请确认' if previous else '找到可能对应的作业，请核对实际名称',
              candidates=[] if missing else [dict(id='candidate',name=name,context=name+' 班级A')]))

    state = dict(notices=[notice()], fail=False)
    posts, errors = [], []

    async def api(route):
        request = route.request
        if request.url.endswith('/attention'):
            data = state['notices']
            status = 200
        elif request.url.endswith('/resolve-target') and request.method=='POST':
            posts.append(request.post_data_json)
            status = 409 if state['fail'] else 200
            data = {'error':'页面候选已变化，请重新查找后再确认'} if state['fail'] else {'control_mode':'AUTO','status':'RUNNING'}
            if not state['fail']:
                state['notices'] = []
        elif request.url.endswith('/resume') and request.method == 'POST':
            posts.append({'action': 'resume'})
            status = 409 if state['fail'] else 200
            data = {'error': '当前任务状态已变化，请刷新'} if state['fail'] else {'status': 'RUNNING'}
            if not state['fail']:
                state['notices'] = []
        else:
            raise AssertionError('Unexpected business API '+request.method+' '+request.url)
        await route.fulfill(status=status,content_type='application/json',body=json.dumps(data))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context = await browser.new_context(viewport={'width':1365,'height':1000})
        await context.add_init_script('localStorage.setItem("access_token","ui-fixture");localStorage.setItem("user_info",'+json.dumps(json.dumps({'user_id':99,'role':'teacher'}))+');')
        page = await context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.route('**/api/**',api)
        await page.route(base+'/target-resolution-smoke',lambda route:route.fulfill(content_type='text/html; charset=utf-8',body=harness))
        await page.goto(base+'/target-resolution-smoke')
        dialog = page.get_by_role('dialog')
        await expect(page.get_by_role('heading',name='请确认找到的课程或作业')).to_be_visible()
        await expect(page.get_by_role('radio')).to_be_checked()
        await expect(dialog).to_contain_text('作业一')
        state['fail'] = True
        await page.get_by_role('button',name='确认所选并继续',exact=True).click()
        await expect(page.get_by_role('alert')).to_contain_text('页面候选已变化')
        await expect(dialog).to_be_visible()
        state['fail'] = False
        await page.get_by_role('button',name='确认所选并继续',exact=True).click()
        await expect(dialog).to_have_count(0)
        await page.wait_for_timeout(5300)
        await expect(dialog).to_have_count(0)
        assert posts[-1] == dict(resolution_id='resolution-1',choice='confirm',candidate_id='candidate')

        state['notices'] = [notice('course',previous=True,epoch=2)]
        await page.evaluate("window.dispatchEvent(new Event('focus'))")
        await expect(dialog).to_contain_text('所在学期')
        await expect(dialog).to_contain_text('2025-2026第三学期')
        out = Path(__file__).resolve().parents[2]/'.runtime/diagnostics'
        out.mkdir(parents=True,exist_ok=True)
        await page.screenshot(path=str(out/'rpa-target-confirmation-desktop.png'))
        await page.set_viewport_size({'width':390,'height':844})
        box = await dialog.bounding_box()
        assert box['x']>=0 and box['x']+box['width']<=390
        await page.screenshot(path=str(out/'rpa-target-confirmation-mobile.png'))
        await page.get_by_role('button',name='都不是，修改名称',exact=True).click()
        await page.get_by_role('textbox',name='正确的课程名称').fill('移动互联网技术导论')
        await page.get_by_role('button',name='修改名称并重新查找',exact=True).click()
        await expect(dialog).to_have_count(0)
        assert posts[-1] == dict(resolution_id='resolution-2',choice='edit',name='移动互联网技术导论')

        state['notices'] = [dict(key='missing:3',kind='human',reason='COURSE_OR_TERM',job_id='test-task',
            course_name='移动互联网导论',assignment_name='作业1',message='当前页面未找到目标课程，请确认课程名或学期')]
        await page.evaluate("window.dispatchEvent(new Event('focus'))")
        await expect(page.get_by_role('heading',name='请确认课程名或学期')).to_be_visible()
        await expect(page.get_by_role('button',name='确认所选并继续',exact=True)).to_have_count(0)
        await expect(dialog).to_contain_text('当前页面未找到目标课程，请确认课程名或学期')
        await expect(dialog).to_contain_text('手动操作')
        count = len(posts)
        await page.get_by_role('button',name='稍后处理',exact=True).click()
        await page.wait_for_timeout(5300)
        await expect(dialog).to_have_count(0)
        assert len(posts)==count and not errors, errors
        state['notices'] = [dict(key='startup:4', kind='human', reason='BROWSER_STARTUP', job_id='test-task',
            course_name='移动互联网导论', assignment_name='作业1', message='官网登录页加载失败，浏览器已保留。')]
        await page.evaluate("window.dispatchEvent(new Event('focus'))")
        await expect(dialog).to_contain_text('无需重新创建抓取任务')
        state['fail'] = True
        await page.get_by_role('button',name='重新加载登录页',exact=True).click()
        await expect(page.get_by_role('alert')).to_contain_text('当前任务状态已变化')
        await expect(dialog).to_be_visible()
        state['fail'] = False
        await page.get_by_role('button',name='重新加载登录页',exact=True).click()
        await expect(dialog).to_have_count(0)
        assert posts[-1] == {'action':'resume'} and not errors, errors
        await browser.close()
    print('PASS: target confirmation, manual semester, mobile, startup recovery popup, failed retry and successful resume')


if __name__=='__main__':
    asyncio.run(main())
