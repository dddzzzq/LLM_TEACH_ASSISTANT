"""Real-browser log search checks. All business/log APIs are intercepted fixtures."""
import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright, expect


async def main():
    base=os.environ.get('RPA_UI_BASE_URL','http://127.0.0.1:6006')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base+'/src/views/LogsView.vue') as response:
        source=response.read().decode()
    imports=re.findall(r'from "([^"]+)"',source)
    vue=next(p for p in imports if '/vue.js' in p)
    router_module=next(p for p in imports if 'vue-router' in p)
    task='11111111-1111-4111-8111-111111111111'
    grading='22222222-2222-4222-8222-222222222222'
    rows=[dict(id=101,time='2026-10-18T14:32:00+08:00',source='task',level='INFO',message='批改状态 CANCELLED：老师取消本次批改',task_ids=[grading,task],line=0,estimated_time=False),
          dict(id=102,time='2026-10-18T12:14:12+08:00',source='backend',level='ERROR',message='download stalled password=[REDACTED] <script>window.__logInjected=1</script> '+('读取附件失败，等待网络恢复。'*24),task_ids=[grading],line=420,estimated_time=False),
          dict(id=103,time='2026-10-18T12:08:18+08:00',source='rpa',level='WARN',message='等待平台打包 1 个附件，已校验 1/2 个',task_ids=[task],line=18,estimated_time=True)]
    sources=[dict(id=k,name=v,available=True,size=1000,indexed_bytes=1000) for k,v in [('task','任务事件'),('backend','后端与批改'),('ai','AI 与模型'),('rpa','教务抓取')]]
    tasks=[dict(id=grading,parent_id=task,kind='批改',course='分布式计算',name='第一次作业',class_name='2CS20510502',status='CANCELLED'),dict(id=task,kind='抓取',course='分布式计算',name='第一次作业',class_name='',status='PARTIAL_SUCCESS')]
    calls=[];errors=[];state={'fail':False}
    harness=f'''<html><head><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app"></div><script type="module">
      import {{createApp,h}} from '{vue}';
      import {{createRouter,createMemoryHistory}} from '{router_module}';
      import App from '/src/App.vue';import Logs from '/src/views/LogsView.vue';
      const router=createRouter({{history:createMemoryHistory(),routes:[{{path:'/logs',name:'logs',component:Logs}},{{path:'/ai-assistant',name:'ai-assistant',component:{{render:()=>h('p','任务控制台')}}}},{{path:'/',redirect:'/logs'}}]}});
      await router.push('/logs?task_id={grading}&period=7d');
      createApp(App).use(router).mount('#app');
    </script></body></html>'''
    async def api(route):
        req=route.request
        assert req.method=='GET','Log lookup must not mutate task state'
        path=urllib.parse.urlparse(req.url).path
        params={k:v[0] for k,v in urllib.parse.parse_qs(urllib.parse.urlparse(req.url).query).items()}
        calls.append((path,params))
        if path.endswith('/sources'):data=dict(sources=sources,retention_days=30,task_scoped=False)
        elif path.endswith('/tasks'):data=tasks
        elif path.endswith('/attention'):data=[]
        elif path.endswith('/context'):data=dict(items=[rows[2],rows[1],rows[0]],permission_filtered=False)
        elif path.endswith('/export'):
            await route.fulfill(content_type='text/plain; charset=utf-8',body='日志导出\npassword=[REDACTED]\n');return
        elif path.endswith('/search'):
            if state['fail']:
                await route.fulfill(status=503,content_type='application/json',body='{"error":"日志索引暂不可用，请稍后重试"}');return
            selected=rows
            if params.get('source','all')!='all':selected=[r for r in selected if r['source']==params['source']]
            if params.get('level','all')!='all':selected=[r for r in selected if r['level']==params['level']]
            if params.get('q'):selected=[r for r in selected if all(term.lower() in r['message'].lower() for term in params['q'].split())]
            if params.get('cursor'):selected=selected[2:];more=False
            else:more=len(selected)>2;selected=selected[:2]
            data=dict(items=selected,next_cursor='fixture-page-2' if more else '',has_more=more)
        else:data=[]
        await route.fulfill(content_type='application/json',body=json.dumps(data))

    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=await browser.new_context(viewport={'width':1440,'height':1050},accept_downloads=True,permissions=['clipboard-read','clipboard-write'])
        await context.add_init_script('localStorage.setItem("access_token","isolated-log-ui");localStorage.setItem("user_info",JSON.stringify({user_id:1,username:"管理员",role:"admin"}));')
        page=await context.new_page()
        page.on('pageerror',lambda e:(errors.append(str(e)),print('Browser error:',str(e))))
        await page.route('**/api/**',api)
        await page.route(base+'/logbook-smoke',lambda route:route.fulfill(content_type='text/html; charset=utf-8',body=harness))
        await page.goto(base+'/logbook-smoke')
        await expect(page.get_by_role('heading',name='日志查阅',exact=True)).to_be_visible()
        await expect(page.locator('tbody tr')).to_have_count(2)
        assert await page.evaluate('window.__logInjected') is None
        await expect(page.get_by_label('关联任务',exact=True)).to_have_value(grading)
        await expect(page.get_by_role('link',name='打开作业控制台')).to_have_attribute('href','/ai-assistant?task='+task)
        assert len([c for c in calls if c[0].endswith('/search')])==1
        await page.get_by_role('button',name='更早日志',exact=True).click()
        await expect(page.locator('tbody tr')).to_have_count(1)
        await expect(page.get_by_text('≈ 时间估算',exact=True)).to_be_visible()
        await expect(page.get_by_role('button',name='更早日志',exact=True)).to_be_disabled()
        await page.get_by_role('button',name='上一页',exact=True).click()
        await expect(page.locator('tbody tr')).to_have_count(2)
        before=[q for p,q in calls if p.endswith('/search')]
        assert before[0]['from']==before[1]['from']==before[2]['from']
        await page.get_by_label('日志级别',exact=True).select_option('ERROR')
        await page.get_by_role('button',name='搜索日志',exact=True).click()
        await expect(page.locator('tbody tr')).to_have_count(1)
        await page.get_by_label('关键词',exact=True).fill('download stalled')
        await page.get_by_role('button',name='搜索日志',exact=True).click()
        await expect(page.locator('mark')).to_have_count(2)
        await page.get_by_role('button',name='展开内容',exact=True).click()
        await expect(page.locator('.message-cell pre')).to_have_class('expanded')
        await page.get_by_role('checkbox',name='每 5 秒刷新',exact=True).check()
        await page.get_by_role('button',name='上下文',exact=True).click()
        await expect(page.get_by_role('dialog')).to_be_visible()
        await expect(page.locator('.context-lines li')).to_have_count(3)
        searches=len([p for p,_ in calls if p.endswith('/search')])
        await page.wait_for_timeout(5500)
        assert len([p for p,_ in calls if p.endswith('/search')])==searches
        await page.get_by_role('button',name='复制上下文',exact=True).click()
        assert 'stalled' in await page.evaluate('navigator.clipboard.readText()')
        await page.keyboard.press('Escape')
        await expect(page.get_by_role('dialog')).to_have_count(0)
        await page.wait_for_timeout(5500)
        assert len([p for p,_ in calls if p.endswith('/search')])>searches
        await page.get_by_role('checkbox',name='每 5 秒刷新',exact=True).uncheck()
        async with page.expect_download() as info:
            await page.get_by_role('button',name='导出筛选结果',exact=True).click()
        download=await info.value
        assert '[REDACTED]' in Path(await download.path()).read_text()
        await page.get_by_label('关键词',exact=True).fill('not found')
        await page.get_by_role('button',name='搜索日志',exact=True).click()
        await expect(page.get_by_role('heading',name='未找到匹配日志')).to_be_visible()
        state['fail']=True
        await page.get_by_label('关键词',exact=True).fill('network failure')
        await page.get_by_role('button',name='搜索日志',exact=True).click()
        await expect(page.get_by_role('alert')).to_have_text('日志索引暂不可用，请稍后重试')
        await expect(page.get_by_role('heading',name='未找到匹配日志')).to_have_count(0)
        state['fail']=False
        await page.get_by_role('button',name='重置',exact=True).click()
        await expect(page.locator('tbody tr')).to_have_count(2)
        await page.get_by_label('时间范围',exact=True).select_option('custom')
        await page.get_by_label('开始时间',exact=True).fill('2026-10-18T14:00')
        await page.get_by_label('结束时间',exact=True).fill('2026-10-18T12:00')
        await page.get_by_role('button',name='搜索日志',exact=True).click()
        await expect(page.get_by_role('alert')).to_have_text('请选择有效的开始、结束时间。')
        await page.get_by_role('button',name='重置',exact=True).click()
        await expect(page.locator('tbody tr')).to_have_count(2)
        output=Path(__file__).resolve().parents[2]/'.runtime/diagnostics'
        await page.screenshot(path=str(output/'logbook-desktop.png'),full_page=True)
        await page.set_viewport_size({'width':390,'height':844})
        await page.screenshot(path=str(output/'logbook-mobile.png'),full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors,errors
        await browser.close()
    print('PASS: task deep link, literal highlighting/XSS safety, stable pagination, context permissions, copy/export, refresh pause, time validation, errors, mobile layout; no task mutations.')


if __name__=='__main__':
    asyncio.run(main())
