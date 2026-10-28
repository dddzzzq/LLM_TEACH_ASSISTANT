"""Fixture-only regression for floating assistant sessions and account isolation."""
import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
from playwright.async_api import async_playwright, expect

BASE = os.environ.get('RPA_UI_BASE_URL', 'http://127.0.0.1:6017')

async def main():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    source = opener.open(BASE + '/src/components/AgentChatWidget.vue', timeout=10).read().decode()
    vue = next(p for p in re.findall(r'from "([^"]+)"', source) if '/vue.js' in p)
    source = opener.open(BASE + '/src/router/index.js', timeout=10).read().decode()
    router_module = next(p for p in re.findall(r'from "([^"]+)"', source) if 'vue-router' in p)
    harness = f'''<html><head><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app"></div><script type="module">
      import {{createApp,h}} from '{vue}';import {{createRouter,createMemoryHistory}} from '{router_module}';
      import App from '/src/App.vue'; import auth from '/src/services/authApi.js'; import Assistant from '/src/views/AIAssistantView.vue';
      const stub={{render:()=>h('p','业务页面')}};
      const router=createRouter({{history:createMemoryHistory(),routes:[{{path:'/assignments',name:'assignments',component:stub}},{{path:'/login',name:'login',component:stub}},{{path:'/ai-assistant',name:'ai-assistant',component:Assistant}},{{path:'/settings/portal',name:'portal-settings',component:stub}}]}});
      window.fixtureRouter=router;window.fixtureAuth=auth;
      localStorage.setItem('access_token','a');localStorage.setItem('user_info',JSON.stringify({{user_id:1,username:'a',role:'teacher'}}));
      localStorage.setItem('current_session_id','legacy-a');
      await router.push('/assignments');createApp(App).use(router).mount('#app');
      </script></body></html>'''
    posts, histories, errors = [], [], []
    started, release = asyncio.Event(), asyncio.Event()
    async def api(route):
        req=route.request; path=urllib.parse.urlsplit(req.url).path
        owner=req.headers.get('authorization','').removeprefix('Bearer ')
        if path=='/api/profile': data={'id':1 if owner=='a' else 2,'username':owner,'role':'teacher'}
        elif path.endswith('/history'):
            histories.append((owner,path))
            session=path.split('/')[-2]
            if session not in ({'legacy-a'} if owner=='a' else {'b-new'}):
                await route.fulfill(status=404,content_type='application/json',body='{}'); return
            data={'messages':[{'role':'assistant','content':owner+'-history'}]}
        elif path=='/api/sessions' and req.method=='POST':
            posts.append((owner,path));data={'id':owner+'-new'}
        elif path=='/api/sessions': data=[{'id':'legacy-a' if owner=='a' else 'b-new','title':owner+' conversation'}]
        elif path.endswith('/runs'): data={'runs':[]}
        elif path=='/api/agent/chat':
            posts.append((owner,path)); started.set(); await release.wait(); data={'reply':'a-late-result','job_id':'a-task','job_type':'rpa_fetch_homework'}
        else:data=[]
        await route.fulfill(content_type='application/json',body=json.dumps(data))
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=await browser.new_context(viewport={'width':1440,'height':1000})
        page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        await page.route('**/api/**',api)
        await page.route(BASE+'/floating-isolation',lambda r:r.fulfill(content_type='text/html',body=harness))
        await page.goto(BASE+'/floating-isolation')
        await page.get_by_role('button',name='AI 教学助手',exact=True).click()
        await expect(page.get_by_text('a-history',exact=True)).to_be_visible()
        assert not posts
        assert await page.evaluate("localStorage.getItem('teacher-assistant-session:1')")=='legacy-a'
        assert await page.evaluate("localStorage.getItem('current_session_id')") is None
        await page.get_by_role('button',name='历史会话',exact=False).click()
        await expect(page.get_by_text('a conversation',exact=True)).to_be_visible()
        await page.get_by_text('a conversation',exact=True).click()
        await page.get_by_role('button',name='执行记录',exact=True).click()
        await expect(page.get_by_role('dialog',name='会话执行记录')).to_be_visible()
        await page.get_by_role('button',name='关闭提醒',exact=True).click()
        await page.get_by_role('button',name='任务控制台',exact=True).click()
        console=page.get_by_role('dialog',name='作业拉取与材料')
        await expect(console).to_be_visible()
        assert await console.get_by_label('下载后操作').locator('option').count()==3
        await page.get_by_role('button',name='收起作业控制台',exact=True).click()
        await page.get_by_role('textbox',name='消息内容').fill('late request')
        await page.get_by_role('button',name='发送',exact=True).click()
        await asyncio.wait_for(started.wait(),5)
        await page.evaluate("async()=>{window.fixtureAuth.logout();await window.fixtureRouter.push('/login');}")
        await expect(page.get_by_role('button',name='AI 教学助手',exact=True)).to_have_count(0)
        await page.evaluate("async()=>{localStorage.setItem('access_token','b');localStorage.setItem('user_info',JSON.stringify({user_id:2,username:'b',role:'teacher'}));await window.fixtureRouter.push('/assignments');}")
        await page.get_by_role('button',name='AI 教学助手',exact=True).click()
        await expect(page.get_by_role('textbox',name='消息内容')).to_be_enabled()
        assert await page.evaluate("localStorage.getItem('teacher-assistant-session:2')")=='b-new'
        assert ('b','/api/sessions/legacy-a/history') not in histories
        release.set();await page.wait_for_timeout(250)
        await expect(page.get_by_text('a-history',exact=True)).to_have_count(0)
        await expect(page.get_by_text('a-late-result',exact=True)).to_have_count(0)
        await expect(page.get_by_role('dialog',name='作业拉取与材料')).to_have_count(0)
        await page.evaluate("async()=>{window.fixtureAuth.logout();await window.fixtureRouter.push('/login');localStorage.setItem('access_token','a');localStorage.setItem('user_info',JSON.stringify({user_id:1,username:'a',role:'teacher'}));await window.fixtureRouter.push('/assignments');}")
        await page.get_by_role('button',name='AI 教学助手',exact=True).click()
        await expect(page.get_by_text('a-history',exact=True)).to_be_visible()
        assert len([p for p in posts if p==('a','/api/sessions')])==0
        assert len([p for p in posts if p==('b','/api/sessions')])==1
        assert not errors,errors
        await browser.close()
    print('PASS: history/execution/console controls, legacy read-only migration, account-scoped restoration, logout UI cleanup and late-response isolation')

if __name__ == '__main__':
    asyncio.run(main())
