"""Verify shortcut intent and role boundaries against intercepted APIs only."""
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
    with opener.open(base+'/src/components/AgentChatWidget.vue') as response: source=response.read().decode()
    vue=next(p for p in re.findall(r'from "([^"]+)"',source) if '/vue.js' in p)
    with opener.open(base+'/src/router/index.js') as response: router_source=response.read().decode()
    router_module=next(p for p in re.findall(r'from "([^"]+)"',router_source) if 'vue-router' in p)
    harness=f'''<html><head><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app"></div><script type="module">
      import {{createApp,h}} from '{vue}';import {{createRouter,createMemoryHistory}} from '{router_module}';
      import App from '/src/App.vue';import Assistant from '/src/views/AIAssistantView.vue';
      const router=createRouter({{history:createMemoryHistory(),routes:[{{path:'/ai-assistant',name:'ai-assistant',component:Assistant}},{{path:'/logs',name:'logs',component:{{render:()=>h('p','日志')}}}},{{path:'/materials',name:'materials',component:{{render:()=>h('p','材料')}}}},{{path:'/assignments',component:{{render:()=>h('p','作业')}}}}]}});
      await router.push('/ai-assistant');createApp(App).use(router).mount('#app');
      </script></body></html>'''
    posts=[];errors=[];chat_ready=asyncio.Event();chat_release=asyncio.Event()

    async def api(route):
        req=route.request;path=urllib.parse.urlparse(req.url).path
        if req.method=='POST':
            posts.append((path,req.post_data_json))
            if path=='/api/agent/chat':
                chat_ready.set();await chat_release.wait();data={'reply':'已按本次请求处理。','session_id':'fixture-session'}
            elif path=='/api/sessions':data={'id':'fixture-new-session'}
            else:raise AssertionError('Shortcut started a business task: '+path)
        elif path.endswith('/history'):data={'messages':[]}
        else:data=[]
        await route.fulfill(content_type='application/json',body=json.dumps(data,ensure_ascii=False))

    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,args=['--no-sandbox'])
        async def open_role(role):
            context=await browser.new_context(viewport={'width':1440,'height':1150})
            user=json.dumps({'user_id':1,'username':'fixture','name':'老师' if role!='student' else '同学','role':role},ensure_ascii=False)
            await context.add_init_script(f'localStorage.setItem("access_token","shortcut-fixture");localStorage.setItem("user_info",JSON.stringify({user}));localStorage.setItem("current_session_id","fixture-session");')
            page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
            await page.route('**/api/**',api)
            await page.route(base+'/assistant-shortcuts-smoke',lambda r:r.fulfill(content_type='text/html; charset=utf-8',body=harness))
            await page.goto(base+'/assistant-shortcuts-smoke')
            await expect(page.locator('.command-options button').first).to_be_enabled()
            return context,page

        context,page=await open_role('teacher')
        await expect(page.locator('.command-options button')).to_have_count(3)
        for name in ['拉取作业并解压','批改作业','查询成绩']:
            await expect(page.get_by_role('button',name=name,exact=True)).to_be_visible()
        assert not posts,'Opening the assistant should not create jobs'
        message=page.get_by_role('textbox',name='消息内容')
        await page.get_by_role('button',name='拉取作业并解压',exact=True).click()
        assert '暂不批改' in await message.input_value()
        assert await message.evaluate('(e)=>e.value.slice(e.selectionStart,e.selectionEnd)')=='课程名称'
        await expect(message).to_be_focused();await page.keyboard.insert_text('分布式计算')
        assert '分布式计算' in await message.input_value()
        await page.get_by_role('button',name='批改作业',exact=True).click()
        assert '目录路径' in await message.input_value()
        await page.get_by_role('button',name='改为拉取、解压并批改',exact=False).click()
        assert '解压后直接批改' in await message.input_value()
        await expect(page.get_by_role('button',name='批改作业',exact=True)).to_have_attribute('aria-pressed','true')
        await page.get_by_role('button',name='改用已有作业目录',exact=False).click()
        assert '目录路径' in await message.input_value()
        assert not posts,'Shortcut selection must only fill a draft'
        diagnostics=Path('/mnt/d/dzq/LLM_TEACH_ASSISTANT/.runtime/diagnostics');diagnostics.mkdir(parents=True,exist_ok=True)
        await page.screenshot(path=str(diagnostics/'assistant-shortcuts-desktop.png'),full_page=True)
        await page.set_viewport_size({'width':390,'height':844})
        await message.scroll_into_view_if_needed()
        await page.screenshot(path=str(diagnostics/'assistant-shortcuts-mobile.png'),full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'),'Mobile overflow'
        for name in ['拉取作业并解压','批改作业','查询成绩']:
            box=await page.get_by_role('button',name=name,exact=True).bounding_box();assert box['width']>=60
        await page.get_by_role('button',name='查询成绩',exact=True).click()
        await page.keyboard.insert_text('20260001')
        assert '20260001' in await message.input_value()
        await page.get_by_role('button',name='发送',exact=True).click()
        await asyncio.wait_for(chat_ready.wait(),5)
        for name in ['拉取作业并解压','批改作业','查询成绩']:
            await expect(page.get_by_role('button',name=name,exact=True)).to_be_disabled()
        assert len(posts)==1 and posts[0][0]=='/api/agent/chat' and '20260001' in posts[0][1]['message']
        chat_release.set();await expect(message).to_be_enabled()
        await page.get_by_role('button',name='新会话',exact=False).click()
        await expect(page.locator('.agent-message')).to_contain_text('拉取并整理作业')
        await context.close()

        admin_context,admin=await open_role('admin')
        await expect(admin.locator('.command-options button')).to_have_count(3)
        await admin_context.close();assert not errors,errors
        await browser.close()
    print('PASS: three teacher/admin entries, extraction-only intent, grading/full-flow switch, editable selection, no automatic sends, loading locks, teacher-only workspace and mobile layout; fixture APIs only.')


asyncio.run(main())
