"""Exercise actual App/router, teacher-only navigation and rubric editing with fixture APIs."""
import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from playwright.async_api import async_playwright, expect

async def main():
    base=os.environ.get('RPA_UI_BASE_URL','http://127.0.0.1:6017')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base+'/src/views/PortalSettingsView.vue') as response: source=response.read().decode()
    vue=next(path for path in re.findall(r'from "([^"]+)"',source) if '/vue.js' in path)
    errors=[];posts=[]
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,args=['--no-sandbox'])
        async def open_role(role):
            context=await browser.new_context(viewport={'width':1440,'height':1100})
            async def api(route):
                request=route.request;path=urllib.parse.urlsplit(request.url).path
                if request.method=='POST': posts.append((path,request.post_data_json))
                if path=='/api/profile': data={'id':7,'username':'fixture-teacher','name':'测试教师','role':role}
                elif path=='/api/sessions': data=[{'id':'fixture-session','title':'作业讨论'}] if request.method=='GET' else {'id':'fixture-session'}
                elif path.endswith('/history'): data={'messages':[]}
                elif path.endswith('/runs'): data={'items':[]}
                elif path=='/api/portal/connection': data={'status':'READY','connection':{'active_snapshot_id':'s1'},'operation':None}
                elif path=='/api/portal/courses': data=[{'id':'c1','title':'移动互联网导论','term':'本学期','snapshot_id':'s1'}]
                elif path.startswith('/api/portal/courses/'): data={'course':{'id':'c1','title':'移动互联网导论'},'snapshot_id':'s1','homeworks':[{'id':'h1','title':'编程小作业1'}],'classes':[]}
                elif path=='/api/grading/workflows': data={'id':'11111111-1111-4111-8111-111111111111','status':'QUEUED'}
                elif path.startswith('/api/grading/workflows/'): data={'workflow':{'status':'QUEUED'},'items':[]}
                elif path=='/api/materials': data={'items':[],'grading_jobs':[]}
                elif path.startswith('/assignments/'): data=[]
                else: data=[]
                await route.fulfill(content_type='application/json',body=json.dumps(data,ensure_ascii=False))
            harness=f'''<html><head><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id=app></div><script type=module>
              import {{createApp}} from '{vue}'; import App from '/src/App.vue'; import router from '/src/router/index.js';
              localStorage.setItem('access_token','fixture-token'); localStorage.setItem('user_info',JSON.stringify({{user_id:7,username:'fixture-teacher',role:'{role}',name:'测试教师'}}));
              localStorage.setItem('current_session_id','fixture-session');window.fixtureRouter=router;
              await router.replace('/ai-assistant'); createApp(App).use(router).mount('#app');
              </script></body></html>'''
            page=await context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
            await page.route('**/api/**',api);await page.route('**/assignments/**',api)
            await page.route(base+'/teacher-smoke',lambda route:route.fulfill(content_type='text/html; charset=utf-8',body=harness))
            await page.goto(base+'/teacher-smoke')
            return context,page
        context,page=await open_role('teacher')
        navigation=page.locator('.sidebar-navigation')
        await expect(navigation.get_by_role('link')).to_have_count(2)
        await expect(navigation.get_by_role('link',name='AI 助手',exact=True)).to_be_visible()
        await expect(navigation.get_by_role('link',name='作业批改',exact=True)).to_be_visible()
        await expect(page.get_by_role('heading',name='AI 助手',exact=True)).to_be_visible()
        assert not re.search('试卷|主观题|考试',await page.locator('body').inner_text())
        await navigation.get_by_role('link',name='作业批改',exact=True).click()
        await expect(page.get_by_role('heading',name='作业批改',exact=True)).to_be_visible()
        await expect(page.get_by_text('从第一份作业开始',exact=True)).to_be_visible()
        await page.get_by_role('link',name='导入与自动批改',exact=True).click()
        for label in ['上传压缩包','指定目录','选择官网作业']:
            await expect(page.get_by_role('radio',name=label,exact=True)).to_be_visible()
        await page.screenshot(path='/mnt/d/dzq/LLM_TEACH_ASSISTANT/.runtime/diagnostics/teacher-workspace-desktop.png',full_page=True)
        await page.get_by_label('课程',exact=True).select_option('c1')
        await page.get_by_label('作业',exact=True).select_option('h1')
        await page.get_by_role('button',name='加入待批改列表',exact=True).click()
        await expect(page.get_by_text('全部班级',exact=False).first).to_be_visible()
        await page.get_by_label('题目要求',exact=True).fill('实现移动互联网编程要求')
        await page.get_by_label('评分项1分值',exact=True).fill('70')
        await expect(page.get_by_text('当前合计 110 分。',exact=False)).to_be_visible()
        await page.get_by_role('button',name='编辑 JSON（高级）',exact=True).click()
        assert json.loads(await page.get_by_label('评分标准（JSON）',exact=True).input_value())['正确性']['score']==70
        await page.get_by_label('评分标准（JSON）',exact=True).fill('invalid-json')
        await page.get_by_role('button',name='返回表格编辑',exact=True).click()
        await expect(page.get_by_role('alert')).to_contain_text('无法转换')
        await page.get_by_label('评分标准（JSON）',exact=True).fill('{"功能正确性":{"description":"完成全部功能","score":100}}')
        await page.get_by_role('button',name='返回表格编辑',exact=True).click()
        await expect(page.get_by_label('评分项1名称',exact=True)).to_have_value('功能正确性')
        diagnostics=Path('/mnt/d/dzq/LLM_TEACH_ASSISTANT/.runtime/diagnostics')
        await page.screenshot(path=str(diagnostics/'teacher-homework-desktop.png'),full_page=True)
        await page.get_by_role('button',name='开始批改',exact=True).click()
        await expect(page.get_by_text('批改流程已受理',exact=False)).to_be_visible()
        submitted=next(body for path,body in posts if path=='/api/grading/workflows')
        assert submitted['items']==[{'homework_id':'h1'}]
        assert submitted['rubric']=={'功能正确性':{'description':'完成全部功能','score':100}}
        for path in ['/assignments','/assignments/new']:
            await page.evaluate('(path)=>window.fixtureRouter.push(path)',path)
            await expect(navigation.get_by_role('link',name='作业批改',exact=True)).to_have_class(re.compile('is-active'))
        await page.evaluate("window.fixtureRouter.push('/exams/1')")
        await expect(page.get_by_role('heading',name='作业批改',exact=True)).to_be_visible()
        await page.set_viewport_size({'width':390,'height':844})
        await page.screenshot(path=str(diagnostics/'teacher-assistant-mobile.png'),full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Mobile overflow'
        await page.evaluate("window.fixtureRouter.push('/grade-homework')")
        await expect(page.get_by_role('button',name='开始批改',exact=True)).to_be_visible()
        await page.screenshot(path=str(diagnostics/'teacher-homework-mobile.png'),full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Mobile grading overflow'
        await context.close()
        context,page=await open_role('admin')
        await expect(page.locator('.sidebar-navigation a')).to_have_count(2)
        await context.close()
        context,page=await open_role('student')
        await expect(page.get_by_role('button',name='登录',exact=True)).to_be_visible()
        await expect(page.get_by_role('alert')).to_contain_text('教师工作区')
        assert await page.locator('.sidebar-navigation').count()==0
        await page.get_by_role('button',name='注册新账户',exact=True).click()
        assert await page.locator('#reg-role').count()==0
        await expect(page.get_by_text('新账户将注册为教师，无需选择角色。',exact=True)).to_be_visible()
        await context.close();await browser.close()
    assert not errors,errors
    print('PASS: real teacher router, two main entries, preserved homework sources, rubric table/JSON roundtrip, all-class workflow, supporting pages, removed exam routes, teacher-only login and mobile layouts.')

asyncio.run(main())
