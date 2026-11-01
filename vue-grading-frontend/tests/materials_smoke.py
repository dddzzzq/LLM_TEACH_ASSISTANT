"""UI checks against fixture APIs only: no real downloads, queue jobs or model calls."""
import asyncio
import copy
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
    with opener.open(base+'/src/components/MaterialBatchPanel.vue') as response: source=response.read().decode()
    vue=next(p for p in re.findall(r'from "([^"]+)"',source) if '/vue.js' in p)
    with opener.open(base+'/src/router/index.js') as response: router_source=response.read().decode()
    router_module=next(p for p in re.findall(r'from "([^"]+)"',router_source) if 'vue-router' in p)
    ready=dict(id='11111111-1111-4111-8111-111111111111',assignment_id=3,assignment_name='第一次作业',class_name='A',created_at='2026-10-18T16:00:00+08:00',status='READY',message='材料已整理，等待老师发起批改',student_count=1,issue_count=0,auto_grade=False,grading_state='NOT_REQUESTED',grading_job_id='',output_dir='D:/dzq/homework/课程/第一次作业/A/批次')
    partial={**ready,'id':'22222222-2222-4222-8222-222222222222','class_name':'B','status':'NEEDS_REVIEW','issue_count':1}
    batches=[ready,partial];posts=[];jobs=[];errors=[];state={'fail':False};regrade_keys=[]
    async def api(route):
        req=route.request;path=urllib.parse.urlparse(req.url).path
        if path.endswith('/attention'):data=[]
        elif req.method=='GET' and path=='/api/materials':data={'items':batches,'grading_jobs':jobs}
        elif req.method=='GET' and path.startswith('/api/materials/'):
            batch=next(b for b in batches if b['id']==path.rsplit('/',1)[1])
            data={'batch':batch,'prepared_dir':batch['output_dir']+'/prepared','manifest':{'issues':[{'path':'20260002_李四.zip','message':'同一学号存在多个提交'}] if batch['issue_count'] else [],'students':[{'student_id':'20260001','student_name':'张三','warnings':[],'files':[{'path':'students/20260001_张三/src/main.py','kind':'code'},{'path':'students/20260001_张三/设计报告.md','kind':'report'}]}]}}
        elif req.method=='POST':
            body=req.post_data_json;posts.append((path,body))
            if path=='/api/materials':
                if state['fail']:
                    await route.fulfill(status=400,content_type='application/json',body=json.dumps({'error':'输出位置不能位于被扫描目录内'}));return
                batch={**ready,'id':f'new-{len(posts)}','status':'PENDING','class_name':'新导入','auto_grade':body.get('auto_grade',False)};batches.append(batch);data=batch
            elif path.endswith('/grade'):
                batch=next(b for b in batches if b['id']==path.split('/')[-2])
                if body.get('regrade'):
                    assert body['previous_job_id']==batch['grading_job_id']
                    regrade_keys.append(req.headers.get('idempotency-key'))
                    assert regrade_keys[-1]
                batch['grading_job_id']='grading-'+batch['id']+('-again' if body.get('regrade') else '')
                jobs.append({'id':batch['grading_job_id'],'status':'PENDING'});data={'message':'材料已确认，等待批改队列'}
            elif path.endswith('/cancel'):
                batch=next(b for b in batches if b['id']==path.split('/')[-2]);batch['status']='CANCELLED';batch['auto_grade']=False;data={'ok':True}
            elif path.endswith('/retry'):
                batch=next(b for b in batches if b['id']==path.split('/')[-2]);batch.update(status='PENDING',auto_grade=False,grading_state='NOT_REQUESTED');data=batch
            else:raise AssertionError('Unexpected mutation: '+path)
        else:data=[]
        await route.fulfill(content_type='application/json',body=json.dumps(data,ensure_ascii=False))
    harness=f'''<html><head><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app"></div><script type="module">
    import {{createApp,h}} from '{vue}';import {{createRouter,createMemoryHistory}} from '{router_module}';
    import App from '/src/App.vue';import Panel from '/src/components/MaterialBatchPanel.vue';
    const router=createRouter({{history:createMemoryHistory(),routes:[{{path:'/materials',name:'materials',component:{{render:()=>h('div',{{class:'surface p-5'}},[h(Panel,{{assignmentId:3}})])}}}},{{path:'/logs',name:'logs',component:{{render:()=>h('p','日志')}}}},{{path:'/assignments/:id',component:{{render:()=>h('p','作业')}}}}]}});await router.push('/materials');createApp(App).use(router).mount('#app');
    </script></body></html>'''
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=await browser.new_context(viewport={'width':1440,'height':1050})
        await context.add_init_script('localStorage.setItem("access_token","materials-fixture");localStorage.setItem("user_info",JSON.stringify({user_id:1,username:"老师",role:"teacher"}));')
        page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('dialog',lambda dialog:dialog.accept())
        await page.route('**/api/**',api);await page.route(base+'/materials-smoke',lambda r:r.fulfill(content_type='text/html; charset=utf-8',body=harness))
        await page.goto(base+'/materials-smoke');await expect(page.locator('.batch-card')).to_have_count(2)
        assert not posts,'Opening materials must never start grading'
        await expect(page.get_by_role('checkbox',name='整理完成后继续批改')).not_to_be_checked()
        cards=page.locator('.batch-card');await expect(cards.nth(1).get_by_role('button',name='开始批改')).to_be_disabled()
        await cards.first.get_by_role('button',name='查看学生与文件清单').click();await expect(page.get_by_role('dialog')).to_be_visible()
        await page.get_by_text('20260001 张三 · 2 个文件',exact=True).click();await expect(page.get_by_text('students/20260001_张三/设计报告.md',exact=False)).to_be_visible();await page.keyboard.press('Escape')
        await page.get_by_label('作业所在目录',exact=True).fill(r'D:\dzq\作业')
        await page.get_by_role('button',name='仅导入并整理',exact=True).click();await expect(cards).to_have_count(3)
        assert posts[-1][1]['auto_grade'] is False and not any(p.endswith('/grade') for p,_ in posts)
        await page.get_by_role('checkbox',name='整理完成后继续批改').check();await page.get_by_role('button',name='整理并批改',exact=True).click();await expect(cards).to_have_count(4)
        assert posts[-1][1]['auto_grade'] is True
        await cards.first.get_by_role('button',name='开始批改',exact=True).click();await expect(cards.first.get_by_text('批改：排队中',exact=False)).to_be_visible();await expect(cards.first.get_by_role('button',name='开始批改')).to_have_count(0)
        await expect(cards.first.get_by_role('button',name='正在批改，请等待',exact=True)).to_be_disabled()
        jobs[0]['status']='SUCCESS'
        previous=ready['grading_job_id']
        await page.get_by_role('button',name='刷新批次',exact=True).click()
        await expect(cards.first.get_by_role('button',name='重新批改',exact=True)).to_be_enabled()
        await cards.first.get_by_role('button',name='重新批改',exact=True).click()
        await expect(cards.first.get_by_role('button',name='正在批改，请等待',exact=True)).to_be_disabled()
        assert posts[-1][1]=={'allow_partial':False,'regrade':True,'previous_job_id':previous}
        assert len(regrade_keys)==1 and ready['grading_job_id']!=previous
        await cards.nth(1).get_by_role('checkbox').check();await expect(cards.nth(1).get_by_role('button',name='开始批改')).to_be_enabled();await cards.nth(1).get_by_role('button',name='开始批改').click();assert posts[-1][1]['allow_partial'] is True
        await cards.nth(2).get_by_role('button',name='取消后续处理').click();await expect(cards.nth(2).get_by_text('已取消',exact=True)).to_be_visible()
        state['fail']=True;await page.get_by_role('button',name='整理并批改',exact=True).click();await expect(page.get_by_role('alert')).to_have_text('输出位置不能位于被扫描目录内')
        batches.append({**ready,'id':'failed-fixture','status':'FAILED','message':'文件处理数量超过30000限制','auto_grade':True,'grading_state':'REQUESTED','grading_job_id':''})
        await page.get_by_role('button',name='刷新批次',exact=True).click();await expect(cards).to_have_count(5)
        await cards.last.get_by_role('button',name='重新解压（不批改）',exact=True).click()
        await expect(cards.last.get_by_text('排队中',exact=True)).to_be_visible()
        assert posts[-1][0]=='/api/materials/failed-fixture/retry' and batches[-1]['auto_grade'] is False
        await expect(page.get_by_role('status')).to_contain_text('无需重新拉取')
        diagnostics=Path('/mnt/d/dzq/LLM_TEACH_ASSISTANT/.runtime/diagnostics');diagnostics.mkdir(exist_ok=True,parents=True)
        await page.screenshot(path=str(diagnostics/'materials-desktop.png'),full_page=True)
        await page.set_viewport_size({'width':390,'height':844});await page.screenshot(path=str(diagnostics/'materials-mobile.png'),full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'),'Mobile page overflow'
        assert not errors,errors
        await browser.close()
    print('PASS: directory preparation defaults to no grading; explicit continuation, manifests, partial consent, single submission, cancellation, errors and mobile layout; fixture APIs only.')

asyncio.run(main())
