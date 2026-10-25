"""Drive actual Vue settings/workbench against synthetic API fixtures only."""
import asyncio
import io
import json
import os
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from playwright.async_api import async_playwright, expect


async def main():
    base = os.environ.get("RPA_UI_BASE_URL", "http://127.0.0.1:6017")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + "/src/views/PortalSettingsView.vue") as response:
        vue_source = response.read().decode()
    vue = next(p for p in re.findall(r'from "([^"]+)"', vue_source) if "/vue.js" in p)
    with opener.open(base + "/src/router/index.js") as response:
        router_source = response.read().decode()
    router_module = next(p for p in re.findall(r'from "([^"]+)"', router_source) if "vue-router" in p)
    flow_id = "11111111-1111-4111-8111-111111111111"
    courses = [{"id":"c1","title":"同名课程","term":"2025-2026第一学期","snapshot_id":"s1"}, {"id":"c2","title":"同名课程","term":"2026-2027第一学期","snapshot_id":"s1"}]
    connection = {"id":"connection-1","account_mask":"te****st","trust_device":True,"active_snapshot_id":"s1","last_synced_at":"2026-10-23T12:00:00+08:00","last_authenticated_at":"2026-10-23T11:00:00+08:00"}
    state = {"status":"READY","operation":{"id":"sync-first","type":"CATALOG_SYNC","status":"SUCCEEDED","message":"目录已更新"},"snapshot":"s1"}
    calls, errors = [], []
    async def api(route):
        request = route.request
        path = urllib.parse.urlsplit(request.url).path
        raw = (request.post_data_buffer or b"").decode("utf-8", errors="replace")
        body = json.loads(raw) if request.headers.get("content-type", "").startswith("application/json") and raw else None
        if request.method != "GET": calls.append((request.method,path,body,raw))
        status = 200
        if path == "/api/profile": data = {"id":7,"username":"teacher","name":"测试教师","role":"teacher"}
        elif path == "/api/portal/connection" and request.method == "GET":
            data = {"connection":{**connection,"active_snapshot_id":state["snapshot"]},"status":state["status"],"has_password":True,"credential_storage_ready":True,"operation":state["operation"]}
        elif path == "/api/portal/connection" and request.method == "PUT":
            state["status"] = "INITIALIZING"; state["operation"] = {"id":"auth-first","type":"AUTH_BOOTSTRAP","status":"WAITING_OTP","message":"请获取并填写本人收到的验证码"}
            data = {"operation_id":"auth-first","operation":state["operation"]}; status = 202
        elif path.endswith("/events"): data = []
        elif path == "/api/rpa/jobs": data = []
        elif path.startswith("/api/rpa/jobs/"):
            task_id = path.split("/")[4]
            current = state["operation"]
            if path.endswith("/resume"):
                state["status"] = "READY"; state["snapshot"] = "s2"
                state["operation"] = {"id":"sync-second","type":"CATALOG_SYNC","status":"SUCCEEDED","message":"Catalog read complete"}
            data = {"job_id":task_id,"task_type":current["type"],"status":"WAITING_USER" if current["status"]=="WAITING_OTP" else current["status"],"stage":"AUTHENTICATE" if current["type"]=="AUTH_BOOTSTRAP" else "DONE","control_mode":"HUMAN" if current["status"]=="WAITING_OTP" else "AUTO","control_epoch":1,"waiting_reason":"LOGIN_CODE","message":current["message"],"target":{"course_name":"Portal","assignment_name":"Authentication"},"grading_files":[],"material_batches":[]}
        elif path == "/api/portal/catalog-syncs":
            state["snapshot"] = "s3"; state["operation"] = {"id":"sync-third","type":"CATALOG_SYNC","status":"SUCCEEDED","message":"目录已更新"}; data = {"operation_id":"sync-third"}; status = 202
        elif path == "/api/portal/courses": data = [{**course,"snapshot_id":state["snapshot"]} for course in courses]
        elif path.startswith("/api/portal/courses/"):
            idx = path.split("/")[-2][-1]
            data = {"course":courses[int(idx)-1],"snapshot_id":state["snapshot"],"homeworks":[{"id":"h"+idx,"title":"第一次作业"}],"classes":[]}
        elif path == "/api/grading/workflows" and request.method == "POST": data = {"id":flow_id,"status":"QUEUED"}; status = 202
        elif path == "/api/grading/workflows/"+flow_id:
            data = {"workflow":{"id":flow_id,"status":"RUNNING"},"items":[{"id":"item1","course_name":"同名课程","homework_name":"第一次作业","class_name":"全部班级","assignment_id":400,"status":"RUNNING","message":"正在整理和评分","rpa_job_id":""}]}
        elif path == "/assignments/" and request.method == "POST": data = {"id":400}; status = 201
        elif path == "/assignments/400/submit": data = {"batch_id":"upload-batch","outcome":"accepted"}; status = 202
        elif path == "/api/materials" and request.method == "POST": data = {"id":"directory-batch","status":"PENDING"}; status = 202
        elif path == "/api/materials": data = {"items":[],"grading_jobs":[]}
        else: raise AssertionError("Unexpected API: " + request.method + " " + path)
        await route.fulfill(status=status,content_type="application/json",body=json.dumps(data,ensure_ascii=False))
    async def socket(ws):
        def message(raw):
            value = json.loads(raw)
            if value['type']=='auth':
                ws.send(json.dumps({'type':'ready'}))
                ws.send(json.dumps({'type':'frame','image':'/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=','revision':1,'width':1440,'height':1000,'page_id':'0','epoch':1,'pages':[{'id':'0','title':'Portal'}]}))
            elif value['type']=='input':
                assert value['action']['text']=='123456'
                ws.send(json.dumps({'type':'input_ack','id':value['id']}))
        ws.on_message(message)
    harness = f'''<html><head><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app"></div><script type="module">
    import {{createApp,h}} from '{vue}'; import {{createRouter,createMemoryHistory,RouterView}} from '{router_module}';
    import Settings from '/src/views/PortalSettingsView.vue'; import Workbench from '/src/views/HomeworkGradingView.vue';
    localStorage.setItem('access_token','fixture-token');localStorage.setItem('user_info',JSON.stringify({{user_id:7,role:'teacher',username:'teacher'}}));
    const router=createRouter({{history:createMemoryHistory(),routes:[{{path:'/settings/portal',component:Settings}},{{path:'/grade-homework',component:Workbench}},{{path:'/assignments/:id',component:{{render:()=>h('p','作业结果')}}}}]}});
    window.fixtureRouter=router;await router.push('/settings/portal');const app=createApp({{render:()=>h(RouterView)}});app.use(router);app.mount('#app');</script></body></html>'''
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width":1280,"height":1000})
        page.on("pageerror",lambda error:errors.append(str(error)))
        await page.route("**/api/**",api)
        await page.route("**/assignments/**",api)
        await page.route(base+"/portal-smoke",lambda route:route.fulfill(content_type="text/html; charset=utf-8",body=harness))
        await page.route_web_socket("**/api/rpa/jobs/*/stream", socket)
        await page.goto(base+"/portal-smoke")
        await expect(page.get_by_text("已同步 2 门课程。更新课程只刷新目录。",exact=True)).to_be_visible()
        await page.get_by_label("教师教务账号",exact=True).fill("teacher-test")
        await page.get_by_label("教务密码",exact=True).fill("test-only-password")
        await page.get_by_role("button",name="保存并连接",exact=True).click()
        await expect(page.get_by_role('region', name='任务浏览器', exact=True)).to_be_visible()
        await expect(page.get_by_label('填写到网页当前输入框的内容', exact=True)).to_be_visible()
        assert await page.get_by_label("教务密码",exact=True).input_value() == ""
        await page.get_by_label('填写到网页当前输入框的内容',exact=True).fill('123456')
        await page.get_by_role('button',name='填写并清空',exact=True).click()
        await page.get_by_role('button',name='交还自动化',exact=True).click()
        await expect(page.get_by_text('Catalog read complete',exact=True).first).to_be_visible()
        assert not any('/auth-runs/' in path or path.endswith('/browser') for _,path,_,_ in calls)
        assert not await page.evaluate("JSON.stringify(localStorage).includes('test-only-password') || JSON.stringify(localStorage).includes('123456')")
        await page.get_by_role("button",name="更新课程",exact=True).click()
        await expect(page.get_by_text("目录已更新",exact=True).first).to_be_visible()
        assert not any('/grading/' in path for _,path,_,_ in calls),"Refreshing courses submitted grading"
        diagnostics=Path('/mnt/d/dzq/LLM_TEACH_ASSISTANT/.runtime/diagnostics');diagnostics.mkdir(parents=True,exist_ok=True)
        await page.screenshot(path=str(diagnostics/'portal-settings-desktop.png'),full_page=True)
        await page.evaluate("window.fixtureRouter.push('/grade-homework')")
        picker=page.get_by_role('region',name='官网课程和作业选择')
        await page.get_by_label("课程",exact=True).select_option('c1')
        await page.get_by_label("作业",exact=True).select_option(['h1'])
        assert await picker.get_by_role('checkbox').count() == 0
        await page.get_by_role('button',name='加入待批改列表',exact=True).click()
        await page.get_by_label("课程",exact=True).select_option('c2')
        await page.get_by_label("作业",exact=True).select_option(['h2'])
        assert await picker.get_by_role('checkbox').count() == 0
        await page.get_by_role('button',name='加入待批改列表',exact=True).click()
        await expect(page.locator('.portal-picker li')).to_have_count(2)
        await page.get_by_label('题目要求',exact=True).fill('实现课程要求')
        await page.get_by_label('评分标准（JSON）',exact=True).fill('{"正确性":100}')
        await page.get_by_role('button',name='开始批改',exact=True).click()
        await expect(page.get_by_text('正在整理和评分',exact=False)).to_be_visible()
        submitted=next(body for method,path,body,_ in calls if path=='/api/grading/workflows')
        assert submitted['snapshot_id']=='s3' and submitted['items']==[{'homework_id':'h1'},{'homework_id':'h2'}]
        assert 'course_name' not in submitted and 'password' not in submitted
        await page.get_by_role('radio',name='上传压缩包',exact=True).check()
        for label,value in [('课程名称','本地课程'),('班级名称','本地班级'),('作业名称','本地作业')]: await page.get_by_label(label,exact=True).fill(value)
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w') as output:output.writestr('20260001_张三/report.txt','test answer')
        await page.get_by_label('上传作业压缩包',exact=True).set_input_files({'name':'homework.zip','mimeType':'application/zip','buffer':archive.getvalue()})
        await page.get_by_role('button',name='开始批改',exact=True).click()
        await expect(page.get_by_text('材料已受理，整理完成后自动批改',exact=True)).to_be_visible()
        upload=next(raw for _,path,_,raw in calls if path=='/assignments/400/submit')
        assert 'auto_grade' in upload and 'true' in upload
        await page.get_by_role('radio',name='指定目录',exact=True).check()
        await page.get_by_label('作业所在目录',exact=False).first.fill(r'D:\dzq\homework\测试作业')
        await page.get_by_role('button',name='开始批改',exact=True).click()
        await page.wait_for_timeout(150)
        directory=next(body for _,path,body,_ in calls if path=='/api/materials' and body)
        assert directory['auto_grade'] is True and directory['assignment_id']==400 and 'source_path' in directory
        await page.set_viewport_size({'width':390,'height':844})
        await page.screenshot(path=str(diagnostics/'portal-workbench-mobile.png'),full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),'Mobile workbench overflow'
        await page.evaluate("window.fixtureRouter.push('/settings/portal')")
        await expect(page.get_by_role('button',name='保存并连接',exact=True)).to_be_visible()
        await page.screenshot(path=str(diagnostics/'portal-settings-mobile.png'),full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),'Mobile settings overflow'
        assert not errors,errors
        await browser.close()
    print('PASS: settings with reused RPA browser/input controls, directory refresh, same-name catalog selection, structured workflow, ZIP/directory automatic grading and mobile layouts; fixture APIs only.')


asyncio.run(main())
