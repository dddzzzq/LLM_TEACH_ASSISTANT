"""Local integration fixture: real Worker/Chromium, intercepted portal, no accounts/models."""
import io
import os
import zipfile

from app.rpa.task_controller import Task
from app.rpa.worker import app, task_for
from fastapi import Request
import uvicorn

submissions = 0
archive = io.BytesIO()
with zipfile.ZipFile(archive, 'w') as bundle:
    bundle.writestr('20260001-test/report.txt', 'integration student answer')
course = '''<h1 class=courseName>操作系统</h1><ul><li id=work><h2>第一次作业</h2>
<a onclick="document.querySelector('#work').hidden=true;document.querySelector('#grade').hidden=false">批阅</a></li></ul>
<div id=grade hidden><h2>第一次作业</h2><ul class=morePop><a onclick="document.querySelector('[role=dialog]').hidden=false">导出作业附件</a></ul></div>
<div role=dialog hidden>导出设置<label class=export-range><input type=checkbox>班级A</label>
<label class=export-range><input type=checkbox>班级B</label>
<a class=confirmDown onclick="fetch('/export');this.closest('[role=dialog]').hidden=true;document.querySelector('#downloadcenter').innerHTML='<table><tbody><tr data-id=task-a><td>第一次作业 班级A 导出成功 <a class=download_ic href=/download>下载</a></td></tr></tbody></table>'">确定</a></div><div id=downloadcenter></div>'''


async def fixture_route(self, route):
    global submissions
    url = route.request.url
    if '/download' in url:
        await route.fulfill(body=archive.getvalue(), content_type='application/zip',
                            headers={'Content-Disposition': 'attachment; filename="homework.zip"'})
        return
    if '/export' in url:
        submissions += 1
        await route.fulfill(body='ok')
        return
    if '/authserver' in url:
        body = '<input type=password><button onclick="location.href=\'https://learning.xidian.edu.cn/portal?ok=1\'">登录</button>'
    elif '/course' in url:
        body = course
    elif '?ok=1' in url:
        body = '<a>个人空间</a><div cname="操作系统"><a target=_blank href="https://learning.xidian.edu.cn/course">操作系统</a></div>'
    else:
        body = '<a class=denglu href="https://ids.xidian.edu.cn/authserver/login">登录</a>'
    await route.fulfill(content_type='text/html; charset=utf-8', body=body)


Task.route = fixture_route


@app.post('/fixture/{job_id}/login')
async def fixture_login(job_id: str, request: Request):
    task = task_for(request, job_id)
    await task.page.locator('button').click()
    await task.page.wait_for_url('**/portal?ok=1')
    return {'ok': True}


@app.get('/fixture/submissions')
async def count():
    return {'count': submissions}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ['RPA_FIXTURE_PORT']), access_log=False, log_level='error')
