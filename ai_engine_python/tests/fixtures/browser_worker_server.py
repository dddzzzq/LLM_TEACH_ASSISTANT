"""Local integration fixture: real Worker/Chromium, intercepted portal, no accounts/models."""

import io
import os
import zipfile

from app.rpa.task_controller import Task
from app.rpa.worker import app, task_for
from fastapi import Request
from tests.fixtures.export_dialog import PORTAL_EXPORT_OPTIONS
import uvicorn

submissions = 0
archive = io.BytesIO()
with zipfile.ZipFile(archive, "w") as bundle:
    bundle.writestr("20260001-test/report.txt", "integration student answer")
course = """<h1 class=courseName>操作系统</h1><ul><li id=work><h2>第一次作业</h2>
<a onclick="document.querySelector('#work').hidden=true;document.querySelector('#grade').hidden=false">批阅</a></li></ul>
<div id=grade hidden><h2>第一次作业</h2><ul class=morePop><a onclick="document.querySelector('[role=dialog]').hidden=false">导出作业附件</a></ul></div>
<div role=dialog hidden>导出设置<label class=export-range><input type=checkbox>班级A</label>
<label class=export-range><input type=checkbox>班级B</label>
<a class=confirmDown onclick="fetch('/export');this.closest('[role=dialog]').hidden=true;document.querySelector('#downloadcenter').innerHTML='<table><tbody><tr data-id=task-a><td>第一次作业 全部班级 导出成功 <a class=download_ic href=/download>下载</a></td></tr></tbody></table>'">确定</a></div><div id=downloadcenter><p class=empty>暂无下载记录</p></div>"""

semantic = os.environ.get("RPA_FIXTURE_VARIANT") == "semantic"
if os.environ.get("RPA_FIXTURE_VARIANT") == "portal-export":
    start = course.index("<div role=dialog hidden>")
    course = (
        course[:start]
        + "<div role=dialog hidden>选择格式"
        + PORTAL_EXPORT_OPTIONS
        + """
    <a class=confirmDown onclick="if(!document.querySelector('.export-range.all .grade_checked') ||
      !document.querySelector('#attachment-option .grade_checked') ||
      !document.querySelector('#default-naming .grade_checked') || window.namingClicks) return;
      fetch('/export');this.closest('[role=dialog]').hidden=true;
      document.querySelector('#downloadcenter').innerHTML='<table><tbody><tr data-id=task-a><td>第一次作业 全部班级 导出成功 <a class=download_ic href=/download>下载</a></td></tr></tbody></table>'">确定</a></div>
    <div id=downloadcenter><table><tbody><tr data-id=old-task><td>第二次作业 已完成</td></tr></tbody></table></div>"""
    )
if os.environ.get("RPA_FIXTURE_VARIANT") == "nested":
    start = course.index("<ul><li id=work>")
    end = course.index("</li></ul>", start) + len("</li></ul>")
    rows = []
    for title in ("Spark编程实验", "MapReduce编程实验", "第三次作业", "第二次作业", "第一次作业"):
        rows.append(f"""<div class=work-row><div class=title><h3><a href=#>{title}</a></h3></div>
          <div class=actions><button>查看</button><button
          onclick="document.querySelector('#work').hidden=true;document.querySelector('#grade').hidden=false">批阅</button></div></div>""")
    course = course[:start] + "<section id=work>" + "".join(rows) + "</section>" + course[end:]
if os.environ.get("RPA_FIXTURE_VARIANT") == "custom":
    course = course.replace(
        "<label class=export-range><input type=checkbox>班级A</label>",
        """<div class=export-range><span class=grade_check onclick="setTimeout(()=>this.parentElement.classList.add('checked'),100)"></span>班级A</div>""",
    )
    course = course.replace(
        "<label class=export-range><input type=checkbox>班级B</label>",
        '<div class=export-range><span class=grade_check onclick="this.parentElement.classList.add(&quot;checked&quot;)"></span>班级B</div>'
        """<div class=out><span class=grade_check onclick="this.parentElement.classList.add('checked')"></span>导出提交附件</div>"""
        "<style>.grade_check{display:inline-block;width:20px;height:20px}</style>",
    )
if semantic:
    course = """<header><div role=heading aria-level=1>操作系统</div></header>
    <section role=list><article role=listitem id=work><h3>第一次作业</h3>
    <div role=button aria-label="查看提交" onclick="document.querySelector('#work').hidden=true;document.querySelector('#grade').hidden=false">进入详情</div></article></section>
    <section id=grade hidden><div role=heading aria-level=1>第一次作业</div>
    <button aria-label="导出附件" onclick="document.querySelector('[role=dialog]').hidden=false">附件导出</button></section>
    <section role=dialog hidden>导出设置<label><input type=checkbox>班级A</label><label><input type=checkbox>班级B</label>
    <button onclick="fetch('/export');this.closest('[role=dialog]').hidden=true;document.querySelector('#downloadcenter').innerHTML='<table><tbody><tr data-id=task-a><td>第一次作业 全部班级 导出成功 <a class=download_ic href=/download>下载</a></td></tr></tbody></table>'">确定</button>
    </section><div id=downloadcenter><p class=empty>暂无下载记录</p></div>"""


async def fixture_route(self, route):
    global submissions
    url = route.request.url
    if "/download" in url:
        await route.fulfill(
            body=archive.getvalue(),
            content_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="homework.zip"'},
        )
        return
    if "/export" in url:
        submissions += 1
        await route.fulfill(body="ok")
        return
    if "/authserver" in url:
        body = "<input type=password><button onclick=\"location.href='https://learning.xidian.edu.cn/portal?ok=1'\">登录</button>"
    elif "/course" in url:
        body = course
    elif "?ok=1" in url:
        body = '<a>个人空间</a><div cname="操作系统"><a target=_blank href="https://learning.xidian.edu.cn/course">操作系统</a></div>'
        if semantic:
            body = """<nav><div role=link>个人空间</div></nav><section role=list><article role=listitem>
                <h3>操作系统</h3><span>2026秋季</span><button aria-label="进入课程"
                onclick="window.open('https://learning.xidian.edu.cn/course','_blank')">进入</button></article></section>"""
    else:
        body = '<a class=denglu href="https://ids.xidian.edu.cn/authserver/login">登录</a>'
    await route.fulfill(content_type="text/html; charset=utf-8", body=body)


Task.route = fixture_route


@app.post("/fixture/{job_id}/login")
async def fixture_login(job_id: str, request: Request):
    task = task_for(request, job_id)
    await task.page.locator("button").click()
    await task.page.wait_for_url("**/portal?ok=1")
    return {"ok": True}


@app.get("/fixture/submissions")
async def count():
    return {"count": submissions}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ["RPA_FIXTURE_PORT"]),
        access_log=False,
        log_level="error",
    )
