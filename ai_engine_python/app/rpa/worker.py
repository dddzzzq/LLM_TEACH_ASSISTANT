"""python 提供rpa节点（Tool）"""
import asyncio
import base64
import contextlib
import fcntl
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from playwright.async_api import async_playwright

from .task_controller import Task, TERMINAL
from .tools import tool_instructions
from .browser_stream import serve_browser_stream


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path(os.environ.get('RUNTIME_DIR', ROOT.parent / 'teaching-runtime'))


def control_token():
    path = Path(os.environ.get('RPA_CONTROL_TOKEN_FILE', RUNTIME / '.rpa-control-token'))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as output:
            output.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    return path.read_text().strip()


class Manager:
    def __init__(self, root=RUNTIME / 'rpa-tasks'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.download_root = Path(os.environ.get('RPA_DOWNLOAD_DIR', ROOT / 'downloads'))
        cfg = dotenv_values(ROOT / 'ai_engine_python/app/.env.rpa')
        self.proxy = os.environ.get('RPA_PROXY_URL', cfg.get('RPA_PROXY_URL'))
        self.tasks = {}
        self.playwright = None
        for path in self.root.glob('*.json'):
            try:
                state = json.loads(path.read_text())
                UUID(state['job_id'])
                if state['job_id'] != path.stem or not isinstance(state.get('target'), dict):
                    raise ValueError('Invalid task journal')
                state['status']
            except (ValueError, KeyError, TypeError):
                # One damaged journal must not prevent other users from recovering tasks.
                path.replace(path.with_suffix('.json.corrupt'))
                continue
            if state['status'] not in TERMINAL:
                state.update(status='INTERRUPTED', control_mode='HUMAN',
                             state_version=state.get('state_version', 0) + 1,
                             message='Worker 已重启，浏览器会话丢失；请重新启动并登录，系统将保留导出及文件检查点')
                self.persist(state)
            self.tasks[state['job_id']] = Task(self, state)

    def persist(self, state):
        path = self.root / (state['job_id'] + '.json')
        tmp = path.with_suffix('.tmp')
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as output:
            json.dump(state, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        tmp.replace(path)

    def start(self, job_id, target, restart=False, checkpoint=None):
        UUID(job_id)
        existing = self.tasks.get(job_id)
        if restart and not existing and checkpoint:
            checkpoint = {**checkpoint, 'job_id':job_id, 'target':target, 'status':'INTERRUPTED'}
            existing = Task(self, checkpoint)
        if existing and not restart:
            return existing
        if restart and (not existing or existing.state['status'] not in {'INTERRUPTED','PARTIAL_SUCCESS','FAILED'}):
            raise ValueError('只有中断或失败的任务可以重新启动')
        if sum(t.state['status'] not in TERMINAL for t in self.tasks.values()) >= int(os.environ.get('RPA_MAX_TASKS', 4)):
            raise ValueError('浏览器任务已满，请等待已有任务结束')
        old = existing.state if existing else {}
        state = {**old, 'job_id': job_id, 'target': target, 'status': 'RUNNING', 'stage': 'OPEN_PORTAL',
                 'control_mode': 'AUTO', 'control_epoch': old.get('control_epoch', 0) + 1,
                 'created_at': old.get('created_at', time.time()), 'updated_at': time.time(),
                 'session_started_at': time.time(),
                 'state_version': old.get('state_version', 0) + 1,
                 'auth_status': 'UNKNOWN', 'waiting_reason': '',
                 'message': '正在启动任务浏览器', 'files': [f for f in old.get('files', []) if f['status'] == 'VERIFIED'], 'exports': old.get('exports', [])}
        task = Task(self, state)
        self.tasks[job_id] = task
        self.persist(state)
        task.runner = asyncio.create_task(task.run())
        return task

    async def watchdog(self):
        while True:
            await asyncio.sleep(5)
            for task in list(self.tasks.values()):
                if task.state['status'] in TERMINAL:
                    continue
                human_idle = task.state['control_mode'] == 'HUMAN' and time.monotonic() - task.last_activity > 900
                total = time.time() - task.state.get('session_started_at', task.state['created_at']) > 7200
                if human_idle or total:
                    await task.cancel('INTERRUPTED')
                    task.event('会话等待或任务总时限已到，已关闭浏览器；可重新启动并登录')


@asynccontextmanager
async def lifespan(app):
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (RUNTIME / 'rpa-worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    app.state.token = control_token()
    async with async_playwright() as playwright:
        manager = Manager()
        manager.playwright = playwright
        app.state.manager = manager
        watchdog = asyncio.create_task(manager.watchdog())
        try:
            yield
        finally:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog
            for task in manager.tasks.values():
                if task.state['status'] not in TERMINAL:
                    await task.cancel('INTERRUPTED')
    lock.close()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware('http')
async def internal_auth(request: Request, call_next):
    supplied = request.headers.get('X-RPA-Token', '')
    if not secrets.compare_digest(supplied, request.app.state.token):
        return JSONResponse(status_code=401, content={'error': 'Unauthorized'})
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.exception_handler(ValueError)
async def invalid(request, exc):
    return JSONResponse(status_code=409, content={'error': str(exc)[:200]})


def task_for(request, job_id):
    task = request.app.state.manager.tasks.get(job_id)
    if not task:
        raise HTTPException(404, '任务不存在')
    return task


@app.get('/health')
async def health():
    return {'ok': True, 'protocol_version': 'browser.v1', 'decision_driver': 'go-eino',
            'browser_stream': 'browser.stream.v1'}


@app.get('/capabilities')
async def capabilities():
    return {'protocol_version': 'browser.v1', 'action_schema': json.loads(tool_instructions())}


@app.get('/tasks/{job_id}/observation')
async def observation(job_id: str, request: Request):
    return await task_for(request, job_id).observe()


@app.post('/tasks/{job_id}')
async def start(job_id: str, request: Request):
    payload = await request.json()
    target = payload['target']
    if not all(isinstance(target.get(k), str) and 0 < len(target[k].strip()) <= 255
               for k in ('course_name', 'assignment_name')):
        raise ValueError('课程和作业名称不能为空')
    existing = request.app.state.manager.tasks.get(job_id)
    if payload.get('restart') and existing and existing.runner and not existing.runner.done() and existing.state['status'] in TERMINAL:
        with contextlib.suppress(asyncio.CancelledError):
            await existing.runner
    return request.app.state.manager.start(job_id, target, payload.get('restart', False), payload.get('checkpoint')).state


@app.get('/tasks/{job_id}')
async def get(job_id: str, request: Request):
    task = task_for(request, job_id)
    await task.refresh_login_state()
    return task.state


@app.post('/tasks/{job_id}/{action}')
async def control(job_id: str, action: str, request: Request):
    task = task_for(request, job_id)
    if action == 'action':
        return await task.apply_action(await request.json())
    elif action == 'decision-failed':
        return await task.decision_failed(await request.json())
    elif action == 'pause':
        await task.pause()
    elif action == 'resume':
        await task.resume()
    elif action == 'cancel':
        await task.cancel()
    elif action == 'input':
        await task.browser_input(await request.json())
        return {'ok': True}
    elif action == 'class-scope':
        async with task.lock:
            if task.state['control_mode'] != 'HUMAN' or task.state.get('export_attempted'):
                raise ValueError('只有导出前的人工阶段可以确认范围')
            payload = await request.json()
            if payload.get('mode') not in ('all', 'selected'):
                raise ValueError('请选择班级范围')
            ids = set(payload.get('class_ids', []))
            options = task.state.get('classes', [])
            available = {o['id']: o for o in options}
            if payload['mode'] == 'all' and not any(o['all'] for o in options):
                raise ValueError('当前导出窗口不支持全部班级')
            if payload['mode'] == 'selected' and (not ids or not ids <= available.keys() or any(available[i]['all'] for i in ids)):
                raise ValueError('班级列表已变化或选择无效')
            task.scope_request = payload
            task.scope_request['option_names'] = {o['id']: o['name'] for o in options}
            task.update(requested_scope=payload)
    elif action == 'bind-exports':
        async with task.lock:
            if task.state['control_mode'] != 'HUMAN' or not task.state.get('export_attempted') or task.state.get('exports'):
                raise ValueError('当前不能绑定导出记录')
            refs = set((await request.json()).get('refs', []))
            rows = await task.adapter.export_rows()
            candidates = {r['ref']: r for r in rows if r['ref'] not in task.state.get('export_baseline', []) and not r.get('ambiguous')}
            if not refs or not refs <= candidates.keys():
                raise ValueError('导出记录已变化，请重新核对')
            task.update(exports=[candidates[r] for r in refs], export_candidates=[])
    elif action == 'confirm-target':
        async with task.lock:
            if task.state['control_mode'] != 'HUMAN' or task.state.get('export_attempted'):
                raise ValueError('当前不能重新确认目标')
            payload = await request.json()
            if any(payload.get(k) != task.state['target'][k] for k in ('course_name', 'assignment_name')):
                raise ValueError('确认内容与任务目标不一致')
            if not await task.adapter.export_entry():
                raise ValueError('请先进入目标作业的批阅页面')
            task.adapter.course_confirmed = task.adapter.assignment_confirmed = True
            task.event('用户明确确认当前页面属于任务目标课程与作业')
    else:
        raise HTTPException(404, '未知操作')
    return task.state


@app.get('/tasks/{job_id}/frame')
async def frame(job_id: str, request: Request):
    data, meta, pages = await task_for(request, job_id).frame()
    return {'image': base64.b64encode(data).decode(), 'revision': meta['revision'],
            'page_id': meta['page_id'], 'epoch': meta['epoch'], 'width': 1440, 'height': 1000, 'pages': pages}


@app.websocket('/tasks/{job_id}/stream')
async def browser_stream(job_id: str, socket: WebSocket):
    # HTTP middleware does not protect WebSocket upgrades.
    supplied = socket.headers.get('X-RPA-Token', '')
    if not secrets.compare_digest(supplied, socket.app.state.token):
        await socket.close(code=4401)
        return
    task = socket.app.state.manager.tasks.get(job_id)
    if not task:
        await socket.close(code=4404)
        return
    await socket.accept()
    await serve_browser_stream(socket, task)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('RPA_CONTROL_PORT', 8765)),
                access_log=False, ws_max_size=16 * 1024)
