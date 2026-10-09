"""Shared Chromium frame producer and bounded, ordered manual input streams."""
import asyncio
import base64
import contextlib
import json
import time
from contextlib import asynccontextmanager

from fastapi import WebSocketDisconnect

from .task_controller import TERMINAL


def replace_latest(queue, value):
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(value)


class BrowserFrames:
    """One CDP screencast per task; slow viewers keep only the newest frame."""
    def __init__(self, task):
        self.task = task
        self.viewers = set()
        self.runner = None
        self.lifecycle = asyncio.Lock()

    @asynccontextmanager
    async def subscribe(self):
        queue = asyncio.Queue(maxsize=1)
        async with self.lifecycle:
            self.viewers.add(queue)
            if self.runner is None or self.runner.done():
                self.runner = asyncio.create_task(self.run())
        try:
            yield queue
        finally:
            async with self.lifecycle:
                self.viewers.discard(queue)
                if not self.viewers and self.runner:
                    self.runner.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.runner
                    self.runner = None

    def publish(self, message):
        for viewer in self.viewers:
            replace_latest(viewer, message)

    async def run(self):
        task = self.task
        while True:
            if task.state['status'] in TERMINAL:
                self.publish({'type': 'ended', 'message': '浏览器任务已结束'})
                return
            try:
                await self.capture_page()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never include browser exceptions: they may contain form data or URLs.
                self.publish({'type': 'error', 'message': '浏览器暂未就绪，正在重新连接'})
                await asyncio.sleep(1)

    async def capture_page(self):
        task = self.task
        async with task.lock:
            page, context, epoch = task.page, task.context, task.state['control_epoch']
            if not page or page.is_closed():
                raise ValueError('浏览器当前不可用')
            url = page.url
        session = await context.new_cdp_session(page)
        events = asyncio.Queue(maxsize=1)
        active = True

        async def receive_frame(event):
            with contextlib.suppress(Exception):
                await session.send('Page.screencastFrameAck', {'sessionId': event['sessionId']})
            if active:
                replace_latest(events, event['data'])

        session.on('Page.screencastFrame', receive_frame)
        try:
            await session.send('Page.startScreencast', {
                'format': 'jpeg', 'quality': 55, 'maxWidth': 1440, 'maxHeight': 1000, 'everyNthFrame': 1})
            last_frame = 0
            while task.state['status'] not in TERMINAL:
                if (page is not task.page or page.is_closed() or url != page.url
                        or epoch != task.state['control_epoch']):
                    return
                try:
                    image = await asyncio.wait_for(events.get(), .25)
                except asyncio.TimeoutError:
                    if time.monotonic() - last_frame < 1:
                        continue
                    # Static pages also need fresh metadata for the three-second click window.
                    async with task.frame_lock:
                        data = await page.screenshot(type='jpeg', quality=55, timeout=5000)
                    image = base64.b64encode(data).decode()
                pages = [{'id': str(i), 'title': (await p.title())[:100]}
                         for i, p in enumerate(context.pages) if not p.is_closed()]
                async with task.lock:
                    if (page is not task.page or page.is_closed() or url != page.url
                            or epoch != task.state['control_epoch']):
                        return
                    meta = task.record_frame(page, url, epoch)
                self.publish({'type': 'frame', 'image': image, 'revision': meta['revision'],
                              'page_id': meta['page_id'], 'epoch': epoch, 'pages': pages,
                              'width': 1440, 'height': 1000})
                last_frame = time.monotonic()
                await asyncio.sleep(1 / 30)
        finally:
            active = False
            session.remove_listener('Page.screencastFrame', receive_frame)
            with contextlib.suppress(Exception):
                await session.send('Page.stopScreencast')
            with contextlib.suppress(Exception):
                await session.detach()


class StreamFailure(Exception):
    def __init__(self, message, code=4409):
        self.message, self.code = message, code


async def serve_browser_stream(socket, task):
    owner = object()
    if task.browser_frames is None:
        task.browser_frames = BrowserFrames(task)
    inputs = asyncio.Queue(maxsize=64)
    replies = asyncio.Queue(maxsize=128)
    available = asyncio.Event()
    latest_frame = None

    def reply(message):
        if replies.full():
            raise StreamFailure('连接过慢，请重新连接', 4429)
        replies.put_nowait(message)
        available.set()

    async def receive():
        last_id = 0
        while True:
            raw = await asyncio.wait_for(socket.receive_text(), 40)
            if len(raw.encode()) > 16 * 1024:
                raise StreamFailure('操作参数过大', 4400)
            try:
                message = json.loads(raw)
            except (ValueError, TypeError):
                raise StreamFailure('操作参数无效', 4400)
            if not isinstance(message, dict):
                raise StreamFailure('操作参数无效', 4400)
            if message.get('type') == 'ping':
                reply({'type': 'pong'})
                continue
            seq = message.get('id')
            if (message.get('type') != 'input' or type(seq) is not int
                    or not last_id < seq <= 2**53 - 1 or not isinstance(message.get('action'), dict)):
                raise StreamFailure('操作顺序或参数无效', 4400)
            if inputs.full():
                raise StreamFailure('操作积压，请重新连接', 4429)
            last_id = seq
            inputs.put_nowait(message)

    async def execute():
        while True:
            message = await inputs.get()
            try:
                await task.browser_input(message['action'], owner=owner)
            except (ValueError, KeyError, TypeError, OverflowError):
                raise StreamFailure('页面或控制权已变化，操作未全部完成，请检查画面后重试')
            except Exception:
                raise StreamFailure('操作结果未确认，请检查画面后重试')
            reply({'type': 'input_ack', 'id': message['id']})

    async def frames(queue):
        nonlocal latest_frame
        while True:
            latest_frame = await queue.get()
            available.set()

    async def send():
        nonlocal latest_frame
        while True:
            await available.wait()
            if not replies.empty():
                message = replies.get_nowait()
            else:
                message, latest_frame = latest_frame, None
            if replies.empty() and latest_frame is None:
                available.clear()
            await asyncio.wait_for(socket.send_json(message), 5)
            if message.get('type') == 'ended':
                return

    reply({'type': 'ready'})
    error = None
    async with task.browser_frames.subscribe() as queue:
        workers = [asyncio.create_task(fn()) for fn in (receive, execute, lambda: frames(queue), send)]
        try:
            done, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
            for worker in done:
                try:
                    worker.result()
                except StreamFailure as exc:
                    error = exc
                except (WebSocketDisconnect, asyncio.TimeoutError):
                    pass
                except Exception:
                    error = StreamFailure('浏览器连接中断，请重新连接')
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task.release_pointer(owner), 3)
    with contextlib.suppress(Exception):
        if error:
            await asyncio.wait_for(socket.send_json({'type': 'error', 'message': error.message}), 1)
        await socket.close(code=error.code if error else 1000)
