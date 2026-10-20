import asyncio
import fcntl
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from app.rpa import worker


class WorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_exception_releases_single_instance_lock(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = Path(root)

            class BrokenPlaywright:
                async def __aenter__(self):
                    raise RuntimeError('synthetic driver failure')

                async def __aexit__(self, *args):
                    pass

            with patch.object(worker, 'RUNTIME', runtime), patch.object(worker, 'control_token', return_value='fixture'), patch.object(worker, 'async_playwright', return_value=BrokenPlaywright()):
                with self.assertRaisesRegex(RuntimeError, 'synthetic driver'):
                    async with worker.lifespan(types.SimpleNamespace(state=types.SimpleNamespace())):
                        self.fail('startup unexpectedly succeeded')
            with (runtime / 'rpa-worker.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)

    async def test_shutdown_marks_interrupted_and_does_not_wait_forever(self):
        class SlowTask:
            state = {'status': 'RUNNING'}

            async def cancel(self, status):
                self.state['status'] = status
                await asyncio.Event().wait()

        task = SlowTask()
        manager = types.SimpleNamespace(tasks={'fixture': task})
        watchdog = asyncio.create_task(asyncio.Event().wait())
        await asyncio.wait_for(worker.shutdown_tasks(manager, watchdog, timeout=.05), .5)
        await asyncio.sleep(0)
        self.assertEqual(task.state['status'], 'INTERRUPTED')
        self.assertTrue(watchdog.cancelled())


if __name__ == '__main__':
    unittest.main()
