"""Real subprocess tests with an isolated fake RPA module, never live services."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HELPER = Path(__file__).resolve().parents[1] / 'service_process.py'
spec = importlib.util.spec_from_file_location('service_process', HELPER)
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)

FAKE_WORKER = '''
import fcntl, os, signal, socket, sys, time
from pathlib import Path
mode=os.environ.get('FIXTURE_MODE','normal')
if mode=='fail': sys.exit(7)
lock=(Path(os.environ['RUNTIME_DIR'])/'rpa-worker.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if mode=='stuck': signal.signal(signal.SIGTERM,signal.SIG_IGN)
if mode=='normal':
    server=socket.socket();server.bind(('127.0.0.1',int(os.environ['FIXTURE_PORT'])));server.listen()
while True: time.sleep(.05)
'''


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = self.root / 'runtime'
        self.runtime.mkdir()
        self.cwd = self.root / 'ai_engine_python'
        module = self.cwd / 'app/rpa/worker.py'
        module.parent.mkdir(parents=True)
        module.write_text(FAKE_WORKER)
        self.service = lifecycle.Service('rpa', self.root, self.runtime)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.env = dict(os.environ, ROOT=str(self.root), RUNTIME_DIR=str(self.runtime),
                        FIXTURE_PORT=str(self.port), PYTHONDONTWRITEBYTECODE='1')
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
        try:
            self.service.stop(.1)
        except lifecycle.ServiceError:
            pass
        self.temp.cleanup()

    def run_helper(self, operation, mode='normal'):
        args = [sys.executable, str(HELPER), operation, 'rpa', '--timeout', '.4']
        if operation == 'start':
            args += ['--port', str(self.port), '--cwd', str(self.cwd), '--',
                     sys.executable, '-u', '-m', 'app.rpa.worker']
        return subprocess.run(args, env=dict(self.env, FIXTURE_MODE=mode),
                              capture_output=True, text=True, timeout=7)

    def spawn_locked_worker(self, mode):
        child = subprocess.Popen([sys.executable, '-u', '-m', 'app.rpa.worker'],
                                 cwd=self.cwd, env=dict(self.env, FIXTURE_MODE=mode),
                                 start_new_session=True)
        self.children.append(child)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            if self.service.lock_owner() == child.pid:
                return child
            time.sleep(.02)
        self.fail('fake worker did not obtain lock')

    def test_stale_pid_cannot_hide_live_lock_or_be_overwritten(self):
        old = self.spawn_locked_worker('stuck')
        self.service.pidfile.write_text('99999999')
        result = self.run_helper('start')
        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(self.service.pidfile.read_text(), '99999999')
        self.assertEqual(self.service.lock_owner(), old.pid)
        result = self.run_helper('stop')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        old.wait(timeout=3)
        self.assertIsNone(self.service.lock_owner())
        self.assertFalse(self.service.pidfile.exists())

    def test_ready_reuse_stop_and_restart(self):
        result = self.run_helper('start')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        pid = int(self.service.pidfile.read_text())
        self.assertEqual(self.service.lock_owner(), pid)
        record = json.loads(self.service.record.read_text())
        self.assertEqual(record['start_ticks'], lifecycle.process(pid)['start_ticks'])
        result = self.run_helper('start')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(int(self.service.pidfile.read_text()), pid)
        result = self.run_helper('stop')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(lifecycle.port_open(self.port))
        result = self.run_helper('start')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(int(self.service.pidfile.read_text()), pid)

    def test_failed_start_does_not_leave_formal_pid(self):
        result = self.run_helper('start', mode='fail')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.service.pidfile.exists())
        self.assertFalse(self.service.starting.exists())
        self.assertFalse(self.service.record.exists())
        self.assertIsNone(self.service.lock_owner())

    def test_unrelated_pid_is_never_signalled(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'],
                                 cwd=self.cwd, start_new_session=True)
        self.children.append(child)
        self.service.pidfile.write_text(str(child.pid))
        for operation in ['start', 'stop']:
            result = self.run_helper(operation)
            self.assertNotEqual(result.returncode, 0)
            self.assertIsNone(child.poll())
            self.assertEqual(self.service.pidfile.read_text(), str(child.pid))

    def test_reused_pid_record_is_rejected_even_with_matching_worker(self):
        child = self.spawn_locked_worker('normal')
        info = lifecycle.process(child.pid)
        self.service.pidfile.write_text(str(child.pid))
        self.service.record.write_text(json.dumps(dict(pid=child.pid, start_ticks=info['start_ticks']-1)))
        result = self.run_helper('stop')
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(child.poll())

    def test_start_allows_proc_identity_to_settle_after_exec(self):
        original = lifecycle.process
        observed = 0

        def transient(pid):
            nonlocal observed
            info = original(pid)
            observed += 1
            if info and observed <= 2:
                return {**info, 'args': ['']}
            return info

        with patch.object(lifecycle, 'process', side_effect=transient), patch.dict(os.environ, self.env):
            self.service.start(self.port, self.cwd, 2, [sys.executable, '-u', '-m', 'app.rpa.worker'])
        self.assertTrue(self.service.pidfile.exists())
        self.assertTrue(lifecycle.port_open(self.port))


if __name__ == '__main__':
    unittest.main()
