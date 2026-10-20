"""Linux/WSL service lifecycle; called while start.sh/kill.bash hold start.lock."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


class ServiceError(Exception):
    pass


def process(pid):
    try:
        path = Path('/proc') / str(pid)
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=pid, pgid=int(stat[2]), start_ticks=int(stat[19]),
                    cwd=os.readlink(path / 'cwd'),
                    args=(path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (OSError, ValueError, IndexError):
        return None


def group_alive(pgid):
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            info = process(int(path.name))
            if info and info['pgid'] == pgid:
                return True
    return False


def port_open(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.3):
            return True
    except OSError:
        return False


class Service:
    def __init__(self, name, root, runtime):
        self.name, self.root, self.runtime = name, Path(root).resolve(), Path(runtime).resolve()
        self.pidfile = self.runtime / (name + '.pid')
        self.record = self.runtime / (name + '.process.json')
        self.starting = self.runtime / (name + '.starting.json')

    def matches(self, info):
        if not info or info['pgid'] != info['pid']:
            return False
        args = info['args']
        cwd = Path(info['cwd']).resolve()
        if self.name == 'rpa':
            return cwd == self.root / 'ai_engine_python' and any(
                args[i:i+2] == ['-m', 'app.rpa.worker'] for i in range(len(args)-1))
        if self.name == 'ai':
            return cwd == self.root / 'ai_engine_python' and any(
                arg in ('app/grpc_server.py', str(self.root / 'ai_engine_python/app/grpc_server.py')) for arg in args[1:])
        if self.name == 'backend':
            return cwd == self.root / 'backend-go' and args[0] == str(self.runtime / 'grading-gateway')
        if self.name == 'frontend':
            return cwd == self.root / 'vue-grading-frontend' and 'node_modules/vite/bin/vite.js' in args[1:]
        if self.name == 'kafka':
            return cwd == self.runtime and 'kafka.Kafka' in args[1:]
        return False

    def lock_owner(self):
        if self.name != 'rpa':
            return None
        with (self.runtime / 'rpa-worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                st = os.fstat(lock.fileno())
                for line in Path('/proc/locks').read_text().splitlines():
                    fields = line.split()
                    if len(fields) < 6 or fields[1] != 'FLOCK':
                        continue
                    try:
                        major, minor, inode = fields[5].split(':')
                        if (int(major, 16), int(minor, 16), int(inode)) == (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino):
                            return int(fields[4])
                    except ValueError:
                        continue
                raise ServiceError('RPA 单实例锁仍被占用，无法核实持锁进程，未启动新实例')
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return None

    def recorded_process(self):
        owner = self.lock_owner()
        if owner:
            info = process(owner)
            if not self.matches(info):
                raise ServiceError('RPA 持锁进程身份不符，未操作该进程')
            self.validate_record(info)
            return info
        paths = [self.pidfile, self.starting, self.record]
        for path in paths:
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text()) if path != self.pidfile else {'pid': int(path.read_text())}
                pid = int(data['pid'])
                if pid <= 1:
                    raise ValueError()
            except (ValueError, KeyError, TypeError):
                raise ServiceError('进程记录无效，已保留记录，未操作进程')
            info = process(pid)
            if info:
                if not self.matches(info) or ('start_ticks' in data and data['start_ticks'] != info['start_ticks']):
                    raise ServiceError('进程身份或启动时间与记录不符，已保留记录，未操作进程')
                self.validate_record(info)
                return info
            if group_alive(pid):
                raise ServiceError('主进程已退出但进程组仍存在，无法核实身份，未启动新实例')
        return None

    def validate_record(self, info):
        for path in (self.record, self.starting):
            if not path.exists():
                continue
            try:
                saved = json.loads(path.read_text())
            except ValueError:
                raise ServiceError('进程启动记录损坏，未操作进程')
            if saved.get('pid') == info['pid'] and saved.get('start_ticks') != info['start_ticks']:
                raise ServiceError('PID 已被复用或记录不符，未操作该进程')

    def clear_records(self):
        for path in [self.pidfile, self.record, self.starting]:
            path.unlink(missing_ok=True)

    def save_ready(self, info):
        data = {k: info[k] for k in ('pid', 'pgid', 'start_ticks')}
        self.starting.write_text(json.dumps(data))
        os.replace(self.starting, self.record)
        temp = self.pidfile.with_suffix('.pid.tmp')
        temp.write_text(str(info['pid']))
        os.replace(temp, self.pidfile)

    def stop_process(self, info, timeout):
        pid, start = info['pid'], info['start_ticks']
        current = process(pid)
        if current:
            if not self.matches(current) or current['start_ticks'] != start:
                raise ServiceError('停止前进程身份发生变化，未发送信号')
            # Let Python shut down its own Playwright/Chromium children first.
            # Sending TERM to the whole group kills the driver before cleanup.
            print(f'{self.name}：发送 SIGTERM（主进程 {pid}）', flush=True)
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while group_alive(pid) and time.monotonic() < deadline:
            time.sleep(.1)
        if group_alive(pid):
            current = process(pid)
            if current and (not self.matches(current) or current['start_ticks'] != start):
                raise ServiceError('停止等待期间 PID 已变化，未强制终止')
            print(f'{self.name}：退出超时，终止已核实的进程组 {pid}', flush=True)
            os.killpg(pid, signal.SIGKILL)
            deadline = time.monotonic() + 3
            while group_alive(pid) and time.monotonic() < deadline:
                time.sleep(.1)
        if group_alive(pid):
            raise ServiceError('进程组尚未退出，保留进程记录')

    def stop(self, timeout):
        info = self.recorded_process()
        if info:
            self.stop_process(info, timeout)
        if self.lock_owner():
            raise ServiceError('RPA 锁尚未释放，保留进程记录')
        self.clear_records()
        print(f'{self.name}：已停止', flush=True)

    def start(self, port, cwd, timeout, command):
        old = self.recorded_process()
        if old:
            if port_open(port):
                self.save_ready(old)
                print(f'{self.name}：复用已核实的进程 {old["pid"]}', flush=True)
                return
            raise ServiceError(f'旧进程 {old["pid"]} 仍在启动或退出，未覆盖 PID；请先运行停止命令')
        if port_open(port):
            raise ServiceError(f'端口 {port} 已被未登记进程占用，未启动新实例')
        self.clear_records()
        with (self.runtime / (self.name + '.log')).open('ab') as log:
            child = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        info = process(child.pid)
        # /proc can briefly expose the pre-exec argv/working directory while
        # a large executable is being loaded. Verify after that transition.
        identity_deadline = time.monotonic() + min(timeout, 2)
        while child.poll() is None and not self.matches(info) and time.monotonic() < identity_deadline:
            time.sleep(.02)
            info = process(child.pid)
        if not self.matches(info):
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=3)
            raise ServiceError('启动命令与服务身份不符')
        self.starting.write_text(json.dumps({k: info[k] for k in ('pid', 'pgid', 'start_ticks')}))
        deadline = time.monotonic() + timeout
        try:
            while child.poll() is None and time.monotonic() < deadline:
                if port_open(port):
                    current = process(child.pid)
                    if current and current['start_ticks'] == info['start_ticks']:
                        self.save_ready(current)
                        print(f'{self.name}：已就绪（端口 {port}，进程 {child.pid}）', flush=True)
                        return
                time.sleep(.2)
            raise ServiceError(f'启动未就绪，请查看 {self.runtime / (self.name + ".log")}')
        except BaseException:
            self.stop_process(info, 3)
            self.clear_records()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['start', 'stop'])
    parser.add_argument('name', choices=['rpa', 'ai', 'backend', 'frontend', 'kafka'])
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--port', type=int)
    parser.add_argument('--cwd')
    args, command = parser.parse_known_args()
    if command[:1] == ['--']:
        command = command[1:]
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    service = Service(args.name, os.environ['ROOT'], os.environ['RUNTIME_DIR'])
    try:
        if args.operation == 'start':
            if not args.port or not args.cwd or not command:
                parser.error('start requires port, cwd, and command')
            service.start(args.port, args.cwd, args.timeout, command)
        else:
            service.stop(args.timeout)
    except (ServiceError, OSError) as error:
        # Application-owned messages only; do not print command lines/environment.
        message = str(error) if isinstance(error, ServiceError) else type(error).__name__
        print(f'{args.name}：{message}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
