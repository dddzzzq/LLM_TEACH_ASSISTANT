"""Exclusive persistent profiles. Authentication secrets never enter task journals."""

import asyncio
import fcntl
import json
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

import httpx

from .errors import RpaIssue


async def take_credentials(ticket):
    if not isinstance(ticket, str) or len(ticket) != 64:
        raise RpaIssue("CREDENTIAL_TICKET_INVALID", "认证票据已失效，请重新认证")
    runtime = Path(
        os.environ.get("RUNTIME_DIR", Path(__file__).resolve().parents[4] / "teaching-runtime")
    )
    socket = os.environ.get("PORTAL_CREDENTIAL_SOCKET", str(runtime / "portal-credentials.sock"))
    transport = httpx.AsyncHTTPTransport(uds=socket)
    async with httpx.AsyncClient(transport=transport, timeout=10, trust_env=False) as client:
        response = await client.post(
            "http://localhost/credentials", headers={"X-Portal-Ticket": ticket}
        )
    if response.status_code != 200:
        raise RpaIssue("CREDENTIAL_TICKET_EXPIRED", "认证票据已失效，请重新认证")
    value = response.json()
    if not value.get("username") or not value.get("password"):
        raise RpaIssue("CREDENTIAL_UNAVAILABLE", "教务凭据不可用")
    return value


class PortalSessions:
    lock_wait_seconds = 15

    def __init__(self, manager, root):
        self.manager = manager
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.version_locks = {}

    def directory(self, connection_id):
        return self.root / str(UUID(connection_id))

    async def fence(self, session):
        directory = self.directory(session["connection_id"])
        generation, version = session.get("profile_generation"), session.get("credential_version")
        if (
            type(generation) is not int
            or type(version) is not int
            or not 1 <= generation < 2**63
            or not 1 <= version < 2**63
        ):
            raise ValueError("教务会话版本无效")
        lock = self.version_locks.setdefault(directory.name, asyncio.Lock())
        async with lock:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = directory / ".fence"
            prior = (
                json.loads(path.read_text()) if path.exists() else {"generation": 0, "version": 0}
            )
            if generation < prior["generation"] or version < prior["version"]:
                raise RpaIssue("SESSION_REVOKED", "教务会话已更新，旧操作不能继续")
            if generation != prior["generation"] or version != prior["version"]:
                tmp = directory / ".fence.tmp"
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as output:
                    json.dump({"generation": generation, "version": version}, output)
                    output.flush()
                    os.fsync(output.fileno())
                tmp.replace(path)

    @asynccontextmanager
    async def open(self, session, downloads):
        await self.fence(session)
        directory = self.directory(session["connection_id"])
        generation = session.get("profile_generation")
        if not isinstance(generation, int) or not 1 <= generation < 2**63:
            raise ValueError("浏览器配置代次无效")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(directory / ".lease", os.O_RDWR | os.O_CREAT, 0o600)
        context = None
        try:
            deadline = time.monotonic() + self.lock_wait_seconds
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RpaIssue(
                            "PORTAL_BUSY", "该教务账号正在处理其他操作，请稍后重试"
                        ) from None
                    await asyncio.sleep(0.1)
            await self.fence(session)
            # Old generations cannot be reopened after an account change or trust revocation.
            for old in directory.iterdir():
                if old.name.isdigit() and old.name != str(generation) and old.is_dir():
                    shutil.rmtree(old)
            profile = directory / str(generation)
            profile.mkdir(mode=0o700, exist_ok=True)
            downloads = Path(downloads)
            downloads.mkdir(parents=True, exist_ok=True, mode=0o700)
            context = await self.manager.playwright.chromium.launch_persistent_context(
                str(profile),
                headless=True,
                viewport={"width": 1440, "height": 1000},
                accept_downloads=True,
                downloads_path=str(downloads),
                args=["--no-sandbox", "--disable-dev-shm-usage"]
                + ([] if self.manager.proxy else ["--no-proxy-server"]),
                proxy={"server": self.manager.proxy} if self.manager.proxy else None,
            )
            yield context
        finally:
            try:
                if context:
                    await asyncio.shield(context.close())
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    async def delete(self, connection_id):
        directory = self.directory(connection_id)
        if not directory.exists():
            return
        fd = os.open(directory / ".lease", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("专用浏览器仍在使用，稍后重试清理") from None
            # Retain the lock inode so another opener can never lock a replacement inode.
            for child in directory.iterdir():
                if child.name in {".lease", ".fence"}:
                    continue
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
