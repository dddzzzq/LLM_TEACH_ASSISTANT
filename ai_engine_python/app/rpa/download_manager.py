import asyncio
import hashlib
import re
import zipfile
import time
from pathlib import Path


def safe_filename(name):
    return re.sub(r'[\\/\x00-\x1f]', '_', name).strip('. ')[:180] or 'homework.zip'


def verify_zip(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError('下载文件为空')
    with zipfile.ZipFile(path) as archive:
        if not archive.infolist():
            raise ValueError('压缩包没有学生附件')
        if archive.testzip() is not None:
            raise ValueError('压缩包校验失败')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'size': path.stat().st_size,
            'sha256': digest.hexdigest(), 'status': 'VERIFIED'}


async def save_download(download, directory, export_ref, progress=None, stall_seconds=120):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = safe_filename(download.suggested_filename)
    path = directory / (hashlib.sha256(export_ref.encode()).hexdigest()[:12] + '-' + name)
    partial = path.with_suffix(path.suffix + '.part')
    transfer = asyncio.create_task(download.save_as(str(partial)))
    try:
        start, last_change, received = time.monotonic(), time.monotonic(), 0
        while not transfer.done():
            await asyncio.wait({transfer}, timeout=1)
            current = progress() if progress else None
            if current is not None and current > received:
                received, last_change = current, time.monotonic()
            if progress and time.monotonic() - last_change > stall_seconds:
                raise TimeoutError('文件传输停滞')
            if time.monotonic() - start > 1800:
                raise TimeoutError('文件传输超过总时限')
        await transfer
        if await download.failure():
            raise ValueError('平台文件传输失败')
        result = await asyncio.to_thread(verify_zip, partial)
        partial.replace(path)
        return {**result, 'path': str(path.resolve()), 'filename': name, 'export_ref': export_ref}
    except BaseException:
        await download.cancel()
        transfer.cancel()
        try:
            await transfer
        except BaseException:
            pass
        partial.unlink(missing_ok=True)
        raise
