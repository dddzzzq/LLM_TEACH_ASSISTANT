"""Generate synthetic fixtures only; py7zr is not a runtime/test dependency.

Run with uv run --no-project --with py7zr==1.0.0 generate_7z.py.
The checked-in base64 files let Go tests run without Python or 7-Zip.
"""
import base64
import io
import os
from pathlib import Path
import tempfile
import zipfile
from unittest.mock import patch

import py7zr

ROOT = Path(__file__).resolve().parent


def make(name, entries, **kwargs):
    buffer = io.BytesIO()
    with py7zr.SevenZipFile(buffer, "w", **kwargs) as archive:
        for path, content in entries:
            archive.writestr(content, path)
    data = buffer.getvalue()
    (ROOT / (name + ".7z.b64")).write_text(base64.b64encode(data).decode() + "\n")
    return data


make("submission", [
    ("src/main.py", "print('程序')\n"),
    ("test/main.py", "print('test')\n"),
    ("设计报告.md", "分析与设计\n"),
    ("empty.txt", ""),
    ("project/node_modules/ignored.js", "dependency"),
    ("project/.venv/ignored.py", "dependency"),
])

nested = io.BytesIO()
with zipfile.ZipFile(nested, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("src/main.py", "print('nested')\n")
make("nested", [("project.zip", nested.getvalue()), ("报告.md", "nested report")])
make("encrypted_headers", [("main.py", "secret")], password="fixture-only", header_encryption=True)
make("encrypted_data", [("main.py", "secret")], password="fixture-only", header_encryption=False)
make("duplicate", [("main.py", "first"), ("main.py", "second")])

# Deliberately unsafe names are test inputs for the Go path boundary.
with patch("py7zr.py7zr.check_archive_path", return_value=True):
    make("traversal", [("../escape.txt", "must not escape")])

raw = bytearray(make("checksum", [("main.py", "print('checksum')\n")], filters=[{"id": py7zr.FILTER_COPY}]))
raw[32] ^= 1  # Corrupt file data while retaining valid archive/header checksums.
(ROOT / "corrupt_data.7z.b64").write_text(base64.b64encode(raw).decode() + "\n")
(ROOT / "checksum.7z.b64").unlink()

with tempfile.TemporaryDirectory() as tmp:
    source = Path(tmp) / "oversize.py"
    with source.open("wb") as stream:
        stream.truncate((256 << 20) + 1)
    buffer = io.BytesIO()
    with py7zr.SevenZipFile(buffer, "w", filters=[{"id": py7zr.FILTER_LZMA2, "preset": 1}]) as archive:
        archive.write(source, "oversize.py")
    (ROOT / "oversize.7z.b64").write_text(base64.b64encode(buffer.getvalue()).decode() + "\n")

    # Real link metadata, never an actual link in a test's output directory.
    target = Path(tmp) / "main.py"
    target.write_text("print('link fixture')\n")
    link = Path(tmp) / "linked.py"
    os.symlink("main.py", link)
    buffer = io.BytesIO()
    with py7zr.SevenZipFile(buffer, "w", dereference=False) as archive:
        archive.write(target, "main.py")
        archive.write(link, "linked.py")
    (ROOT / "symlink.7z.b64").write_text(base64.b64encode(buffer.getvalue()).decode() + "\n")

print("Generated", len(list(ROOT.glob("*.7z.b64"))), "synthetic 7z fixtures")
