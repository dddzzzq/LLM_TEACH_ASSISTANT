"""Synthetic native-worker protocol for concurrency and cancellation tests."""

import json
import os
from pathlib import Path
import time


def fixture_ocr_worker(connection, settings):
    try:
        while True:
            path = connection.recv()
            if path is None:
                connection.send({"ok": True, "results": []})
                continue
            value = Path(path).read_text() if Path(path).is_file() else path
            if value == "crash":
                os._exit(3)
            if value == "error":
                connection.send({"ok": False, "error": "fixture failure"})
                continue
            started = time.monotonic()
            time.sleep(10 if value == "slow" else 0.1)
            text = json.dumps({"pid": os.getpid(), "start": started, "end": time.monotonic()})
            connection.send({"ok": True, "results": [[[], (text, 1.0)]]})
    except EOFError:
        pass
    finally:
        connection.close()
