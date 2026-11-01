"""Bounded OCR workers, each owning a separate Paddle predictor process."""

import atexit
import logging
import multiprocessing
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .ocr_runtime import OCRCancelled, OCRTimeout, OCRUnavailable, run_ocr_worker

logger = logging.getLogger(__name__)


class _Worker:
    def __init__(self, target, settings):
        self.target, self.settings = target, settings
        self.process = self.connection = None

    def predict(self, path, check):
        try:
            check()
            if self.process is None:
                context = multiprocessing.get_context("spawn")
                self.connection, child = context.Pipe()
                self.process = context.Process(target=self.target, args=(child, self.settings))
                try:
                    self.process.start()
                finally:
                    child.close()
            self.connection.send(path)
            while not self.connection.poll(0.05):
                check()
                if not self.process.is_alive():
                    raise OCRUnavailable("OCR worker exited")
            check()
            response = self.connection.recv()
            if not response["ok"]:
                raise OCRUnavailable(response["error"])
            return response["results"]
        except (OCRCancelled, OCRTimeout):
            self.close()
            raise
        except (EOFError, OSError) as exc:
            self.close()
            raise OCRUnavailable("OCR worker connection lost") from exc
        except BaseException:
            # Deadline cancellation also stops native inference.
            self.close()
            raise

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            if self.process.pid is not None:
                if self.process.is_alive():
                    self.process.terminate()
                self.process.join(timeout=1)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1)
            self.process.close()
            self.process = None


class OCRService:
    def __init__(self, *, workers=None, worker_target=run_ocr_worker):
        count = int(workers if workers is not None else os.environ.get("OCR_WORKERS", "2"))
        if not 1 <= count <= 8:
            raise ValueError("OCR_WORKERS must be between 1 and 8")
        settings = {
            "device": os.environ.get("OCR_DEVICE", "cpu"),
            "cpu_threads": int(
                os.environ.get(
                    "OCR_CPU_THREADS", str(max(1, min(8, (os.cpu_count() or 2) // count)))
                )
            ),
        }
        if settings["cpu_threads"] < 1:
            raise ValueError("OCR_CPU_THREADS must be positive")
        self.timeout = float(os.environ.get("OCR_TASK_TIMEOUT_SECONDS", "120"))
        if self.timeout <= 0:
            raise ValueError("OCR_TASK_TIMEOUT_SECONDS must be positive")
        self.workers = [_Worker(worker_target, settings) for _ in range(count)]
        self.available = queue.Queue(maxsize=count)
        for worker in self.workers:
            self.available.put(worker)
        self.closed = threading.Event()
        atexit.register(self.close)

    def _execute(self, path, *, cancelled=None, timeout=None):
        started = time.monotonic()
        deadline = started + (self.timeout if timeout is None else min(timeout, self.timeout))

        def check():
            if self.closed.is_set():
                raise OCRUnavailable("OCR service is closed")
            if cancelled and cancelled():
                raise OCRCancelled("OCR request cancelled")
            if time.monotonic() >= deadline:
                raise OCRTimeout("OCR request deadline exceeded")

        while True:
            check()
            try:
                worker = self.available.get(timeout=0.05)
                break
            except queue.Empty:
                continue
        acquired = time.monotonic()
        try:
            result = worker.predict(path, check)
            logger.info(
                "OCR completed worker_pid=%s queue_ms=%.0f processing_ms=%.0f",
                worker.process.pid,
                (acquired - started) * 1000,
                (time.monotonic() - acquired) * 1000,
            )
            return result
        finally:
            self.available.put(worker)

    def process_image(self, file_path, *, cancelled=None, timeout=None):
        return self._execute(file_path, cancelled=cancelled, timeout=timeout)

    def get_concatenated_text(self, image_path_list, *, cancelled=None, timeout=None):
        deadline = time.monotonic() + (
            self.timeout if timeout is None else min(timeout, self.timeout)
        )
        all_text = []
        for index, path in enumerate(image_path_list, 1):
            results = self.process_image(
                path, cancelled=cancelled, timeout=deadline - time.monotonic()
            )
            all_text.append(f"--- 图片 {index} 开始 ---")
            all_text.extend(line[1][0] for line in results)
            if not results:
                all_text.append("[未识别到文本]")
            all_text.append(f"--- 图片 {index} 结束 ---")
        return "\n".join(all_text)

    def warmup(self):
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            pending = [executor.submit(self._execute, None) for _ in self.workers]
            for future in pending:
                future.result()
        logger.info("OCR ready: %d isolated workers", len(self.workers))

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        for worker in self.workers:
            worker.close()


# Importing RPC contracts never loads model weights.
ocr_service_instance = OCRService()
