"""Real spawned processes and gRPC deadlines without model downloads."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import grpc

from app.rpc.compute import ComputeServicer, ComputeServices
from app.rpc.protocol import pb2, pb2_grpc
from app.services.ocr_runtime import OCRCancelled, OCRTimeout, OCRUnavailable
from app.services.ocr_service import OCRService
from tests.fixtures.ocr_worker import fixture_ocr_worker


class OCRConcurrencyTests(unittest.TestCase):
    def service(self, count=2):
        service = OCRService(workers=count, worker_target=fixture_ocr_worker)
        self.addCleanup(service.close)
        service.warmup()
        return service

    def test_independent_processes_overlap_and_bound_parallelism(self):
        service = self.service()
        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(service.process_image, ["image"] * 6))
        intervals = [json.loads(result[0][1][0]) for result in results]
        self.assertEqual(len({item["pid"] for item in intervals}), 2)
        events = sorted(
            [(item["start"], 1) for item in intervals] + [(item["end"], -1) for item in intervals]
        )
        active = maximum = 0
        for _, change in events:
            active += change
            maximum = max(maximum, active)
        self.assertEqual(maximum, 2)
        for pid in {item["pid"] for item in intervals}:
            same_worker = sorted(
                (i for i in intervals if i["pid"] == pid), key=lambda i: i["start"]
            )
            self.assertTrue(
                all(a["end"] <= b["start"] for a, b in zip(same_worker, same_worker[1:]))
            )

    def test_deadline_terminates_inference_and_replaces_only_failed_worker(self):
        service = self.service(1)
        old_pid = service.workers[0].process.pid
        with self.assertRaises(OCRTimeout):
            service.process_image("slow", timeout=0.15)
        self.assertIsNone(service.workers[0].process)
        result = service.process_image("image", timeout=5)
        self.assertNotEqual(json.loads(result[0][1][0])["pid"], old_pid)

    def test_cancelled_requests_and_predictor_crashes_do_not_poison_pool(self):
        service = self.service(1)
        with self.assertRaises(OCRCancelled):
            service.process_image("image", cancelled=lambda: True)
        with self.assertRaises(OCRUnavailable):
            service.process_image("crash")
        self.assertTrue(service.process_image("image", timeout=5))

    def test_rpc_deadline_cancels_worker_and_always_removes_temp_file(self):
        service = self.service(1)
        executor = ThreadPoolExecutor(max_workers=2)
        server = grpc.server(executor)
        pb2_grpc.add_ComputeServiceServicer_to_server(
            ComputeServicer(ComputeServices(None, None, None, service)), server
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        stub = pb2_grpc.ComputeServiceStub(channel)
        original = tempfile.NamedTemporaryFile
        try:
            with (
                tempfile.TemporaryDirectory() as folder,
                patch(
                    "app.rpc.compute.tempfile.NamedTemporaryFile",
                    side_effect=lambda **kwargs: original(dir=folder, **kwargs),
                ),
            ):
                with self.assertRaises(grpc.RpcError) as failure:
                    stub.ExtractText(
                        pb2.ExtractRequest(filename="scan.png", file_content=b"slow"), timeout=0.15
                    )
                self.assertEqual(failure.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
                deadline = time.monotonic() + 3
                while list(Path(folder).iterdir()) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(list(Path(folder).iterdir()), [])
                result = stub.ExtractText(
                    pb2.ExtractRequest(filename="scan.png", file_content=b"image"), timeout=5
                )
                self.assertIn('"pid"', result.text_content)
                self.assertEqual(list(Path(folder).iterdir()), [])
                with self.assertRaises(grpc.RpcError) as failure:
                    stub.ExtractText(
                        pb2.ExtractRequest(filename="scan.png", file_content=b"error"), timeout=5
                    )
                self.assertEqual(failure.exception.code(), grpc.StatusCode.UNAVAILABLE)
                self.assertEqual(list(Path(folder).iterdir()), [])
        finally:
            channel.close()
            server.stop(0).wait(timeout=5)
            executor.shutdown(wait=True)
