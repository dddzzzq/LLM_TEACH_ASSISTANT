import logging
import multiprocessing
import os
import signal
import sys
from concurrent import futures

import grpc

# 动态引入根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app.rpc.protocol import pb2, pb2_grpc
from app.rpc.compute import ComputeServices, ComputeServicer as _ComputeServicer
from app.rpc.limits import GRPC_OPTIONS

# 拦截子进程无意义的导入，只有主进程才加载大模型
if multiprocessing.current_process().name == "MainProcess":
    from app.services.plagiarism_service import plagiarism_service
    from app.services.aigc_service import aigc_detector_service
    from app.services.deepseek_service import deepseek_service
    from app.services.ocr_service import ocr_service_instance
else:
    plagiarism_service = None
    aigc_detector_service = None
    deepseek_service = None
    ocr_service_instance = None

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)


class ComputeServicer(_ComputeServicer):
    """Keep the entry module's no-argument constructor for existing callers."""

    def __init__(self):
        super().__init__(
            ComputeServices(
                plagiarism_service=plagiarism_service,
                aigc_detector_service=aigc_detector_service,
                deepseek_service=deepseek_service,
                ocr_service_instance=ocr_service_instance,
            )
        )


def serve():
    logging.info("正在初始化 AI 模型节点...")
    # 强制预热模型
    _ = aigc_detector_service
    _ = plagiarism_service
    ocr_service_instance.warmup()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(os.environ.get("AI_GRPC_WORKERS", "10"))),
        options=GRPC_OPTIONS,
    )
    pb2_grpc.add_ComputeServiceServicer_to_server(ComputeServicer(), server)

    address = os.environ.get("AI_GRPC_BIND", "127.0.0.1:50051")
    if not server.add_insecure_port(address):
        raise RuntimeError(f"无法监听 AI gRPC 地址：{address}")
    server.start()
    logging.info("Python AI 节点启动成功，监听 %s", address)

    def stop_server(signum, frame):
        server.stop(5)

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    try:
        server.wait_for_termination()
    finally:
        server.stop(0).wait(timeout=5)
        ocr_service_instance.close()


if __name__ == "__main__":
    serve()
