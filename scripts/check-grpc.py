"""Start the real compute server and check local RPCs, then stop only that server."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai_engine_python/app/pb2"))


def main():
    import grpc
    import compute_service_pb2 as pb
    import compute_service_pb2_grpc as rpc
    from dotenv import dotenv_values
    runtime = Path(os.environ.get("RUNTIME_DIR", ROOT / ".runtime"))
    runtime.mkdir(exist_ok=True)
    env = dict(os.environ)
    for key, value in dotenv_values(ROOT / "ai_engine_python/app/.env").items():
        if value is not None:
            env.setdefault(key, value)
    for key, value in {
        "HF_HOME": runtime / "cache/huggingface",
        "PADDLE_PDX_CACHE_HOME": runtime / "cache/paddlex",
        "PADDLE_HOME": runtime / "cache/paddle",
        "XDG_CACHE_HOME": runtime / "cache",
        "TMPDIR": runtime / "tmp",
        "TEMP": runtime / "tmp",
        "TMP": runtime / "tmp",
    }.items():
        env[key] = str(value)
    env["PYTHONUTF8"] = "1"
    env["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
    env["AI_GRPC_BIND"] = "127.0.0.1:50051"
    # Refuse to run against someone else's existing service.
    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 50051))
    report = {"ok": False, "checks": {}}
    with (runtime / "grpc-check-server.log").open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "app/grpc_server.py"],
            cwd=ROOT / "ai_engine_python", env=env, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            with grpc.insecure_channel("127.0.0.1:50051") as channel:
                grpc.channel_ready_future(channel).result(timeout=90)
                stub = rpc.ComputeServiceStub(channel)
                image = (ROOT / ".runtime/ocr-smoke.png").read_bytes()
                result = stub.ExtractText(pb.ExtractRequest(filename="ocr-smoke.png", file_content=image), timeout=60)
                assert "2026" in result.text_content, result.text_content
                report["checks"]["ExtractText"] = "passed"
                text = "操作系统通过进程调度分配计算资源，保障任务公平执行。" * 5
                result = stub.DetectAIGC(pb.AIGCRequest(text_content=text), timeout=60)
                data = json.loads(result.aigc_report_json)
                assert "error" not in data and 0 <= data["ai_probability"] <= 1, data
                report["checks"]["DetectAIGC"] = "passed"
            report["ok"] = True
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            process.terminate()
            process.wait(timeout=15)
            (runtime / "grpc-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
