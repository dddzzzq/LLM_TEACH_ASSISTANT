"""Exercise real local models without calling any paid/cloud API."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai_engine_python"))
sys.path.insert(0, str(ROOT / "ai_engine_python/app/pb2"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/local-ai-check.json")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / "ai_engine_python/app/.env")
    cache = ROOT / ".runtime/cache"
    for key, value in {
        "HF_HOME": cache / "huggingface",
        "PADDLE_PDX_CACHE_HOME": cache / "paddlex",
        "PADDLE_HOME": cache / "paddle",
        "XDG_CACHE_HOME": cache,
    }.items():
        os.environ.setdefault(key, str(value))
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("OCR_DEVICE", "cpu")
    report = {"python": sys.version.split()[0], "checks": {}}
    try:
        import compute_service_pb2 as pb
        import compute_service_pb2_grpc
        assert pb.ExtractRequest(filename="sample.png", file_content=b"test").filename == "sample.png"
        report["checks"]["protobuf_grpc_import"] = "passed"
        from app.core.paths import model_path
        report["models"] = {name: model_path(name) for name in (
            "bert-base-chinese", "unixcoder-base",
            "chatgpt-detector-roberta-chinese", "chatgpt-detector-roberta",
        )}
        from app.core.config import settings
        report["cloud_key_configured"] = bool(settings.DEEPSEEK_API_KEY)
        if args.models:
            import torch
            report["torch"] = torch.__version__
            report["cuda_available"] = torch.cuda.is_available()
            if args.require_cuda and not report["cuda_available"]:
                raise RuntimeError("CUDA is unavailable on the deployment GPU")
            if report["cuda_available"]:
                report["gpu"] = torch.cuda.get_device_name(0)
                report["compute_capability"] = torch.cuda.get_device_capability(0)
                matrix = torch.ones((16, 16), device="cuda")
                assert float((matrix @ matrix).sum().cpu()) == 4096
                report["checks"]["cuda_tensor_compute"] = "passed"
            from app.services.plagiarism_service import plagiarism_service
            for kind, content in (("text", "操作系统通过进程调度分配计算资源。"), ("code", "def add(a, b): return a + b")):
                embedding = plagiarism_service._get_embedding(content, kind)
                assert embedding.shape == (1, 768)
                report["checks"]["embedding_" + kind] = "passed"
            from app.services.aigc_service import aigc_detector_service
            for kind, content in (("prose", "操作系统通过进程调度分配计算资源，保障任务公平执行。" * 5), ("code", "def add(a, b):\n    return a + b\n" * 5)):
                result = aigc_detector_service._detect_single_part(content, kind)
                assert "error" not in result, result
                assert 0 <= result["ai_probability"] <= 1, result
                report["checks"]["aigc_" + kind] = "passed"
            if report["cuda_available"]:
                report["gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        if args.ocr:
            # Match grpc_server's import order for Windows DLL initialization.
            import torch
            import paddle
            report["paddle"] = paddle.__version__
            report["paddle_cuda_build"] = paddle.is_compiled_with_cuda()
            from PIL import Image, ImageDraw, ImageFont
            image_path = ROOT / ".runtime/ocr-smoke.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (800, 140), "white")
            fonts = [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
            font = next((ImageFont.truetype(str(p), 44) for p in fonts if p.exists()), ImageFont.load_default())
            ImageDraw.Draw(image).text((30, 35), "LOCAL OCR TEST 2026", fill="black", font=font)
            image.save(image_path)
            from app.services.ocr_service import ocr_service_instance
            assert ocr_service_instance is not None, "OCR initialization failed"
            text = ocr_service_instance.get_concatenated_text([str(image_path)])
            assert "2026" in text, text
            report["checks"]["ocr_image_text"] = "passed"
        report["ok"] = True
    except Exception as error:
        report["ok"] = False
        report["error"] = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
