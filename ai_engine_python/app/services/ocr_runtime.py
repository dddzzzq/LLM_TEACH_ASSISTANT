"""Spawn-safe predictor entry point; no model imports in the parent process."""

import logging
import os


class OCRCancelled(Exception):
    pass


class OCRTimeout(TimeoutError):
    pass


class OCRUnavailable(RuntimeError):
    pass


def run_ocr_worker(connection, settings):
    try:
        os.environ["OMP_NUM_THREADS"] = str(settings["cpu_threads"])
        os.environ["MKL_NUM_THREADS"] = str(settings["cpu_threads"])
        os.environ["KMP_WARNINGS"] = "0"
        if os.name == "nt":
            import torch  # Load Torch's Windows DLLs before Paddle.
        from paddleocr import PaddleOCR

        engine = PaddleOCR(
            device=settings["device"],
            cpu_threads=settings["cpu_threads"],
            mkldnn_cache_capacity=1,
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            lang="ch",
        )
        while True:
            path = connection.recv()
            results = []
            if path is not None:
                page = next(iter(engine.predict(path)), None)
                if page is not None:
                    for poly, text, score in zip(
                        page.get("rec_polys", []),
                        page.get("rec_texts", []),
                        page.get("rec_scores", []),
                    ):
                        results.append([poly.tolist(), (str(text), float(score))])
            connection.send({"ok": True, "results": results})
    except EOFError:
        pass
    except Exception as exc:
        logging.getLogger(__name__).exception("OCR worker failed")
        try:
            connection.send({"ok": False, "error": f"OCR predictor failed ({type(exc).__name__})"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
