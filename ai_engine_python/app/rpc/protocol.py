"""Load generated bindings without modifying protoc output or loading AI models."""

import sys
from pathlib import Path

_bindings_dir = str(Path(__file__).resolve().parents[1] / "pb2")
if _bindings_dir not in sys.path:
    sys.path.append(_bindings_dir)

# protoc generates an absolute import in compute_service_pb2_grpc.
import compute_service_pb2 as pb2
import compute_service_pb2_grpc as pb2_grpc

__all__ = ["pb2", "pb2_grpc"]
