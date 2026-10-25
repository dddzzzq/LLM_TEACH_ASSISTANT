"""Resolve deployment paths independently of the process working directory."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def model_path(name: str) -> str:
    # The repository and models are siblings on Windows and under /mnt/d in WSL.
    root = project_path(os.environ.get("TEACH_MODELS_DIR", "../models"))
    path = root / name
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"模型配置不存在：{path / 'config.json'}；请检查 TEACH_MODELS_DIR")
    return str(path)


def model_cache_dir() -> str:
    path = project_path(os.environ.get("HF_HOME", ".runtime/cache/huggingface"))
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
