#!/usr/bin/env bash
# Linux paths below live inside the WSL virtual disk stored on D:.
TEACH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RUNTIME_DIR="${RUNTIME_DIR:-/opt/teaching-runtime}"
export PATH="$RUNTIME_DIR/tools/uv:$RUNTIME_DIR/tools/go/bin:$RUNTIME_DIR/tools/node/bin:/usr/lib/wsl/lib:$PATH"
export UV_CACHE_DIR="$RUNTIME_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$RUNTIME_DIR/python"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_DIR/python-env"
export PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export GOPATH="$RUNTIME_DIR/go"
export GOMODCACHE="$RUNTIME_DIR/cache/go-mod"
export GOCACHE="$RUNTIME_DIR/cache/go-build"
export npm_config_cache="$RUNTIME_DIR/cache/npm"
export HF_HOME="$RUNTIME_DIR/cache/huggingface"
export PADDLE_PDX_CACHE_HOME="$RUNTIME_DIR/cache/paddlex"
export PADDLE_PDX_MODEL_SOURCE=BOS
export PLAYWRIGHT_BROWSERS_PATH="$RUNTIME_DIR/cache/playwright"
export TMPDIR="$RUNTIME_DIR/tmp"
export TEACH_SKILLS_DIR="$TEACH_ROOT/skills"
export PYTHONDONTWRITEBYTECODE=1
export GOTOOLCHAIN=local
export TZ=Asia/Shanghai
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
mkdir -p "$TMPDIR"
