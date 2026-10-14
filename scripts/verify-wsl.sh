#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/wsl-env.sh"
cd "$TEACH_ROOT"
"$PYTHON" scripts/check-local-ai.py --models --ocr --require-cuda --output "$RUNTIME_DIR/local-ai-check.json"
"$PYTHON" scripts/check-grpc.py
cd "$TEACH_ROOT/ai_engine_python"
"$PYTHON" -m unittest discover -s tests
cd "$TEACH_ROOT/backend-go"
RPA_BROWSER_PYTHON="$PYTHON" go test ./...
echo 'WSL local inference, RPC and browser tests passed.'
