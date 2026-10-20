#!/usr/bin/env bash
# Stop the services launched by start.sh; keep MySQL and Redis running.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${RUNTIME_DIR:-$(dirname "$ROOT")/teaching-runtime}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
[[ "$STOP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo 'STOP_TIMEOUT 必须为正整数（秒）' >&2
  exit 1
}
[[ -d "$RUNTIME_DIR" ]] || { echo '没有运行目录，无需停止'; exit 0; }
PYTHON="${PYTHON:-$ROOT/ai_engine_python/.venv/bin/python}"
if [[ ! -x "$PYTHON" && -x "$RUNTIME_DIR/python-env/bin/python" ]]; then
  PYTHON="$RUNTIME_DIR/python-env/bin/python"
fi
[[ -x "$PYTHON" ]] || { echo "缺少 Python 环境：$PYTHON" >&2; exit 1; }
export ROOT RUNTIME_DIR

# Share the startup lock so startup and shutdown cannot run together.
exec 9>"$RUNTIME_DIR/start.lock"
flock -n 9 || { echo '启动或停止脚本正在运行，请稍后重试' >&2; exit 1; }

stop() {
  "$PYTHON" "$ROOT/scripts/service_process.py" stop "$1" --timeout "$STOP_TIMEOUT" 9>&-
}

status=0
for name in frontend backend ai rpa kafka; do
  if ! stop "$name"; then
    status=1
  fi
done
exit "$status"
