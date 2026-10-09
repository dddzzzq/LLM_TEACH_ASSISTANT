#!/usr/bin/env bash
# AutoDL/Linux: start installed services without reinstalling dependencies.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${RUNTIME_DIR:-$(dirname "$ROOT")/teaching-runtime}"
PYTHON="${PYTHON:-$ROOT/ai_engine_python/.venv/bin/python}"
KAFKA_HOME="${KAFKA_HOME:-$RUNTIME_DIR/kafka_2.13-4.1.2}"
KAFKA_CONFIG="${KAFKA_CONFIG:-$RUNTIME_DIR/kafka.properties}"
export ROOT RUNTIME_DIR PYTHON KAFKA_HOME KAFKA_CONFIG
export TEACH_SKILLS_DIR="${TEACH_SKILLS_DIR:-$ROOT/skills}"
export RPA_CONTROL_TOKEN_FILE="${RPA_CONTROL_TOKEN_FILE:-$RUNTIME_DIR/.rpa-control-token}"
export RPA_CONTROL_PORT="${RPA_CONTROL_PORT:-8765}"
export RPA_CONTROL_URL="${RPA_CONTROL_URL:-http://127.0.0.1:$RPA_CONTROL_PORT}"
mkdir -p "$RUNTIME_DIR"
chmod 700 "$RUNTIME_DIR"
[[ -x "$PYTHON" ]] || { echo "缺少 Python 环境：$PYTHON" >&2; exit 1; }
# Parse dotenv as data, never execute it as shell code. Persist JWT secrets locally.
if [[ "${1:-}" != --internal-env-ready ]]; then
  exec "$PYTHON" - "$ROOT/scripts/start.sh" <<'PY'
import os, sys, secrets
from pathlib import Path
from dotenv import dotenv_values
root = Path(os.environ['ROOT'])
for k, v in dotenv_values(root / 'ai_engine_python/app/.env').items():
    if v is not None and k not in os.environ:
        os.environ[k] = v
if not os.environ.get('DEEPSEEK_API_KEY'):
    sys.exit('请先在 ai_engine_python/app/.env 配置 DEEPSEEK_API_KEY')
path = Path(os.environ['RUNTIME_DIR']) / '.jwt.env'
if not path.exists():
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        for key in ('JWT_ACCESS_SECRET', 'JWT_REFRESH_SECRET'):
            f.write(key + '=' + secrets.token_hex(32) + '\n')
for k, v in dotenv_values(path).items():
    os.environ.setdefault(k, v)
os.execvpe('bash', ['bash', sys.argv[1], '--internal-env-ready'], os.environ)
PY
fi
exec 9>"$RUNTIME_DIR/start.lock"
flock -n 9 || { echo '另一个启动脚本正在运行'; exit 1; }
trap 'echo "启动失败（行 $LINENO），请查看 $RUNTIME_DIR 中对应服务的日志。已启动服务保留运行。" >&2' ERR
for cmd in go node java mysql redis-cli service curl; do
  command -v "$cmd" >/dev/null || { echo "缺少命令：$cmd" >&2; exit 1; }
done
[[ -x "$KAFKA_HOME/bin/kafka-server-start.sh" && -f "$KAFKA_CONFIG" ]]
[[ -f "$ROOT/vue-grading-frontend/node_modules/vite/bin/vite.js" ]]
export PYTHONDONTWRITEBYTECODE=1
export VITE_CACHE_DIR="$RUNTIME_DIR/vite-cache"
export HF_HOME="${HF_HOME:-$(dirname "$ROOT")/.hf-cache}"
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-$(dirname "$ROOT")/.paddlex}"
export PADDLE_PDX_MODEL_SOURCE="${PADDLE_PDX_MODEL_SOURCE:-BOS}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$(dirname "$ROOT")/.playwright}"
export TMPDIR="${TMPDIR:-$(dirname "$ROOT")/.tmp}"
export KAFKA_HEAP_OPTS="${KAFKA_HEAP_OPTS:--Xms256m -Xmx512m}"
mkdir -p "$TMPDIR"
port_open() { "$PYTHON" - "$1" <<'PY'
import socket, sys
try:
    with socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=1): pass
except OSError: sys.exit(1)
PY
}
start() {
  local name="$1" port="$2" cwd="$3" limit="$4"; shift 4
  if port_open "$port"; then
    echo "$name：端口 $port 已在监听，复用现有服务"
    return
  fi
  # start_new_session detaches children; close the startup lock in the child.
  "$PYTHON" - "$name" "$cwd" "$@" 9>&- <<'PY'
import os, subprocess, sys
from pathlib import Path
name, cwd, *cmd = sys.argv[1:]
runtime = Path(os.environ['RUNTIME_DIR'])
with (runtime / (name + '.log')).open('ab') as log:
    p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
(runtime / (name + '.pid')).write_text(str(p.pid))
PY
  local deadline=$((SECONDS + limit)) pid
  pid="$(cat "$RUNTIME_DIR/$name.pid")"
  until port_open "$port"; do
    if ! kill -0 "$pid" 2>/dev/null || (( SECONDS >= deadline )); then
      echo "$name 未就绪，请查看 $RUNTIME_DIR/$name.log" >&2
      return 1
    fi
    sleep 2
  done
  echo "$name：已就绪（端口 $port）"
}
if ! port_open 3306; then service mysql start 9>&- >"$RUNTIME_DIR/mysql.log" 2>&1; fi
port_open 3306
if ! port_open 6379; then service redis-server start 9>&- >"$RUNTIME_DIR/redis.log" 2>&1; fi
if [[ -n "${REDIS_PASSWORD:-}" ]]; then
  [[ "$(REDISCLI_AUTH="$REDIS_PASSWORD" redis-cli -h 127.0.0.1 ping)" == PONG ]]
else
  [[ "$(env -u REDISCLI_AUTH redis-cli -h 127.0.0.1 ping)" == PONG ]]
fi
start kafka 9092 "$RUNTIME_DIR" 120 "$KAFKA_HOME/bin/kafka-server-start.sh" "$KAFKA_CONFIG"
for topic in topic_grading_homework topic_grading_exam topic_rpa_fetch; do
  "$KAFKA_HOME/bin/kafka-topics.sh" --bootstrap-server localhost:9092 \
    --create --if-not-exists --topic "$topic" --partitions 1 --replication-factor 1 >>"$RUNTIME_DIR/kafka-topics.log" 2>&1
done
start rpa "$RPA_CONTROL_PORT" "$ROOT/ai_engine_python" 60 "$PYTHON" -u -m app.rpa.worker
# A listening port may belong to the old Python decision loop. Verify the
# authenticated execution-only protocol before starting its Go decision driver.
"$PYTHON" - <<'PY'
import json, os, sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    token = Path(os.environ['RPA_CONTROL_TOKEN_FILE']).read_text().strip()
    request = Request(os.environ['RPA_CONTROL_URL'].rstrip('/') + '/health',
                      headers={'X-RPA-Token': token})
    with urlopen(request, timeout=10) as response:
        health = json.load(response)
except (OSError, ValueError, HTTPError, URLError):
    sys.exit('浏览器 Worker 健康检查失败，请检查控制地址、令牌文件及 rpa.log')
if (not isinstance(health, dict) or health.get('ok') is not True
        or health.get('protocol_version') != 'browser.v1'
        or health.get('decision_driver') != 'go-eino'
        or health.get('browser_stream') != 'browser.stream.v1'):
    sys.exit('浏览器 Worker 与 Go/Eino 不兼容，请用 scripts/kill.bash 停止旧服务后重新启动')
print('rpa：已验证 browser.v1 和流式浏览器，页面决策由 Go/Eino 执行')
PY
start ai 50051 "$ROOT/ai_engine_python" "${AI_START_TIMEOUT:-300}" "$PYTHON" -u app/grpc_server.py
if ! port_open 8000; then
  echo '编译 Go 后端…'
  (cd "$ROOT/backend-go" && go build -o "$RUNTIME_DIR/grading-gateway" .)
fi
start backend 8000 "$ROOT/backend-go" 90 "$RUNTIME_DIR/grading-gateway"
# Protected route must respond with 401 when called without authentication.
[[ "$(curl --silent --show-error --max-time 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/profile)" == 401 ]]
start frontend 6006 "$ROOT/vue-grading-frontend" 60 node node_modules/vite/bin/vite.js --host 0.0.0.0 --port 6006 --strictPort
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:6006/ >/dev/null
printf '\n系统已启动：http://127.0.0.1:6006\n日志目录：%s\n' "$RUNTIME_DIR"
if [[ -n "${AutoDLService6006URL:-}" ]]; then printf 'AutoDL 入口：%s\n' "$AutoDLService6006URL"; fi
