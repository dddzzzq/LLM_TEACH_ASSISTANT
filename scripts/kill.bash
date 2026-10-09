#!/usr/bin/env bash
# Stop the services launched by start.sh; keep MySQL and Redis running.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${RUNTIME_DIR:-$(dirname "$ROOT")/teaching-runtime}"
STOP_TIMEOUT="${STOP_TIMEOUT:-60}"
[[ "$STOP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo 'STOP_TIMEOUT 必须为正整数（秒）' >&2
  exit 1
}
[[ -d "$RUNTIME_DIR" ]] || { echo '没有运行目录，无需停止'; exit 0; }

# Share the startup lock so startup and shutdown cannot run together.
exec 9>"$RUNTIME_DIR/start.lock"
flock -n 9 || { echo '启动或停止脚本正在运行，请稍后重试' >&2; exit 1; }

group_alive() {
  # Zombies have already exited and cannot respond to signals.
  ps -eo pgid=,stat= | awk -v pgid="$1" \
    '$1 == pgid && $2 !~ /^Z/ { alive = 1 } END { exit !alive }'
}

stop() {
  local name="$1" pidfile="$RUNTIME_DIR/$1.pid" pid cmd cwd pgid deadline
  [[ -f "$pidfile" ]] || { echo "$name：没有 PID 文件，跳过"; return 0; }
  pid="$(cat -- "$pidfile")"
  [[ "$pid" =~ ^[1-9][0-9]*$ && "$pid" -gt 1 ]] || {
    echo "$name：PID 无效，保留文件并跳过" >&2
    return 1
  }
  if [[ ! -d "/proc/$pid" ]]; then
    if group_alive "$pid"; then
      echo "$name：主进程已退出，但进程组仍存在，无法核实身份，跳过" >&2
      return 1
    fi
    rm -f -- "$pidfile"
    echo "$name：已退出"
    return 0
  fi

  # Check identity before using a PID that may have been reused.
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  cwd="$(readlink -- "/proc/$pid/cwd" || true)"
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  [[ "$pgid" == "$pid" ]] || {
    echo "$name：进程组与启动记录不符，跳过" >&2
    return 1
  }
  case "$name:$cwd:$cmd" in
    "frontend:$ROOT/vue-grading-frontend:"*"node_modules/vite/bin/vite.js"*) ;;
    "backend:$ROOT/backend-go:$RUNTIME_DIR/grading-gateway"*) ;;
    "ai:$ROOT/ai_engine_python:"*"app/grpc_server.py"*) ;;
    "rpa:$ROOT/ai_engine_python:"*"app.rpa.worker"*) ;;
    "kafka:$RUNTIME_DIR:"*"kafka.Kafka"*) ;;
    *) echo "$name：进程身份与启动记录不符，跳过" >&2; return 1 ;;
  esac

  echo "$name：发送 SIGTERM（进程组 $pid）"
  kill -TERM -- "-$pid" 2>/dev/null || true
  deadline=$((SECONDS + STOP_TIMEOUT))
  while group_alive "$pid"; do
    if (( SECONDS >= deadline )); then
      echo "$name：等待 ${STOP_TIMEOUT} 秒后仍未退出，保留 PID 文件，未强制终止" >&2
      return 1
    fi
    sleep 1
  done
  rm -f -- "$pidfile"
  echo "$name：已停止"
}

status=0
for name in frontend backend ai rpa kafka; do
  if ! stop "$name"; then
    status=1
  fi
done
exit "$status"
