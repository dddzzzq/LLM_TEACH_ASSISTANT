#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $EUID == 0 ]] || { echo 'Run bootstrap as the WSL root user.' >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl sudo python3 antiword poppler-utils mysql-server redis-server \
  openjdk-21-jre-headless build-essential unzip fonts-noto-cjk libgomp1
id teacher >/dev/null 2>&1 || useradd --create-home --shell /bin/bash teacher
RUNTIME=/opt/teaching-runtime
mkdir -p "$RUNTIME"/{tools,cache,tmp,downloads}
chown teacher:teacher "$RUNTIME" "$RUNTIME"/{tools,cache,tmp,downloads}
chmod 700 "$RUNTIME"
cat >/etc/wsl.conf <<'CONF'
[boot]
systemd=true
[user]
default=teacher
CONF
printf 'teacher ALL=(root) NOPASSWD: /usr/sbin/service mysql start, /usr/sbin/service redis-server start\n' >/etc/sudoers.d/teaching-services
chmod 440 /etc/sudoers.d/teaching-services
visudo -cf /etc/sudoers.d/teaching-services

fetch() {
  local url="$1" destination="$2"
  [[ -s "$destination" ]] || curl --fail --location --retry 3 --connect-timeout 20 --max-time 900 "$url" -o "$destination.part"
  if [[ -f "$destination.part" ]]; then mv "$destination.part" "$destination"; fi
}
if [[ ! -x "$RUNTIME/tools/uv/uv" ]]; then
  fetch https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-unknown-linux-gnu.tar.gz "$RUNTIME/tools/uv.tar.gz"
  echo "1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154  $RUNTIME/tools/uv.tar.gz" | sha256sum -c -
  mkdir -p "$RUNTIME/tools/uv"
  tar -xzf "$RUNTIME/tools/uv.tar.gz" -C "$RUNTIME/tools/uv" --strip-components=1
fi
if [[ ! -x "$RUNTIME/tools/go/bin/go" ]]; then
  fetch https://go.dev/dl/go1.25.5.linux-amd64.tar.gz "$RUNTIME/tools/go.tar.gz"
  curl --fail --location --retry 3 'https://go.dev/dl/?mode=json&include=all' -o "$RUNTIME/tools/go-releases.json"
  # The distribution's system Python is only used for bootstrap metadata.
  python3 - "$RUNTIME/tools/go-releases.json" "$RUNTIME/tools/go.tar.gz" <<'PY'
import hashlib, json, sys
releases = json.load(open(sys.argv[1]))
entry = next(f for r in releases for f in r['files'] if f['filename'] == 'go1.25.5.linux-amd64.tar.gz')
with open(sys.argv[2], 'rb') as source:
    assert hashlib.file_digest(source, 'sha256').hexdigest() == entry['sha256'], 'Go checksum mismatch'
PY
  tar -xzf "$RUNTIME/tools/go.tar.gz" -C "$RUNTIME/tools"
fi
if [[ ! -x "$RUNTIME/tools/node/bin/node" ]]; then
  fetch https://nodejs.org/dist/v24.19.0/node-v24.19.0-linux-x64.tar.xz "$RUNTIME/tools/node.tar.xz"
  curl --fail --location --retry 3 https://nodejs.org/dist/v24.19.0/SHASUMS256.txt -o "$RUNTIME/tools/node-shasums.txt"
  expected="$(awk '$2 == "node-v24.19.0-linux-x64.tar.xz" {print $1}' "$RUNTIME/tools/node-shasums.txt")"
  [[ -n "$expected" ]]
  echo "$expected  $RUNTIME/tools/node.tar.xz" | sha256sum -c -
  mkdir -p "$RUNTIME/tools/node"
  tar -xJf "$RUNTIME/tools/node.tar.xz" -C "$RUNTIME/tools/node" --strip-components=1
fi
if [[ ! -x "$RUNTIME/kafka_2.13-4.1.2/bin/kafka-server-start.sh" ]]; then
  fetch https://dlcdn.apache.org/kafka/4.1.2/kafka_2.13-4.1.2.tgz "$RUNTIME/tools/kafka.tgz"
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 60 https://dlcdn.apache.org/kafka/4.1.2/kafka_2.13-4.1.2.tgz.sha512 -o "$RUNTIME/tools/kafka.sha512"
  python3 - "$RUNTIME/tools/kafka.sha512" "$RUNTIME/tools/kafka.tgz" <<'PY'
import hashlib, pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
expected = ''.join(re.findall(r'\b[0-9a-fA-F]{8,128}\b', text)).lower()
assert len(expected) == 128, 'Unrecognized Apache SHA512 file'
with open(sys.argv[2], 'rb') as source:
    assert hashlib.file_digest(source, 'sha512').hexdigest() == expected, 'Kafka checksum mismatch'
PY
  tar -xzf "$RUNTIME/tools/kafka.tgz" -C "$RUNTIME"
fi
if [[ ! -f "$RUNTIME/kafka.properties" ]]; then
  cat >"$RUNTIME/kafka.properties" <<CONF
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@127.0.0.1:9093
listeners=PLAINTEXT://127.0.0.1:9092,CONTROLLER://127.0.0.1:9093
advertised.listeners=PLAINTEXT://127.0.0.1:9092
controller.listener.names=CONTROLLER
listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
inter.broker.listener.name=PLAINTEXT
log.dirs=$RUNTIME/kafka-data
num.partitions=1
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
group.initial.rebalance.delay.ms=0
CONF
fi
if [[ ! -f "$RUNTIME/kafka-data/meta.properties" ]]; then
  cluster_id="$("$RUNTIME/kafka_2.13-4.1.2/bin/kafka-storage.sh" random-uuid)"
  "$RUNTIME/kafka_2.13-4.1.2/bin/kafka-storage.sh" format -t "$cluster_id" -c "$RUNTIME/kafka.properties"
fi
service mysql start
service redis-server start
if [[ ! -f "$RUNTIME/.database.env" ]]; then
  python3 - "$RUNTIME" <<'PY'
import os, pathlib, secrets, subprocess, sys
password = secrets.token_hex(24)
sql = f"""CREATE DATABASE IF NOT EXISTS grading_system CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
CREATE USER IF NOT EXISTS 'teaching'@'127.0.0.1' IDENTIFIED BY '{password}';
ALTER USER 'teaching'@'127.0.0.1' IDENTIFIED BY '{password}';
GRANT ALL PRIVILEGES ON grading_system.* TO 'teaching'@'127.0.0.1';"""
subprocess.run(['mysql', '-uroot'], input=sql, text=True, check=True)
path = pathlib.Path(sys.argv[1]) / '.database.env'
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as output:
    output.write(f'DATABASE_DSN=teaching:{password}@tcp(127.0.0.1:3306)/grading_system?charset=utf8mb4&parseTime=True&loc=Local\n')
PY
fi
chown -R teacher:teacher "$RUNTIME"
runuser -u teacher -- bash "$ROOT/scripts/setup-wsl-app.sh"
"$RUNTIME/python-env/bin/python" -m playwright install-deps chromium
