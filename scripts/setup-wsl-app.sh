#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/wsl-env.sh"
uv python install 3.11.17
cd "$TEACH_ROOT/ai_engine_python"
uv sync --locked --no-dev
# Install browser libraries as root separately; browser cache belongs to teacher.
sudo -n service mysql start
cd "$TEACH_ROOT/backend-go"
go mod download
go run ./cmd/browser-worker --install
go build -o "$RUNTIME_DIR/browser-worker" ./cmd/browser-worker
go build -o "$RUNTIME_DIR/grading-gateway" .
cd "$TEACH_ROOT/vue-grading-frontend"
npm ci
npm run build
echo 'Application dependencies prepared. Run the local checks before startup.'
