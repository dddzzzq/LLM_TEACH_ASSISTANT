#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/wsl-env.sh"
exec bash "$TEACH_ROOT/scripts/start.sh"
