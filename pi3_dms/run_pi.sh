#!/usr/bin/env bash
# Start the Pi 3 DMS core.
# The server is reachable from another device on the same network.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PORT="${PORT:-8000}"

exec python3 app.py \
  --host 0.0.0.0 \
  --port "$PORT" \
  --camera-backend auto \
  --capture-width 1920 \
  --capture-height 1080 \
  --camera-fps 15 \
  --stream-width 960 \
  --stream-height 540 \
  --process-width 480 \
  --process-height 270 \
  "$@"
