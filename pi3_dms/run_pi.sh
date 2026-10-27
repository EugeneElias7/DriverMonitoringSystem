#!/usr/bin/env bash
# Start the Pi 3 DMS. The face core is always enabled; object inference is opt-in.
# Examples:
#   bash run_pi.sh
#   OBJECTS=1 bash run_pi.sh
#   OBJECTS=1 OBJECT_BACKEND=http OBJECT_SERVICE_URL=http://127.0.0.1:8010/detect bash run_pi.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PORT="${PORT:-8000}"
TINT_MODE="${TINT_MODE:-soft}"

ARGS=(
  --host 0.0.0.0
  --port "$PORT"
  --camera-backend auto
  --capture-width 1920
  --capture-height 1080
  --camera-fps 15
  --stream-width 960
  --stream-height 540
  --process-width 480
  --process-height 270
  --tint-mode "$TINT_MODE"
)

if [ "${OBJECTS:-0}" = "1" ]; then
  ARGS+=(--objects --object-backend "${OBJECT_BACKEND:-mediapipe}")
  if [ -n "${OBJECT_MODEL:-}" ]; then
    ARGS+=(--object-model "$OBJECT_MODEL")
  fi
  if [ -n "${OBJECT_CLASSES:-}" ]; then
    ARGS+=(--object-classes "$OBJECT_CLASSES")
  fi
  if [ -n "${OBJECT_SERVICE_URL:-}" ]; then
    ARGS+=(--object-service-url "$OBJECT_SERVICE_URL")
  fi
fi

exec python3 app.py "${ARGS[@]}" "$@"
