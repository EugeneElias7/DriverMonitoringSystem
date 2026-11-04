#!/usr/bin/env bash
# Start the driver-monitoring app with the Pi profile.
# From the Pi desktop it opens the preview; over SSH it runs headless.
# The latest run is also written to ~/dms_logs/last_run.log.
set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="${HOME}/dms_logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/last_run.log"
: > "$LOG"

if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    exec .venv/bin/python run.py --config config_pi.json --source picamera "$@" 2>&1 | tee "$LOG"
fi

exec .venv/bin/python run.py --config config_pi.json --source picamera --headless "$@" 2>&1 | tee "$LOG"
