#!/usr/bin/env bash
# Start the driver-monitoring app with the Pi profile.
# From the Pi desktop it opens the preview; over SSH it runs headless.
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    exec .venv/bin/python run.py --config config_pi.json --source picamera "$@"
fi

exec .venv/bin/python run.py --config config_pi.json --source picamera --headless "$@"
