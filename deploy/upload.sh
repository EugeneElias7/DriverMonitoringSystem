#!/usr/bin/env bash
# Upload the app to the Raspberry Pi over SSH.
# Run from the project root:  bash deploy/upload.sh
#
# Override the defaults from the environment if needed:
#   PI_USER=pi PI_HOST=192.168.1.50 PI_DIR=/home/pi/dms bash deploy/upload.sh
set -euo pipefail

PI_USER="${PI_USER:-sansolution}"
PI_HOST="${PI_HOST:-192.168.1.14}"
PI_DIR="${PI_DIR:-/home/sansolution/dms}"

ssh "$PI_USER@$PI_HOST" "mkdir -p '$PI_DIR'"
tar --exclude='__pycache__' --exclude='*.pyc' -czf - \
    run.py requirements.txt config_pi.json dms models audio \
    | ssh "$PI_USER@$PI_HOST" "tar -xzf - -C '$PI_DIR'"

echo "uploaded to $PI_HOST:$PI_DIR"
