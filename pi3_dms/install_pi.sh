#!/usr/bin/env bash
# Install the Pi 3 DMS core on 64-bit Raspberry Pi OS.
# Run this once from the pi3_dms directory.
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "WARNING: this script is intended for 64-bit Raspberry Pi OS (aarch64)." >&2
fi

sudo apt update
sudo apt install -y python3-venv python3-pip python3-picamera2 \
  libgl1 libglib2.0-0 libgomp1 || true
sudo apt install -y libcamera-apps || sudo apt install -y rpicam-apps || true

python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

python - <<'PY'
import cv2
import flask
import mediapipe
print("OpenCV", cv2.__version__)
print("Flask", flask.__version__)
print("MediaPipe", mediapipe.__version__)
PY

echo
echo "Install complete."
echo "Test the camera first:  libcamera-still -o camera-test.jpg"
echo "  (Bookworm:            rpicam-still -o camera-test.jpg)"
echo "Start the DMS:          bash run_pi.sh"
