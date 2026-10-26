# Pi 3 DMS - Core

A self-contained Raspberry Pi 3 demo for driver face, eye, mouth and head
monitoring. It uses **one model**:

```text
models/face_landmarker.task
```

It does not use the old smoking, phone, bottle or dashboard models. Those were
removed from this deployment path because they are too heavy for a Pi 3 and are
not validated DMS classes. The core is the part that can run at a useful rate
on the Pi.

The demo is a research prototype. It is not a safety device and makes no
accuracy, precision, recall or fatigue-diagnosis claim.

## What is different in this version

- 1080p camera module is supported for capture. The app keeps a **960x540**
  browser stream and runs inference on **480x270** by default.
- Capture and inference are separate threads. The processor always takes the
  newest frame, not an old queued frame. This is what removes most of the
  perceived lag.
- Long metric text is no longer drawn on the video. The video has a large,
  high-contrast state badge and only one short measurement line. Full numbers
  are in the HTML side panel.
- No heavy optional detectors run on the Pi.
- Eye closure, smoothing, head-angle guards and uncertainty handling are
  included in the local state engine.
- `DROWSY` is only raised for a current sustained eye closure. A blink, a
  missing face or poor eye geometry cannot become drowsiness.

## Folder contents

```text
pi3_dms/
  app.py                  # camera, face model, state engine, web UI, overlay
  models/
    face_landmarker.task  # the only model (3.6 MB)
  requirements.txt
  install_pi.sh
  run_pi.sh
  README.md
```

## 1. Test it on a desktop first

From the repository root:

```powershell
cd pi3_dms
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py --open
```

The default OpenCV webcam is camera 0. The page is
`http://127.0.0.1:8000`.

To test without a camera:

```powershell
python app.py --image C:\path\to\a\face.jpg --open
```

If the page opens but the video is blank, check the terminal and the
`/healthz` endpoint. The camera error is also shown in the Performance card.

## 2. Copy it to the Pi 3

The simple route is to copy this folder over SSH:

```bash
scp -r pi3_dms pi@raspberrypi.local:~/
ssh pi@raspberrypi.local
cd ~/pi3_dms
bash install_pi.sh
bash run_pi.sh
```

Then open this from a laptop or phone on the same network. The page has no
login; keep the Pi on a trusted local network:

```text
http://raspberrypi.local:8000
```

If `.local` does not resolve, use the Pi's IP address:

```text
http://192.168.1.50:8000
```

### Important Pi 3 requirement

Use **64-bit Raspberry Pi OS**. The current MediaPipe Python wheels are an
aarch64/64-bit target; a 32-bit Pi OS install is not the supported path.

The installer uses the system Picamera2 package (`--system-site-packages`) and
installs Flask, NumPy, OpenCV and MediaPipe (MediaPipe also pulls its own Python
dependencies). A CPython 3.11 aarch64 MediaPipe 0.10.18 wheel is available from
PyPI, and Raspberry Pi OS Bookworm uses Python 3.11. That is not proof that the
runtime is fast enough on a Pi 3; it still has to be measured on the target
board. If `pip install mediapipe` cannot find a matching wheel, do not pretend
the install succeeded. Use a 64-bit image and a Python version for which
MediaPipe publishes a wheel, or run the desktop version on a more capable
machine.

## 3. Camera setup

### Raspberry Pi Camera Module (CSI, 1080p)

The default `run_pi.sh` uses:

```text
camera backend : auto (Picamera2 first, OpenCV fallback)
capture        : 1920x1080 at 15 fps
browser stream : 960x540
inference      : 480x270
```

Check that the camera is visible:

```bash
libcamera-hello --list-cameras    # older Raspberry Pi OS
rpicam-hello --list-cameras      # Bookworm
libcamera-still -o camera-test.jpg
rpicam-still -o camera-test.jpg  # Bookworm
```

The equivalent explicit Picamera2 command is:

```bash
python3 app.py \
  --host 0.0.0.0 --port 8000 \
  --camera-backend picamera2 \
  --capture-width 1920 --capture-height 1080 --camera-fps 15 \
  --stream-width 960 --stream-height 540 \
  --process-width 480 --process-height 270
```

The camera can be a 1080p module. The app does not try to run face inference on
1080p. It captures the camera stream, displays a 960x540 stream, and processes a
480x270 frame. This is the practical split for a Pi 3.

### USB camera

```bash
python3 app.py --camera-backend opencv --camera 0 \
  --capture-width 1920 --capture-height 1080 --camera-fps 15 \
  --process-width 480 --process-height 270
```

If the USB camera negotiates a low frame rate at 1080p, use 1280x720 capture.
The browser stream and inference size remain the same.

### Night / IR camera

```bash
python3 app.py --night --camera-backend picamera2
```

`--night` converts to grayscale and applies CLAHE. For a colour camera with a
purple cast, the default gray-world correction is already on. Use
`--no-tint-fix` to inspect the raw image.

### Camera focus and image quality

- Camera Module 3 has autofocus. `app.py` requests continuous autofocus through
  Picamera2. Make sure the driver's face is in the focus area.
- Clean the lens. A dirty lens is a common cause of an unclear face image.
- Avoid a bright window directly behind the driver.
- A face should occupy roughly 25-50 percent of the frame width.
- If the face is small, the app reports `TRACKING UNCERTAIN` instead of making
  an unreliable eye decision.

## 4. Performance on a Pi 3

A Pi 3 has four slow Cortex-A53 cores, no GPU/NPU acceleration for this
MediaPipe task, and 1 GB RAM. **1080p inference at 30 fps is not realistic.**
A useful target for this core is roughly 6-12 processed frames per second,
depending on the image, cooling and OS build. Do not promise 30 fps for the Pi
build.

The UI shows:

- processed FPS
- processing latency
- capture-to-process frame age
- camera backend and capture resolution
- stream and inference resolution

If it is below the target, tune in this order:

```bash
# Lower inference work first
python3 app.py --process-width 320 --process-height 180

# Then lower capture load
python3 app.py --capture-width 1280 --capture-height 720 --camera-fps 10

# Then lower display/JPEG cost
python3 app.py --stream-width 800 --stream-height 450 --jpeg-quality 75

# Then reduce face frames deliberately (optional)
python3 app.py --camera-fps 10
```

`--process-width 320` is faster but less precise for a distant face.
`--process-width 480` is the default compromise. Measure on the actual Pi;
desktop timings do not transfer.

## 5. Read the display

### Video overlay

The video deliberately shows only:

- a large state badge (`ATTENTIVE`, `EYES CLOSED`, `DROWSY`, etc.)
- a short reason
- eye and mouth landmark outlines
- a head-orientation arrow
- one compact line with smoothed EAR, eye state, closure and head direction
- processed FPS, latency and frame age

The detailed values are in the side panel, where the browser can render text
crisply. This avoids the old problem of squeezing tiny text into the video.

### State rules

| State | Meaning |
|---|---|
| `ATTENTIVE` | Face tracked, eyes open, head forward |
| `TRACKING UNCERTAIN` | Face missing, face too small, or eye geometry unreliable |
| `BLINK` | Very short closure (below 500 ms); the detail panel still shows `closed` |
| `EYES CLOSED` | Current closure is 500-899 ms, below the sustained alert threshold |
| `DROWSY` | Current closure >= 900 ms (microsleep reported at 1500 ms) |
| `YAWNING` | Smoothed MAR >= 0.60 for >= 800 ms |
| `GLANCE` | Head turned, but less than 1.5 s |
| `DISTRACTED` | Head turned for >= 1.5 s |

EAR uses a 3-frame median. Reopening is immediate: an open-eye frame is never
held closed by the median. The closed threshold is capped at `0.20` so a
drifting high baseline cannot call a normal open eye closed, and floored at
`0.15` so a low personal baseline cannot hide a real closure. A missing face
resets the closure timer and reports uncertainty instead of drowsiness.

PERCLOS is shown separately as a 60-second measurement. It is not turned into
`DROWSY` by itself.

## 6. Tuning flags

```text
--process-width 480        inference width; lower is faster
--process-height 270
--stream-width 960         browser display width; higher is clearer but heavier
--stream-height 540
--camera-fps 15            requested capture rate
--jpeg-quality 82          MJPEG quality
--drowsy-ms 900            closure needed for DROWSY
--microsleep-ms 1500
--yawn-mar 0.60
--yawn-ms 800
--distracted-ms 1500
--min-face-width 40        uncertainty if face is too small in inference pixels
--mirror                   mirror the image
--night                    grayscale + CLAHE for IR/night
--no-tint-fix              disable gray-world colour correction
--image path.jpg           still-image test mode
```

## 7. Run as a service (optional)

Create `/etc/systemd/system/dms.service`:

```ini
[Unit]
Description=Pi 3 DMS core
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pi3_dms
ExecStart=/home/pi/pi3_dms/.venv/bin/python /home/pi/pi3_dms/app.py --host 0.0.0.0 --port 8000 --camera-backend picamera2 --capture-width 1920 --capture-height 1080 --camera-fps 15 --stream-width 960 --stream-height 540 --process-width 480 --process-height 270
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now dms.service
sudo systemctl status dms.service
journalctl -u dms.service -f
```

## 8. Model and licence

`models/face_landmarker.task` is the MediaPipe Face Landmarker task model used
by the app. SHA-256:

```text
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
```

Keep the model file next to `app.py` under `models/`. The app has no network
download step.

MediaPipe and its model files have their own licence and model-card terms.
Review them before redistribution or commercial use. Camera frames can contain
personal/biometric data; keep them local and obtain consent before recording.

## 9. Troubleshooting

**Page opens, video is black**
- Check the terminal for `[camera]` errors.
- Run `libcamera-still -o test.jpg`.
- Try `--camera-backend opencv --camera 0`.
- Make sure no other process is using the camera.

**`TRACKING UNCERTAIN` all the time**
- Move the face closer / improve lighting.
- Check that the face is at least `--min-face-width` pixels wide in the
  inference frame.
- Turn the head back toward the camera.
- Try `--process-width 640` on a desktop to confirm the model works, then use
  the largest Pi resolution that still meets the FPS target.

**FPS is low**
- Use `--process-width 320 --process-height 180`.
- Use 1280x720 or 640x480 capture.
- Use `--stream-width 800 --stream-height 450`.
- Close other applications; use a heatsink/fan; check for throttling with
  `vcgencmd get_throttled`.
- Do not add smoking/phone/COCO models to the Pi build.

**The image has a purple cast**
- Try `--night`.
- Try `--no-tint-fix` to compare.
- Fix the camera white balance/IR filter first; software cannot fully repair a
  bad sensor image.

**MediaPipe import fails**
- Confirm 64-bit OS: `uname -m` should print `aarch64`.
- Confirm Python version: `python3 --version`.
- Check that `pip install mediapipe` found a wheel; a failed build is not a
  supported install.
- If no wheel exists for that Pi OS/Python combination, use a newer Pi or a
  desktop for the demo; do not claim the Pi path is working until it runs.

## 10. Scope and next steps

This build is deliberately limited to what can plausibly run on a Pi 3:

- face landmarks
- EAR blink/closure
- MAR yawning
- head orientation and sustained distraction
- clear uncertainty reporting

It does **not** infer alcohol, seatbelt use, smoking, phone use or drinking on
the Pi. Those need separate validated models/sensors and are not part of this
folder.
