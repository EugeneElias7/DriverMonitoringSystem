# Pi 3 DMS

A Raspberry Pi 3 driver-monitoring research prototype with:

1. an always-on **face core** (face tracking, eyes, blinks, closure/drowsiness,
   yawning, head pose, looking-away, PERCLOS); and
2. an **optional object/behaviour module** (phone, bottle/drinking and, when a
   suitable model is supplied, cigarette/smoking evidence).

The face core and the object module share **one camera capture path**. Object
inference is periodic, asynchronous and disabled by default. A missing,
broken or slow object model cannot prevent the face core from starting.

This is not a safety device, medical device or production DMS. No accuracy,
precision, recall, fatigue-diagnosis, Pi 3 performance or safety-certification
claim is made. See [Support matrix](#support-matrix) before using a feature.

## What is implemented

### Face core

- MediaPipe Face Landmarker retained as the face model.
- One capture thread; the processor always consumes the newest frame.
- Adaptive EAR closure threshold with a conservative floor/cap.
- One- and two-frame raw blink counting plus smoothed closure/drowsiness.
- Sustained closure is required for `DROWSY`; PERCLOS is reported separately.
- Yawning is independent of eye state, with a start threshold and release
  hysteresis so the state does not flicker around the boundary.
- Drowsiness, yawning and attention are separate states. The large badge is a
  display priority, not a replacement for the independent state payload.
- Head pose dead zone to avoid a permanently “turned” state from camera mounting.
- A transparent **tracking-confidence heuristic** from face size, pose and
  continuity. It is **not** a MediaPipe probability and is not a validation
  metric.
- Duplicate/out-of-order observation timestamps are rejected.
- Face loss or failed quality gates produce `TRACKING UNCERTAIN`, not
  `ATTENTIVE`.

### Optional object/behaviour module

- Local `mediapipe` TFLite backend using the shipped COCO control model.
- HTTP backend for a separate object-inference service.
- One shared frame source; no second camera and no unbounded detection queue.
- Periodically samples the newest inference frame and rejects stale results.
- Detected objects are displayed separately from suspected behaviours.
- Phone use, drinking and smoking are independent and can be active together.
- Behaviour evidence requires an explicit matching class, face/mouth proximity,
  persistence and multiple observations.
- A bottle is never relabelled as a cigarette. Smoking requires an explicit
  `cigarette` detector label.
- Face loss, object staleness and unavailable modules clear behaviour evidence
  instead of producing an alert.

### Camera and colour

- 1080p capture, 960x540 browser stream, 480x270 inference by default.
- Picamera2 ISP controls are configurable: AWB mode, red/blue gains, exposure
  and analogue gain.
- Software colour modes: raw (`off`), soft gray-world (`soft`), full gray-world
  (`grayworld`) and grayscale + CLAHE (`night`).
- The **raw, uncorrected frame is preserved** and served at
  `/snapshot_raw.jpg` for debugging.
- Conservative defaults. Do not use aggressive colour correction to hide a
  hardware/IR-filter problem. See [Purple/magenta tint](#purplemagenta-tint).

## Quick start on a laptop

From the repository root:

```powershell
cd pi3_dms
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py --open
```

The default webcam is camera 0. The browser opens at `http://127.0.0.1:8000`.
The face core runs with objects disabled.

Enable the shipped experimental object control model:

```powershell
python app.py --objects
```

The model in `models/object/efficientdet_lite0.tflite` is an official COCO
control model. It supports phone and bottle classes; it has **no cigarette
class**. It is disabled by default and is not validated for driver monitoring.

### Useful desktop options

```text
--camera 0                         camera index
--camera-backend auto|picamera2|opencv
--process-width 480 --process-height 270
--tint-mode off|soft|grayworld|night
--no-tint-fix                      same as --tint-mode off
--night                            grayscale + CLAHE
--mirror                           mirror the displayed/processed image
--objects                          enable object inference
--object-backend mediapipe|http
--object-classes phone,bottle      accepted canonical classes
--object-interval-ms 500           periodic object inference
--object-stale-ms 1500             detector result age limit
--object-phone-ms 1200             phone-use persistence window
--object-drink-ms 1000             drinking persistence window
--object-smoke-ms 1000             smoking persistence window
--object-min-observations 3
```

Run `python app.py --help` for the full list.

If the dashboard reports `face too small`, move the camera closer or increase
`--process-width/--process-height`. Lowering `--min-face-width` alone can make
the landmarks noisier; the tracking confidence and quality gates remain
conservative.

## Quick start on a Raspberry Pi 3

Use 64-bit Raspberry Pi OS. Camera Module support is installed through
`python3-picamera2`/`libcamera`; the face path can fall back to OpenCV for a USB
camera.

```bash
cd ~/pi3_dms
bash install_pi.sh
bash run_pi.sh
```

Open `http://<pi-address>:8000`. The default launch is the face core only.
Object inference is deliberately off because its Pi 3 cost is unverified.

To opt in on the Pi:

```bash
OBJECTS=1 bash run_pi.sh
# or explicitly:
bash run_pi.sh --objects --object-backend mediapipe
# separate service:
bash run_pi.sh --objects --object-backend http \
  --object-service-url http://127.0.0.1:8010/detect
```

Start with the core-only path, measure it, then try the object path. Do not
assume desktop timings transfer to a Pi 3.

### Pi camera setup

`run_pi.sh` uses camera backend `auto` (Picamera2 first, OpenCV fallback),
requests 1920x1080 at 15 fps, displays 960x540 and infers at 480x270. Check
that the camera is visible:

```bash
libcamera-hello --list-cameras    # older Raspberry Pi OS
rpicam-hello --list-cameras       # Bookworm
libcamera-still -o camera-test.jpg
rpicam-still -o camera-test.jpg   # Bookworm
```

The equivalent explicit Picamera2 launch is:

```bash
python3 app.py \
  --host 0.0.0.0 --port 8000 \
  --camera-backend picamera2 \
  --capture-width 1920 --capture-height 1080 --camera-fps 15 \
  --stream-width 960 --stream-height 540 \
  --process-width 480 --process-height 270
```

For a USB camera:

```bash
python3 app.py --camera-backend opencv --camera 0 \
  --capture-width 1280 --capture-height 720 --camera-fps 15 \
  --process-width 480 --process-height 270
```

Use 64-bit Raspberry Pi OS (`uname -m` should print `aarch64`). The installer
uses the system Picamera2 package with `--system-site-packages`. MediaPipe must
have an aarch64 wheel for the installed Python; a source build or a failed wheel
install is not a supported installation. The page has no login, so run it only
on a trusted network or behind an authenticated reverse proxy.

Camera Module 3 autofocus is requested through Picamera2. Clean the lens, avoid
a bright window directly behind the driver, and aim for a face that occupies
roughly 25-50 percent of the frame width. See
[Purple/magenta tint](#purplemagenta-tint) before changing colour settings.

## Running as a service (optional)

Create `/etc/systemd/system/dms.service` using the same interpreter and model
paths as the working install:

```ini
[Unit]
Description=Pi 3 DMS face core
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pi3_dms
ExecStart=/home/pi/pi3_dms/.venv/bin/python /home/pi/pi3_dms/app.py --host 0.0.0.0 --port 8000 --camera-backend picamera2 --capture-width 1920 --capture-height 1080 --camera-fps 15 --stream-width 960 --stream-height 540 --process-width 480 --process-height 270 --tint-mode soft
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

Object inference is deliberately absent from this unit. Add `--objects` (and
measure the Pi first) only if you accept the extra CPU/RAM cost.

## Architecture

```text
CameraWorker (one capture thread)
        |
        v
FrameStore: newest FramePacket only (stream, process, raw, timestamp, sequence)
        |                                   |
        v                                   v
Processor thread                      Optional ObjectWorker (periodic)
  preprocess (configurable)             preprocess (same mode)
  face model                             local TFLite or HTTP detector
  MeasurementEngine                      timestamped detections
        |                                   |
        +---------------+-------------------+
                        v
              FusionEngine / BehaviourFusion
                        |
                        v
          /status, annotated stream, raw snapshot
```

Key modules:

| File | Responsibility |
|---|---|
| `dms/core.py` | MediaPipe face model, landmark geometry, temporal eye/mouth/head measurements |
| `dms/camera.py` | Single camera path, newest-frame store, colour preprocessing |
| `dms/objects.py` | Optional local/HTTP detectors, canonical labels, periodic worker |
| `dms/fusion.py` | Independent behaviour evidence and central temporal fusion |
| `dms/resources.py` | Optional CPU/RAM/temperature sampling |
| `app.py` | Flask dashboard, orchestration, annotations, status payload |

## Object model and service

### Local model

Default path:

```text
models/object/efficientdet_lite0.tflite
SHA-256 2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b
```

It is an Apache-2.0 TensorFlow/TFLite model-collection control model. Its COCO
classes include phone and bottle, but **not cigarette**. It is included only as
an experimental control and must not be described as a smoking detector.

To use another TFLite model with metadata:

```bash
python app.py --objects \
  --object-backend mediapipe \
  --object-model /path/to/model.tflite \
  --object-classes phone,bottle,cigarette
```

Only add a model after checking labels, licence/provenance, hardware
compatibility and real-world performance. Do not restore the old cigarette
checkpoints: their recorded licence/provenance is AGPL/unknown/unclear and they
are not validated for this package. See `models/object/README.md`.

### Separate HTTP service

The client and protocol are documented in
[`docs/object_service.md`](docs/object_service.md). The service receives the
already-captured JPEG and returns timestamped, normalized detections. It must
not open a second camera. This keeps a heavy or separately licensed model out of
the face process:

```bash
python app.py --objects --object-backend http \
  --object-service-url http://127.0.0.1:8010/detect
```

The object module reports `disabled`, `unavailable`, `starting`, `running`,
`stale` or `error` explicitly. Core start-up does not depend on the service.

## Behaviour rules and limits

A detection is not a behaviour. The fusion layer requires:

- phone: explicit `phone` label near the face, persistence and observations;
- drinking: explicit `bottle` label near the mouth, persistence and observations;
- smoking: explicit `cigarette` label near the mouth, persistence and observations.

The shipped COCO model cannot produce the `cigarette` class. Therefore the
shipped configuration cannot validate or report smoking. A phone or bottle
never creates a smoking state. Low-confidence, old and face-unassociated
detections are rejected. Behaviour output is labelled `suspected`, not proven.

There is no hand landmark model, no cigarette-specific validated model and no
labelled driver-video dataset in this repository. Do not claim validated
smoking, phone-use or drinking accuracy.

## Purple/magenta tint

A purple or magenta image is usually an ISP white-balance issue, a missing IR
cut filter, mixed IR illumination, or a camera without useful AWB. Software
gray-world correction can mask the symptom and can make skin/eye landmarks
worse. The app therefore:

- exposes Picamera2 AWB/colour-gain controls;
- defaults to a **soft** gray-world mode, not an aggressive full correction;
- allows `--tint-mode off` / `--no-tint-fix` for raw inference;
- keeps `/snapshot_raw.jpg` available for comparison.

Investigate in this order:

1. Check the camera module's IR-cut filter and IR illuminator.
2. Test daylight, low light and IR separately.
3. Compare `--tint-mode off`, `soft`, `grayworld` and `night` on the same scene.
4. Only then adjust AWB/gains or exposure.

Do not use colour correction as evidence that the camera hardware is healthy.

## Face model and licence

`models/face_landmarker.task` is the MediaPipe Face Landmarker task model used by
the core. SHA-256:

```text
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
```

The app does not download models at runtime. MediaPipe and its model files have
their own licence and model-card terms; review them before redistribution or
commercial use. Camera frames can contain personal/biometric data: keep them
local, obtain consent and follow applicable privacy law.

## Troubleshooting

**Page opens but the video is black**

- Check the terminal for `[camera]` errors and open `/healthz`.
- Run `rpicam-still -o test.jpg` (or `libcamera-still`).
- Try `--camera-backend opencv --camera 0`.
- Make sure no other process is using the camera.

**`TRACKING UNCERTAIN` all the time**

- Improve lighting and move the face closer or increase
  `--process-width/--process-height`.
- Check the face is at least `--min-face-width` pixels wide in the inference
  frame.
- Turn the head back toward the camera. A missing head pose is reported as
  uncertain rather than `ATTENTIVE`.
- Lowering `--min-face-width` alone is not a substitute for adequate resolution.

**FPS is low**

- Reduce inference first: `--process-width 320 --process-height 180`.
- Try 1280x720 or 640x480 capture.
- Reduce display cost: `--stream-width 800 --stream-height 450`.
- Check for throttling with `vcgencmd get_throttled`; use cooling.
- Start with objects disabled and measure again before enabling them.

**Purple/magenta image**

- Check the IR-cut filter and IR illuminator; test daylight, low light and IR
  separately.
- Compare `--tint-mode off` and `--tint-mode soft` on the same scene.
- Check `/snapshot_raw.jpg`; do not use software correction to hide a sensor or
  white-balance problem.
- Use Picamera2 AWB/gain controls only after the hardware check.

**MediaPipe import fails**

- Confirm 64-bit OS (`uname -m` prints `aarch64`) and a Python version with an
  aarch64 MediaPipe wheel.
- A failed source build is not a supported install. Do not claim the Pi path is
  working until the model loads and a camera frame is processed.

## Tests

Simulated tests do not open a camera and do not require a face image:

```bash
cd pi3_dms
python -m unittest discover -s tests -t . -v
```

The current suite contains 42 tests and passes in this workspace. It covers
closure/blink timing, yawn hysteresis, attention/drowsiness independence, stale
timestamps, object status, proximity/persistence, bottle vs cigarette
separation, simultaneous behaviours, HTTP parsing, colour helpers and
configuration.

The camera smoke test is separate and requires a real camera or image:

```bash
python camera_smoke_test.py --camera 0 --seconds 8
python camera_smoke_test.py --image face.jpg --seconds 2
python camera_smoke_test.py --camera 0 --objects --seconds 8
```

It verifies that the single camera path produces processed frames. It does not
assert that a face is present or that a detection is accurate.

## Benchmarks

Run on the target board with a real face image:

```bash
python benchmark.py --image face.jpg --iterations 20 --width 480 --height 270
python benchmark.py --image face.jpg --objects --iterations 20 \
  --json-out benchmark.json --markdown-out benchmark.md
```

Only benchmark numbers actually measured are reported. Desktop results are in
[`BENCHMARKS.md`](BENCHMARKS.md). **Raspberry Pi 3 results are pending** because
no Pi 3/Camera Module hardware or Pi footage was available in this workspace.
Desktop numbers must not be presented as Pi performance.

## Support matrix

| Capability | Status | Notes |
|---|---|---|
| Face tracking and eye/mouth/head measurements | Implemented, simulated tests | MediaPipe core; real-world accuracy not validated |
| Blink/closure/drowsiness/PERCLOS | Implemented, simulated tests | Sustained closure required for `DROWSY`; PERCLOS separate |
| Yawning | Implemented, simulated tests | Threshold/hysteresis; talking-vs-yawn accuracy unvalidated |
| Looking-away/attention | Implemented, simulated tests | Head pose only; no eye-gaze model |
| Tracking confidence | Implemented | Transparent heuristic, not a model probability |
| Single camera / no duplicate capture | Implemented | One `CameraWorker` and newest-frame store |
| Phone detection | Experimental | COCO control model; not validated for DMS |
| Bottle detection / drinking evidence | Experimental | Requires explicit bottle class and mouth proximity |
| Cigarette detection / smoking evidence | **Unavailable in shipped config** | No rights-cleared validated cigarette model; HTTP extension only |
| Alcohol | **Unavailable** | No physical breath sensor; video cannot measure it |
| Raspberry Pi 3 performance | **Unmeasured** | Desktop benchmark only; target-board measurement required |
| Separate object service | Client + protocol implemented | No service/model shipped; user supplies it |
| Production/safety certification | **None** | Research prototype |

## Privacy and safety

- Keep camera frames and recordings local unless the user explicitly configures
  otherwise.
- Obtain consent from drivers/operators and follow applicable workplace/privacy
  law.
- Do not make disciplinary, medical or safety decisions from this prototype.
- The dashboard shows “suspected” behaviour evidence only; it does not prove
  that an event occurred.
