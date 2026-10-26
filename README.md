# Driver Monitoring - Raspberry Pi 3 Core

This repository now contains one clean deployment path for a truck-driver DMS
on a Raspberry Pi 3 with a Camera Module (including a 1080p module):

```text
pi3_dms/
```

The Pi package is self-contained and uses exactly one model:

```text
pi3_dms/models/face_landmarker.task
```

It tracks the face, eyes, mouth and head orientation. It does not run the old
smoking, phone, bottle or dashboard detectors. Those models were removed from
this deployment because they are too heavy for a Pi 3 and are not validated DMS
classes.

## Quick start on a laptop

From this folder:

```powershell
cd pi3_dms
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py --open
```

The default webcam is camera 0. Open `http://127.0.0.1:8000`.

## Quick start on a Pi 3

Copy the `pi3_dms` folder to the Pi, then:

```bash
cd ~/pi3_dms
bash install_pi.sh
bash run_pi.sh
```

Open `http://raspberrypi.local:8000` from another device on the same network.

Use **64-bit Raspberry Pi OS**. See
[`pi3_dms/README.md`](pi3_dms/README.md) for camera setup, the exact launch
command, performance tuning, night/IR mode and a systemd service.

## Video and latency design

- Camera capture may use a 1080p module: the app requests a 1920x1080 camera
  stream where the backend supports it.
- The browser receives a 960x540 MJPEG stream so overlay text is large and
  readable.
- Inference runs at 480x270 by default, not at 1080p.
- Capture and processing are separate threads. The processor always consumes
  the newest captured frame, never an old queue, so the display does not drift
  behind the driver.
- The video shows only a large state badge, a short reason and key
  measurements. Detailed values are in the HTML side panel where text is
  always crisp.
- The heavy optional models are not present, so there is no COCO/smoking
  inference competing with the face model.

## Realistic performance

A Pi 3 is not a 30 FPS 1080p inference device. The useful target for this
face-only core is roughly 6-12 processed frames per second, depending on
cooling, camera settings and the chosen inference size. The UI reports the
actual processed FPS, processing latency and capture-to-process frame age.

If it is too slow, lower inference first:

```bash
python3 app.py --process-width 320 --process-height 180
```

Then lower capture or stream size. Do not add the removed object models back to
the Pi and expect real-time operation.

## What the states mean

- `ATTENTIVE`: face tracked, eyes open, head forward.
- `TRACKING UNCERTAIN`: missing face, face too small, or unreliable eye
  geometry. This is not drowsiness.
- `BLINK`: a normal short closure below 500 ms.
- `EYES CLOSED`: current closure is 500-899 ms, below the sustained alert
  threshold.
- `DROWSY`: current closure at or above 900 ms; microsleep is counted at
  1500 ms.
- `YAWNING`: smoothed mouth opening above threshold for at least 800 ms.
- `GLANCE`: head turned briefly.
- `DISTRACTED`: head turned for at least 1.5 s.

PERCLOS is displayed separately and is not used by itself to raise `DROWSY`.

## Scope and limits

This is a research prototype, not a safety or medical device. No accuracy,
precision, recall, fatigue-diagnosis or safety claim is made from unlabelled
camera sessions. Alcohol cannot be inferred from video and needs a physical
sensor. Phone, smoking and drinking detection need separate validated models
and are intentionally not part of the Pi 3 build.

A CPython 3.11 aarch64 MediaPipe wheel is available, but the Pi 3 runtime,
camera integration and the 6-12 FPS target still need to be measured on the
actual board. Do not claim otherwise until that has been done.

## Repository layout

```text
.
  README.md
  .gitignore
  .gitattributes
  pi3_dms/
    app.py
    README.md
    requirements.txt
    install_pi.sh
    run_pi.sh
    models/
      face_landmarker.task
      README.md
```
