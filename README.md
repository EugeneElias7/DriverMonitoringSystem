# Driver Monitoring - Raspberry Pi 3

This repository contains a research prototype for a Raspberry Pi 3 driver
monitoring system:

```text
pi3_dms/
```

The always-on face core uses the MediaPipe Face Landmarker and tracks the face,
eyes, mouth and head pose. An optional object module adds experimental
phone/bottle detections and separate behaviour evidence. It is disabled by
default and cannot block the face core.

## Quick start (laptop)

```powershell
cd pi3_dms
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py --open
```

Open `http://127.0.0.1:8000`. The default webcam is camera 0.

Enable the optional object control model:

```powershell
python app.py --objects
```

## Quick start (Pi 3)

```bash
cd ~/pi3_dms
bash install_pi.sh
bash run_pi.sh
```

The default Pi launch runs the face core only. Object inference is opt-in because
its Raspberry Pi 3 cost has not been measured:

```bash
OBJECTS=1 bash run_pi.sh
```

## Honest scope

| Area | Status |
|---|---|
| Face/eye/mouth/head core | Implemented; simulated tests pass; real-world accuracy unvalidated |
| Optional phone/bottle detection | Experimental COCO control model; not validated for DMS |
| Cigarette/smoking detection | Unavailable in shipped configuration; no rights-cleared validated model |
| Alcohol | Unavailable; requires a physical breath sensor |
| Raspberry Pi 3 FPS/CPU/RAM/thermals | Not yet measured; desktop benchmark only |
| Production or safety certification | None |

The shipped `efficientdet_lite0.tflite` has phone and bottle classes but **no
cigarette class**. A bottle or phone is never relabelled as a cigarette.
Detected objects are shown separately from suspected behaviours, and every
behaviour needs an explicit matching class, face/mouth proximity, persistence
and multiple observations.

## Documentation

- [`pi3_dms/README.md`](pi3_dms/README.md): architecture, flags, camera/tint
  investigation, tests, benchmarks and support matrix.
- [`pi3_dms/BENCHMARKS.md`](pi3_dms/BENCHMARKS.md): actual desktop measurements
  and the Pi 3 pending status.
- [`pi3_dms/models/object/README.md`](pi3_dms/models/object/README.md): object
  model provenance and restrictions.
- [`pi3_dms/docs/object_service.md`](pi3_dms/docs/object_service.md): optional
  HTTP object-service protocol.

## Tests

```bash
cd pi3_dms
python -m unittest discover -s tests -t . -v
```

Camera tests are intentionally separate:

```bash
python camera_smoke_test.py --camera 0 --seconds 8
```

This is a research prototype. It is not a safety device and makes no medical,
fatigue-diagnosis, accuracy or production-readiness claim.
