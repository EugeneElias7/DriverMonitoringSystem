# Driver Monitoring System (DMS)

A small, local driver-monitoring app for a PC webcam and a Raspberry Pi 3. It
watches the driver's face and the cabin, shows a clean live preview, and speaks
a warning when it sees drowsiness, phone use, or smoking.

The face path uses OpenCV YuNet + Google's 478-point face-landmark ONNX on
both PC and Pi (no MediaPipe runtime, which requires ARM AES instructions the
Pi 3 lacks). Object detection is OpenCV DNN over ONNX YOLO models.

## What it does

| Area    | Signal                                 | Spoken alert          |
| ------- | -------------------------------------- | --------------------- |
| Face    | eyes closed, PERCLOS, yawning          | "Please take a break" |
| Face    | no face for 2 seconds                  | status only           |
| Objects | phone at the ear or held near the face | "Do not use phone"    |
| Objects | cigarette near the mouth               | "Stop smoking"        |
| Objects | drinking, eating, no seatbelt          | status only           |
| Head    | looking away from the road             | status only           |

The spoken warnings repeat every `alert_cooldown_s` (2.5 s by default) while
the condition stays active. Smoking and phone conditions are held through a
missed detector pass, so a dropped frame does not silence a warning. The
alert stops when the condition clears.

The preview shows:

- the driver state as one word: **Attentive**, **Drowsy**, **Sleeping**,
  **Distracted - phone**, **Distracted - looking away**, **Smoking**,
  **Drinking**, **Eating**, or **No driver**;
- live **EAR** (eye aspect ratio) and **MAR** (mouth aspect ratio) values;
- face corner brackets and fresh object boxes;
- the active warning lines at the bottom.

Stale detector boxes (older than 1.5 s) are not drawn, because on a Pi one
YOLO pass can take several seconds.

## Layout

```text
.
├── run.py                    # entry point
├── config_pc.json            # PC webcam profile
├── config_pi.json            # Raspberry Pi profile
├── requirements.txt
├── dms/                      # application package
│   ├── app.py                # camera loop and preview
│   ├── face.py               # face observation type + fallback analyzer
│   ├── face_onnx.py          # YuNet + 478-point landmarks (active path)
│   ├── objects.py            # YOLOv8/YOLO11 ONNX detectors
│   ├── behaviors.py          # drowsiness, smoking, phone rules
│   ├── pipeline.py           # perception and analysis wiring
│   ├── state.py              # one-word driver state
│   ├── alerts.py             # alert de-duplication
│   ├── audio.py              # WAV playback
│   ├── config.py             # typed configuration
│   ├── geometry.py           # EAR, MAR, head pose
│   └── temporal.py           # time windows and holds
├── models/                   # flat; no subdirectories
├── audio/                    # spoken WAV phrases
├── deploy/upload.sh          # copy the app to the Pi
└── tests/test_smoke.py       # config and model-load checks
```

## Models

| File                                         | Job                 | Input → output              | Classes                                      |
| -------------------------------------------- | ------------------- | ---------------------------- | -------------------------------------------- |
| `models/face_detection_yunet_2023mar.onnx` | face boxes          | OpenCV YuNet                 | face                                         |
| `models/face_landmarks.onnx`               | 478 face landmarks  | 1×256×256×3 → 478×3         | eyes, mouth, head pose                       |
| `models/2d106det.onnx`                     | fallback landmarks  | 192 × 192                   | 106 points                                   |
| `models/best.onnx`                         | driver objects      | 1×3×640×640 → 1×9×8400 | Drinking, Eating, Seatbelt (phone and cigarette excluded) |
| `models/cigarette_smokers_v4.onnx`         | dedicated cigarette | 1×3×640×640 → 1×5×8400 | cigarette                                    |
| `models/phone_yolo11n.onnx`                | phone               | 1×3×640×640 → 1×84×8400 | COCO `cell phone` only                       |

`models/driver_classes.txt`, `models/cigarette_v4_classes.txt`, and
`models/coco_classes.txt` must stay in the same order as their model's class
ids. The runtime lowercases names.

The runtime expects the Ultralytics detection layout
`[1, 4 + classes, 8400]` (no embedded NMS). It letterboxes to 640, scales
`1/255`, converts BGR → RGB, applies per-class confidence floors, and runs NMS.

### Adding or replacing the cigarette model

1. Export YOLOv8/YOLO11 to ONNX with `imgsz=640`, static batch 1, and no NMS.
2. Put the file in `models/` (flat, no subfolder).
3. Update `models/cigarette_v4_classes.txt` to the model's class order.
4. Point `detectors[1].model_path` and `detectors[1].classes_path` in the
   config at the new files.
5. If the new model replaces `best.onnx`, keep its class order and update
   `models/driver_classes.txt`.

The dedicated cigarette model is the only source of cigarette detections.
`best.onnx` still lists a cigarette class but it is masked out: on webcam
footage it produced false boxes, including occasional full-frame ones. The
dedicated model runs on both PC and Pi; `config_pi.json` samples it at 0.5 Hz
and lengthens the confirmation time accordingly.

## Run on the PC

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py --config config_pc.json --source 0
```

Press `q` or `Esc` to quit. Useful flags:

```text
--headless            no preview window
--max-seconds N       stop after N seconds
--realtime            play video files at native speed
--record PATH         write the annotated preview to an MP4
--log alerts.csv      optional CSV alert log (off by default)
```

## Run on the Raspberry Pi 3

Use 64-bit Raspberry Pi OS. The defaults in `deploy/upload.sh` are
`sansolution@192.168.1.14:/home/sansolution/dms`; override with `PI_USER`,
`PI_HOST`, and `PI_DIR` if your setup differs.

From the PC:

```powershell
bash deploy/upload.sh
```

On the Pi:

```bash
ssh sansolution@192.168.1.14
cd /home/sansolution/dms
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -r requirements.txt
python run.py --config config_pi.json --source picamera
```

Use `--source 0` for a USB webcam. The `picamera` source needs
`sudo apt install python3-picamera2` and a venv created with
`--system-site-packages`.

If `opencv-python` does not install on the Pi, use
`opencv-python-headless`; the preview window is then unavailable, so run with
`--headless` and read the status lines.

Do **not** install MediaPipe on the Pi 3. The OpenCV face path is the supported
path.

## Audio

`audio/` contains:

| File                        | Used for              |
| --------------------------- | --------------------- |
| `please_take_a_break.wav` | drowsiness / sleeping |
| `do_not_use_phone.wav`    | phone use             |
| `stop_smoking.wav`        | smoking               |
| `system_on.wav`           | startup confirmation  |

Windows uses `winsound`. Linux uses `aplay`; set `audio.device` to the right
ALSA device if the default is not the speaker (for example
`"plughw:Headphones"`). If a WAV is missing, the app falls back to SAPI on
Windows or `espeak`/`espeak-ng` on Linux.

## Configuration

`config_pc.json` and `config_pi.json` are the only profiles. Important keys:

- `face.detector_model_path`, `face.landmark_model_path`,
  `face.min_detection_confidence`
- `detectors[]`: `name`, `model_path`, `classes_path`, `conf_threshold`,
  `class_conf_overrides`, `target_hz`, `enabled`
- `thresholds.cigarette_s`, `cigarette_high_conf`, `cigarette_low_conf`,
  `cigarette_low_s`, `cigarette_gate_span`, `cigarette_max_size_span`
- `thresholds.phone_call_s`, `phone_in_hand_s`, `phone_hold_s`
- `thresholds.ear_closed`, `microsleep_s`, `yawn_mar`, `yawn_s`
- `alert_cooldown_s`: repeat cadence while a condition stays active
- `audio.device`: ALSA device on Linux
- `display`: `false` for headless

Smoking is deliberately conservative. A cigarette box must sit within 0.30
face-heights of the mouth, be no larger than 0.55 face-heights, and stay
detected for 1.0 s at confidence 0.60+ (or 1.5 s at 0.45+). The dedicated
model runs at 2 Hz on the PC, so this spans at least two detector passes. On
the Pi the same bar is 2.5 s / 3.0 s because the model runs at 0.5 Hz. A real
puff stays at the lips long enough to alert; a single flickering box cannot.

A disabled detector does not need its model or class file to exist, so a
profile can drop a model without deleting files.

## Tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

The smoke test loads both configs, checks that the model files exist, runs the
face and object models once on a blank frame, and verifies the alert hold.

## Notes and limits

- Object detectors run on background threads, so a slow Pi does not freeze the
  face loop. Each detector has its own `target_hz`.
- Pi 3 frame rate, latency, and thermal behaviour are not measured here.
- The models are trained for a driver-facing camera. A laptop webcam is a
  different view; treat the PC test as a functional check, not an accuracy
  result.
- The Ultralytics exports are AGPL-3.0. Check the model and dataset licences
  before redistributing or using this commercially.
- The app does not record video or upload anything. The optional CSV log is off
  unless `--log` is passed.
