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

The spoken warnings repeat every `alert_cooldown_s` (2.5 s on the PC, 5 s on
the Pi) while the condition stays active. When several conditions rise at
once the order is fixed: drowsiness first, then phone, then smoking. A phrase
already being spoken is allowed to finish; pending phrases are held in a
priority queue. Smoking and phone conditions are held through a missed
detector pass, so a dropped frame does not silence a warning. The alert stops
when the condition clears.

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
dedicated model runs on both PC and Pi. `config_pi.json` gives it a 2 Hz
start-to-start target; the pass itself is CPU-bound, so the effective cadence
is about one pass plus a short rest. The Pi profile also drops the `best.onnx`
detector entirely: drinking, eating and seatbelt are status-only, and a third
640-pixel pass costs another ~2.5 s on the Cortex-A53.

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

The deployed copy is `/home/sansolution/dms` on 64-bit Raspberry Pi OS
(trixie, Python 3.13). The venv uses the system packages because `picamera2`
and the distro OpenCV are installed by apt:

```bash
ssh sansolution@192.168.1.14
cd /home/sansolution/dms
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run.py --config config_pi.json --source picamera
```

Run it from the Pi desktop for the preview window. Over SSH add `--headless`;
the app probes the display first and falls back to headless instead of hanging
when XWayland is not reachable from the session.

`config_pi.json` enables only the cigarette and phone detectors. The Pi
profile also:

- serializes detector passes through a priority gate, so two YOLO forwards
  never fight over the four Cortex-A53 cores;
- caps OpenCV at `opencv_threads: 2`, which leaves cores for the face path;
- captures at `target_fps: 5`, enough for the face loop without running the
  camera pipeline flat out;
- uses face-first scheduling (`face_first: true`): the moment the eyes close
  or the face disappears, detector workers stop starting new passes until the
  face is back. A pass already in flight finishes, but it cannot delay the
  confirming face frame by another whole pass.

Measured headless on this Pi 3: ~1.3 processed frames/s, ~2.1 s for one
cigarette pass at the full 1.2 GHz clock, 3.9-5.0 s while the board is
actively throttled to 600 MHz, and ~410 MB resident memory. Detector results
are held between passes, so the overlay marks boxes older than 1.5 s as
`STALE`.

The CSI night-vision module (OV5647 without an IR-cut filter) renders daylight
magenta under auto white balance. `config_pi.json` disables AWB and fixes
`ColourGains` at `[0.95, 1.0]`, the neutral point measured on this camera; the
app applies it through `Picamera2.set_controls`. The same values can be tested
without the app:

```bash
rpicam-still --nopreview --width 640 --height 480 --timeout 1200 \
  --awbgains 0.95,1.00 -o /tmp/test.jpg
```

### Power supply check

The Pi 3 has no input-voltage sensor, so use both the firmware flags and a
multimeter:

```bash
vcgencmd get_throttled        # must be 0x0
vcgencmd measure_clock arm    # should stay 1200000000 under load
vcgencmd measure_temp
dmesg | grep -i -E 'voltage|throttl'
```

Run those in a loop while the app is working:

```bash
for i in $(seq 1 30); do
  date +%T; vcgencmd get_throttled; vcgencmd measure_clock arm; sleep 2
done
```

`0x50005` means under-voltage is happening **now** (bit 0), the ARM clock is
capped (bit 2), and both have happened before (bits 16 and 18). The board then
drops from 1.2 GHz to 600 MHz, which roughly doubles every inference pass. The
software cannot compensate for that; measured pass times alternate between
~2.1 s and ~4.2 s for exactly this reason.

Measure the 5 V rail with a multimeter between TP1 (5 V) and TP2 (GND) on the
board, or GPIO pin 2/4 and pin 6, **while the app is running**. It must stay at
or above 4.8 V; 5.0-5.1 V is the target. `vcgencmd measure_volts core` is the
SoC core rail (~1.2 V), not the 5 V input, so it cannot diagnose the supply.

A 12 V adapter with a buck converter can work, but the converter must be rated
for 5 V at 3 A continuous (15 W) and the 12 V adapter for at least 2 A (24 W)
to leave margin. Many small buck modules are only 2 A, and thin or long wires
and a poor micro-USB cable drop the rest. Set the converter to 5.1 V measured
at the Pi under load, use short 20 AWG (or thicker) leads, and check the
voltage at both ends of the cable. If it still sags, replace the converter or
the cable. The simplest known-good fix is a Raspberry Pi 5.1 V 2.5 A supply
(or a good 5 V 3 A micro-USB supply) with a short cable. Powering through the
GPIO 5 V pins bypasses the micro-USB connector and polyfuse, but only do that
with a regulated 5.1 V supply, a fuse, and correct polarity.

Use `--source 0` for a USB webcam. Do **not** install MediaPipe on the Pi 3:
the OpenCV face path is the supported path.

The defaults in `deploy/upload.sh` are
`sansolution@192.168.1.14:/home/sansolution/dms`; override with `PI_USER`,
`PI_HOST`, and `PI_DIR` if your setup differs. From the PC:

```powershell
bash deploy/upload.sh
```

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
- `target_fps`, `opencv_threads`, `serialize_detectors`, `face_first`
- `thresholds.cigarette_s`, `cigarette_high_conf`, `cigarette_low_conf`,
  `cigarette_low_s`, `cigarette_gate_span`, `cigarette_max_size_span`,
  `cigarette_min_aspect`, `cigarette_mouth_pad_span`
- `thresholds.phone_call_s`, `phone_in_hand_s`, `phone_hold_s`
- `thresholds.ear_closed`, `microsleep_s`, `microsleep_min_frames`,
  `perclos_min_samples`, `yawn_mar`, `yawn_s`
- `alert_cooldown_s`: repeat cadence while a condition stays active
- `audio.device`: ALSA device on Linux
- `display`: `false` for headless

Smoking is deliberately conservative. On the PC a cigarette box must sit
within 0.30 face-heights of the mouth, be no larger than 0.55 face-heights,
and stay detected for 1.0 s at confidence 0.60+ (or 1.5 s at 0.45+). The
dedicated model runs at 2 Hz there, so this spans at least two detector
passes. The Pi profile tightens the geometry to 0.24 / 0.50 face-heights,
requires an elongated box (aspect ≥ 1.15) whose mouth pad is ≤ 0.12
face-heights, and uses the same 1.0 s / 1.5 s confirmation with a 4 s hold.
A real puff stays at the lips long enough to alert; a watch, ring or pen on
the cheek does not pass the geometry.

A disabled detector does not need its model or class file to exist, so a
profile can drop a model without deleting files.

Drowsiness ignores frames with an impossible EAR (> 0.50), an implausible MAR
(> 1.20) or an extreme head pose, which is what a hand over the face produces.
The Pi profile needs 3 consecutive closed samples, both eyelids below 0.10,
and 1.2 s, so a real closure alerts in roughly 1.5-2 s at 1-2 fps; a normal
0.2-0.4 s blink does not reach the bar, and a one-eye landmark glitch cannot
start the run. Face-first scheduling pauses detector passes as soon as a
closed run starts, which is what keeps those samples coming quickly.
PERCLOS still needs `perclos_min_samples` samples (10) in its window, so on a
1-2 fps Pi it stays quiet instead of firing on a handful of noisy samples;
microsleep remains the fast path.

## Tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

The tests load both configs, check that the model files exist, run the face
and object models once on a blank frame, and cover the alert hold, the
drowsiness validity/run guards, the both-eyes 3-frame bar, the cigarette
geometry gates, detector serialization/pause, and the alert/audio priority
order.

## Notes and limits

- Object detectors run on background threads, so a slow Pi does not freeze the
  face loop. Each detector has its own `target_hz`; on the Pi one shared gate
  serializes passes, and face-first scheduling pauses new passes while the
  eyes are closed or no face is visible.
- The face path is the priority. Alert audio is held in a priority queue with
  drowsiness first, phone second, smoking third. A phrase already playing is
  not interrupted, so the worst extra delay is one ~1.9 s phrase.
- Measured on this Pi 3: ~1.3 processed frames/s headless, ~2.1 s per
  640-pixel cigarette pass at 1.2 GHz (3.9-5.0 s while actively throttled),
  ~410 MB resident memory, 50-64 °C. The CPU and the 5 V supply are the
  limits, not RAM or disk.
- The preview window is optional. If highgui cannot open a display, the app
  prints a note and continues headless instead of blocking.
- The models are trained for a driver-facing camera. A laptop webcam is a
  different view; treat the PC test as a functional check, not an accuracy
  result.
- The Ultralytics exports are AGPL-3.0. Check the model and dataset licences
  before redistributing or using this commercially.
- The app does not record video or upload anything. The optional CSV log is off
  unless `--log` is passed.
