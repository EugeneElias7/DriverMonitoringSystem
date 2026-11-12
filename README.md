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
├── config_events.json        # vehicle number for evidence events
├── .env.example              # Telegram credential template (copy to .env)
├── requirements.txt
├── THIRD_PARTY_NOTICES.md    # model/dataset credits and licence status
├── licenses/                 # full third-party licence texts
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
│   ├── events/               # evidence capture, JSONL log, Telegram
│   ├── geometry.py           # EAR, MAR, head pose
│   └── temporal.py           # time windows and holds
├── models/                   # flat; no subdirectories
├── audio/                    # spoken WAV phrases
├── deploy/upload.sh          # copy the app to the Pi
├── tools/telegram_live_test.py  # manual Telegram test: synthetic, image, camera
└── tests/                    # automated tests
    ├── test_smoke.py         # config and model-load checks
    └── test_events.py        # evidence, JSONL, Telegram tests
```

## Models

| File | Job | Profiles | Origin / author | Licence |
| --- | --- | --- | --- | --- |
| `models/face_detection_yunet_2023mar.onnx` | face boxes | PC + Pi | OpenCV Zoo YuNet, Shiqi Yu | MIT |
| `models/face_landmarks.onnx` | **active** 478 face landmarks: eyes, mouth, head pose | PC + Pi | Google MediaPipe Face Mesh, converted to ONNX | Apache-2.0 |
| `models/2d106det.onnx` | **fallback** 106 landmarks only | PC + Pi | InsightFace `2d106det` | Non-commercial research only |
| `models/best.onnx` | driver objects: drinking, eating, seatbelt | PC only | Project-trained on `data/data.yaml`, exported with Ultralytics YOLO | AGPL-3.0 metadata; dataset terms unknown |
| `models/cigarette_smokers_v4.onnx` | dedicated cigarette | PC + Pi | Project-trained on `cigarette-smokers-clean/data.yaml`, exported with Ultralytics YOLO | AGPL-3.0 metadata; dataset terms unknown |
| `models/phone_yolo11n.onnx` | phone; COCO `cell phone` only | PC + Pi | Ultralytics YOLO11n, COCO-pretrained | AGPL-3.0 |

`models/driver_classes.txt`, `models/cigarette_v4_classes.txt`, and
`models/coco_classes.txt` must stay in the same order as their model's class
ids. The runtime lowercases names.

`face_landmarks.onnx` is the active landmark model on both profiles. The
`face.landmark_model_path` config key still names `2d106det.onnx` so the
fallback analyzer can load it; `create_face_analyzer()` prefers
`face_landmarks.onnx` and only falls back to 2d106 when the 478 model is
missing or fails to load.

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
`best.onnx` still lists Cigarette and Phone classes, but they are masked out:
on webcam footage the general model produced false boxes, including occasional
full-frame ones. The
dedicated model runs on both PC and Pi. Both Pi detectors use a 1 Hz
start-to-start target so each worker rests between passes instead of one model
monopolizing the CPU; the pass itself is CPU-bound, so the effective cadence is
roughly one pass each. The Pi profile also drops the `best.onnx` detector
entirely: drinking, eating and seatbelt are status-only, and a third 640-pixel
pass costs another ~2.5 s on the Cortex-A53.

## Credits and licensing

The full model inventory, exact attribution, dataset provenance, and
commercial checklist are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
The upstream licence texts are in [`licenses/`](licenses/).

In short:

- **YuNet** face detector: MIT, Copyright (c) 2020 Shiqi Yu. Cite Wu, Peng and
  Yu, *YuNet: A Tiny Millisecond-Level Face Detector* (2023).
- **MediaPipe** 478-point face landmarks: Apache-2.0, Google LLC. The ONNX file
  is a conversion, not an official Google release.
- **InsightFace 2d106** fallback: the InsightFace Model Zoo says all models are
  for non-commercial research only. It is not part of a commercial build unless
  separately licensed or replaced.
- **Ultralytics YOLO**: `phone_yolo11n.onnx` is COCO-pretrained, and
  `best.onnx` plus `cigarette_smokers_v4.onnx` are project-trained models
  exported with Ultralytics. All three carry AGPL-3.0 metadata. A commercial
  embedded product needs either AGPL-3.0 compliance or an Ultralytics
  Enterprise License.
- **Project-trained models**: `best.onnx` was trained on `data/data.yaml`;
  `cigarette_smokers_v4.onnx` was trained on
  `cigarette-smokers-clean/data.yaml`. The datasets are not included in this
  repository, and their licences and consent status are not recorded here.
  The ONNX `author` field says `Ultralytics` because the export tool wrote it;
  the model descriptions identify these as project-trained models, not stock
  Ultralytics releases.

The project is a real-world feasibility prototype and claims no ownership of
third-party models, datasets, code, names, or trademarks. It does not sell the
models themselves. Before commercial distribution, verify every upstream
licence and the dataset rights. This is not legal advice.

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
--event-config PATH   vehicle number config (default: config_events.json)
--no-events           disable evidence images, JSONL, and Telegram
--credits             print third-party model credits and exit
```

## Run on the Raspberry Pi 3

The deployed copy is `/home/sansolution/dms` on 64-bit Raspberry Pi OS
(trixie, Python 3.13). The venv uses the system packages because `picamera2`
and the distro OpenCV are installed by apt:

Double-click the **DMS** desktop icon, or run one command:

```bash
cd /home/sansolution/dms
.venv/bin/python run.py --config config_pi.json --source picamera
```

Over SSH add `--headless`:

```bash
cd /home/sansolution/dms
.venv/bin/python run.py --config config_pi.json --source picamera --headless
```

The app probes the display first and falls back to headless instead of hanging
when XWayland is not reachable from the session. Pass normal app flags after
the command, e.g. `--max-seconds 60`.

The test suite lives in the source tree on the PC; the Pi deployment keeps
only the runtime files.

`config_pi.json` enables only the cigarette and phone detectors. The Pi
profile also:

- runs the two YOLO models on separate worker threads, each capped at
  `opencv_threads: 1`, so both passes finish in roughly the time of one
  (~6.4 s and ~6.8 s measured concurrently on this throttled Pi 3) instead of
  sharing a serialized gate (~12.5 s for a full cycle);
- captures at `target_fps: 5`, enough for the face loop without running the
  camera pipeline flat out;
- uses face-first scheduling (`face_first: true`): the moment the eyes close
  or the face disappears, detector workers stop starting new passes until the
  face is back. A pass already in flight finishes, but it cannot delay the
  confirming face frame by another whole pass;
- skips the model self-test on the Pi (`self_test: false`) but keeps the
  short startup phrase (`audio.announce_startup: true`) so the speaker path is
  proven before an alert is needed.

Measured headless on this Pi 3: ~1.5 processed frames/s, ~1.8 s for one
cigarette pass at the full 1.2 GHz clock with 2 threads, roughly double that
while the board is actively throttled to 600 MHz, and ~287 MB resident
memory. Three threads measured 2.1-2.4 s per pass and caused a larger current
spike, so the Pi profile uses two. Detector results are held between passes,
so the overlay marks boxes older than 1.5 s as `STALE`.

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

## Evidence capture and Telegram alerts

When the behaviour layer confirms **drowsiness**, **yawning**, **phone use**,
or **smoking**, the app captures that camera frame once, saves a JPEG, appends
one JSONL record, and sends a Telegram photo. Detection thresholds and the
existing monitors are unchanged.

| Confirmed signal | `alert_type` | Folder |
| --- | --- | --- |
| `drowsy_microsleep` / `drowsy_perclos` | `DROWSINESS` | `evidence/drowsiness/` |
| `drowsy_yawning` | `YAWNING` | `evidence/yawning/` |
| `phone_use` | `PHONE_USAGE` | `evidence/phone/` |
| `smoking` | `SMOKING` | `evidence/smoking/` |

Microsleep and PERCLOS share one drowsiness event. A category must stay clear
for 2 seconds before it can fire again, so a one-frame signal gap does not
create a duplicate. The frame is copied on the rising edge and handed to a
background worker; JPEG writing and Telegram requests never run in the camera
loop. The image is the raw, unannotated camera frame at the resolution
delivered by the camera (normally 640×480); the downscale used for detection
does not apply to evidence.

```text
evidence/
    drowsiness/EVT-20261111-000001.jpg
    yawning/
    phone/
    smoking/

logs/events.jsonl
```

Each line is one event:

```json
{
  "event_id": "EVT-20261111-000001",
  "vehicle_number": "KA03MN1234",
  "alert_type": "DROWSINESS",
  "timestamp": "2026-11-11T17:30:20",
  "image_path": "evidence/drowsiness/EVT-20261111-000001.jpg",
  "notification_status": "SENT"
}
```

`notification_status` is `SENT` or `FAILED`. A notification failure still leaves
the image and JSONL record, with a `notification_error` field; an image-save
failure is logged with an empty `image_path`. Telegram being offline or
unconfigured never stops the DMS. The final record is written after the Telegram
attempt so its status is accurate.

### Vehicle number

Edit `config_events.json`:

```json
{
  "vehicle_number": "KA03MN1234"
}
```

No other vehicle, driver, company, or fleet fields are stored. The file can be
overridden with `--event-config path/to/file.json`. Use `--no-events` to turn
the whole feature off for a run.

### Telegram credentials

Copy `.env.example` to `.env` once and fill in the two values:

```text
TELEGRAM_BOT_TOKEN=123456:your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

The app loads `.env` from the current directory or the project root. Only
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are read; other entries are
ignored. Real shell or service environment variables always win over `.env`,
so a one-off override still works.

PowerShell:

```powershell
$env:TELEGRAM_BOT_TOKEN = "your-bot-token"
$env:TELEGRAM_CHAT_ID = "your-chat-id"
```

Linux / Raspberry Pi:

```bash
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

Set `DMS_ENV_FILE=/path/to/file` to load a different dotenv file. `.env` is
ignored by Git and is **never** uploaded by `deploy/upload.sh`. The bot token
and chat ID are never written to `config_events.json`, logs, evidence,
captions, or error messages. Do not commit `.env`, paste it into a screenshot,
or share it.

The laptop `.env` cannot be read by the Pi. The Pi needs the credentials once,
in a local file or service environment (see the deployment section below). If
the Pi has no credentials, evidence and JSONL records are still written with
`notification_status: "FAILED"`; the DMS itself keeps running.

The notifier uses Telegram's `sendPhoto` endpoint with an HTML caption. It
retries transient failures up to 3 times after the first attempt, with a short
backoff. A 401/403 or a missing image fails immediately. If the environment
variables are absent, the app prints a note and continues; events are logged as
`FAILED`.

```text
🚨 DMS ALERT

Vehicle Number: KA03MN1234
Alert Type: 😴 DROWSINESS
Time: 11 Nov 2026, 05:30 PM
Event ID: EVT-20261111-000001
```

### Windows test

```powershell
cd "C:\D drive\DMS-Model-Testing"
.\.venv\Scripts\Activate.ps1
# .env in the project root supplies the token and chat ID
python run.py --config config_pc.json --source 0
```

Trigger a confirmed event and check:

```text
evidence/drowsiness/EVT-...jpg
evidence/yawning/EVT-...jpg
evidence/phone/EVT-...jpg
evidence/smoking/EVT-...jpg
logs/events.jsonl
```

The automated tests use a fake notifier and mocked HTTP requests, so they never
send a real message. For a real Telegram check, run the separate manual script:

```powershell
# One live laptop-camera frame
python tools/telegram_live_test.py --camera

# An existing evidence image
python tools/telegram_live_test.py --image evidence/smoking/EVT-...jpg

# A small synthetic test image
python tools/telegram_live_test.py
```

`--camera` defaults to index 0; use `--camera 1` for a second camera. Close the
DMS first if it is already using the camera. Exit code 0 means Telegram accepted
the photo; 1 means the request failed; 2 means credentials, the camera, or the
input image are missing. The token is not printed.

### Raspberry Pi deployment

`deploy/upload.sh` copies the app and `.env.example`, but deliberately skips
`.env`:

```bash
bash deploy/upload.sh
ssh sansolution@192.168.1.14
```

The Pi cannot read the laptop `.env`, so provide the credentials there once,
outside the project. The cleanest option is a root-owned environment file:

```bash
sudo install -m 600 /dev/null /etc/dms-telegram.env
sudo nano /etc/dms-telegram.env
```

Add the two lines:

```text
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Then load it in the shell that starts the app:

```bash
cd /home/sansolution/dms
set -a
. /etc/dms-telegram.env
set +a
.venv/bin/python run.py --config config_pi.json --source picamera --headless
```

A systemd unit can point at the same file with
`EnvironmentFile=/etc/dms-telegram.env`, so the secret is not duplicated in the
unit or the project. The loader imports only the two allowed keys, so unrelated
entries in that file are ignored. If you prefer a project-local file, copy
`.env.example` to `.env` on the Pi and fill it in; `deploy/upload.sh` will not
upload, overwrite, or delete it.

`evidence/` and `logs/` are created relative to the working directory (the
project root in the commands above). The Pi needs outbound HTTPS access to
`api.telegram.org`; if it is offline, images and JSONL records are still
written and the notification status is `FAILED`. Set the environment variables
in the same shell or service that launches the app.

After a confirmed event, verify the evidence and record:

```bash
ls -R evidence
tail -n 5 logs/events.jsonl
```

## Configuration

`config_pc.json` and `config_pi.json` are the only profiles. Important keys:

- `face.detector_model_path`, `face.landmark_model_path`,
  `face.min_detection_confidence`. `face.landmark_model_path` is the fallback
  path (2d106); the active 478 model is loaded from
  `models/face_landmarks.onnx` when present.
- `detectors[]`: `name`, `model_path`, `classes_path`, `conf_threshold`,
  `class_conf_overrides`, `target_hz`, `enabled`
- `target_fps`, `opencv_threads`, `serialize_detectors`, `face_first`
- `thresholds.cigarette_s`, `cigarette_high_conf`, `cigarette_low_conf`,
  `cigarette_low_s`, `cigarette_gate_span`, `cigarette_max_size_span`,
  `cigarette_min_aspect`, `cigarette_mouth_pad_span`,
  `phone_cigarette_resolve`, `cigarette_suppress_conf`, `phone_strong_conf`,
  `phone_cigarette_margin`, `phone_cigarette_iou`, `phone_strong_size_span`
- `thresholds.phone_call_s`, `phone_in_hand_s`, `phone_hold_s`,
  `phone_min_size_span`, `phone_max_size_span`, `phone_face_margin_span`
- `thresholds.ear_closed`, `microsleep_s`, `microsleep_min_frames`,
  `perclos_min_samples`, `yawn_mar`, `yawn_s`
- `alert_cooldown_s`: repeat cadence while a condition stays active
- `audio.device`: ALSA device on Linux
- `audio.announce_startup`: play the startup phrase (enabled on the Pi)
- `self_test`: run the models once at startup and print timings (`false` on
  the Pi, where it costs ~5 s before the first frame)
- `display`: `false` for headless

Smoking is deliberately conservative. On the PC a cigarette box must pass the
mouth/size gate, be no larger than 0.55 face-heights, and stay detected for
1.0 s at confidence 0.60+ (or 1.5 s at 0.45+). The dedicated model runs at
2 Hz there, so this spans at least two detector passes. The Pi profile uses a
mouth-edge distance of 0.28 face-heights, a maximum box size of 0.80
face-heights, and an aspect ratio of at least 1.15 so a real end-on cigarette
still passes. It confirms in 0.3 s at 0.50+ (or 0.6 s at 0.25+), bridges a
one-frame miss for 1.0 s, and holds the alert for 4 s. The Pi republishes the
last detector result until the next ~6 s pass, so these short onset bars reduce
the delay after a pass without changing which detections are accepted; geometry
and confidence still do the filtering. A real puff stays at the lips long
enough to alert; a watch, ring or pen on the cheek does not pass the geometry.

The Pi also enables `phone_cigarette_resolve`, because its camera frequently
makes the COCO phone model fire on a cigarette, the fingers holding it, or the
cheek beside it while the dedicated cigarette model fires on the same object.
Two independent guards keep those false phone alerts away:

- `phone_min_size_span` rejects phone boxes smaller than 0.35 face-heights on
  the Pi. A real handset at the ear measured 0.73 face-heights; cigarette and
  hand boxes measured 0.23-0.31. The gate applies to both call-at-ear and
  in-hand checks. The PC profile leaves it at 0 (off) to preserve behaviour.
- `phone_max_size_span` is raised to 1.5 on the Pi. A handset held close to
  the camera can be larger than the face (live boxes measured 1.04-1.15);
  the old 1.0 limit rejected them. A full-frame model failure is still far
  above 1.5.
- `phone_face_margin_span` is raised to 0.75 on the Pi, so a phone held just
  beside the face is not missed by the horizontal edge of the in-hand gate.
- When a cigarette box at 0.25+ passes the full mouth gate, the resolver
  compares it with each phone box at 0.15+:
  - a phone box that overlaps the cigarette or is centred at the lips is
    treated as the same object; a smaller phone box at the ear is too, because
    that is where the cigarette/hand/cheek often lands;
  - a large phone box (0.55+ face-heights) at the ear is a real handset even
    when its confidence is below 0.60, so it is not hidden by the cigarette;
  - a cigarette-sized phone box is dropped even at 0.60+ confidence, because
    its size is stronger evidence than its score;
  - a cigarette wins over a weaker phone, so holding a cigarette does not also
    raise a phone alert;
  - a strong phone (0.60+) that beats the cigarette by 0.15 confidence and
    passes the size test wins, so a real phone is not hidden by a weak
    cigarette box;
  - when both labels are strong and the phone is large, both are kept rather
    than suppressing a real alert.

The PC profile leaves the resolver off so its existing detection behaviour is
unchanged. Phone calls and phones held in hand confirm in 0.5 s on the Pi.

A disabled detector does not need its model or class file to exist, so a
profile can drop a model without deleting files.

Drowsiness ignores frames with an impossible EAR (> 0.50), an implausible MAR
(> 1.20) or an extreme head pose, which is what a hand over the face produces.
The Pi profile needs 2 consecutive closed samples with the averaged EAR below
0.20, and the microsleep timer runs for 0.3 s. At ~1 fps a single closed frame
is a normal blink: the status line shows `closed=yes run=1/2`, and only a
second consecutive closed sample can alert. A closed/open/closed pattern
correctly stays Attentive. The average is already robust against one bad
eyelid: it can only fall below 0.20 when both eyes are at least partly shut.
The per-eye values are printed in the status line while the eyes are near
closure, because a steep camera angle can make the far eye read high even
when its lid is shut. Face-first
scheduling pauses detector passes as soon as a closed run starts, which is
what keeps those samples coming quickly.
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
drowsiness validity/run guards, the steep-yaw close-eye case, the cigarette
geometry gates, detector serialization/pause, the alert/audio priority
order, evidence deduplication, JSONL IDs, mocked Telegram retries, and `.env`
credential loading/precedence. The event tests are in `tests/test_events.py`;
no test sends a real Telegram message.

## Notes and limits

- Object detectors run on background threads, so a slow Pi does not freeze the
  face loop. Each detector has its own `target_hz`; the Pi runs the cigarette
  and phone models in parallel, and face-first scheduling pauses new passes
  while the eyes are closed or no face is visible.
- The face path is the priority. Alert audio is held in a priority queue with
  drowsiness first, phone second, smoking third. A phrase already playing is
  not interrupted, so the worst extra delay is one ~1.9 s phrase.
- Measured on this Pi 3 while throttled to 600 MHz: ~1.0 face frame/s headless,
  ~6.4 s for the cigarette pass and ~6.8 s for the phone pass running
  concurrently with `opencv_threads: 1`, ~287 MB resident memory, 50-64 °C.
  The CPU and the 5 V supply are the limits, not RAM or disk.
- Object latency is set by the two 640-pixel forwards: both refresh about every
  7 s while throttled, so a phone or cigarette alert typically appears in
  7-9 s. A single serialized cycle was ~12.5 s. The exported ONNX graphs are
  fixed at 640 input, so lowering `input_size` is not an option without
  re-exporting the models. Sub-second object alerts need a Pi 4/5, a smaller
  re-exported model, or an accelerator; the sleep path is the
  only one designed to confirm in under a second on this board.
- The preview window is optional. If highgui cannot open a display, the app
  prints a note and continues headless instead of blocking.
- The models are trained for a driver-facing camera. A laptop webcam is a
  different view; treat the PC test as a functional check, not an accuracy
  result.
- The Ultralytics exports are AGPL-3.0 and `2d106det.onnx` is non-commercial
  research only. See `THIRD_PARTY_NOTICES.md` for the model credits, licence
  texts, and commercial checklist before redistributing or selling a device.
- The app writes evidence JPEGs and `logs/events.jsonl` when a confirmed event
  fires, and sends Telegram photos only when `TELEGRAM_BOT_TOKEN` and
  `TELEGRAM_CHAT_ID` are set. It does not upload anything except those Telegram
  photos. Video recording is opt-in with `--record`, and the optional CSV alert
  log is off unless `--log` is passed.
- Evidence files are not pruned automatically. Archive or delete old files on a
  schedule, and treat both `evidence/` and `logs/` as sensitive personal data.
