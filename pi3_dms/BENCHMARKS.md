# Measured benchmarks

Only measurements actually taken in this workspace are listed. Nothing in this
file is a Raspberry Pi 3 result.

## Desktop, 2026-10-27

Environment recorded by `benchmark.py`:

- Windows 11 (`Windows-11-10.0.26200-SP0`), AMD64
- Python 3.14.3
- OpenCV 5.0.0, NumPy 2.5.4
- 15,710 MB system RAM
- Input image: one frontal face, 960x540 inference frame
- Face model SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- Object model SHA-256: `2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b`
- Iterations: 20, one warm-up excluded

| Path | Mean | Median | p95 | Throughput | Process CPU | Process RSS | System CPU after | System RAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Face core only | 9.73 ms | 9.68 ms | 10.69 ms | 102.7 fps | 152.5% | 138.8 MB | 20.0% | 96.8% |
| Object model only | 35.68 ms | 34.76 ms | 42.81 ms | 28.0 fps | 100.7% | 168.0 MB | 38.5% | 95.9% |
| Combined, sequential | 42.47 ms | 41.88 ms | 44.83 ms | 23.5 fps | 110.4% | 168.8 MB | 25.9% | 95.9% |

Raw JSON: [`benchmarks/desktop_2026-10-27_960x540.json`](benchmarks/desktop_2026-10-27_960x540.json).

### How to read these numbers

- **Process CPU can exceed 100%** because MediaPipe/TFLite/XNNPACK can use more
  than one CPU thread. This is not a percentage of total machine capacity and
  is not a Pi 3 utilization measurement.
- **System RAM** is the whole Windows machine. It was already heavily loaded
  (about 96%), so do not interpret it as the DMS footprint. Use the process RSS
  column for the DMS process.
- **Combined, sequential** runs face and object inference one after another in
  the same loop. The app actually runs object inference periodically in a
  separate worker and skips old frames, so this is a conservative sequential
  upper bound, not the dashboard's concurrent throughput.
- The object model is the shipped COCO phone/bottle control model. This benchmark
  measures speed, not accuracy. It is not validated for driver monitoring and it
  has no cigarette class.
- These numbers do **not** predict Pi 3 performance. A Pi 3 has much less CPU and
  memory bandwidth and no desktop XNNPACK tuning guarantee.

## Desktop integration checks

- Image mode (`face_open.jpg`, 960x540 inference, face model + object model):
  `/status` reported `mode=running`, `main_state=ATTENTIVE`,
  tracking confidence `0.59`, object status `running`, and object latency
  around `36 ms`. No object was invented; the blank/face frame produced no
  phone/bottle detections.
- Flask test client returned HTTP 200 for `/`, `/status`, `/snapshot.jpg` and
  `/snapshot_raw.jpg`. The raw endpoint returned the unannotated frame.
- A Windows webcam smoke run opened the OpenCV camera and produced processed
  frames. The test view did not contain a detectable face, so it is a camera-path
  check only; it is not face-accuracy evidence. The source reported about 1 fps
  in that environment, so it is not used as a performance result.
- `camera_smoke_test.py` is the separate manual camera check. Simulated tests do
  not open a camera.

## Raspberry Pi 3 status: pending

No Pi 3/Camera Module hardware or Pi-recorded footage was available. The
following required measurements are **not** available:

- core-only and combined FPS on the Pi;
- per-stage and total latency;
- CPU per core / total utilization;
- RAM and process RSS;
- temperature and throttling;
- object detection accuracy and false positives on real cabin footage.

Do not copy the desktop table into a Pi report. Run `benchmark.py` and the camera
smoke test on the target board before enabling object inference there.
