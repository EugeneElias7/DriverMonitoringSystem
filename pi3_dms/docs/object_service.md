# Optional HTTP object-inference service

The face core can run with the object module disabled. When enabled with
`--object-backend http`, the core sends the already-captured inference frame to
a separate service. This keeps a heavy, separately licensed or
hardware-specific detector outside the face process.

The service **must not open a second camera**. The core owns the camera and sends
the frame it has already captured.

## Start the core

```bash
python app.py --objects \
  --object-backend http \
  --object-service-url http://127.0.0.1:8010/detect \
  --object-classes phone,bottle
```

`--object-interval-ms` controls the maximum call rate. The core samples the
newest frame and skips frames if the service is slow; it does not accumulate a
queue. A response older than `--object-stale-ms` is rejected.

## Request

`POST /detect` with `Content-Type: application/json`:

```json
{
  "timestamp": 12345.678,
  "frame_id": 42,
  "image_b64": "<base64 JPEG bytes>"
}
```

- `timestamp`: monotonic capture timestamp in seconds. Return this value, or a
  newer one, with detections so the fusion layer can reject stale results.
- `frame_id`: the core frame sequence; useful for logging.
- `image_b64`: JPEG of the inference frame, not the raw capture. It may already
  have the configured soft/night preprocessing applied.

## Response

```json
{
  "detections": [
    {
      "label": "phone",
      "confidence": 0.81,
      "bbox": [0.42, 0.25, 0.55, 0.45],
      "timestamp": 12345.678
    }
  ]
}
```

- `label` must be one of `phone`, `bottle` or `cigarette`. Aliases such as
  `cell phone` and `mobile phone` are accepted by the client and canonicalized.
  Unknown labels are ignored; a bottle is never converted to a cigarette.
- `confidence` is the detector's own score in `[0, 1]`. The core does not invent
  a score if it is missing. Keep the service's score semantics documented.
- `bbox` is normalized `[x1, y1, x2, y2]` relative to the submitted image. Pixel
  boxes are not accepted.
- `timestamp` is optional; if omitted the request timestamp is used. The client
  filters individual detections older than `--object-stale-ms`.

The core filters the returned labels against `--object-classes` and applies
minimum confidence and persistence rules in `dms/fusion.py`. The service should
still filter its own output and report its model name/limitations in logs.

## Minimal service contract

A valid service:

1. decodes the JPEG;
2. runs a detector whose licence/provenance and labels are understood;
3. returns only normalized boxes and explicit labels;
4. avoids holding old detections;
5. does not claim a behaviour; the core's temporal fusion decides that;
6. never returns `cigarette` unless the model actually has a cigarette class.

The service may be implemented in Python, C++, or another runtime. It is not
included because no rights-cleared, validated cigarette model is bundled. A
service that returns simulated or hard-coded detections would invalidate the
monitoring result and must not be used as a detector.

## Failure behaviour

- HTTP errors/timeouts set the object module status to `error`.
- A missing service at startup does not stop the face core.
- After `--object-stale-ms` without a fresh result, the object status becomes
  `stale` and behaviour evidence is cleared.
- If face tracking is unreliable, object evidence is reported as `uncertain`
  rather than creating a behaviour alert.
