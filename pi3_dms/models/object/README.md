# Optional object model

This folder is **not** required by the face core. The app starts and runs when
the object module is disabled or when this file is missing.

## `efficientdet_lite0.tflite`

| Property | Value |
|---|---|
| Source | TensorFlow Lite Task Library official EfficientDet-Lite0 model |
| SHA-256 | `2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b` |
| Size | 4,563,519 bytes |
| Licence | Apache-2.0 (TensorFlow/TFLite model collection; verify the model card before redistribution) |
| Classes | COCO-90, including `bottle` and `cell phone` |
| Cigarette | **No cigarette class** |
| Runtime | MediaPipe Tasks / TFLite CPU, ARM-compatible in principle |
| Status | Experimental COCO control model, **not validated for driver monitoring** |

It is included only so the optional phone/bottle path can be exercised without
downloading a model at runtime. It must not be described as a cigarette
detector. Generic COCO models have been observed to mislabel cigarettes as
toothbrushes or phones; this app filters to explicit `phone` and `bottle`
classes, so no such label can create a smoking alert.

The model's accuracy and speed on a Raspberry Pi 3 have **not** been measured.
Run `benchmark.py` on the target board and treat the object path as
experimental.

## Local cigarette and phone models

The app accepts a TFLite model with label metadata through `--object-model` and
`--object-classes`. It also accepts a separate HTTP object service. Do not add a
checkpoint to this folder unless:

1. the class labels are known and tested;
2. the model licence and training-data provenance are acceptable for your use;
3. the runtime is compatible with the target hardware;
4. actual performance and false-positive behaviour are measured.

The previously tested cigarette checkpoints are not shipped. Their records
include AGPL metadata, unknown provenance or unclear rights, and they were not
validated for this deployment.
