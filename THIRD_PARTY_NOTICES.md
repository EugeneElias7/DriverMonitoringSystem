# Third-Party Notices and Model Credits

This file records the origin of the third-party models, datasets, and software
used by this Driver Monitoring System (DMS) prototype.

The project does **not** claim ownership of any third-party model, dataset,
code, name, or trademark. It does not sell the models as standalone products.
The model files are used as components of a driver-monitoring system.

This is an attribution and provenance record, not legal advice. Licence terms
must be checked against the upstream sources before any commercial
distribution. See [Commercial checklist](#commercial-checklist) for the known
issues in this repository.

## Model inventory

| File | Role in this project | Used on | Origin / author | Licence status |
| --- | --- | --- | --- | --- |
| `models/face_detection_yunet_2023mar.onnx` | Face detection; every face metric starts from this box | PC + Pi | OpenCV Zoo **YuNet**, Shiqi Yu | **MIT** |
| `models/face_landmarks.onnx` | Active 478-point face landmarks: EAR, MAR, mouth anchor, pose | PC + Pi (preferred) | Google **MediaPipe** Face Mesh / Face Landmarker, converted to ONNX | **Apache-2.0** (verify the model card) |
| `models/2d106det.onnx` | Fallback 106-point landmarks, used only if the 478 model is missing or fails to load | PC + Pi (fallback) | **InsightFace** `2d106det` | **Non-commercial research only** per the InsightFace model zoo |
| `models/best.onnx` | Driver objects: drinking, eating, seatbelt | PC only | **Project-trained model**, trained on `data/data.yaml`, exported with Ultralytics YOLO | Embedded metadata says **AGPL-3.0**; dataset terms unknown |
| `models/cigarette_smokers_v4.onnx` | Dedicated cigarette detector | PC + Pi | **Project-trained model**, trained on `/content/cigarette-smokers-clean/data.yaml`, exported with Ultralytics YOLO | Embedded metadata says **AGPL-3.0**; dataset terms unknown |
| `models/phone_yolo11n.onnx` | Phone detector; only the COCO `cell phone` class is enabled | PC + Pi | **Ultralytics YOLO11n**, COCO-pretrained | **AGPL-3.0** |

`models/best.onnx` still contains Cigarette and Phone classes, but the config
masks them. Cigarette and phone alerts come from the dedicated models.

The config files still name `models/2d106det.onnx` in
`face.landmark_model_path`. That field is the fallback path: at runtime,
`create_face_analyzer()` prefers `models/face_landmarks.onnx` and only falls
back to 2d106 when the 478 model is missing or fails to load.

## Third-party models

### 1. YuNet face detector

- File: `models/face_detection_yunet_2023mar.onnx`
- Origin: OpenCV Zoo, YuNet
- Author: Shiqi Yu
- Copyright notice: **Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>**
- Licence: **MIT License**
- Source: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- Upstream licence: <https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE>
- Full text included at: `licenses/YuNet-MIT.txt`

Citation requested by the upstream project:

> Wu, Wei, Peng, Hanyang, and Yu, Shiqi. "YuNet: A Tiny Millisecond-Level Face
> Detector." *Machine Intelligence Research* 20, no. 5 (2023): 656-665.
> <https://doi.org/10.1007/s11633-023-1423-y>

### 2. Google MediaPipe 478-point face landmarks

- File: `models/face_landmarks.onnx`
- Origin: Google MediaPipe Face Mesh / Face Landmarker 478-point model,
  converted from the MediaPipe TFLite model to ONNX for OpenCV DNN
- Author / project: Google LLC, MediaPipe
- Licence: **Apache License 2.0**
- MediaPipe source: <https://github.com/google-ai-edge/mediapipe>
- MediaPipe licence: <https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE>
- Face Landmarker documentation:
  <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker>
- Current Face Mesh V2 model card, for reference:
  <https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf>
- Full MediaPipe licence text included at: `licenses/MediaPipe-LICENSE.txt`

The ONNX conversion is not an official Google release, and the exact MediaPipe
model version is not recorded in the ONNX metadata. The upstream model and the
model card that matches the converted file remain the authoritative source for
the weights and their terms. Verify that model card before redistribution.

### 3. InsightFace 2d106 landmark model

- File: `models/2d106det.onnx`
- Origin: InsightFace Model Zoo, 2D face alignment, 106 points
- Authors / project: Jia Guo, Jiankang Deng, and the InsightFace contributors
- Licence status: **non-commercial research only**, according to the InsightFace
  Model Zoo README
- Source: <https://github.com/deepinsight/insightface>
- Model zoo: <https://github.com/deepinsight/insightface/tree/master/model_zoo>
- Exact upstream statement:
  <https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md>

> **ALL models are available for non-commercial research purposes only.**

This is the most important commercial blocker in the current model set. The
file is only a fallback, but it is still shipped in `models/` and must be
removed, replaced with a commercially licensed landmark model, or separately
licensed from the rights holder before a commercial build is distributed.
Deleting the file is not enough if a commercial build still relies on it as a
fallback; the fallback must be removed or replaced as well.

### 4. Ultralytics YOLO models and exports

Files:

- `models/phone_yolo11n.onnx` — Ultralytics YOLO11n, COCO-pretrained
- `models/best.onnx` — project-trained model exported with Ultralytics YOLO
- `models/cigarette_smokers_v4.onnx` — project-trained model exported with
  Ultralytics YOLO

Licence: **GNU Affero General Public License v3.0 (AGPL-3.0)**.

- Ultralytics project: <https://github.com/ultralytics/ultralytics>
- Ultralytics licence:
  <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
- Ultralytics licensing page: <https://www.ultralytics.com/license>
- Ultralytics Enterprise terms:
  <https://www.ultralytics.com/legal/enterprise-software-license>
- Full licence text included at: `licenses/Ultralytics-AGPL-3.0.txt`

The embedded metadata in all three ONNX files contains the exact string:

> `AGPL-3.0 License (https://ultralytics.com/license)`

The embedded descriptions are:

- `best.onnx`:
  `Ultralytics best model trained on data/data.yaml`
- `cigarette_smokers_v4.onnx`:
  `Ultralytics cigarette smokers.v4-best model trained on /content/cigarette-smokers-clean/data.yaml`
- `phone_yolo11n.onnx`:
  `Ultralytics YOLO11n model trained on /usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml`

`best.onnx` and `cigarette_smokers_v4.onnx` are **project-trained models**,
not stock Ultralytics releases. Their metadata records `Ultralytics` as the
author because the export tool wrote that field. The training data and the
project's own contribution are described in the dataset section below.

Ultralytics' own licensing page states that a commercial product, proprietary
application, or embedded deployment requires either:

1. compliance with AGPL-3.0, including making the complete corresponding
   source available under AGPL-3.0, or
2. an Ultralytics Enterprise License.

This applies to custom-trained and fine-tuned YOLO models as well as stock
pretrained weights. The distributor must choose one path before distributing a
commercial device or application. If the AGPL path is chosen, the notices and
source offer must cover the whole derivative work as required by that licence.

### 5. Project-trained models

These are the project's own trained/fine-tuned weights:

- `models/best.onnx` — trained on `data/data.yaml`
- `models/cigarette_smokers_v4.onnx` — trained on
  `/content/cigarette-smokers-clean/data.yaml`

They are not third-party stock models. The project does not claim that the
datasets, the Ultralytics training/export code, or the base architectures are
its own. The ONNX files carry Ultralytics AGPL metadata because they were
produced with the Ultralytics toolchain.

This repository does not contain a project-wide `LICENSE` file. This notices
file does not grant a license for the project's own code or trained weights.
The project owner should choose and add a project license before distributing
the source or granting rights to others.

## Datasets and training data

The repository does **not** include the training images or the dataset YAML
files. The following provenance is recorded only in the embedded ONNX metadata
or in the project history:

| Dataset reference | Used by | Status |
| --- | --- | --- |
| `data/data.yaml` | `models/best.onnx` | Source URL, licence, and consent record are not recorded in this repository |
| `/content/cigarette-smokers-clean/data.yaml` | `models/cigarette_smokers_v4.onnx` | Source URL, licence, and consent record are not recorded in this repository |
| COCO (`coco.yaml`) | `models/phone_yolo11n.onnx` | COCO dataset terms apply; verify image-level licences |

Before commercial use or redistribution:

- identify the source of every training image and annotation;
- record the dataset licence and any attribution requirements;
- confirm that people shown in the images gave the necessary permissions;
- check whether the dataset permits commercial use and model training;
- keep the dataset citation and licence with the product documentation.

COCO references:

- <https://cocodataset.org/>
- <https://cocodataset.org/#termsofuse>
- Lin, Tsung-Yi, Maire, Michael, Belongie, Serge, et al. "Microsoft COCO:
  Common Objects in Context." *ECCV* 2014. <https://arxiv.org/abs/1405.0312>

The absence of a dataset licence record is a blocker, not a permission. Do not
assume that a dataset is commercially usable because it was downloadable.

## Other runtime software

The application also depends on:

- **OpenCV** — Apache-2.0. <https://opencv.org/license/>
- **NumPy** — BSD-3-Clause. <https://numpy.org/doc/stable/license.html>
- **Python** — PSF-2.0. <https://docs.python.org/3/license.html>
- **Raspberry Pi OS / libcamera / ALSA / aplay** — system components under
  their own distribution licences.

These are installed or supplied separately and are not relicensed by this
project. A commercial device image should include the licence notices that
come with the operating-system packages it ships.

## Suggested short credits for an About screen

The following text can be shown in an About or Credits screen. It is a summary,
not a substitute for the full notices and licence texts.

```text
Third-party components
- YuNet face detector (c) 2020 Shiqi Yu, MIT License.
- MediaPipe face landmarks, Google LLC, Apache License 2.0.
- InsightFace 2d106 fallback, non-commercial research use only.
- Ultralytics YOLO models and exports, AGPL-3.0.
  See https://www.ultralytics.com/license for commercial licensing.
- COCO dataset, https://cocodataset.org.

Custom project models
- best.onnx: project-trained on data/data.yaml, exported with Ultralytics.
- cigarette_smokers_v4.onnx: project-trained on
  cigarette-smokers-clean/data.yaml, exported with Ultralytics.

No ownership is claimed over third-party models, datasets, code, names, or
trademarks. Full notices: THIRD_PARTY_NOTICES.md.
```

## Commercial checklist

Before selling or distributing a device or application that contains these
files:

- [ ] Remove `models/2d106det.onnx` from the commercial build, or obtain
      commercial permission and replace the fallback. The 478 model is the
      active path, so removing the fallback does not change normal operation.
- [ ] Resolve the Ultralytics path: comply with AGPL-3.0 (including source
      disclosure and notices) or obtain an Ultralytics Enterprise License.
- [ ] Verify the licence and consent status of `data/data.yaml`,
      `cigarette-smokers-clean/data.yaml`, and COCO images.
- [ ] Keep `THIRD_PARTY_NOTICES.md` and `licenses/` with the distributed
      software or device documentation.
- [ ] Add an About/Credits screen or printed notice that names the models and
      their licences, and explains how to obtain the full notices.
- [ ] Confirm the MediaPipe Face Mesh model card and the YuNet attribution
      requirements against the exact files being shipped.
- [ ] Do not use third-party names or logos as product branding without
      permission.

This checklist is a practical summary of the upstream terms, not legal advice.
For a commercial product, have a qualified lawyer review the final build and
the distribution model.
