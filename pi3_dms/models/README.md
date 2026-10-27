# Models

The **required** face-core model is:

```text
face_landmarker.task
```

SHA-256:

```text
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
```

It is the MediaPipe Face Landmarker task model. The app loads it locally; there
is no download at startup.

The optional object model is:

```text
object/efficientdet_lite0.tflite
```

SHA-256:

```text
2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b
```

It is an official COCO EfficientDet-Lite0 control model (Apache-2.0 model
collection). It has **no cigarette class** and is not validated for driver
monitoring. The object module is disabled by default. See
[`object/README.md`](object/README.md).

The model and MediaPipe runtime have their own licence and model-card terms.
Review them before redistributing this folder or using it commercially. The
models are used here for research/demo purposes only and are not validated for
safety-critical driver monitoring.
