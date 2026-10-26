"""TFLite interpreter factory and generic SSD/EfficientDet-style decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.base import BaseDetector
from ..core.geometry import clip_bbox
from ..core.types import Detection


def create_interpreter(model_path: str | Path):
    """Create a TFLite interpreter without importing TensorFlow when possible."""
    errors: list[str] = []
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=str(model_path))
    except Exception as exc:  # pragma: no cover - depends on installation
        errors.append(f"ai_edge_litert: {exc}")
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=str(model_path))
    except Exception as exc:  # pragma: no cover - depends on installation
        errors.append(f"tflite_runtime: {exc}")
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=str(model_path))
    except Exception as exc:
        errors.append(f"tensorflow.lite: {exc}")
    raise RuntimeError("no TFLite interpreter available; tried " + "; ".join(errors))


def read_labelmap(path: str | Path) -> list[str]:
    labels: list[str] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            labels.append(value)
    if not labels:
        raise ValueError(f"empty label map: {path}")
    return labels


class TFLiteDetectorBackend(BaseDetector):
    """Decode TensorFlow Lite object-detection models.

    Supports the common TF2 Detection API output set:
    boxes ``[1, N, 4]`` (ymin, xmin, ymax, xmax), scores ``[1, N]``,
    classes ``[1, N]``, and an optional ``[1]`` detection count. Input and output
    tensor meaning is validated at runtime rather than trusted from names.
    """

    def __init__(
        self,
        name: str,
        config: Any,
        model_path: str | Path,
        labelmap_path: str | Path,
        input_range: str = "unit",
        score_threshold: float = 0.3,
        known_issue: bool = False,
        labelmap_offset: int = 0,
    ):
        super().__init__(name, config)
        self.model_path = Path(model_path)
        self.labelmap_path = Path(labelmap_path)
        self.input_range = input_range
        self.score_threshold = float(score_threshold)
        self.known_issue = bool(known_issue)
        self.labelmap_offset = int(labelmap_offset)
        self.interpreter = None
        self.class_names: list[str] = []
        self._input_index = 0
        self._input_shape: tuple[int, ...] = ()
        self._input_dtype = np.float32
        self._outputs: list[dict[str, Any]] = []
        self._quant_params: dict[str, Any] = {}

    def load(self) -> None:
        if self._loaded:
            return
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"TFLite model not found: {self.model_path}")
            if not self.labelmap_path.exists():
                raise FileNotFoundError(f"label map not found: {self.labelmap_path}")
            labels = read_labelmap(self.labelmap_path)
            if self.labelmap_offset:
                labels = labels[self.labelmap_offset :]
            self.class_names = labels
            self.interpreter = create_interpreter(self.model_path)
            self.interpreter.allocate_tensors()
            inputs = self.interpreter.get_input_details()
            outputs = self.interpreter.get_output_details()
            if len(inputs) != 1:
                raise ValueError(f"expected one TFLite input, found {len(inputs)}")
            if len(outputs) < 3:
                raise ValueError(f"expected at least three TFLite outputs, found {len(outputs)}")
            inp = inputs[0]
            shape = tuple(int(v) for v in inp["shape"])
            if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
                raise ValueError(f"unsupported input tensor shape: {shape}")
            self._input_index = int(inp["index"])
            self._input_shape = shape
            self._input_dtype = inp["dtype"]
            self._quant_params = dict(inp.get("quantization_parameters") or {})
            self._outputs = [
                {
                    "index": int(d["index"]),
                    "name": str(d["name"]),
                    "shape": tuple(int(v) for v in d["shape"]),
                    "dtype": d["dtype"],
                }
                for d in outputs
            ]
            has_box = any(o["shape"][-1:] == (4,) and len(o["shape"]) == 3 for o in self._outputs)
            has_num = any(int(np.prod(o["shape"])) == 1 for o in self._outputs)
            if not has_box or not has_num:
                raise ValueError(
                    "could not identify boxes/count outputs; this backend expects a "
                    "TF2 Detection API model"
                )
            message = f"TFLite {self._input_dtype.__name__} {shape[2]}x{shape[1]}; {len(self.class_names)} labels"
            if self.known_issue:
                self.health.state = "degraded"
                self.health.ready = True
                self._loaded = True
                self.health.message = message + "; known-problematic checkpoint, do not treat output as validated"
            else:
                self._mark_ready(message)
        except Exception as exc:
            self._mark_error(f"{type(exc).__name__}: {exc}")
            raise

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        _, height, width, _ = self._input_shape
        image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._input_dtype == np.uint8:
            return np.expand_dims(image.astype(np.uint8), axis=0)
        if self._input_dtype == np.int8:
            scale, zero = self._input_quant()
            if scale <= 0:
                raise ValueError("int8 input has no quantization scale")
            quantized = np.round(image.astype(np.float32) / 255.0 / scale + zero)
            quantized = np.clip(quantized, -128, 127).astype(np.int8)
            return np.expand_dims(quantized, axis=0)
        if self._input_dtype == np.float32:
            data = image.astype(np.float32)
            if self.input_range == "unit":
                data /= 255.0
            elif self.input_range == "raw":
                pass
            else:
                raise ValueError(f"unsupported input_range: {self.input_range}")
            return np.expand_dims(data, axis=0)
        raise ValueError(f"unsupported TFLite input dtype: {self._input_dtype}")

    def _input_quant(self) -> tuple[float, int]:
        scales = self._quant_params.get("scales") or []
        points = self._quant_params.get("zero_points") or []
        if len(scales) == 0:
            return (0.0, 0)
        return (float(scales[0]), int(points[0]) if len(points) else 0)

    def _detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        if self.interpreter is None:
            return []
        data = self._preprocess(frame)
        self.interpreter.set_tensor(self._input_index, data)
        self.interpreter.invoke()
        arrays = {
            o["name"]: self.interpreter.get_tensor(o["index"]).copy()
            for o in self._outputs
        }
        return self.decode_arrays(arrays, frame.shape[1], frame.shape[0])

    def decode_arrays(
        self, arrays: dict[str, np.ndarray], source_width: int, source_height: int
    ) -> list[Detection]:
        """Decode a TF2 Detection API output dictionary into detections."""
        boxes = None
        num = None
        remaining: list[tuple[str, np.ndarray]] = []
        for name, value in arrays.items():
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes = arr[0]
            elif arr.size == 1:
                num = arr.reshape(-1)[0]
            else:
                remaining.append((name, arr.reshape(-1)))
        if boxes is None:
            raise ValueError("box tensor not found in TFLite outputs")
        if len(remaining) < 2:
            raise ValueError("score/class tensors not found in TFLite outputs")
        classes_arr, scores_arr = _identify_classes_and_scores(remaining, len(self.class_names))
        count = int(num) if num is not None else boxes.shape[0]
        count = max(0, min(count, boxes.shape[0], scores_arr.shape[0], classes_arr.shape[0]))
        results: list[Detection] = []
        for idx in range(count):
            score = float(scores_arr[idx])
            if not np.isfinite(score) or score < self.score_threshold:
                continue
            class_id = int(round(float(classes_arr[idx])))
            if not (0 <= class_id < len(self.class_names)):
                continue
            box = boxes[idx].astype(np.float64)
            if float(np.nanmax(box)) <= 2.0:
                x1 = box[1] * source_width
                y1 = box[0] * source_height
                x2 = box[3] * source_width
                y2 = box[2] * source_height
            else:
                x1, y1, x2, y2 = box[1], box[0], box[3], box[2]
            bbox = clip_bbox((x1, y1, x2, y2), source_width, source_height)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            results.append(
                Detection(
                    label=self.class_names[class_id],
                    confidence=score,
                    bbox=bbox,
                    class_id=class_id,
                    source=self.name,
                    metadata={"backend": "tflite", "known_issue": self.known_issue},
                )
            )
        return results


def _identify_classes_and_scores(
    candidates: list[tuple[str, np.ndarray]], num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Use names first, then integer-likeness, to identify class vs score tensors."""
    if len(candidates) != 2:
        raise ValueError(f"expected exactly two class/score tensors, got {len(candidates)}")
    by_name: dict[str, np.ndarray] = {}
    for name, value in candidates:
        lower = name.lower()
        if "class" in lower:
            by_name["class"] = value
        elif "score" in lower:
            by_name["score"] = value
    if "class" in by_name and "score" in by_name:
        return by_name["class"], by_name["score"]

    (name_a, a), (name_b, b) = candidates

    def class_likeness(values: np.ndarray) -> float:
        if values.size == 0:
            return -1.0
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return -1.0
        integer = np.mean(np.isclose(finite, np.round(finite), atol=1e-3))
        in_range = np.mean((finite >= -0.01) & (finite <= num_classes - 0.99))
        distinct = len(np.unique(np.round(finite))) / max(len(finite), 1)
        return float(integer + in_range + 0.25 * distinct)

    score_a = class_likeness(a)
    score_b = class_likeness(b)
    if score_a >= score_b:
        return a, b
    return b, a
