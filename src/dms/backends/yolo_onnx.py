"""YOLO-family ONNX Runtime backend (YOLOv5/v8/v11-style outputs)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.base import BaseDetector
from ..core.geometry import (
    letterbox,
    nms_classwise,
    normalize_label,
    unletterbox_bbox,
    xywh_to_xyxy,
)
from ..core.types import Detection


class YoloOnnxBackend(BaseDetector):
    """Class-aware YOLO ONNX decoder with strict class-name validation.

    Generic COCO checkpoints have no cigarette class. This backend therefore
    refuses to run if the checkpoint metadata and configured labels disagree;
    it never silently maps an arbitrary class to ``cigarette``.
    """

    def __init__(self, name: str, config: Any, model_path: str | Path):
        super().__init__(name, config)
        self.model_path = Path(model_path)
        self.session = None
        self.class_names: list[str] = []
        self.input_name = ""
        self.output_name = ""
        self.input_width = int(config.get("smoking.input_size", 640))
        self.input_height = int(config.get("smoking.input_size", 640))
        self.confidence_threshold = float(config.get("smoking.confidence_threshold", 0.25))
        self.nms_iou = float(config.get("smoking.nms_iou", 0.45))
        self.class_thresholds = {
            normalize_label(k): float(v)
            for k, v in (config.get("smoking.class_thresholds") or {}).items()
        }
        self._output_shape: tuple[Any, ...] = ()

    def load(self) -> None:
        if self._loaded:
            return
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if len(inputs) != 1 or len(outputs) < 1:
                raise ValueError("expected one ONNX input and at least one output")
            inp = inputs[0]
            if len(inp.shape) != 4:
                raise ValueError(f"unsupported ONNX input shape: {inp.shape}")
            self.input_name = inp.name
            self.output_name = outputs[0].name
            self._output_shape = tuple(outputs[0].shape)
            if isinstance(inp.shape[2], int) and inp.shape[2] > 0:
                self.input_height = int(inp.shape[2])
            if isinstance(inp.shape[3], int) and inp.shape[3] > 0:
                self.input_width = int(inp.shape[3])
            if inp.shape[1] != 3:
                raise ValueError(f"expected NCHW 3-channel input, got {inp.shape}")
            metadata = self.session.get_modelmeta().custom_metadata_map or {}
            meta_names = _parse_names(metadata.get("names"))
            configured = [
                str(v) for v in (self.config.get("smoking.class_names") or [])
            ]
            if meta_names:
                self.class_names = meta_names
                if configured and [
                    normalize_label(v) for v in configured
                ] != [normalize_label(v) for v in meta_names]:
                    raise ValueError(
                        "configured smoking.class_names do not match checkpoint metadata: "
                        f"configured={configured!r}, checkpoint={meta_names!r}"
                    )
            elif configured:
                self.class_names = configured
            else:
                raise ValueError("no class names in ONNX metadata or configuration")
            target = normalize_label(str(self.config.get("smoking.expected_target_class", "cigarette")))
            if target not in [normalize_label(v) for v in self.class_names]:
                raise ValueError(
                    f"target class {target!r} not present in checkpoint classes {self.class_names!r}"
                )
            is_ultralytics = "ultralytics" in str(metadata.get("description", "")).lower() or "ultralytics" in str(
                metadata.get("author", "")
            ).lower()
            self._mark_ready(
                f"ONNX {self.input_width}x{self.input_height}; classes={self.class_names}; "
                f"metadata={'ultralytics' if is_ultralytics else 'unknown'}"
            )
        except Exception as exc:
            self._mark_error(f"{type(exc).__name__}: {exc}")
            raise

    def _threshold_for(self, class_id: int) -> float:
        label = normalize_label(self.class_names[class_id]) if 0 <= class_id < len(self.class_names) else ""
        return float(self.class_thresholds.get(label, self.confidence_threshold))

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, Any]:
        padded, meta = letterbox(
            frame,
            (self.input_height, self.input_width),
            color=(114, 114, 114),
            scale_up=True,
            stride=32,
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor), meta

    def _detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        if self.session is None:
            return []
        tensor, meta = self._preprocess(frame)
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return self.decode_output(raw, meta, frame.shape[1], frame.shape[0])

    def decode_output(
        self,
        raw: np.ndarray,
        meta: Any,
        source_width: int,
        source_height: int,
    ) -> list[Detection]:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(f"unsupported batch size: {arr.shape}")
            arr = arr[0]
        nc = len(self.class_names)
        end2end = False
        if arr.ndim == 2 and arr.shape[1] == 6 and arr.shape[0] <= 1000:
            # Common NMS/end-to-end export: x1, y1, x2, y2, score, class.
            end2end = True
            boxes_xyxy = arr[:, :4].copy()
            scores_all = arr[:, 4]
            class_ids = arr[:, 5].astype(np.int32)
        else:
            if arr.ndim != 2:
                raise ValueError(f"unsupported YOLO output shape: {arr.shape}")
            if arr.shape[0] <= arr.shape[1]:
                arr = arr.T
            channels = arr.shape[1]
            if channels == 4 + nc:
                boxes_xywh = arr[:, :4]
                class_scores = arr[:, 4:]
                scores_all = class_scores.max(axis=1)
                class_ids = class_scores.argmax(axis=1).astype(np.int32)
            elif channels == 5 + nc:
                boxes_xywh = arr[:, :4]
                objectness = arr[:, 4]
                class_scores = arr[:, 5:]
                scores_all = objectness * class_scores.max(axis=1)
                class_ids = class_scores.argmax(axis=1).astype(np.int32)
            else:
                raise ValueError(
                    f"YOLO output has {channels} channels, expected {4 + nc} or {5 + nc} "
                    f"for {nc} classes (shape {arr.shape})"
                )
            boxes_xyxy = np.asarray(
                [xywh_to_xyxy(*[float(v) for v in box]) for box in boxes_xywh],
                dtype=np.float32,
            )

        keep_mask = []
        for idx, score in enumerate(scores_all):
            if not np.isfinite(score):
                keep_mask.append(False)
                continue
            cid = int(class_ids[idx])
            if not (0 <= cid < nc):
                keep_mask.append(False)
                continue
            keep_mask.append(float(score) >= self._threshold_for(cid))
        mask = np.asarray(keep_mask, dtype=bool)
        if not mask.any():
            return []
        boxes_xyxy = boxes_xyxy[mask]
        scores_all = scores_all[mask]
        class_ids = class_ids[mask]
        if not end2end:
            keep = nms_classwise(boxes_xyxy, scores_all, class_ids, self.nms_iou)
            boxes_xyxy = boxes_xyxy[keep]
            scores_all = scores_all[keep]
            class_ids = class_ids[keep]
        results: list[Detection] = []
        for box, score, cid in zip(boxes_xyxy, scores_all, class_ids):
            bbox = unletterbox_bbox(box, meta, source_width, source_height)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            label = self.class_names[int(cid)]
            results.append(
                Detection(
                    label=label,
                    confidence=float(score),
                    bbox=bbox,
                    class_id=int(cid),
                    source=self.name,
                    metadata={"backend": "onnx", "end2end": end2end},
                )
            )
        results.sort(key=lambda d: d.confidence, reverse=True)
        return results


def _parse_names(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        mapping = raw
    else:
        text = str(raw).strip()
        if not text:
            return []
        mapping = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, dict):
                    mapping = parsed
                    break
            except Exception:
                continue
        if mapping is None:
            return []
    try:
        return [str(mapping[key]) for key in sorted(mapping, key=lambda v: int(v))]
    except Exception:
        return [str(v) for _, v in sorted(mapping.items())]
