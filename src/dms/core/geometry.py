"""Bounding-box, letterbox and NMS helpers used by all detector backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class LetterboxMeta:
    ratio: float
    pad_x: float
    pad_y: float
    input_width: int
    input_height: int


def clip_bbox(bbox: Iterable[float], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1 = min(max(x1, 0.0), max(width - 1.0, 0.0))
    y1 = min(max(y1, 0.0), max(height - 1.0, 0.0))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return (x1, y1, x2, y2)


def xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> BBox:
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def xyxy_to_xywh(bbox: BBox) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2 - x1, y2 - y1)


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def expand_bbox(bbox: BBox, padding_ratio: float, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    return clip_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def point_to_bbox_distance(point: tuple[float, float], bbox: BBox) -> float:
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    scale_up: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize with centred padding, returning the image and mapping metadata."""
    if image is None or image.size == 0:
        raise ValueError("letterbox received an empty image")
    height, width = image.shape[:2]
    target_h, target_w = int(new_shape[0]), int(new_shape[1])
    ratio = min(target_w / width, target_h / height)
    if not scale_up:
        ratio = min(ratio, 1.0)
    new_w = max(1, int(round(width * ratio)))
    new_h = max(1, int(round(height * ratio)))
    if stride > 1:
        new_w = max(stride, int(round(new_w / stride) * stride))
        new_h = max(stride, int(round(new_h / stride) * stride))
        new_w = min(new_w, target_w)
        new_h = min(new_h, target_h)
        ratio = min(new_w / width, new_h / height)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_x = pad_w / 2.0
    pad_y = pad_h / 2.0
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if padded.shape[0] != target_h or padded.shape[1] != target_w:
        padded = cv2.resize(padded, (target_w, target_h))
    return padded, LetterboxMeta(ratio, pad_x, pad_y, target_w, target_h)


def unletterbox_bbox(bbox: Sequence[float], meta: LetterboxMeta, source_width: int, source_height: int) -> BBox:
    """Map a box from letterboxed network coordinates back to source pixels."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1 = (x1 - meta.pad_x) / meta.ratio
    x2 = (x2 - meta.pad_x) / meta.ratio
    y1 = (y1 - meta.pad_y) / meta.ratio
    y2 = (y2 - meta.pad_y) / meta.ratio
    return clip_bbox((x1, y1, x2, y2), source_width, source_height)


def nms_classwise(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    class_ids: Sequence[int],
    iou_threshold: float,
) -> list[int]:
    """Greedy per-class NMS. Returns indices into the input arrays."""
    if len(boxes) == 0:
        return []
    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    classes_arr = np.asarray(class_ids, dtype=np.int32)
    keep: list[int] = []
    for class_id in np.unique(classes_arr):
        indices = np.where(classes_arr == class_id)[0]
        xywh = np.column_stack(
            [
                boxes_arr[indices, 0],
                boxes_arr[indices, 1],
                boxes_arr[indices, 2] - boxes_arr[indices, 0],
                boxes_arr[indices, 3] - boxes_arr[indices, 1],
            ]
        )
        selected = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores_arr[indices].tolist(),
            score_threshold=0.0,
            nms_threshold=float(iou_threshold),
        )
        if selected is None or len(selected) == 0:
            continue
        for local_idx in np.asarray(selected).reshape(-1):
            keep.append(int(indices[int(local_idx)]))
    keep.sort(key=lambda idx: float(scores_arr[idx]), reverse=True)
    return keep


def normalize_label(label: str) -> str:
    return "".join(str(label).strip().lower().replace("_", " ").split())
