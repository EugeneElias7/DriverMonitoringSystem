"""Object detection: YOLOv8/YOLO11 ONNX graphs through OpenCV's DNN module.

The app can run more than one detector: the 5-class driver model supplies
cigarette, drinking, eating, phone and seatbelt, while an optional one-class
model can add a dedicated cigarette detector. `merge_detections` drops
same-label duplicates from different models; `BackgroundDetector` runs each
forward pass off the frame thread.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DetectorConfig

# Class files vary in spelling ("Phone" vs "cell phone"); the behaviour
# monitors match on lowercase names.
PHONE_LABELS = frozenset({"cell phone", "phone"})


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in source-frame pixels
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", self.label.lower())

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def merge_detections(
    detections: list[Detection], iou_threshold: float = 0.5
) -> list[Detection]:
    """Merge results from several detectors, dropping same-label duplicates.

    The driver model and the dedicated cigarette model can both see the same
    cigarette. Keeping the higher-confidence box avoids drawing two boxes on
    one object; different labels and different objects are kept.
    """
    merged: list[Detection] = []
    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        duplicate = any(
            detection.label == kept.label
            and _iou(detection, kept) >= iou_threshold
            for kept in merged
        )
        if not duplicate:
            merged.append(detection)
    return merged


def _letterbox(
    image: np.ndarray, size: int, pad_value: int = 114
) -> tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, left, top


class YoloDetector:
    """YOLOv8 ONNX detector with a fixed square input (640 by default).

    Frame content is letterboxed because the exported graph has a fixed input
    size; the reverse transform runs in `_decode`.
    """

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.enabled = cfg.enabled
        self._source = Path(cfg.model_path).stem
        self._num_classes: int | None = None
        if not self.enabled:
            # A disabled detector must not require its files to exist.
            self._labels: list[str] = []
            self._net = None
            self._output_names: list[str] = []
            return

        self._labels = self._apply_whitelist(
            self._load_labels(cfg.classes_path), cfg.classes_whitelist
        )
        path = Path(cfg.model_path)
        if not path.exists():
            raise FileNotFoundError(f"detector model not found: {path}")

        self._net = cv2.dnn.readNetFromONNX(str(path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._output_names = self._net.getUnconnectedOutLayersNames()

    @staticmethod
    def _load_labels(classes_path: str | None) -> list[str]:
        if not classes_path:
            raise ValueError(
                "detector.classes_path must point at the model's class list "
                "(one name per line, in the export's class order)"
            )
        path = Path(classes_path)
        if not path.exists():
            raise FileNotFoundError(f"classes file not found: {path}")
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [n.lower() for n in names if n]

    @staticmethod
    def _apply_whitelist(labels: list[str], whitelist: list[str] | None) -> list[str]:
        # Keep class indices aligned with the model's own output: masked
        # entries become empty labels that _decode skips.
        if not whitelist:
            return labels
        keep = {name.lower() for name in whitelist}
        return [name if name in keep else "" for name in labels]

    def _class_floors(self, num_classes: int) -> np.ndarray:
        """Per-class acceptance floor, aligned with the model's class ids."""
        overrides = self.cfg.class_conf_overrides or {}
        floors = np.full(num_classes, self.cfg.conf_threshold, dtype=np.float32)
        if not overrides:
            return floors
        for label, floor in overrides.items():
            for index, name in enumerate(self._labels[:num_classes]):
                if name == label.lower():
                    floors[index] = float(floor)
        return floors

    @property
    def input_size(self) -> int:
        return self.cfg.input_size

    @property
    def source(self) -> str:
        return self._source

    @property
    def name(self) -> str:
        return self.cfg.name or self._source

    @property
    def num_classes(self) -> int | None:
        """Class count the graph actually emits, once a pass has run."""
        return self._num_classes

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def has_class(self, name: str) -> bool:
        return name in self._labels

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if not self.enabled:
            return []
        size = self.input_size
        canvas, scale, pad_x, pad_y = _letterbox(frame_bgr, size)
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 255.0, (size, size), swapRB=True, crop=False
        )
        self._net.setInput(blob)
        output = self._net.forward(self._output_names)[0]
        return self._decode(output, size, scale, pad_x, pad_y, frame_bgr.shape[:2])

    def close(self) -> None:
        self._net = None

    def _decode(
        self,
        output: np.ndarray,
        size: int,
        scale: float,
        pad_x: int,
        pad_y: int,
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        # (1, 4 + num_classes, N) -> (N, 4 + num_classes)
        predictions = output[0].T if output.ndim == 3 else output.T
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        # Ultralytics exports emit cx,cy,w,h in input pixels, but some
        # re-exports normalise them to 0..1. Pixel boxes are never below a few
        # pixels, so a maximum under 2 identifies the normalised form.
        if boxes.max() <= 2.0:
            boxes = boxes * size

        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        self._num_classes = int(scores.shape[1])

        floors = self._class_floors(scores.shape[1])
        keep = confidences >= floors[class_ids]
        if not keep.any():
            return []
        boxes, confidences, class_ids = boxes[keep], confidences[keep], class_ids[keep]

        cx, cy, bw, bh = boxes.T
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        rects = np.stack([x1, y1, bw / scale, bh / scale], axis=1).tolist()
        # NMS drops boxes below its score threshold, so it must accept the
        # lowest per-class floor, not the global threshold.
        indices = cv2.dnn.NMSBoxes(
            rects, confidences.tolist(), float(floors.min()), self.cfg.iou_threshold
        )
        if len(indices) == 0:
            return []

        h, w = frame_shape
        detections: list[Detection] = []
        for i in np.array(indices).flatten():
            bx, by, bwd, bht = rects[int(i)]
            left = int(max(0, bx))
            top = int(max(0, by))
            right = int(min(w, bx + bwd))
            bottom = int(min(h, by + bht))
            if right <= left or bottom <= top:
                continue
            # A box covering nearly the whole frame is a model failure, not a
            # phone/cigarette/seatbelt. Drop it before it reaches the preview
            # or the behaviour gates.
            if (right - left) * (bottom - top) >= 0.80 * w * h:
                continue
            label_id = int(class_ids[int(i)])
            label = self._labels[label_id] if label_id < len(self._labels) else str(label_id)
            if not label:
                continue
            detections.append(
                Detection(label, float(confidences[int(i)]), (left, top, right, bottom), source=self._source)
            )
        return detections


class BackgroundDetector:
    """Runs detection on a worker thread and republishes the latest results.

    A 640x640 YOLOv8 forward pass takes seconds on a Pi, so running it inline
    would starve the face loop and destroy drowsiness tracking. All consumers
    here only care about sustained presence over seconds, so a stale-by-a-bit
    result is fine — `results_at` lets the overlay expose its age.
    """

    def __init__(self, detector, target_hz: float = 2.0):
        self._detector = detector
        self._period = 1.0 / max(0.1, target_hz)
        self._frame: np.ndarray | None = None
        self._frame_seq = 0
        self._lock = threading.Lock()
        self._results: list[Detection] = []
        self._results_seq = 0
        # time.monotonic() at which the published results were computed. On a
        # Pi 3 a full pass takes 12-16 s, so boxes can belong to a frame that
        # is long gone; the overlay needs the age to avoid painting them onto
        # a current frame as if they were live.
        self._results_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_latency_ms = 0.0
        # The worker swallows exceptions so a bad pass cannot kill the loop;
        # this keeps the reason visible to the status line instead of leaving
        # detections silently empty.
        self.last_error: str | None = None

    @property
    def name(self) -> str:
        return getattr(self._detector, "name", self._detector.source)

    def start(self) -> None:
        if not self._detector.enabled:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"detector-{self.name}", daemon=True
        )
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray) -> None:
        if not self._detector.enabled:
            return
        with self._lock:
            self._frame = frame_bgr.copy()
            self._frame_seq += 1

    @property
    def latest(self) -> list[Detection]:
        with self._lock:
            return list(self._results)

    @property
    def results_at(self) -> float | None:
        """Monotonic timestamp of the last published pass, None before the first."""
        with self._lock:
            return self._results_at

    def snapshot(self) -> tuple[list[Detection], float | None]:
        """Results and their timestamp read atomically, so the age matches."""
        with self._lock:
            return list(self._results), self._results_at

    def _run(self) -> None:
        processed = 0
        while not self._stop.is_set():
            with self._lock:
                frame, seq = self._frame, self._frame_seq
            if frame is None or seq == processed:
                self._stop.wait(self._period)
                continue
            processed = seq
            started = time.perf_counter()
            try:
                results = self._detector.detect(frame)
            except Exception as exc:  # keep the loop alive; a dropped pass is harmless
                results = []
                self.last_error = f"{type(exc).__name__}: {exc}"
            else:
                self.last_error = None
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._results = results
                self._results_seq += 1
                self._results_at = time.monotonic()
            self._stop.wait(self._period)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Wait for the pass in flight instead of timing out after 2 s: a
            # 640x640 forward takes ~5 s on a Pi 3, and exiting while OpenCV
            # is inside it aborts the interpreter with "terminate called
            # without an active exception" after the run summary.
            self._thread.join()
            self._thread = None

    def __enter__(self) -> "BackgroundDetector":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()
