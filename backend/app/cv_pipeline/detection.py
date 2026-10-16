"""
Player / referee / ball detection using an Ultralytics YOLO model.

Notes for production use
-------------------------
The default `yolo11n.pt` weights are COCO-pretrained (generic "person" class,
no ball/referee distinction). For real broadcast-football accuracy you should
fine-tune on a football dataset (e.g. SoccerNet-Tracking) and point
`YOLO_MODEL_PATH` at that checkpoint. The interfaces below already assume a
3-class model (player, referee, ball) via `CLASS_MAP` so swapping the weights
is a drop-in change.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import get_settings

settings = get_settings()

# Expected class layout for a football-finetuned model. For the generic COCO
# fallback model, only "person" (0) exists and everyone is treated as a player.
CLASS_MAP = {0: "player", 1: "referee", 2: "ball"}


@dataclass
class Detection:
    frame_number: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, self.y2  # foot position, not box centroid

    @property
    def as_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])


class PlayerDetector:
    """Thin, swappable wrapper around an Ultralytics YOLO model with dual-mode class mapping."""

    def __init__(
        self, model_path: Optional[str] = None, confidence: Optional[float] = None
    ):
        from ultralytics import (
            YOLO,
        )  # imported lazily so the API can boot without torch installed

        model_str = model_path or settings.YOLO_MODEL_PATH
        if not Path(model_str).is_file():
            candidates = [
                Path(model_str),
                Path(__file__).resolve().parents[2] / model_str,
                Path(__file__).resolve().parents[3] / model_str,
                settings.WEIGHTS_DIR / model_str,
                Path(__file__).resolve().parents[2] / "weights" / model_str,
            ]
            for cand in candidates:
                if cand.is_file():
                    model_str = str(cand)
                    break

        self.model = YOLO(model_str)
        self.confidence = (
            confidence
            if confidence is not None
            else settings.CONFIDENCE_THRESHOLD
        )
        self.names = getattr(self.model, "names", {})

    def _resolve_class_name(self, cls_id: int) -> Optional[str]:
        if not self.names:
            return CLASS_MAP.get(cls_id)

        if cls_id not in self.names:
            return None

        raw_name = str(self.names[cls_id]).lower()

        # Check for fine-tuned or domain-specific names
        if "ball" in raw_name:
            return "ball"
        if "ref" in raw_name:
            return "referee"
        if "player" in raw_name or "goalkeeper" in raw_name or "keeper" in raw_name:
            return "player"

        # COCO 80-class mapping
        if cls_id == 0 or raw_name == "person":
            return "player"
        if cls_id == 32 or "sports ball" in raw_name:
            return "ball"

        return None

    def _boxes_to_detections(self, results, frame_number: int) -> list[Detection]:
        detections: list[Detection] = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = self._resolve_class_name(cls_id)
            if class_name is None:
                continue

            detections.append(
                Detection(
                    frame_number=frame_number,
                    class_name=class_name,
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    def detect(self, frame: np.ndarray, frame_number: int) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=settings.IOU_THRESHOLD,
            imgsz=settings.INFERENCE_IMGSZ,
            verbose=False,
        )[0]
        return self._boxes_to_detections(results, frame_number)

    def detect_batch(
        self, frames: list[np.ndarray], frame_numbers: list[int]
    ) -> list[list[Detection]]:
        """Batched inference for throughput; one result list per input frame."""
        if not frames:
            return []
        results = self.model.predict(
            frames,
            conf=self.confidence,
            iou=settings.IOU_THRESHOLD,
            imgsz=settings.INFERENCE_IMGSZ,
            verbose=False,
        )
        return [
            self._boxes_to_detections(res, frame_numbers[i])
            for i, res in enumerate(results)
        ]
