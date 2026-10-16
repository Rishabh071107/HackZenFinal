"""
Dedicated Kalman-based Ball Tracker with confidence scoring, velocity gating,
and occlusion handling for football broadcast video.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from app.cv_pipeline.detection import Detection


class BallState(str, Enum):
    OBSERVED = "observed"
    PREDICTED = "predicted"
    LOST = "lost"


def _new_ball_kf(cx: float, cy: float) -> KalmanFilter:
    """Kalman filter for ball motion: state [x, y, vx, vy], measurement [x, y]."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    kf.H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    kf.x[:2, 0] = [cx, cy]
    kf.P *= 10.0
    kf.R *= 4.0
    kf.Q *= 0.5
    return kf


@dataclass
class BallSample:
    frame_number: int
    timestamp_seconds: float
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    state: BallState
    pitch_x: float = 0.0
    pitch_y: float = 0.0


class BallTracker:
    """
    Tracks football across frames with occlusion handling, maximum velocity
    filtering, and trajectory recording.
    """

    def __init__(self, max_lost_frames: int = 15, max_speed_px_per_frame: float = 120.0):
        self.kf: Optional[KalmanFilter] = None
        self.state: BallState = BallState.LOST
        self.time_since_update: int = 0
        self.max_lost_frames = max_lost_frames
        self.max_speed_px = max_speed_px_per_frame
        self.last_bbox: Optional[tuple[float, float, float, float]] = None
        self.last_confidence: float = 0.0
        self.history: list[BallSample] = []

    def update(
        self,
        ball_detections: list[Detection],
        frame_number: int,
        timestamp_s: float,
        pitch_homography=None,
    ) -> Optional[BallSample]:
        """
        Updates the ball tracker for the current frame.
        Picks the best candidate matching Kalman prediction within velocity bounds.
        """
        if self.kf is not None:
            self.kf.predict()
            self.time_since_update += 1

        matched_det: Optional[Detection] = None

        if ball_detections:
            # Sort candidate detections by confidence descending
            candidates = sorted(ball_detections, key=lambda d: d.confidence, reverse=True)

            if self.kf is None or self.state == BallState.LOST:
                # Initialize or reacquire on highest confidence candidate
                matched_det = candidates[0]
            else:
                pred_x, pred_y = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
                # Find closest candidate within speed gating
                best_dist = float("inf")
                for cand in candidates:
                    cx = (cand.x1 + cand.x2) / 2.0
                    cy = (cand.y1 + cand.y2) / 2.0
                    dist = float(np.hypot(cx - pred_x, cy - pred_y))
                    if dist <= self.max_speed_px and dist < best_dist:
                        best_dist = dist
                        matched_det = cand

        sample: Optional[BallSample] = None

        if matched_det is not None:
            cx = (matched_det.x1 + matched_det.x2) / 2.0
            cy = (matched_det.y1 + matched_det.y2) / 2.0
            w = max(10.0, matched_det.x2 - matched_det.x1)
            h = max(10.0, matched_det.y2 - matched_det.y1)

            if self.kf is None:
                self.kf = _new_ball_kf(cx, cy)
            else:
                self.kf.update(np.array([[cx], [cy]]))

            self.state = BallState.OBSERVED
            self.time_since_update = 0
            self.last_bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            self.last_confidence = matched_det.confidence

            px, py = (0.0, 0.0)
            if pitch_homography is not None:
                px, py = pitch_homography.image_to_pitch(cx, cy)

            sample = BallSample(
                frame_number=frame_number,
                timestamp_seconds=timestamp_s,
                x1=self.last_bbox[0],
                y1=self.last_bbox[1],
                x2=self.last_bbox[2],
                y2=self.last_bbox[3],
                confidence=matched_det.confidence,
                state=BallState.OBSERVED,
                pitch_x=px,
                pitch_y=py,
            )
            self.history.append(sample)

        elif self.kf is not None and self.time_since_update <= self.max_lost_frames:
            # Predict position during temporary occlusion/miss
            cx = float(self.kf.x[0, 0])
            cy = float(self.kf.x[1, 0])
            w = (self.last_bbox[2] - self.last_bbox[0]) if self.last_bbox else 20.0
            h = (self.last_bbox[3] - self.last_bbox[1]) if self.last_bbox else 20.0

            self.state = BallState.PREDICTED
            decayed_conf = max(0.10, self.last_confidence * (0.90 ** self.time_since_update))
            self.last_bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

            px, py = (0.0, 0.0)
            if pitch_homography is not None:
                px, py = pitch_homography.image_to_pitch(cx, cy)

            sample = BallSample(
                frame_number=frame_number,
                timestamp_seconds=timestamp_s,
                x1=self.last_bbox[0],
                y1=self.last_bbox[1],
                x2=self.last_bbox[2],
                y2=self.last_bbox[3],
                confidence=decayed_conf,
                state=BallState.PREDICTED,
                pitch_x=px,
                pitch_y=py,
            )
            self.history.append(sample)

        else:
            self.state = BallState.LOST

        return sample

    def get_trajectory_points(self) -> list[list[float]]:
        """Returns [[timestamp, pitch_x, pitch_y], ...] for observed/predicted ball path."""
        return [
            [round(s.timestamp_seconds, 3), round(s.pitch_x, 2), round(s.pitch_y, 2)]
            for s in self.history
            if s.pitch_x != 0.0 or s.pitch_y != 0.0
        ]
