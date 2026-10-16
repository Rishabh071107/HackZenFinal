"""
Image-plane to pitch-plane coordinate mapping via OpenCV homography.

A single homography is only valid for a fixed (non-panning/zooming)
broadcast camera. This module supports that baseline case: the frontend
calibration step lets a user click 4 known pitch landmarks (e.g. corner
flags, penalty box corners) once per video, and every subsequent frame is
mapped through the resulting 3x3 matrix. For a moving broadcast camera,
this same interface can be re-computed per shot once shot-boundary
detection + line detection are added -- flagged here as a clear extension
point rather than silently pretending pan/zoom is handled.
"""

import cv2
import numpy as np


class PitchHomography:
    def __init__(
        self,
        image_points: list[tuple[float, float]],
        pitch_points: list[tuple[float, float]],
    ):
        if len(image_points) < 4 or len(pitch_points) < 4:
            raise ValueError("Homography requires at least 4 point correspondences.")

        src = np.array(image_points, dtype=np.float32)
        dst = np.array(pitch_points, dtype=np.float32)
        self.matrix, self._mask = cv2.findHomography(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        if self.matrix is None:
            raise ValueError(
                "Homography could not be computed from the given correspondences."
            )

    def image_to_pitch(self, x: float, y: float) -> tuple[float, float]:
        point = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.matrix)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def batch_image_to_pitch(self, points: np.ndarray) -> np.ndarray:
        """points: (N, 2) array of image-plane (x, y). Returns (N, 2) pitch-plane meters."""
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        mapped = cv2.perspectiveTransform(pts, self.matrix)
        return mapped.reshape(-1, 2)


DEFAULT_PITCH_CORNERS_M = [
    (0.0, 0.0),
    (105.0, 0.0),
    (105.0, 68.0),
    (0.0, 68.0),
]
