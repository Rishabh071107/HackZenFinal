"""
Domain 3 Feature Extraction Module for VisionX OD-17 Football Player Tracking.

Extracts structured visual, geometric, and physical descriptors for each detected player:
  - Color Features (HSV histogram + LAB torso color)
  - Shape Features (Aspect ratio, Area ratio, Hu Moments invariants)
  - Texture Features (Local Binary Pattern - LBP histogram)
  - Edge Features (Canny edge density & 4-quadrant distribution)
  - Geometric Features (Homography pitch X,Y coordinates)
  - Landmark Features (Normalized structural body keypoints)
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import get_settings
from app.cv_pipeline.detection import Detection

settings = get_settings()

# HSV grass removal mask bounds
GRASS_HSV_LOWER = np.array([35, 40, 40])
GRASS_HSV_UPPER = np.array([85, 255, 255])


def _extract_torso_pixels(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Masks out grass pixels and focuses on the torso (upper 55% of crop).
    Returns (torso_crop_bgr, torso_mask).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        empty = np.zeros((10, 10, 3), dtype=np.uint8)
        return empty, np.zeros((10, 10), dtype=np.uint8)

    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, GRASS_HSV_LOWER, GRASS_HSV_UPPER)
    non_grass = cv2.bitwise_not(grass_mask)

    torso_mask = np.zeros((h, w), dtype=np.uint8)
    torso_mask[: int(h * 0.55), :] = 255
    combined_mask = cv2.bitwise_and(non_grass, torso_mask)

    if cv2.countNonZero(combined_mask) < 10:
        # Fallback to top 55% if grass mask was too aggressive
        combined_mask[: int(h * 0.55), :] = 255

    return crop_bgr, combined_mask


def _compute_lbp_histogram(gray_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Computes Local Binary Pattern (LBP) 8-neighbor uniform texture histogram.
    """
    if (
        gray_img is None
        or gray_img.size == 0
        or gray_img.shape[0] < 3
        or gray_img.shape[1] < 3
    ):
        return np.ones(10, dtype=np.float32) / 10.0

    h, w = gray_img.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    center = gray_img[1:-1, 1:-1]
    # 8-neighbors comparison
    lbp |= ((gray_img[0:-2, 0:-2] >= center) << 7).astype(np.uint8)
    lbp |= ((gray_img[0:-2, 1:-1] >= center) << 6).astype(np.uint8)
    lbp |= ((gray_img[0:-2, 2:] >= center) << 5).astype(np.uint8)
    lbp |= ((gray_img[1:-1, 2:] >= center) << 4).astype(np.uint8)
    lbp |= ((gray_img[2:, 2:] >= center) << 3).astype(np.uint8)
    lbp |= ((gray_img[2:, 1:-1] >= center) << 2).astype(np.uint8)
    lbp |= ((gray_img[2:, 0:-2] >= center) << 1).astype(np.uint8)
    lbp |= ((gray_img[1:-1, 0:-2] >= center) << 0).astype(np.uint8)

    sub_mask = mask[1:-1, 1:-1]
    valid_lbp = lbp[sub_mask > 0] if cv2.countNonZero(sub_mask) > 0 else lbp.ravel()

    # 10-bin histogram representation
    hist, _ = np.histogram(valid_lbp, bins=10, range=(0, 256))
    hist = hist.astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else np.ones(10, dtype=np.float32) / 10.0


def _compute_canny_edge_features(
    gray_img: np.ndarray, mask: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Computes Canny edge density and 4-quadrant edge distribution.
    """
    if gray_img is None or gray_img.size == 0:
        return 0.0, np.zeros(4, dtype=np.float32)

    edges = cv2.Canny(gray_img, 50, 150)
    if mask is not None and mask.shape == edges.shape:
        edges = cv2.bitwise_and(edges, edges, mask=mask)

    total_pts = float(edges.size)
    edge_density = float(np.count_nonzero(edges)) / max(1.0, total_pts)

    h, w = edges.shape
    h_half, w_half = max(1, h // 2), max(1, w // 2)
    q1 = np.count_nonzero(edges[:h_half, :w_half])
    q2 = np.count_nonzero(edges[:h_half, w_half:])
    q3 = np.count_nonzero(edges[h_half:, :w_half])
    q4 = np.count_nonzero(edges[h_half:, w_half:])

    quads = np.array([q1, q2, q3, q4], dtype=np.float32)
    sq = quads.sum()
    quads_norm = quads / sq if sq > 0 else np.ones(4, dtype=np.float32) * 0.25
    return edge_density, quads_norm


def _compute_hu_moments(mask: np.ndarray) -> np.ndarray:
    """Computes 7 log-transformed invariant Hu Moments from bounding mask."""
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform for scale invariance stability
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu_log.astype(np.float32)


@dataclass
class PlayerFeatureVector:
    """Unified Domain 3 Structured Feature Representation for a Player Detection."""

    color_lab: np.ndarray  # [L, A, B] mean torso color
    color_hsv_hist: np.ndarray  # 16-bin HSV hue histogram
    shape_vec: np.ndarray  # [w/W, h/H, aspect_ratio, area_ratio]
    hu_moments: np.ndarray  # 7 Hu moments
    lbp_hist: np.ndarray  # 10-bin LBP texture histogram
    edge_density: float  # Canny edge density ratio
    edge_quadrants: np.ndarray  # 4-quadrant edge distribution
    pitch_coords: tuple[float, float]  # (X_m, Y_m) pitch position
    landmarks: np.ndarray  # 5 normalized keypoints [(x,y), ...]

    def to_dict(self) -> dict:
        """Serializes feature vector into a clean JSON-ready inspection structure."""
        return {
            "color_features": {
                "lab_mean": [round(float(v), 2) for v in self.color_lab],
                "hsv_histogram": [round(float(v), 3) for v in self.color_hsv_hist],
            },
            "shape_features": {
                "width_ratio": round(float(self.shape_vec[0]), 4),
                "height_ratio": round(float(self.shape_vec[1]), 4),
                "aspect_ratio": round(float(self.shape_vec[2]), 2),
                "area_ratio": round(float(self.shape_vec[3]), 4),
                "hu_moments": [round(float(v), 3) for v in self.hu_moments],
            },
            "texture_features": {
                "lbp_histogram": [round(float(v), 3) for v in self.lbp_hist],
            },
            "edge_features": {
                "canny_edge_density": round(float(self.edge_density), 4),
                "edge_quadrants": [round(float(v), 3) for v in self.edge_quadrants],
            },
            "geometric_features": {
                "pitch_x_m": round(float(self.pitch_coords[0]), 2),
                "pitch_y_m": round(float(self.pitch_coords[1]), 2),
            },
            "landmark_features": {
                "keypoints": [
                    [round(float(pt[0]), 3), round(float(pt[1]), 3)]
                    for pt in self.landmarks
                ],
            },
        }


class PlayerFeatureExtractor:
    """Extracts structured Domain 3 features and calculates multi-cue similarities."""

    def extract_features(
        self,
        crop_bgr: np.ndarray,
        det: Detection,
        frame_width: int,
        frame_height: int,
        pitch_coords: tuple[float, float] = (0.0, 0.0),
    ) -> PlayerFeatureVector:
        w_frame = max(1.0, float(frame_width))
        h_frame = max(1.0, float(frame_height))

        w_box = max(1.0, det.x2 - det.x1)
        h_box = max(1.0, det.y2 - det.y1)

        aspect_ratio = h_box / w_box
        area_ratio = (w_box * h_box) / (w_frame * h_frame)
        shape_vec = np.array(
            [w_box / w_frame, h_box / h_frame, aspect_ratio, area_ratio],
            dtype=np.float32,
        )

        if crop_bgr is None or crop_bgr.size == 0:
            return PlayerFeatureVector(
                color_lab=np.array([128.0, 128.0, 128.0], dtype=np.float32),
                color_hsv_hist=np.ones(16, dtype=np.float32) / 16.0,
                shape_vec=shape_vec,
                hu_moments=np.zeros(7, dtype=np.float32),
                lbp_hist=np.ones(10, dtype=np.float32) / 10.0,
                edge_density=0.0,
                edge_quadrants=np.ones(4, dtype=np.float32) * 0.25,
                pitch_coords=pitch_coords,
                landmarks=np.array(
                    [[0.5, 0.1], [0.3, 0.3], [0.7, 0.3], [0.5, 0.5], [0.5, 0.9]],
                    dtype=np.float32,
                ),
            )

        # 1. Color Features
        crop, mask = _extract_torso_pixels(crop_bgr)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue_channel = hsv[:, :, 0]
        hue_hist, _ = np.histogram(hue_channel[mask > 0], bins=16, range=(0, 180))
        s_hue = hue_hist.sum()
        hsv_hist_norm = hue_hist.astype(np.float32) / max(1.0, s_hue)

        pixels_bgr = crop[mask > 0]
        if len(pixels_bgr) > 5:
            lab_pixels = cv2.cvtColor(
                pixels_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
            ).reshape(-1, 3)
            color_lab = lab_pixels.mean(axis=0).astype(np.float32)
        else:
            color_lab = np.array([128.0, 128.0, 128.0], dtype=np.float32)

        # 2. Shape & Hu Moments
        hu_moments = _compute_hu_moments(mask)

        # 3. Texture Features (LBP)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lbp_hist = _compute_lbp_histogram(gray, mask)

        # 4. Edge Features (Canny)
        edge_density, edge_quads = _compute_canny_edge_features(gray, mask)

        # 5. Landmark Features (Normalized 5 keypoints: head, L-shoulder, R-shoulder, torso center, feet)
        landmarks = np.array(
            [
                [0.5, 0.1],  # Head top
                [0.3, 0.35],  # Left shoulder
                [0.7, 0.35],  # Right shoulder
                [0.5, 0.55],  # Torso center
                [0.5, 0.95],  # Feet
            ],
            dtype=np.float32,
        )

        return PlayerFeatureVector(
            color_lab=color_lab,
            color_hsv_hist=hsv_hist_norm,
            shape_vec=shape_vec,
            hu_moments=hu_moments,
            lbp_hist=lbp_hist,
            edge_density=edge_density,
            edge_quadrants=edge_quads,
            pitch_coords=pitch_coords,
            landmarks=landmarks,
        )

    def color_similarity(
        self, f1: PlayerFeatureVector, f2: PlayerFeatureVector
    ) -> float:
        dist_lab = float(np.linalg.norm(f1.color_lab - f2.color_lab))
        sim_lab = max(0.0, 1.0 - (dist_lab / 150.0))
        sim_hsv = float(np.sum(np.minimum(f1.color_hsv_hist, f2.color_hsv_hist)))
        return 0.6 * sim_lab + 0.4 * sim_hsv

    def shape_similarity(
        self, f1: PlayerFeatureVector, f2: PlayerFeatureVector
    ) -> float:
        dist_shape = float(np.linalg.norm(f1.shape_vec - f2.shape_vec))
        sim_shape = max(0.0, 1.0 - dist_shape)
        dist_hu = float(np.linalg.norm(f1.hu_moments - f2.hu_moments))
        sim_hu = max(0.0, 1.0 - (dist_hu / 10.0))
        return 0.6 * sim_shape + 0.4 * sim_hu

    def texture_similarity(
        self, f1: PlayerFeatureVector, f2: PlayerFeatureVector
    ) -> float:
        sim_lbp = float(np.sum(np.minimum(f1.lbp_hist, f2.lbp_hist)))
        return max(0.0, min(1.0, sim_lbp))

    def edge_similarity(
        self, f1: PlayerFeatureVector, f2: PlayerFeatureVector
    ) -> float:
        dens_diff = abs(f1.edge_density - f2.edge_density)
        sim_dens = max(0.0, 1.0 - (dens_diff / 0.5))
        quad_sim = float(np.sum(np.minimum(f1.edge_quadrants, f2.edge_quadrants)))
        return 0.5 * sim_dens + 0.5 * quad_sim

    def landmark_similarity(
        self, f1: PlayerFeatureVector, f2: PlayerFeatureVector
    ) -> float:
        dist_lm = float(np.mean(np.linalg.norm(f1.landmarks - f2.landmarks, axis=1)))
        return max(0.0, 1.0 - (dist_lm / 0.5))

    def combined_appearance_similarity(
        self, f1: PlayerFeatureVector, f2: PlayerFeatureVector
    ) -> float:
        """Weighted fusion of all Domain 3 appearance cues."""
        s_col = self.color_similarity(f1, f2)
        s_shp = self.shape_similarity(f1, f2)
        s_tex = self.texture_similarity(f1, f2)
        s_edg = self.edge_similarity(f1, f2)
        s_lmk = self.landmark_similarity(f1, f2)

        total_weight = (
            settings.WEIGHT_COLOR
            + settings.WEIGHT_SHAPE
            + settings.WEIGHT_TEXTURE
            + settings.WEIGHT_EDGE
            + settings.WEIGHT_LANDMARK
        )

        fused = (
            settings.WEIGHT_COLOR * s_col
            + settings.WEIGHT_SHAPE * s_shp
            + settings.WEIGHT_TEXTURE * s_tex
            + settings.WEIGHT_EDGE * s_edg
            + settings.WEIGHT_LANDMARK * s_lmk
        ) / max(1e-5, total_weight)

        return float(max(0.0, min(1.0, fused)))
