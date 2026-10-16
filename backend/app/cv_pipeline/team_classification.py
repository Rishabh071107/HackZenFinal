"""
Team classification from jersey color.

Pipeline (per problem statement):
  crop player -> mask out grass (HSV green range) -> HSV + LAB color spaces ->
  extract dominant color of the remaining pixels -> KMeans into 2 clusters
  across the whole match -> Team A / Team B assignment.

Referees are excluded upstream (detected class != "player") but a jersey
frequently seen alone as a third color cluster is handled by fitting K=3 and
dropping the smallest cluster, which is usually the referee/keeper kit.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans

# HSV range that covers common pitch-grass greens under broadcast lighting.
GRASS_HSV_LOWER = np.array([35, 40, 40])
GRASS_HSV_UPPER = np.array([85, 255, 255])


def _remove_grass(crop_bgr: np.ndarray) -> np.ndarray:
    """Returns the crop's non-grass pixels as an (N, 3) BGR array."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, GRASS_HSV_LOWER, GRASS_HSV_UPPER)
    non_grass_mask = cv2.bitwise_not(grass_mask)

    # Focus on the torso (upper 55% of the box) to avoid shorts/socks/pitch lines.
    h = crop_bgr.shape[0]
    torso_mask = np.zeros_like(non_grass_mask)
    torso_mask[: int(h * 0.55), :] = 255
    mask = cv2.bitwise_and(non_grass_mask, torso_mask)

    pixels = crop_bgr[mask > 0]
    if pixels.size == 0:
        # Fallback: whole crop, grass-removal was too aggressive.
        pixels = crop_bgr.reshape(-1, 3)
    return pixels


def get_appearance_vector(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts a normalized 3D LAB color feature vector from a player image crop.
    Focuses on torso pixels by masking out grass in HSV space and converting to LAB space.
    Returns a 3-element float32 array in LAB color space [L, A, B].
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    pixels = _remove_grass(crop_bgr)
    if len(pixels) < 5:
        mean_bgr = (
            pixels.mean(axis=0)
            if len(pixels)
            else np.array([128.0, 128.0, 128.0], dtype=np.float32)
        )
        lab = cv2.cvtColor(
            mean_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
        ).reshape(3)
        return lab.astype(np.float32)

    lab_pixels = cv2.cvtColor(
        pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3)

    k = min(2, len(lab_pixels))
    km = KMeans(n_clusters=k, n_init=4, random_state=42).fit(lab_pixels)
    counts = np.bincount(km.labels_)
    dominant_lab = km.cluster_centers_[counts.argmax()]
    return dominant_lab.astype(np.float32)


def appearance_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes normalized similarity score in [0.0, 1.0] between two LAB color vectors.
    Uses Euclidean distance in LAB space (CIELAB Delta E approximation).
    """
    if vec1 is None or vec2 is None:
        return 0.0
    dist = float(np.linalg.norm(vec1 - vec2))
    sim = max(0.0, 1.0 - (dist / 150.0))
    return sim


def extract_dominant_color(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Returns the dominant jersey color as a BGR triplet, using KMeans (k=2) on
    the LAB-converted torso pixels for perceptual accuracy, then mapping the
    winning cluster center back to BGR.
    """
    dominant_lab = get_appearance_vector(crop_bgr)
    dominant_bgr = cv2.cvtColor(
        dominant_lab.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2BGR
    ).reshape(3)
    return dominant_bgr.astype(np.float32)


@dataclass
class TeamAssignment:
    tracking_id: int
    team_label: str  # "Team A" / "Team B"


def assign_teams(
    track_dominant_colors: dict[int, np.ndarray],
) -> tuple[dict[int, str], dict[str, str]]:
    """
    Clusters every track's average dominant jersey color into two teams
    across the whole match (more stable than per-frame classification).

    Returns
    -------
    assignment: {tracking_id: "Team A" | "Team B"}
    team_colors: {"Team A": "#rrggbb", "Team B": "#rrggbb"}
    """
    if not track_dominant_colors:
        return {}, {}

    ids = list(track_dominant_colors.keys())
    colors_bgr = np.array([track_dominant_colors[i] for i in ids])

    # Fit with k=3 to absorb a referee/goalkeeper kit as its own tiny cluster,
    # then keep the two largest clusters as the two teams.
    k = min(3, len(ids)) if len(ids) >= 3 else min(2, len(ids))
    km = KMeans(n_clusters=k, n_init=6, random_state=42).fit(colors_bgr)

    counts = np.bincount(km.labels_)
    top_two_clusters = np.argsort(counts)[::-1][:2]

    label_map = {top_two_clusters[0]: "Team A"}
    if len(top_two_clusters) > 1:
        label_map[top_two_clusters[1]] = "Team B"

    assignment: dict[int, str] = {}
    for track_id, cluster_id in zip(ids, km.labels_):
        assignment[track_id] = label_map.get(
            cluster_id, "Unassigned"
        )  # e.g. referee cluster

    team_colors = {}
    for cluster_id, label in label_map.items():
        b, g, r = km.cluster_centers_[cluster_id]
        team_colors[label] = f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    return assignment, team_colors
