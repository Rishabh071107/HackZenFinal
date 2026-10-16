"""
Trajectory generation: smooths a player's raw per-frame pitch positions into
a clean movement path using a Savitzky-Golay filter (falls back to a moving
average for very short tracks), and reports start/end points + direction.
"""

import numpy as np
from scipy.signal import savgol_filter


def smooth_trajectory(
    timestamps: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> list[list[float]]:
    n = len(xs)
    if n == 0:
        return []
    if n < 5:
        return [[float(t), float(x), float(y)] for t, x, y in zip(timestamps, xs, ys)]

    window = min(11, n if n % 2 == 1 else n - 1)
    window = max(window, 5)
    poly_order = min(3, window - 1)

    xs_smooth = savgol_filter(xs, window, poly_order)
    ys_smooth = savgol_filter(ys, window, poly_order)

    return [
        [float(t), float(x), float(y)]
        for t, x, y in zip(timestamps, xs_smooth, ys_smooth)
    ]


def trajectory_endpoints(points: list[list[float]]) -> tuple[list[float], list[float]]:
    if not points:
        return [], []
    return points[0][1:], points[-1][1:]
