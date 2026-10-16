"""
Movement heatmaps via Gaussian kernel density estimation over pitch
coordinates, rendered on top of a football pitch outline.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering, no display server in the container
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from app.core.config import get_settings

settings = get_settings()

GRID_RESOLUTION = 200


def _draw_pitch(ax):
    L, W = settings.PITCH_LENGTH_M, settings.PITCH_WIDTH_M
    ax.set_facecolor("#0b3d14")
    ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], color="white", linewidth=1.2)
    ax.plot([L / 2, L / 2], [0, W], color="white", linewidth=1.0)
    center_circle = plt.Circle(
        (L / 2, W / 2), 9.15, color="white", fill=False, linewidth=1.0
    )
    ax.add_patch(center_circle)
    # Penalty boxes
    for x0, sign in [(0, 1), (L, -1)]:
        box_x = x0 + sign * 16.5
        ax.plot(
            [x0, box_x, box_x, x0],
            [W / 2 - 20.16, W / 2 - 20.16, W / 2 + 20.16, W / 2 + 20.16],
            color="white",
            linewidth=1.0,
        )
    ax.set_xlim(-2, L + 2)
    ax.set_ylim(-2, W + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def compute_density_grid(pitch_x: np.ndarray, pitch_y: np.ndarray) -> dict:
    """Returns a serializable KDE grid for the interactive frontend heatmap layer."""
    L, W = settings.PITCH_LENGTH_M, settings.PITCH_WIDTH_M
    xx, yy = np.mgrid[
        0 : L : complex(GRID_RESOLUTION), 0 : W : complex(GRID_RESOLUTION)
    ]

    if len(pitch_x) < 2:
        density = np.zeros_like(xx)
    else:
        try:
            x_data = pitch_x
            y_data = pitch_y
            # Check if variance is zero or very small in either dimension, or if points are collinear
            if (
                np.var(pitch_x) < 1e-6
                or np.var(pitch_y) < 1e-6
                or len(np.unique(pitch_x)) < 2
            ):
                # Add small random noise (jitter) to ensure covariance is positive-definite
                x_data = pitch_x + np.random.normal(0, 1e-3, size=pitch_x.shape)
                y_data = pitch_y + np.random.normal(0, 1e-3, size=pitch_y.shape)

            kde = gaussian_kde(np.vstack([x_data, y_data]), bw_method=0.25)
            density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            density = density / density.max() if density.max() > 0 else density
        except (ValueError, np.linalg.LinAlgError):
            density = np.zeros_like(xx)

    return {
        "pitch_length_m": L,
        "pitch_width_m": W,
        "grid_resolution": GRID_RESOLUTION,
        "density": density.tolist(),
    }


def render_heatmap_png(
    pitch_x: np.ndarray, pitch_y: np.ndarray, output_path: Path, title: str = ""
) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=150)
    _draw_pitch(ax)

    if len(pitch_x) >= 2:
        try:
            x_data = pitch_x
            y_data = pitch_y
            if (
                np.var(pitch_x) < 1e-6
                or np.var(pitch_y) < 1e-6
                or len(np.unique(pitch_x)) < 2
            ):
                x_data = pitch_x + np.random.normal(0, 1e-3, size=pitch_x.shape)
                y_data = pitch_y + np.random.normal(0, 1e-3, size=pitch_y.shape)

            kde = gaussian_kde(np.vstack([x_data, y_data]), bw_method=0.25)
            L, W = settings.PITCH_LENGTH_M, settings.PITCH_WIDTH_M
            xx, yy = np.mgrid[
                0 : L : complex(GRID_RESOLUTION), 0 : W : complex(GRID_RESOLUTION)
            ]
            density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, density, levels=25, cmap="inferno", alpha=0.75)
        except (ValueError, np.linalg.LinAlgError):
            # Degenerate case: all points identical (player barely moved).
            ax.scatter(pitch_x, pitch_y, color="red", s=40)
    else:
        ax.scatter(pitch_x, pitch_y, color="red", s=40)

    if title:
        ax.set_title(title, color="white", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output_path
