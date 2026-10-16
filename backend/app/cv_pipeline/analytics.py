"""
Football analytics computed from a player's (or team's) pitch-coordinate
tracking samples over time. All distances in meters, speeds in m/s.
"""

import numpy as np

from app.core.config import get_settings

settings = get_settings()

STATIONARY_SPEED_THRESHOLD_MPS = 0.3  # below this, treat player as not moving


def _thirds_boundaries() -> tuple[float, float]:
    L = settings.PITCH_LENGTH_M
    return L / 3, 2 * L / 3


def get_zone_index(x: float, y: float) -> int:
    """
    Standard 18-zone football pitch grid.
    Length (X): 105m divided into 6 equal columns (each 17.5m wide).
    Width (Y): 68m divided into 3 rows:
      - Left Wing: 0 to 17m
      - Center: 17m to 51m
      - Right Wing: 51m to 68m

    Zones are numbered 1 to 18:
      Col 1 (0 - 17.5m):   Left=1,  Center=2,  Right=3
      Col 2 (17.5 - 35m):  Left=4,  Center=5,  Right=6
      Col 3 (35 - 52.5m):  Left=7,  Center=8,  Right=9
      Col 4 (52.5 - 70m):  Left=10, Center=11, Right=12
      Col 5 (70 - 87.5m):  Left=13, Center=14, Right=15 (Zone 14 is Attacking Center)
      Col 6 (87.5 - 105m): Left=16, Center=17, Right=18 (Zone 17 is Penalty Center)
    """
    x = max(0.0, min(104.99, x))
    y = max(0.0, min(67.99, y))

    col = int(x // 17.5)

    if y < 17.0:
        row = 0
    elif y < 51.0:
        row = 1
    else:
        row = 2

    return col * 3 + row + 1


def compute_player_analytics(
    timestamps: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> dict:
    """
    timestamps, xs, ys: aligned arrays of a single player's samples across
    the whole match (already mapped to pitch meters via homography).
    """
    if len(xs) < 2:
        return _empty_player_metrics()

    dt = np.diff(timestamps)
    dt = np.where(dt <= 0, 1e-6, dt)
    dx = np.diff(xs)
    dy = np.diff(ys)
    step_distances = np.hypot(dx, dy)

    total_distance_m = float(step_distances.sum())
    speeds_mps = step_distances / dt
    avg_speed_mps = float(speeds_mps.mean()) if len(speeds_mps) else 0.0
    max_speed_mps = float(speeds_mps.max()) if len(speeds_mps) else 0.0

    accelerations = (
        np.diff(speeds_mps) / dt[1:] if len(speeds_mps) > 1 else np.array([])
    )
    avg_acceleration = (
        float(np.abs(accelerations).mean()) if len(accelerations) else 0.0
    )

    time_stationary_s = float(dt[speeds_mps < STATIONARY_SPEED_THRESHOLD_MPS].sum())
    match_duration_s = (
        float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    )

    third1, third2 = _thirds_boundaries()
    defensive_time = float(dt[xs[:-1] < third1].sum())
    middle_time = float(dt[(xs[:-1] >= third1) & (xs[:-1] < third2)].sum())
    attacking_time = float(dt[xs[:-1] >= third2].sum())

    W = settings.PITCH_WIDTH_M
    left_time = float(dt[ys[:-1] < W / 3].sum())
    central_time = float(dt[(ys[:-1] >= W / 3) & (ys[:-1] < 2 * W / 3)].sum())
    right_time = float(dt[ys[:-1] >= 2 * W / 3].sum())

    avg_position = [float(xs.mean()), float(ys.mean())]

    # Calculate 18-zone tactical grid occupancy
    zone_grid_occupancy = {str(z): 0.0 for z in range(1, 19)}
    for i in range(len(xs) - 1):
        z = get_zone_index(xs[i], ys[i])
        zone_grid_occupancy[str(z)] += float(dt[i])

    return {
        "distance_covered_m": round(total_distance_m, 1),
        "average_speed_mps": round(avg_speed_mps, 2),
        "max_speed_mps": round(max_speed_mps, 2),
        "average_acceleration_mps2": round(avg_acceleration, 2),
        "time_stationary_s": round(time_stationary_s, 1),
        "match_duration_s": round(match_duration_s, 1),
        "average_position": avg_position,
        "zone_occupancy": {
            "defensive_third_s": round(defensive_time, 1),
            "middle_third_s": round(middle_time, 1),
            "attacking_third_s": round(attacking_time, 1),
            "left_wing_s": round(left_time, 1),
            "central_s": round(central_time, 1),
            "right_wing_s": round(right_time, 1),
        },
        "zone_grid_occupancy": {k: round(v, 2) for k, v in zone_grid_occupancy.items()},
        "estimated_role": _estimate_role(defensive_time, middle_time, attacking_time),
    }


def _estimate_role(
    defensive_time: float, middle_time: float, attacking_time: float
) -> str:
    """Heuristic role estimate purely from time-in-thirds distribution."""
    total = defensive_time + middle_time + attacking_time
    if total == 0:
        return "Unknown"
    d, m, a = defensive_time / total, middle_time / total, attacking_time / total
    if d > 0.5:
        return "Defender"
    if a > 0.45:
        return "Forward"
    if m > 0.45:
        return "Midfielder"
    return "Midfielder"


def _empty_player_metrics() -> dict:
    return {
        "distance_covered_m": 0.0,
        "average_speed_mps": 0.0,
        "max_speed_mps": 0.0,
        "average_acceleration_mps2": 0.0,
        "time_stationary_s": 0.0,
        "match_duration_s": 0.0,
        "average_position": [0.0, 0.0],
        "zone_occupancy": {
            "defensive_third_s": 0.0,
            "middle_third_s": 0.0,
            "attacking_third_s": 0.0,
            "left_wing_s": 0.0,
            "central_s": 0.0,
            "right_wing_s": 0.0,
        },
        "zone_grid_occupancy": {str(z): 0.0 for z in range(1, 19)},
        "estimated_role": "Unknown",
    }


def compute_team_analytics(
    player_metrics: list[dict], player_positions: list[tuple[float, float]]
) -> dict:
    if not player_metrics:
        return {}

    positions = np.array(player_positions) if player_positions else np.empty((0, 2))
    avg_position = positions.mean(axis=0).tolist() if len(positions) else [0.0, 0.0]

    width = (
        float(positions[:, 1].max() - positions[:, 1].min())
        if len(positions) > 1
        else 0.0
    )
    depth = (
        float(positions[:, 0].max() - positions[:, 0].min())
        if len(positions) > 1
        else 0.0
    )
    compactness = (
        float(np.mean(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))
        if len(positions) > 1
        else 0.0
    )

    total_defensive = sum(
        m["zone_occupancy"]["defensive_third_s"] for m in player_metrics
    )
    total_middle = sum(m["zone_occupancy"]["middle_third_s"] for m in player_metrics)
    total_attacking = sum(
        m["zone_occupancy"]["attacking_third_s"] for m in player_metrics
    )
    total_time = total_defensive + total_middle + total_attacking or 1.0

    # Aggregate 18-zone tactical grid metrics for the team
    team_zone_occupancy_s = {str(z): 0.0 for z in range(1, 19)}
    for m in player_metrics:
        if "zone_grid_occupancy" in m:
            for z_str, s in m["zone_grid_occupancy"].items():
                team_zone_occupancy_s[z_str] += s

    total_team_time = sum(team_zone_occupancy_s.values()) or 1.0
    team_zone_percentages = {
        z_str: round(100.0 * s / total_team_time, 1)
        for z_str, s in team_zone_occupancy_s.items()
    }

    return {
        "average_position": avg_position,
        "team_width_m": round(width, 1),
        "team_depth_m": round(depth, 1),
        "compactness_m": round(compactness, 1),
        "attacking_percentage": round(100 * total_attacking / total_time, 1),
        "midfield_percentage": round(100 * total_middle / total_time, 1),
        "defensive_percentage": round(100 * total_defensive / total_time, 1),
        "total_distance_covered_m": round(
            sum(m["distance_covered_m"] for m in player_metrics), 1
        ),
        "zone_percentages": team_zone_percentages,
    }


def compute_match_analytics(
    total_players: int,
    total_frames: int,
    duration_s: float,
    avg_conf: float,
    avg_track_acc: float,
    confirmed_tracks: int = 0,
    id_switches: int = 0,
    recovered_tracks: int = 0,
    rejected_false_detections: int = 0,
) -> dict:
    return {
        "total_players_detected": total_players,
        "total_frames": total_frames,
        "match_duration_s": round(duration_s, 1),
        "average_detection_confidence": round(avg_conf, 3),
        "average_tracking_accuracy": round(avg_track_acc, 3),
        "confirmed_tracks": confirmed_tracks,
        "id_switches": id_switches,
        "recovered_tracks": recovered_tracks,
        "rejected_false_detections": rejected_false_detections,
    }
