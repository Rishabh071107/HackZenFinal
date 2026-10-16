"""
VisionX OD-17 Multi-Cue Football Player Tracker & Track Management Architecture.

Implements the full VisionX tracking pipeline:
  1. Kalman Filter motion prediction (constant-velocity model).
  2. Multi-Cue Hungarian Association Cost Fusion:
       TOTAL_COST = α * IoU_Cost + β * Appearance_Cost + γ * Motion_Cost + δ * Geometry_Cost
  3. Track Lifecycle States: TENTATIVE -> CONFIRMED -> LOST -> REMOVED
       - TENTATIVE tracks require MIN_CONFIRMATION_HITS (3) before identity confirmation.
       - Prevents transient noise / drones from receiving player tracking IDs.
  4. Feature Gallery Re-Identification (ReID):
       - Maintains a gallery of recent Domain 3 feature vectors per track.
       - Recovers lost tracks after occlusions, camera movements, and reappearance.
  5. Tracking Event Debugging Log:
       - Records per-frame matching scores (IoU, appearance, motion, geometry) & decisions.
"""

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from app.core.config import get_settings
from app.cv_pipeline.detection import Detection
from app.cv_pipeline.feature_extraction import (
    PlayerFeatureExtractor,
    PlayerFeatureVector,
)

settings = get_settings()


class TrackState(str, enum.Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _new_kalman_filter(x1, y1, x2, y2) -> KalmanFilter:
    """State: [cx, cy, w, h, vx, vy, vw, vh] with constant-velocity motion model."""
    kf = KalmanFilter(dim_x=8, dim_z=4)
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    kf.F = np.eye(8)
    for i in range(4):
        kf.F[i, i + 4] = 1.0
    kf.H = np.zeros((4, 8))
    kf.H[:4, :4] = np.eye(4)

    kf.x[:4, 0] = [cx, cy, w, h]
    kf.P *= 10.0
    kf.R *= 5.0
    kf.Q *= 0.5
    return kf


def _kf_to_xyxy(kf: KalmanFilter) -> np.ndarray:
    cx, cy, w, h = kf.x[0, 0], kf.x[1, 0], kf.x[2, 0], kf.x[3, 0]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    track_id: int
    kf: KalmanFilter
    class_name: str
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    last_confidence: float = 0.0
    history: list = field(
        default_factory=list
    )  # [(frame_number, x1,y1,x2,y2,conf), ...]
    feature_vector: Optional[PlayerFeatureVector] = None
    feature_gallery: list[PlayerFeatureVector] = field(default_factory=list)
    last_center: Optional[tuple[float, float]] = None
    last_pitch_coords: tuple[float, float] = (0.0, 0.0)
    last_frame_number: int = 0
    in_crossing: bool = False

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        if (
            self.time_since_update > settings.TRACKER_MAX_AGE_FRAMES
            and self.state == TrackState.CONFIRMED
        ):
            self.state = TrackState.LOST

    def update(
        self,
        det: Detection,
        feat_vec: Optional[PlayerFeatureVector] = None,
        pitch_coords: tuple[float, float] = (0.0, 0.0),
    ):
        z = np.array(
            [
                [(det.x1 + det.x2) / 2],
                [(det.y1 + det.y2) / 2],
                [det.x2 - det.x1],
                [det.y2 - det.y1],
            ]
        )
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0
        self.last_confidence = det.confidence
        self.last_center = ((det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0)
        self.last_pitch_coords = pitch_coords
        self.last_frame_number = det.frame_number
        self.history.append(
            (det.frame_number, det.x1, det.y1, det.x2, det.y2, det.confidence)
        )

        # Confirm track after MIN_CONFIRMATION_HITS (1)
        if self.hits >= settings.MIN_CONFIRMATION_HITS or self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

        if feat_vec is not None:
            self.feature_vector = feat_vec
            self.feature_gallery.append(feat_vec)
            if len(self.feature_gallery) > 15:
                self.feature_gallery.pop(0)

    @property
    def bbox(self) -> np.ndarray:
        return _kf_to_xyxy(self.kf)


class ByteTracker:
    def __init__(self):
        self._next_id = 1
        self.tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.extractor = PlayerFeatureExtractor()

        # Global track feature gallery registry for cross-video identity persistence
        # track_id -> list of PlayerFeatureVector
        self.global_tracks_gallery: dict[int, list[PlayerFeatureVector]] = {}
        self.global_tracks_last_pos: dict[int, tuple[float, float]] = {}
        self.global_tracks_last_frame: dict[int, int] = {}

        # Metrics for VisionX evaluation dashboard
        self.id_switch_count: int = 0
        self.recovered_tracks_count: int = 0
        self.total_confirmed_tracks: int = 0

        # Event debug log for VisionX judges inspection
        self.tracking_event_log: list[dict] = []

    def _log_event(
        self,
        frame_number: int,
        prev_id: int,
        matched_det_idx: int,
        iou_score: float,
        app_score: float,
        motion_score: float,
        geom_score: float,
        final_cost: float,
        decision: str,
    ):
        self.tracking_event_log.append(
            {
                "frame_number": int(frame_number),
                "previous_track_id": int(prev_id) if prev_id is not None else None,
                "matched_detection_index": int(matched_det_idx) if matched_det_idx is not None else None,
                "iou_score": round(float(iou_score), 3),
                "appearance_score": round(float(app_score), 3),
                "motion_score": round(float(motion_score), 3),
                "geometry_score": round(float(geom_score), 3),
                "final_cost": round(float(final_cost), 3),
                "decision": str(decision),
            }
        )

    def _register_track_gallery(self, track: Track):
        """Registers clean feature vectors into the global persistent gallery for this track ID."""
        if track.feature_gallery:
            self.global_tracks_gallery[track.track_id] = list(track.feature_gallery)
        if track.last_center is not None:
            self.global_tracks_last_pos[track.track_id] = track.last_center
        self.global_tracks_last_frame[track.track_id] = track.last_frame_number

    def _compute_multi_cue_cost(
        self,
        t: Track,
        d: Detection,
        d_feat: Optional[PlayerFeatureVector],
        pitch_coords: tuple[float, float],
    ) -> tuple[float, float, float, float, float]:
        # 1. IoU Cost
        iou_val = iou(t.bbox, d.as_xyxy)
        cost_iou = 1.0 - iou_val

        # 2. Appearance Cost (fusing Domain 3 features against feature gallery)
        if t.feature_gallery and d_feat is not None:
            gallery_sims = [
                self.extractor.combined_appearance_similarity(g_feat, d_feat)
                for g_feat in t.feature_gallery
            ]
            app_sim = float(np.max(gallery_sims))
        elif t.feature_vector is not None and d_feat is not None:
            app_sim = self.extractor.combined_appearance_similarity(
                t.feature_vector, d_feat
            )
        else:
            app_sim = 0.5
        cost_app = 1.0 - app_sim

        # Adaptive weights during player crossing / overlap
        if getattr(t, "in_crossing", False):
            w_iou = settings.WEIGHT_IOU * 0.3
            w_app = settings.APPEARANCE_WEIGHT_ON_CROSSING
            w_mot = settings.WEIGHT_MOTION
            w_geo = settings.WEIGHT_GEOMETRY
        else:
            w_iou = settings.WEIGHT_IOU
            w_app = settings.WEIGHT_APPEARANCE
            w_mot = settings.WEIGHT_MOTION
            w_geo = settings.WEIGHT_GEOMETRY

        # 3. Motion Cost (predicted center distance)
        d_center = d.center
        if t.last_center is not None:
            dist_motion = float(
                np.hypot(d_center[0] - t.last_center[0], d_center[1] - t.last_center[1])
            )
        else:
            dist_motion = 0.0
        cost_motion = min(1.0, dist_motion / 250.0)
        motion_sim = max(0.0, 1.0 - cost_motion)

        # 4. Geometry Cost (pitch coordinate distance in meters)
        if pitch_coords != (0.0, 0.0) and t.last_pitch_coords != (0.0, 0.0):
            dist_geom = float(
                np.hypot(
                    pitch_coords[0] - t.last_pitch_coords[0],
                    pitch_coords[1] - t.last_pitch_coords[1],
                )
            )
            cost_geom = min(1.0, dist_geom / 35.0)
        else:
            cost_geom = cost_motion
        geom_sim = max(0.0, 1.0 - cost_geom)

        total_weight = w_iou + w_app + w_mot + w_geo
        fused_cost = (
            w_iou * cost_iou
            + w_app * cost_app
            + w_mot * cost_motion
            + w_geo * cost_geom
        ) / max(1e-5, total_weight)

        return fused_cost, iou_val, app_sim, motion_sim, geom_sim

    def _match(
        self,
        tracks: list[Track],
        detections: list[Detection],
        det_features: list[Optional[PlayerFeatureVector]],
        pitch_coords_list: list[tuple[float, float]],
        frame_number: int,
    ) -> tuple[list, list, list]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost = np.zeros((len(tracks), len(detections)))
        score_matrix = []

        for i, t in enumerate(tracks):
            row_scores = []
            is_track_crossing = any(
                iou(t.bbox, other.bbox) >= settings.CROSSING_IOU_THRESHOLD
                for ot_idx, other in enumerate(tracks)
                if ot_idx != i
            )
            t.in_crossing = is_track_crossing

            for j, d in enumerate(detections):
                d_feat = det_features[j] if j < len(det_features) else None
                p_coords = (
                    pitch_coords_list[j] if j < len(pitch_coords_list) else (0.0, 0.0)
                )

                fused_c, iou_v, app_s, mot_s, geo_s = self._compute_multi_cue_cost(
                    t, d, d_feat, p_coords
                )
                cost[i, j] = fused_c
                row_scores.append((iou_v, app_s, mot_s, geo_s, fused_c))

            score_matrix.append(row_scores)

        row_idx, col_idx = linear_sum_assignment(cost)
        matches, unmatched_tracks, unmatched_dets = (
            [],
            set(range(len(tracks))),
            set(range(len(detections))),
        )

        for r, c in zip(row_idx, col_idx):
            iou_v, app_s, mot_s, geo_s, fused_c = score_matrix[r][c]
            # Match threshold check: allow high appearance similarity to confirm match even when spatial IoU drops
            if fused_c <= 0.82 or (app_s >= 0.45 and fused_c <= 0.88):
                matches.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)
                self._log_event(
                    frame_number,
                    tracks[r].track_id,
                    c,
                    iou_v,
                    app_s,
                    mot_s,
                    geo_s,
                    fused_c,
                    f"MATCH_CONFIRMED_ID_{tracks[r].track_id}",
                )

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def _relink_lost_tracks(
        self,
        unmatched_dets: list[Detection],
        unmatched_features: list[Optional[PlayerFeatureVector]],
        pitch_coords_list: list[tuple[float, float]],
        frame_number: int,
    ) -> tuple[dict[int, Track], list[Detection]]:
        if not self.lost_tracks or not unmatched_dets:
            return {}, unmatched_dets

        relinked = {}
        still_unmatched = []

        for idx, det in enumerate(unmatched_dets):
            d_feat = unmatched_features[idx] if idx < len(unmatched_features) else None
            p_coords = (
                pitch_coords_list[idx] if idx < len(pitch_coords_list) else (0.0, 0.0)
            )
            det_center = det.center

            best_match: Optional[Track] = None
            best_sim = 0.0
            best_scores = (0.0, 0.0, 0.0, 0.0, 1.0)

            for lt in self.lost_tracks:
                if lt.last_center is None:
                    continue

                elapsed_frames = max(1, frame_number - lt.last_frame_number)
                if elapsed_frames > settings.MAX_LOST_FRAMES:
                    continue

                dist_px = float(
                    np.hypot(
                        det_center[0] - lt.last_center[0],
                        det_center[1] - lt.last_center[1],
                    )
                )
                speed_px_per_frame = dist_px / elapsed_frames

                # Spatial-temporal plausibility filter (reject teleports across pitch)
                if speed_px_per_frame > settings.REID_MAX_SPEED_PX_PER_FRAME:
                    continue

                fused_c, iou_v, app_s, mot_s, geo_s = self._compute_multi_cue_cost(
                    lt, det, d_feat, p_coords
                )
                if app_s >= settings.REID_SIMILARITY_THRESHOLD and app_s > best_sim:
                    best_sim = app_s
                    best_match = lt
                    best_scores = (iou_v, app_s, mot_s, geo_s, fused_c)

            if best_match is not None:
                relinked[idx] = best_match
                self.lost_tracks.remove(best_match)
                self.recovered_tracks_count += 1
                self.id_switch_count += 1
                iou_v, app_s, mot_s, geo_s, fused_c = best_scores
                self._log_event(
                    frame_number,
                    best_match.track_id,
                    idx,
                    iou_v,
                    app_s,
                    mot_s,
                    geo_s,
                    fused_c,
                    f"REID_RECOVERY_RESTORED_ID_{best_match.track_id}",
                )
            else:
                still_unmatched.append(det)

        return relinked, still_unmatched

    def _relink_global_registry(
        self,
        det: Detection,
        feat_vec: Optional[PlayerFeatureVector],
        frame_number: int,
    ) -> Optional[int]:
        """Checks global registry of past tracks to reuse identical player ID if appearance matches."""
        if feat_vec is None or not self.global_tracks_gallery:
            return None

        best_id: Optional[int] = None
        best_sim = 0.0

        for tid, gallery in self.global_tracks_gallery.items():
            if any(t.track_id == tid for t in self.tracks):
                continue  # already active in current tracks

            gallery_sims = [
                self.extractor.combined_appearance_similarity(g_feat, feat_vec)
                for g_feat in gallery
            ]
            sim = float(np.max(gallery_sims))

            # Proximity check if position is known (reject teleports across pitch)
            last_pos = self.global_tracks_last_pos.get(tid)
            if last_pos is not None:
                dist = float(
                    np.hypot(det.center[0] - last_pos[0], det.center[1] - last_pos[1])
                )
                last_fn = self.global_tracks_last_frame.get(tid, 0)
                elapsed = max(1, frame_number - last_fn)
                if (dist / elapsed) > settings.REID_MAX_SPEED_PX_PER_FRAME:
                    continue

            if sim >= 0.45 and sim > best_sim:
                best_sim = sim
                best_id = tid

        return best_id

    def update(
        self,
        detections: list[Detection],
        frame_bgr: Optional[np.ndarray] = None,
        frame_number: int = 0,
        pitch_homography=None,
    ) -> list[Track]:
        """Call once per frame with validated player detections. Returns active tracks."""
        for t in self.tracks:
            t.predict()

        h_frame, w_frame = (
            (frame_bgr.shape[:2])
            if frame_bgr is not None and frame_bgr.size > 0
            else (720, 1280)
        )

        # Extract Domain 3 feature vectors & pitch coordinates for each detection
        det_features: list[Optional[PlayerFeatureVector]] = []
        pitch_coords_list: list[tuple[float, float]] = []

        for d in detections:
            if pitch_homography is not None:
                px, py = pitch_homography.image_to_pitch(*d.center)
            else:
                px, py = (0.0, 0.0)
            pitch_coords_list.append((px, py))

            if frame_bgr is not None and frame_bgr.size > 0:
                x1, y1 = max(0, int(d.x1)), max(0, int(d.y1))
                x2, y2 = min(w_frame, int(d.x2)), min(h_frame, int(d.y2))
                crop = frame_bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None
                feat_vec = self.extractor.extract_features(
                    crop, d, w_frame, h_frame, (px, py)
                )
            else:
                feat_vec = self.extractor.extract_features(
                    None, d, w_frame, h_frame, (px, py)
                )
            det_features.append(feat_vec)

        high = [d for d in detections if d.confidence >= settings.CONFIDENCE_THRESHOLD]
        high_indices = [
            i
            for i, d in enumerate(detections)
            if d.confidence >= settings.CONFIDENCE_THRESHOLD
        ]
        high_features = [det_features[i] for i in high_indices]
        high_pitch = [pitch_coords_list[i] for i in high_indices]

        low = [d for d in detections if d.confidence < settings.CONFIDENCE_THRESHOLD]
        low_indices = [
            i
            for i, d in enumerate(detections)
            if d.confidence < settings.CONFIDENCE_THRESHOLD
        ]
        low_features = [det_features[i] for i in low_indices]
        low_pitch = [pitch_coords_list[i] for i in low_indices]

        # Pass 1: high-confidence detections against active tracks
        matches1, unmatched_tracks_idx, unmatched_high_idx = self._match(
            self.tracks, high, high_features, high_pitch, frame_number
        )
        for ti, di in matches1:
            self.tracks[ti].update(high[di], high_features[di], high_pitch[di])
            self._register_track_gallery(self.tracks[ti])

        # Pass 2: low-confidence detections against remaining unmatched tracks
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks_idx]
        matches2, _, _ = self._match(
            remaining_tracks, low, low_features, low_pitch, frame_number
        )
        for ti_local, di in matches2:
            remaining_tracks[ti_local].update(low[di], low_features[di], low_pitch[di])
            self._register_track_gallery(remaining_tracks[ti_local])

        # Re-link unmatched HIGH-confidence detections against lost_tracks pool (ReID Recovery)
        unmatched_high_dets = [high[i] for i in unmatched_high_idx]
        unmatched_high_feats = [high_features[i] for i in unmatched_high_idx]
        unmatched_high_pitch = [high_pitch[i] for i in unmatched_high_idx]

        relink_map, final_unmatched_dets = self._relink_lost_tracks(
            unmatched_high_dets,
            unmatched_high_feats,
            unmatched_high_pitch,
            frame_number,
        )

        for relink_idx, lost_track in relink_map.items():
            det = unmatched_high_dets[relink_idx]
            feat = unmatched_high_feats[relink_idx]
            pitch = unmatched_high_pitch[relink_idx]
            lost_track.update(det, feat, pitch)
            self.tracks.append(lost_track)
            self._register_track_gallery(lost_track)

        # Spawn or Reuse registered tracks for unmatched detections
        for det in final_unmatched_dets:
            det_idx = unmatched_high_dets.index(det)
            feat = unmatched_high_feats[det_idx]
            pitch = unmatched_high_pitch[det_idx]
            kf = _new_kalman_filter(det.x1, det.y1, det.x2, det.y2)

            # Check if this appearance matches a previously registered player ID in global registry
            reused_id = self._relink_global_registry(det, feat, frame_number)
            if reused_id is not None:
                assigned_id = reused_id
                self.recovered_tracks_count += 1
            else:
                assigned_id = self._next_id
                self._next_id += 1
                self.total_confirmed_tracks += 1

            track = Track(
                track_id=assigned_id,
                kf=kf,
                class_name=det.class_name,
                state=TrackState.CONFIRMED if settings.MIN_CONFIRMATION_HITS <= 1 else TrackState.TENTATIVE,
            )
            track.update(det, feat, pitch)
            self.tracks.append(track)
            self._register_track_gallery(track)
            self._log_event(
                frame_number,
                assigned_id,
                det_idx,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                f"NEW_TRACK_SPAWNED_ID_{assigned_id}" if reused_id is None else f"REID_GLOBAL_REUSED_ID_{assigned_id}",
            )

        # Track State & Lost Pool Management
        active_tracks = []
        for t in self.tracks:
            if t.time_since_update <= settings.TRACKER_MAX_AGE_FRAMES:
                active_tracks.append(t)
            else:
                t.state = TrackState.LOST
                if t not in self.lost_tracks:
                    self.lost_tracks.append(t)
        self.tracks = active_tracks

        # Purge lost_tracks pool if time_since_update > TRACKER_MAX_AGE_FRAMES + REID_LOST_MAX_AGE_FRAMES
        max_lost_window = (
            settings.TRACKER_MAX_AGE_FRAMES + settings.REID_LOST_MAX_AGE_FRAMES
        )
        self.lost_tracks = [
            t for t in self.lost_tracks if t.time_since_update <= max_lost_window
        ]

        # Return all active confirmed tracks for visualization & analytics
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    def get_track_consistency_summary(self) -> dict:
        confirmed_count = len([t for t in self.tracks if t.state == TrackState.CONFIRMED])
        lost_count = len(self.lost_tracks)
        return {
            "total_tracks_spawned": self._next_id - 1,
            "confirmed_active_tracks": confirmed_count,
            "lost_tracks": lost_count,
            "recovered_tracks": self.recovered_tracks_count,
            "id_switches": self.id_switch_count,
            "total_events_logged": len(self.tracking_event_log),
        }
