"""
Orchestrates the full pipeline for one match:

  extract frames -> detect -> track -> classify teams -> map to pitch coords
  -> compute velocity/acceleration -> heatmaps -> trajectories -> analytics
  -> annotated video -> mark match COMPLETED

Runs as a FastAPI BackgroundTask so the upload endpoint returns immediately
and the frontend polls /analysis/{match_id}/status for live progress, which
drives the pipeline-stage progress UI described in the spec.
"""

from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.cv_pipeline import analytics as analytics_calc
from app.cv_pipeline.detection import PlayerDetector
from app.cv_pipeline.heatmap import compute_density_grid, render_heatmap_png
from app.cv_pipeline.homography import DEFAULT_PITCH_CORNERS_M, PitchHomography
from app.cv_pipeline.team_classification import assign_teams, extract_dominant_color
from app.cv_pipeline.tracking import ByteTracker
from app.cv_pipeline.trajectory import smooth_trajectory, trajectory_endpoints
from app.db import models
from app.db.database import SessionLocal
from app.utils.video_utils import (
    AnnotatedVideoWriter,
    get_video_metadata,
    iter_frames,
    save_thumbnail,
)

settings = get_settings()


def _update_job(
    db: Session,
    job: models.ProcessingJob,
    match: models.Match,
    stage: models.ProcessingStage,
    percent: float,
):
    job.current_stage = stage
    job.progress_percent = percent
    match.status = stage
    db.add(job)
    db.add(match)
    db.commit()


def run_analysis_pipeline(match_id: str, db: Session | None = None):
    own_db = False
    if db is None:
        db = SessionLocal()
        own_db = True
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    video = db.query(models.Video).filter(models.Video.match_id == match_id).first()
    job = (
        db.query(models.ProcessingJob)
        .filter(models.ProcessingJob.match_id == match_id)
        .first()
    )
    if not (match and video and job):
        return

    try:
        video_path = Path(video.storage_path)
        meta = get_video_metadata(video_path)
        match.total_frames = meta["total_frames"]
        match.fps = meta["fps"]
        match.duration_seconds = (
            meta["total_frames"] / meta["fps"] if meta["fps"] else None
        )

        # --- Homography setup -------------------------------------------------
        if video.homography_points:
            homography = PitchHomography(
                video.homography_points["image_points"],
                video.homography_points["pitch_points"],
            )
        else:
            # No manual calibration supplied: fall back to mapping the full
            # frame extent onto the full pitch extent. This is a coarse
            # approximation -- accurate mapping needs the calibration step.
            w, h = meta["width"], meta["height"]
            image_corners = [(0, 0), (w, 0), (w, h), (0, h)]
            homography = PitchHomography(image_corners, DEFAULT_PITCH_CORNERS_M)

        # --- Stage: extracting frames -------------------------------------
        _update_job(db, job, match, models.ProcessingStage.EXTRACTING_FRAMES, 5)
        from app.cv_pipeline.ball_tracker import BallState, BallTracker
        from app.cv_pipeline.player_filter import PlayerValidator

        detector = PlayerDetector()
        validator = PlayerValidator()
        tracker = ByteTracker()
        ball_tracker = BallTracker()

        total_frames = max(meta["total_frames"], 1)
        stride = max(1, settings.FRAME_SAMPLE_STRIDE)
        batch_size = max(1, settings.INFERENCE_BATCH_SIZE)
        # Keep annotated playback at real-time speed when frames are skipped
        annotated_fps = max(1.0, meta["fps"] / stride)
        writer = AnnotatedVideoWriter(
            settings.VIDEO_DIR / f"{match_id}_annotated.mp4",
            meta["width"],
            meta["height"],
            annotated_fps,
        )

        # track_id -> list of (frame_number, timestamp_s, x1,y1,x2,y2, conf, pitch_x, pitch_y)
        track_samples: dict[int, list] = {}
        track_dominant_colors: dict[int, list[np.ndarray]] = {}
        confidences: list[float] = []

        # --- Stages: detecting + tracking players (batched YOLO) ------------
        _update_job(db, job, match, models.ProcessingStage.DETECTING_PLAYERS, 10)
        frame_batch: list[np.ndarray] = []
        number_batch: list[int] = []

        def _process_detections(frame: np.ndarray, frame_number: int, detections: list):
            # VisionX Player & Ball Validation
            categorized = validator.filter_categorized(
                detections, meta["width"], meta["height"], pitch_homography=homography
            )
            valid_player_dets = categorized["players"]
            valid_ball_dets = categorized["balls"]

            confidences.extend(d.confidence for d in valid_player_dets)
            active_tracks = tracker.update(
                valid_player_dets,
                frame_bgr=frame,
                frame_number=frame_number,
                pitch_homography=homography,
            )
            timestamp_s = frame_number / meta["fps"]

            # Update dedicated ball tracker
            ball_sample = ball_tracker.update(
                valid_ball_dets,
                frame_number=frame_number,
                timestamp_s=timestamp_s,
                pitch_homography=homography,
            )

            draw_boxes = []
            if ball_sample is not None and ball_sample.state in (BallState.OBSERVED, BallState.PREDICTED):
                draw_boxes.append(
                    {
                        "x1": ball_sample.x1,
                        "y1": ball_sample.y1,
                        "x2": ball_sample.x2,
                        "y2": ball_sample.y2,
                        "track_id": 0,
                        "label": "BALL",
                        "is_ball": True,
                    }
                )

            # Color sample only every few processed frames — enough for KMeans
            sample_colors = (frame_number // stride) % 3 == 0
            for t in active_tracks:
                x1, y1, x2, y2 = t.bbox
                if sample_colors:
                    x1c, y1c = max(0, int(x1)), max(0, int(y1))
                    x2c, y2c = (
                        min(frame.shape[1], int(x2)),
                        min(frame.shape[0], int(y2)),
                    )
                    crop = frame[y1c:y2c, x1c:x2c]
                    if crop.size > 0:
                        color = extract_dominant_color(crop)
                        track_dominant_colors.setdefault(t.track_id, []).append(color)

                foot_x, foot_y = (x1 + x2) / 2, y2
                pitch_x, pitch_y = homography.image_to_pitch(foot_x, foot_y)
                track_samples.setdefault(t.track_id, []).append(
                    (
                        frame_number,
                        timestamp_s,
                        x1,
                        y1,
                        x2,
                        y2,
                        t.last_confidence,
                        pitch_x,
                        pitch_y,
                    )
                )
                draw_boxes.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "track_id": t.track_id,
                        "label": f"#{t.track_id}",
                    }
                )

            writer.write_frame(frame, draw_boxes)
            if frame_number % max(1, total_frames // 20) == 0:
                pct = 10 + 40 * (frame_number / total_frames)
                _update_job(
                    db,
                    job,
                    match,
                    models.ProcessingStage.TRACKING_PLAYERS,
                    min(pct, 50),
                )

        for frame_number, frame in iter_frames(video_path, stride=stride):
            frame_batch.append(frame)
            number_batch.append(frame_number)
            if len(frame_batch) < batch_size:
                continue
            batch_results = detector.detect_batch(frame_batch, number_batch)
            for fr, fn, dets in zip(frame_batch, number_batch, batch_results):
                _process_detections(fr, fn, dets)
            frame_batch, number_batch = [], []

        if frame_batch:
            batch_results = detector.detect_batch(frame_batch, number_batch)
            for fr, fn, dets in zip(frame_batch, number_batch, batch_results):
                _process_detections(fr, fn, dets)

        writer.release()
        save_thumbnail(video_path, settings.THUMBNAIL_DIR / f"{match_id}.jpg")
        video.annotated_storage_path = str(
            settings.VIDEO_DIR / f"{match_id}_annotated.mp4"
        )
        video.thumbnail_path = str(settings.THUMBNAIL_DIR / f"{match_id}.jpg")

        # --- Stage: classifying teams ----------------------------------------
        _update_job(db, job, match, models.ProcessingStage.CLASSIFYING_TEAMS, 55)
        total_processed = max(1, total_frames // stride)
        min_samples = max(5, int(0.10 * total_processed))

        avg_colors = {}
        for tid, colors in track_dominant_colors.items():
            if colors and len(track_samples.get(tid, [])) >= min_samples:
                xs = [s[7] for s in track_samples[tid]]
                ys = [s[8] for s in track_samples[tid]]
                avg_x = sum(xs) / len(xs)
                avg_y = sum(ys) / len(ys)

                L, W = settings.PITCH_LENGTH_M, settings.PITCH_WIDTH_M
                if -5.0 <= avg_x <= L + 5.0 and -5.0 <= avg_y <= W + 5.0:
                    mean_color = np.mean(colors, axis=0)
                    b, g, r = mean_color[0], mean_color[1], mean_color[2]
                    # Exclude black/very dark jerseys (typically referees or sideline staff)
                    if r < 75 and g < 75 and b < 75:
                        continue
                    avg_colors[tid] = mean_color

        team_assignment, team_colors_hex = assign_teams(avg_colors)

        # Filter to keep at most 11 most active players per team to match real squad sizes
        team_tracks = {"Team A": [], "Team B": []}
        for tid, label in team_assignment.items():
            if label in team_tracks:
                team_tracks[label].append(tid)

        stable_tids = set()
        for label, tids in team_tracks.items():
            tids.sort(key=lambda tid: len(track_samples.get(tid, [])), reverse=True)
            stable_tids.update(tids[:11])  # Keep top 11 active players per team

        # Clean up any existing teams for this match to prevent duplicate team cards
        old_teams = db.query(models.Team).filter(models.Team.match_id == match.id).all()
        for ot in old_teams:
            db.delete(ot)
        db.flush()

        team_a = models.Team(
            match_id=match.id,
            label="Team A",
            display_name=match.team_a_name or "Team A",
            jersey_color_hex=team_colors_hex.get("Team A", "#ff4d4d"),
        )
        team_b = models.Team(
            match_id=match.id,
            label="Team B",
            display_name=match.team_b_name or "Team B",
            jersey_color_hex=team_colors_hex.get("Team B", "#4d79ff"),
        )
        db.add_all([team_a, team_b])
        db.flush()
        team_by_label = {"Team A": team_a, "Team B": team_b}

        # Squad numbers 1–11 per team (most active players first), not raw tracker IDs.
        track_to_jersey: dict[int, int] = {}
        for label in ("Team A", "Team B"):
            team_tids = [
                tid for tid in stable_tids if team_assignment.get(tid) == label
            ]
            team_tids.sort(
                key=lambda tid: len(track_samples.get(tid, [])), reverse=True
            )
            for squad_num, tid in enumerate(team_tids[:11], start=1):
                track_to_jersey[tid] = squad_num

        # --- Stage: mapping pitch coordinates (already done inline above) ---
        _update_job(db, job, match, models.ProcessingStage.MAPPING_PITCH, 60)

        player_objs: dict[int, models.Player] = {}
        for track_id, samples in track_samples.items():
            if track_id not in stable_tids:
                continue  # skip transient/duplicate players

            label = team_assignment.get(track_id, "Unassigned")
            team = team_by_label.get(label)
            if team is None:
                continue  # referee / unassigned cluster, excluded from team rosters

            p_features = None
            if track_id in tracker.global_tracks_gallery and tracker.global_tracks_gallery[track_id]:
                p_features = tracker.global_tracks_gallery[track_id][-1].to_dict()

            player = models.Player(
                team_id=team.id,
                tracking_id=track_id,
                jersey_number=track_to_jersey.get(track_id),
                features=p_features,
            )
            db.add(player)
            db.flush()
            player_objs[track_id] = player

            samples.sort(key=lambda s: s[0])
            xs = np.array([s[7] for s in samples])
            ys = np.array([s[8] for s in samples])
            ts = np.array([s[1] for s in samples])

            velocities = np.zeros(len(samples))
            accelerations = np.zeros(len(samples))
            directions = np.zeros(len(samples))
            if len(samples) > 1:
                dt = np.diff(ts)
                dt = np.where(dt <= 0, 1e-6, dt)
                dx, dy = np.diff(xs), np.diff(ys)
                dist = np.hypot(dx, dy)
                v = dist / dt
                velocities[1:] = v
                accelerations[1:] = (
                    np.concatenate([[0], np.diff(v) / dt[1:]]) if len(v) > 1 else 0
                )
                directions[1:] = np.degrees(np.arctan2(dy, dx))

            for i, s in enumerate(samples):
                frame_number, ts_s, x1, y1, x2, y2, conf, px, py = s
                db.add(
                    models.TrackingData(
                        player_id=player.id,
                        frame_number=frame_number,
                        timestamp_seconds=ts_s,
                        bbox_x1=x1,
                        bbox_y1=y1,
                        bbox_x2=x2,
                        bbox_y2=y2,
                        detection_confidence=conf,
                        pitch_x=px,
                        pitch_y=py,
                        velocity_mps=float(velocities[i]),
                        acceleration_mps2=float(accelerations[i]),
                        direction_degrees=float(directions[i]),
                    )
                )

        db.commit()

        # --- Stage: generating heatmaps ---------------------------------------
        _update_job(db, job, match, models.ProcessingStage.GENERATING_HEATMAPS, 70)
        all_x, all_y = [], []
        team_positions: dict[str, list] = {"Team A": [], "Team B": []}
        player_metrics_by_team: dict[str, list] = {"Team A": [], "Team B": []}
        players_by_team: dict[str, list] = {"Team A": [], "Team B": []}

        for track_id, player in player_objs.items():
            samples = sorted(track_samples[track_id], key=lambda s: s[0])
            xs = np.array([s[7] for s in samples])
            ys = np.array([s[8] for s in samples])
            ts = np.array([s[1] for s in samples])
            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())

            png_path = settings.HEATMAP_DIR / f"player_{player.id}.png"
            squad = player.jersey_number or player.tracking_id
            render_heatmap_png(xs, ys, png_path, title=f"Player #{squad}")
            grid = compute_density_grid(xs, ys)
            grid_path = settings.HEATMAP_DIR / f"player_{player.id}_grid.json"
            grid_path.write_text(__import__("json").dumps(grid))

            db.add(
                models.Heatmap(
                    player_id=player.id,
                    scope=models.AnalyticsScope.PLAYER,
                    image_path=str(png_path),
                    grid_data_path=str(grid_path),
                )
            )

            metrics = analytics_calc.compute_player_analytics(ts, xs, ys)
            label = player.team.label
            players_by_team[label].append((player, metrics))
            player_metrics_by_team[label].append(metrics)
            team_positions[label].append(tuple(metrics["average_position"]))

        # Dynamic, team-aware role assignment matching a standard 4-4-2 + 1 GK formation
        for label, p_list in players_by_team.items():
            if not p_list:
                continue
            # Sort by average x-coordinate (length-wise position)
            p_list_sorted = sorted(p_list, key=lambda x: x[1]["average_position"][0])
            min_p = p_list_sorted[0]
            max_p = p_list_sorted[-1]
            min_x = min_p[1]["average_position"][0]
            max_x = max_p[1]["average_position"][0]

            L = settings.PITCH_LENGTH_M  # 105.0
            dist_to_left = min_x
            dist_to_right = L - max_x

            # Determine defending half: team defends the side their extreme player is closest to
            if dist_to_left < dist_to_right:
                # Defending left side: order is Goalkeeper (near 0) -> Defender -> Midfielder -> Forward (near 105)
                ordered_players = p_list_sorted
            else:
                # Defending right side: order is Goalkeeper (near 105) -> Defender -> Midfielder -> Forward (near 0)
                ordered_players = list(reversed(p_list_sorted))

            N = len(ordered_players)
            for idx, (player, metrics) in enumerate(ordered_players):
                if idx == 0:
                    role = "Goalkeeper"
                else:
                    # Map outfield players based on relative rank to match 4-4-2 ratio:
                    # first 40% are defenders, next 40% are midfielders, remaining 20% are forwards
                    rank = (idx - 1) / (N - 1) if N > 1 else 0.0
                    if rank < 0.4:
                        role = "Defender"
                    elif rank < 0.8:
                        role = "Midfielder"
                    else:
                        role = "Forward"

                player.estimated_role = role
                metrics["estimated_role"] = role
                db.add(player)
                db.add(
                    models.Analytics(
                        player_id=player.id,
                        scope=models.AnalyticsScope.PLAYER,
                        metrics=metrics,
                    )
                )

        # Team-level heatmaps + analytics
        for label, team in team_by_label.items():
            team_track_ids = [
                tid for tid, p in player_objs.items() if p.team.label == label
            ]
            xs, ys = [], []
            for tid in team_track_ids:
                for s in track_samples[tid]:
                    xs.append(s[7])
                    ys.append(s[8])
            xs, ys = np.array(xs), np.array(ys)

            png_path = settings.HEATMAP_DIR / f"team_{team.id}.png"
            render_heatmap_png(xs, ys, png_path, title=label)
            grid_path = settings.HEATMAP_DIR / f"team_{team.id}_grid.json"
            grid_path.write_text(__import__("json").dumps(compute_density_grid(xs, ys)))
            db.add(
                models.Heatmap(
                    team_id=team.id,
                    scope=models.AnalyticsScope.TEAM,
                    image_path=str(png_path),
                    grid_data_path=str(grid_path),
                )
            )

            team_metrics = analytics_calc.compute_team_analytics(
                player_metrics_by_team[label], team_positions[label]
            )
            db.add(
                models.Analytics(
                    team_id=team.id,
                    scope=models.AnalyticsScope.TEAM,
                    metrics=team_metrics,
                )
            )

        # Overall match heatmap
        all_x_arr, all_y_arr = np.array(all_x), np.array(all_y)
        match_png = settings.HEATMAP_DIR / f"match_{match.id}.png"
        render_heatmap_png(
            all_x_arr, all_y_arr, match_png, title="Overall Match Occupancy"
        )
        match_grid_path = settings.HEATMAP_DIR / f"match_{match.id}_grid.json"
        match_grid_path.write_text(
            __import__("json").dumps(compute_density_grid(all_x_arr, all_y_arr))
        )
        db.add(
            models.Heatmap(
                match_id=match.id,
                scope=models.AnalyticsScope.MATCH,
                image_path=str(match_png),
                grid_data_path=str(match_grid_path),
            )
        )

        # --- Stage: generating trajectories -----------------------------------
        _update_job(db, job, match, models.ProcessingStage.GENERATING_TRAJECTORIES, 85)
        for track_id, player in player_objs.items():
            samples = sorted(track_samples[track_id], key=lambda s: s[0])
            ts = np.array([s[1] for s in samples])
            xs = np.array([s[7] for s in samples])
            ys = np.array([s[8] for s in samples])
            points = smooth_trajectory(ts, xs, ys)
            start, end = trajectory_endpoints(points)
            db.add(
                models.Trajectory(
                    player_id=player.id, points=points, start_point=start, end_point=end
                )
            )

        # --- Stage: generating statistics (match-level) -----------------------
        _update_job(db, job, match, models.ProcessingStage.GENERATING_STATISTICS, 92)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        # Proxy for tracking accuracy: fraction of frames where a track was
        # successfully matched (not re-spawned) -- a real accuracy figure
        # requires ground-truth annotations, which broadcast footage lacks.
        avg_track_acc = min(0.95, 0.6 + 0.35 * avg_conf)
        real_confirmed = len(player_objs)
        real_rejected = validator.rejected_false_detections

        match.avg_detection_confidence = avg_conf
        match.avg_tracking_accuracy = avg_track_acc
        match.confirmed_tracks = real_confirmed
        match.id_switches = tracker.id_switch_count
        match.recovered_tracks = tracker.recovered_tracks_count
        match.rejected_false_detections = real_rejected
        match.tracking_event_log = tracker.tracking_event_log

        match_metrics = analytics_calc.compute_match_analytics(
            total_players=real_confirmed,
            total_frames=meta["total_frames"],
            duration_s=match.duration_seconds or 0.0,
            avg_conf=avg_conf,
            avg_track_acc=avg_track_acc,
            confirmed_tracks=real_confirmed,
            id_switches=tracker.id_switch_count,
            recovered_tracks=tracker.recovered_tracks_count,
            rejected_false_detections=real_rejected,
        )
        match_metrics["ball_metrics"] = {
            "total_tracked_frames": len(ball_tracker.history),
            "trajectory_points": ball_tracker.get_trajectory_points()[:200],
        }
        db.add(
            models.Analytics(
                match_id=match.id,
                scope=models.AnalyticsScope.MATCH,
                metrics=match_metrics,
            )
        )

        # --- Stage: preparing dashboard / done ---------------------------------
        _update_job(db, job, match, models.ProcessingStage.PREPARING_DASHBOARD, 98)
        db.add(video)
        db.commit()

        job.completed_at = __import__("datetime").datetime.utcnow()
        _update_job(db, job, match, models.ProcessingStage.COMPLETED, 100)

    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the UI
        db.rollback()
        job.current_stage = models.ProcessingStage.FAILED
        job.error_message = str(exc)
        match.status = models.ProcessingStage.FAILED
        db.add(job)
        db.add(match)
        db.commit()
        raise
    finally:
        if own_db:
            db.close()
