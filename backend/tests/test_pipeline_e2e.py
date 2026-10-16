"""
End-to-end integration test for the full computer vision analysis pipeline:
- Generates synthetic video with moving players & ball
- Mocks detector output for deterministic frame-by-frame inputs
- Executes run_analysis_pipeline
- Validates Match lifecycle, Team classification, Tracking consistency log,
  Domain 3 Player features, Ball metrics, Heatmaps, and Trajectories.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pytest
from unittest.mock import patch

from app.core.config import get_settings
from app.cv_pipeline.detection import Detection
from app.db import models
from app.db.database import Base, engine, SessionLocal
from app.services.analysis_orchestrator import run_analysis_pipeline

settings = get_settings()


@pytest.fixture(scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def synthetic_video_path(tmp_path):
    vid_file = tmp_path / "synthetic_match.mp4"
    width, height, fps = 640, 360, 25
    total_frames = 15

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_file), fourcc, fps, (width, height))

    for f in range(total_frames):
        # Green pitch background
        frame = np.full((height, width, 3), (35, 120, 45), dtype=np.uint8)

        # Player 1 (Red jersey)
        p1_x = int(120 + f * 4)
        p1_y = 180
        cv2.rectangle(frame, (p1_x, p1_y), (p1_x + 30, p1_y + 60), (30, 30, 220), -1)

        # Player 2 (Blue jersey)
        p2_x = int(350 + f * 2)
        p2_y = 200
        cv2.rectangle(frame, (p2_x, p2_y), (p2_x + 30, p2_y + 60), (220, 80, 20), -1)

        # Ball (White circle)
        b_x = int(160 + f * 8)
        b_y = 210
        cv2.circle(frame, (b_x, b_y), 6, (255, 255, 255), -1)

        out.write(frame)

    out.release()
    return vid_file


def test_full_pipeline_e2e(setup_db, synthetic_video_path):
    db = setup_db
    match_id = "test-match-e2e-001"

    # Clean existing
    db.query(models.TrackingData).delete()
    db.query(models.Trajectory).delete()
    db.query(models.Heatmap).delete()
    db.query(models.Analytics).delete()
    db.query(models.Player).delete()
    db.query(models.Team).delete()
    db.query(models.ProcessingJob).filter(models.ProcessingJob.match_id == match_id).delete()
    db.query(models.Video).filter(models.Video.match_id == match_id).delete()
    db.query(models.Match).filter(models.Match.id == match_id).delete()
    db.commit()

    # Create Match
    match = models.Match(
        id=match_id,
        name="E2E Pipeline Test Match",
        team_a_name="Red FC",
        team_b_name="Blue City",
        status=models.ProcessingStage.UPLOADING,
    )
    db.add(match)

    # Create Video
    video = models.Video(
        match_id=match_id,
        original_filename="synthetic_match.mp4",
        storage_path=str(synthetic_video_path),
        size_bytes=synthetic_video_path.stat().st_size,
    )
    db.add(video)

    # Create ProcessingJob
    job = models.ProcessingJob(
        match_id=match_id,
        current_stage=models.ProcessingStage.UPLOADING,
        progress_percent=0.0,
    )
    db.add(job)
    db.commit()

    # Mock detection to emit player 1, player 2, and ball
    def mock_detect_batch(self, frames, frame_numbers):
        batch_results = []
        for fn in frame_numbers:
            p1_x = 120 + fn * 4
            p1_y = 180
            p2_x = 350 + fn * 2
            p2_y = 200
            b_x = 160 + fn * 8
            b_y = 210

            dets = [
                Detection(fn, "player", 0.92, p1_x, p1_y, p1_x + 30, p1_y + 60),
                Detection(fn, "player", 0.91, p2_x, p2_y, p2_x + 30, p2_y + 60),
                Detection(fn, "ball", 0.88, b_x - 5, b_y - 5, b_x + 5, b_y + 5),
            ]
            batch_results.append(dets)
        return batch_results

    with patch("app.cv_pipeline.detection.PlayerDetector.detect_batch", side_effect=mock_detect_batch, autospec=True):
        run_analysis_pipeline(match_id, db=db)

    # Reload Match from DB
    db.expire_all()
    updated_match = db.query(models.Match).filter(models.Match.id == match_id).first()
    updated_job = db.query(models.ProcessingJob).filter(models.ProcessingJob.match_id == match_id).first()

    # 1. Pipeline Completion
    assert updated_match.status == models.ProcessingStage.COMPLETED
    assert updated_job.current_stage == models.ProcessingStage.COMPLETED
    assert updated_job.progress_percent == 100.0

    # 2. Tracking telemetry
    assert updated_match.confirmed_tracks >= 2
    assert updated_match.tracking_event_log is not None
    assert len(updated_match.tracking_event_log) > 0
    # Check that events contain previous_track_id and decision
    assert "previous_track_id" in updated_match.tracking_event_log[0]
    assert "decision" in updated_match.tracking_event_log[0]

    # 3. Teams and Players
    teams = db.query(models.Team).filter(models.Team.match_id == match_id).all()
    assert len(teams) == 2
    players = db.query(models.Player).all()
    assert len(players) >= 2

    # Verify Domain 3 features are stored on player
    for p in players:
        assert p.jersey_number is not None
        assert "color_features" in p.features
        assert "shape_features" in p.features
        assert "lab_mean" in p.features["color_features"]
        assert "aspect_ratio" in p.features["shape_features"]

    # 4. Tracking Data
    tracking_data = db.query(models.TrackingData).all()
    assert len(tracking_data) > 0

    # 5. Heatmaps & Trajectories
    heatmaps = db.query(models.Heatmap).filter(models.Heatmap.match_id == match_id).all()
    assert len(heatmaps) >= 1

    trajectories = db.query(models.Trajectory).all()
    assert len(trajectories) >= 2

    # 6. Analytics & Ball Metrics
    analytics = db.query(models.Analytics).filter(models.Analytics.match_id == match_id).first()
    assert analytics is not None
    assert "ball_metrics" in analytics.metrics
    assert analytics.metrics["ball_metrics"]["total_tracked_frames"] > 0

    # 7. Annotated Video Output
    updated_video = db.query(models.Video).filter(models.Video.match_id == match_id).first()
    assert updated_video.annotated_storage_path is not None
    assert Path(updated_video.annotated_storage_path).exists()
    assert Path(updated_video.annotated_storage_path).stat().st_size > 0
