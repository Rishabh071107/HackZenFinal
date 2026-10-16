"""
Unit tests for VisionX OD-17 Architecture:
  - PlayerValidator (Non-Player & Drone Rejection)
  - PlayerFeatureExtractor (Domain 3 Multi-Cue Descriptors)
  - Extended ByteTracker (Multi-Cue Hungarian Association, ReID, Lost Tracks)
"""

import numpy as np

from app.cv_pipeline.detection import Detection
from app.cv_pipeline.feature_extraction import PlayerFeatureExtractor
from app.cv_pipeline.player_filter import PlayerValidator
from app.cv_pipeline.tracking import ByteTracker


def create_solid_crop(
    bgr_color: tuple[int, int, int], height: int = 80, width: int = 40
) -> np.ndarray:
    """Creates a synthetic image crop filled with a specific BGR color."""
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    crop[:, :] = bgr_color
    return crop


def make_frame_with_det(
    det: Detection,
    bgr_color: tuple[int, int, int],
    canvas_size: tuple[int, int] = (2000, 2000),
) -> np.ndarray:
    """Creates a full frame with a colored crop at the detection's bounding box coordinates."""
    frame = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    h = int(det.y2 - det.y1)
    w = int(det.x2 - det.x1)
    crop = create_solid_crop(bgr_color, height=h, width=w)
    frame[int(det.y1) : int(det.y2), int(det.x1) : int(det.x2)] = crop
    return frame


def test_player_validator_drone_rejection():
    validator = PlayerValidator()

    # Valid player detection (aspect ratio ~2.0, plausible area)
    valid_player = Detection(
        frame_number=1,
        class_name="player",
        confidence=0.85,
        x1=100,
        y1=100,
        x2=140,
        y2=180,
    )
    is_valid, reason = validator.is_valid_player(valid_player, 1920, 1080)
    assert is_valid is True

    # Drone detection (class = 'drone') -> MUST be rejected
    drone_det = Detection(
        frame_number=1,
        class_name="drone",
        confidence=0.90,
        x1=200,
        y1=200,
        x2=280,
        y2=230,
    )
    is_valid_d, reason_d = validator.is_valid_player(drone_det, 1920, 1080)
    assert is_valid_d is False
    assert "non-player" in reason_d

    # Non-human aspect ratio (wide object: width 100, height 20 -> AR = 0.2 < 1.1) -> MUST be rejected
    wide_det = Detection(
        frame_number=1,
        class_name="player",
        confidence=0.85,
        x1=100,
        y1=100,
        x2=200,
        y2=120,
    )
    is_valid_w, reason_w = validator.is_valid_player(wide_det, 1920, 1080)
    assert is_valid_w is False
    assert "Aspect ratio" in reason_w


def test_domain3_feature_extractor():
    extractor = PlayerFeatureExtractor()
    red_crop = create_solid_crop((0, 0, 220), height=80, width=40)
    blue_crop = create_solid_crop((220, 0, 0), height=80, width=40)

    det = Detection(
        frame_number=1,
        class_name="player",
        confidence=0.9,
        x1=100,
        y1=100,
        x2=140,
        y2=180,
    )

    feat_red = extractor.extract_features(red_crop, det, 1920, 1080, (35.0, 20.0))
    feat_blue = extractor.extract_features(blue_crop, det, 1920, 1080, (35.0, 20.0))

    assert feat_red.color_lab.shape == (3,)
    assert feat_red.lbp_hist.shape == (10,)
    assert len(feat_red.hu_moments) == 7
    assert feat_red.landmarks.shape == (5, 2)

    # Combined appearance similarity should be high for identical crops
    sim_self = extractor.combined_appearance_similarity(feat_red, feat_red)
    assert sim_self >= 0.90

    # Similarity between red and blue crops should be lower
    sim_diff = extractor.combined_appearance_similarity(feat_red, feat_blue)
    assert sim_diff < sim_self


def test_lost_tracks_pool_and_relink():
    tracker = ByteTracker()
    red_color = (0, 0, 220)

    # Frame 1: Initial detection (requires 3 hits for full confirmation in state machine)
    det1 = Detection(
        frame_number=1,
        class_name="player",
        confidence=0.9,
        x1=100,
        y1=100,
        x2=140,
        y2=180,
    )
    frame1 = make_frame_with_det(det1, red_color)
    for f in range(1, 4):
        active = tracker.update([det1], frame_bgr=frame1, frame_number=f)

    assert len(active) == 1
    assigned_id = active[0].track_id

    # Simulate missing frames (player occluded/out of frame)
    for f in range(4, 38):
        tracker.update([], frame_bgr=frame1, frame_number=f)

    assert len(tracker.lost_tracks) >= 1

    # Frame 38: Player reappears near last position
    det_reappear = Detection(
        frame_number=38,
        class_name="player",
        confidence=0.85,
        x1=110,
        y1=105,
        x2=150,
        y2=185,
    )
    frame38 = make_frame_with_det(det_reappear, red_color)
    active_reappear = tracker.update([det_reappear], frame_bgr=frame38, frame_number=38)

    assert len(active_reappear) == 1
    assert active_reappear[0].track_id == assigned_id
    assert tracker.recovered_tracks_count >= 1


def test_reject_implausible_teleport():
    tracker = ByteTracker()
    blue_color = (220, 0, 0)

    det1 = Detection(
        frame_number=1,
        class_name="player",
        confidence=0.9,
        x1=50,
        y1=100,
        x2=90,
        y2=180,
    )
    frame1 = make_frame_with_det(det1, blue_color)
    for f in range(1, 4):
        active = tracker.update([det1], frame_bgr=frame1, frame_number=f)
    orig_id = active[0].track_id

    for f in range(4, 38):
        tracker.update([], frame_bgr=frame1, frame_number=f)

    # Teleport to x=3500 (speed > 80 px/frame)
    det_teleport = Detection(
        frame_number=38,
        class_name="player",
        confidence=0.9,
        x1=3500,
        y1=100,
        x2=3540,
        y2=180,
    )
    frame38 = make_frame_with_det(det_teleport, blue_color, canvas_size=(4000, 4000))
    for f in range(38, 41):
        active2 = tracker.update([det_teleport], frame_bgr=frame38, frame_number=f)

    # Re-link MUST be rejected due to spatial-temporal speed plausibility filter, spawning a new track
    assert len(active2) == 1
    assert active2[0].track_id != orig_id


def test_crossing_paths_guard():
    tracker = ByteTracker()
    red_color = (0, 0, 220)
    blue_color = (220, 0, 0)

    # Players starting at x=100 and x=200, moving toward each other
    # Frame 1
    det_a1 = Detection(
        frame_number=1,
        class_name="player",
        confidence=0.9,
        x1=100,
        y1=100,
        x2=140,
        y2=180,
    )
    det_b1 = Detection(
        frame_number=1,
        class_name="player",
        confidence=0.9,
        x1=200,
        y1=100,
        x2=240,
        y2=180,
    )
    frame1 = np.zeros((500, 500, 3), dtype=np.uint8)
    frame1[100:180, 100:140] = create_solid_crop(red_color, height=80, width=40)
    frame1[100:180, 200:240] = create_solid_crop(blue_color, height=80, width=40)
    tracker.update([det_a1, det_b1], frame_bgr=frame1, frame_number=1)

    # Frame 2
    det_a2 = Detection(
        frame_number=2,
        class_name="player",
        confidence=0.9,
        x1=115,
        y1=100,
        x2=155,
        y2=180,
    )
    det_b2 = Detection(
        frame_number=2,
        class_name="player",
        confidence=0.9,
        x1=185,
        y1=100,
        x2=225,
        y2=180,
    )
    frame2 = np.zeros((500, 500, 3), dtype=np.uint8)
    frame2[100:180, 115:155] = create_solid_crop(red_color, height=80, width=40)
    frame2[100:180, 185:225] = create_solid_crop(blue_color, height=80, width=40)
    tracker.update([det_a2, det_b2], frame_bgr=frame2, frame_number=2)

    # Frame 3: Confirmed tracks
    det_a3 = Detection(
        frame_number=3,
        class_name="player",
        confidence=0.9,
        x1=130,
        y1=100,
        x2=170,
        y2=180,
    )
    det_b3 = Detection(
        frame_number=3,
        class_name="player",
        confidence=0.9,
        x1=170,
        y1=100,
        x2=210,
        y2=180,
    )
    frame3 = np.zeros((500, 500, 3), dtype=np.uint8)
    frame3[100:180, 130:170] = create_solid_crop(red_color, height=80, width=40)
    frame3[100:180, 170:210] = create_solid_crop(blue_color, height=80, width=40)
    active3 = tracker.update([det_a3, det_b3], frame_bgr=frame3, frame_number=3)
    assert len(active3) == 2

    # Frame 4: Players cross and overlap (x1=145..185 and x1=150..190 -> high IoU overlap)
    det_a4 = Detection(
        frame_number=4,
        class_name="player",
        confidence=0.9,
        x1=145,
        y1=100,
        x2=185,
        y2=180,
    )
    det_b4 = Detection(
        frame_number=4,
        class_name="player",
        confidence=0.9,
        x1=150,
        y1=100,
        x2=190,
        y2=180,
    )
    frame4 = np.zeros((500, 500, 3), dtype=np.uint8)
    frame4[100:180, 145:185] = create_solid_crop(red_color, height=80, width=40)
    frame4[100:180, 150:190] = create_solid_crop(blue_color, height=80, width=40)

    active4 = tracker.update([det_a4, det_b4], frame_bgr=frame4, frame_number=4)
    assert len(active4) == 2
    assert any(t.in_crossing for t in active4)


def test_persistent_id_during_detection_misses():
    """Verify player ID remains 100% constant and bounding box is returned even when YOLO misses 3 frames."""
    tracker = ByteTracker()
    green_color = (0, 220, 0)

    # Frame 1: Player detected
    det1 = Detection(
        frame_number=1,
        class_name="player",
        confidence=0.88,
        x1=300,
        y1=300,
        x2=340,
        y2=380,
    )
    frame1 = make_frame_with_det(det1, green_color)
    active1 = tracker.update([det1], frame_bgr=frame1, frame_number=1)
    assert len(active1) == 1
    p_id = active1[0].track_id

    # Frames 2..4: YOLO misses detection
    for f in range(2, 5):
        active_miss = tracker.update([], frame_bgr=frame1, frame_number=f)
        # Tracker MUST continue returning predicted bounding box and identical player ID without dropping or flickering
        assert len(active_miss) == 1
        assert active_miss[0].track_id == p_id
        assert active_miss[0].time_since_update == f - 1

    # Frame 5: Player detected again
    det5 = Detection(
        frame_number=5,
        class_name="player",
        confidence=0.89,
        x1=304,
        y1=300,
        x2=344,
        y2=380,
    )
    active5 = tracker.update([det5], frame_bgr=frame1, frame_number=5)
    assert len(active5) == 1
    assert active5[0].track_id == p_id
    assert active5[0].time_since_update == 0
