"""
Unit tests for Detection and Categorized Validation:
- COCO and fine-tuned class resolution
- Player, Ball, and Referee validation
- Drone and non-game object rejection
"""

from app.cv_pipeline.detection import Detection, PlayerDetector
from app.cv_pipeline.player_filter import PlayerValidator


def test_coco_and_finetuned_class_resolution():
    detector = PlayerDetector.__new__(PlayerDetector)
    detector.names = {0: "person", 32: "sports ball"}

    assert detector._resolve_class_name(0) == "player"
    assert detector._resolve_class_name(32) == "ball"
    assert detector._resolve_class_name(1) is None  # bicycle skipped

    # Fine-tuned names
    detector.names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    assert detector._resolve_class_name(0) == "ball"
    assert detector._resolve_class_name(1) == "player"
    assert detector._resolve_class_name(2) == "player"
    assert detector._resolve_class_name(3) == "referee"


def test_validator_categorization_and_drone_rejection():
    validator = PlayerValidator()

    player_det = Detection(1, "player", 0.85, 100, 100, 140, 180)
    ball_det = Detection(1, "ball", 0.75, 200, 200, 215, 215)
    ref_det = Detection(1, "referee", 0.80, 300, 100, 340, 180)
    drone_det = Detection(1, "drone", 0.90, 400, 20, 450, 40)
    banner_det = Detection(1, "banner", 0.70, 0, 0, 500, 50)

    dets = [player_det, ball_det, ref_det, drone_det, banner_det]
    categorized = validator.filter_categorized(dets, 1920, 1080)

    assert len(categorized["players"]) == 1
    assert categorized["players"][0].class_name == "player"

    assert len(categorized["balls"]) == 1
    assert categorized["balls"][0].class_name == "ball"

    assert len(categorized["referees"]) == 1
    assert categorized["referees"][0].class_name == "referee"

    assert validator.rejected_false_detections >= 2  # drone and banner rejected
