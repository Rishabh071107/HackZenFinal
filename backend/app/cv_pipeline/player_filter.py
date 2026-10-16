"""
Player validation and non-player / drone rejection module for VisionX OD-17.

Filters raw YOLO detections to ensure only legitimate football player bounding boxes
enter the tracking and feature extraction pipeline:

  1. Class Filter: Allows 'player' or 'person', explicitly rejecting drones, cameras, balls, birds.
  2. Aspect Ratio Filter: Enforces human player aspect ratio (height / width between 1.1 and 4.5).
  3. Frame Area Scale Filter: Rejects implausibly tiny noise (< 0.01% frame) or huge visual artifacts (> 15% frame).
  4. Homography ROI Filter: Validates that player foot points fall within or near pitch boundaries.
"""

from app.core.config import get_settings
from app.cv_pipeline.detection import Detection

settings = get_settings()

REJECTED_CLASSES = {
    "drone",
    "camera",
    "skycam",
    "bird",
    "car",
    "banner",
    "ad_board",
}


class PlayerValidator:
    """Robust detection validator supporting players, referees, and the ball."""

    def __init__(self):
        self.total_processed_detections: int = 0
        self.rejected_false_detections: int = 0

    def is_valid_player(
        self,
        det: Detection,
        frame_width: int,
        frame_height: int,
        pitch_homography=None,
    ) -> tuple[bool, str]:
        """
        Validates a single detection against multi-cue physical & visual heuristics for players.
        Returns (is_valid: bool, rejection_reason: str).
        """
        self.total_processed_detections += 1

        cls_lower = det.class_name.lower()
        if cls_lower in REJECTED_CLASSES:
            return False, f"Class '{det.class_name}' is in non-player rejection list."

        if cls_lower not in {"player", "person", "goalkeeper"}:
            return False, f"Class '{det.class_name}' is not recognized as a player."

        if det.confidence < settings.CONFIDENCE_THRESHOLD:
            return (
                False,
                f"Confidence {det.confidence:.2f} < threshold {settings.CONFIDENCE_THRESHOLD}.",
            )

        w = max(1.0, det.x2 - det.x1)
        h = max(1.0, det.y2 - det.y1)
        aspect_ratio = h / w

        # Aspect Ratio Filter (Standing/running players have height > width, e.g. 1.1 to 4.5)
        if not (settings.ASPECT_RATIO_MIN <= aspect_ratio <= settings.ASPECT_RATIO_MAX):
            return (
                False,
                f"Aspect ratio {aspect_ratio:.2f} outside player bounds [{settings.ASPECT_RATIO_MIN}, {settings.ASPECT_RATIO_MAX}].",
            )

        # Scale & Area Filter relative to frame dimensions
        frame_area = float(max(1, frame_width * frame_height))
        box_area = float(w * h)
        area_ratio = box_area / frame_area

        if not (
            settings.PLAYER_MIN_AREA_RATIO
            <= area_ratio
            <= settings.PLAYER_MAX_AREA_RATIO
        ):
            return (
                False,
                f"Area ratio {area_ratio:.4f} outside plausible bounds [{settings.PLAYER_MIN_AREA_RATIO}, {settings.PLAYER_MAX_AREA_RATIO}].",
            )

        # Pitch ROI Filter (if homography is provided)
        if pitch_homography is not None:
            foot_x, foot_y = det.center
            pitch_x, pitch_y = pitch_homography.image_to_pitch(foot_x, foot_y)
            margin = 15.0  # meters margin outside touchline/goal line
            if not (-margin <= pitch_x <= settings.PITCH_LENGTH_M + margin) or not (
                -margin <= pitch_y <= settings.PITCH_WIDTH_M + margin
            ):
                return (
                    False,
                    f"Pitch coords ({pitch_x:.1f}, {pitch_y:.1f}) outside pitch bounds.",
                )

        return True, "Valid"

    def is_valid_ball(
        self,
        det: Detection,
        frame_width: int,
        frame_height: int,
    ) -> tuple[bool, str]:
        """Validates a ball detection by circularity/aspect ratio and small size."""
        cls_lower = det.class_name.lower()
        if cls_lower not in {"ball", "sports ball"}:
            return False, f"Class '{det.class_name}' is not a ball."

        if det.confidence < 0.15:
            return False, f"Ball confidence {det.confidence:.2f} < 0.15"

        w = max(1.0, det.x2 - det.x1)
        h = max(1.0, det.y2 - det.y1)
        aspect_ratio = h / w

        # A football in broadcast video is approximately square in aspect ratio (0.5 to 1.8)
        if not (0.5 <= aspect_ratio <= 1.8):
            return False, f"Ball aspect ratio {aspect_ratio:.2f} not circular."

        frame_area = float(max(1, frame_width * frame_height))
        area_ratio = (w * h) / frame_area
        # Ball should occupy at most 0.5% of the frame
        if area_ratio > 0.005:
            return False, f"Ball area ratio {area_ratio:.4f} too large."

        return True, "Valid Ball"

    def is_valid_referee(
        self,
        det: Detection,
        frame_width: int,
        frame_height: int,
    ) -> tuple[bool, str]:
        """Validates referee detection."""
        cls_lower = det.class_name.lower()
        if cls_lower not in {"referee", "ref"}:
            return False, "Not referee class."

        w = max(1.0, det.x2 - det.x1)
        h = max(1.0, det.y2 - det.y1)
        aspect_ratio = h / w
        if not (settings.ASPECT_RATIO_MIN <= aspect_ratio <= settings.ASPECT_RATIO_MAX):
            return False, "Aspect ratio outside bounds."
        return True, "Valid Referee"

    def filter_detections(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
        pitch_homography=None,
    ) -> list[Detection]:
        """Filters a list of detections for one frame. Returns valid player detections."""
        valid_list: list[Detection] = []
        for d in detections:
            valid, reason = self.is_valid_player(
                d, frame_width, frame_height, pitch_homography
            )
            if valid:
                valid_list.append(d)
            elif d.class_name.lower() in REJECTED_CLASSES:
                self.rejected_false_detections += 1
            elif d.class_name.lower() not in {"ball", "referee"}:
                # Count true false detections
                self.rejected_false_detections += 1
        return valid_list

    def filter_categorized(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
        pitch_homography=None,
    ) -> dict[str, list[Detection]]:
        """Categorizes all detections in a frame into players, balls, and referees."""
        categorized = {"players": [], "balls": [], "referees": []}
        for d in detections:
            cls_lower = d.class_name.lower()
            if cls_lower in REJECTED_CLASSES:
                self.rejected_false_detections += 1
                continue

            if cls_lower in {"ball", "sports ball"}:
                valid, _ = self.is_valid_ball(d, frame_width, frame_height)
                if valid:
                    categorized["balls"].append(d)
                else:
                    self.rejected_false_detections += 1
            elif cls_lower in {"referee", "ref"}:
                valid, _ = self.is_valid_referee(d, frame_width, frame_height)
                if valid:
                    categorized["referees"].append(d)
                else:
                    self.rejected_false_detections += 1
            else:
                valid, _ = self.is_valid_player(
                    d, frame_width, frame_height, pitch_homography
                )
                if valid:
                    categorized["players"].append(d)
                else:
                    self.rejected_false_detections += 1

        return categorized
