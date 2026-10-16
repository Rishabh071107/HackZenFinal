"""
Application configuration, loaded from environment variables.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Football Analytics Platform"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # Database
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "football_analytics"
    USE_SQLITE_FALLBACK: bool = True

    @property
    def DATABASE_URL(self) -> str:
        if self.POSTGRES_HOST.lower() in ("sqlite", "none", ""):
            return f"sqlite:///{self.SQLITE_DB_PATH}"
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Storage
    STORAGE_ROOT: Path = Path(__file__).resolve().parents[2] / "storage"
    SQLITE_DB_PATH: Path = STORAGE_ROOT / "football_analytics.db"
    VIDEO_DIR: Path = STORAGE_ROOT / "videos"
    HEATMAP_DIR: Path = STORAGE_ROOT / "heatmaps"
    TRAJECTORY_DIR: Path = STORAGE_ROOT / "trajectories"
    THUMBNAIL_DIR: Path = STORAGE_ROOT / "thumbnails"
    WEIGHTS_DIR: Path = STORAGE_ROOT / "weights"

    # CV / DL
    YOLO_MODEL_PATH: str = "yolo11n.pt"  # generic pretrained weights; swap for a
    # football-finetuned checkpoint for production accuracy
    # VisionX OD-17 Tracking & Filtering Configuration
    CONFIDENCE_THRESHOLD: float = 0.30
    IOU_THRESHOLD: float = 0.20
    MIN_CONFIRMATION_HITS: int = (
        1  # immediate track confirmation to eliminate frame-1 assignment lag
    )
    MAX_LOST_FRAMES: int = (
        90  # extended lost buffer window for camera pans & reappearance
    )

    # Player Validation & Non-Player/Drone Rejection Filters
    PLAYER_MIN_AREA_RATIO: float = 0.0001  # min bounding box area relative to frame
    PLAYER_MAX_AREA_RATIO: float = 0.15  # max bounding box area relative to frame
    ASPECT_RATIO_MIN: float = 1.1  # min height-to-width ratio for valid player
    ASPECT_RATIO_MAX: float = 4.5  # max height-to-width ratio for valid player

    # Multi-Cue Hungarian Association Weights
    WEIGHT_IOU: float = 0.25
    WEIGHT_APPEARANCE: float = 0.50
    WEIGHT_MOTION: float = 0.15
    WEIGHT_GEOMETRY: float = 0.10

    # Domain 3 Feature Component Weights
    WEIGHT_COLOR: float = 0.35
    WEIGHT_TEXTURE: float = 0.25
    WEIGHT_SHAPE: float = 0.15
    WEIGHT_EDGE: float = 0.15
    WEIGHT_LANDMARK: float = 0.10

    # Re-ID & Feature Extraction Options
    REID_LOST_MAX_AGE_FRAMES: int = 90
    REID_SIMILARITY_THRESHOLD: float = 0.55
    REID_MAX_SPEED_PX_PER_FRAME: float = 80.0
    CROSSING_IOU_THRESHOLD: float = 0.20
    APPEARANCE_WEIGHT_ON_CROSSING: float = 0.65
    ENABLE_LANDMARKS: bool = True
    ENABLE_OPTICAL_FLOW: bool = False

    # Legacy & Pipeline compatibility fields
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.30
    TRACKER_MAX_AGE_FRAMES: int = 30
    TRACKER_IOU_THRESHOLD: float = 0.2
    FRAME_SAMPLE_STRIDE: int = 1
    INFERENCE_IMGSZ: int = 416
    INFERENCE_BATCH_SIZE: int = 4

    # Pitch dimensions in meters (standard FIFA pitch)
    PITCH_LENGTH_M: float = 105.0
    PITCH_WIDTH_M: float = 68.0

    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for d in (
        settings.VIDEO_DIR,
        settings.HEATMAP_DIR,
        settings.TRAJECTORY_DIR,
        settings.THUMBNAIL_DIR,
        settings.WEIGHTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
    return settings
