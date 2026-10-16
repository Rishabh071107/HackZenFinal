"""
Pydantic schemas for API request/response bodies.
Kept separate from ORM models so the API contract can evolve independently
of the database schema.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from app.db.models import AnalyticsScope, ProcessingStage


class MatchCreate(BaseModel):
    name: str
    team_a_name: Optional[str] = None
    team_b_name: Optional[str] = None
    tournament: Optional[str] = None
    match_date: Optional[datetime] = None


class MatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    team_a_name: Optional[str]
    team_b_name: Optional[str]
    tournament: Optional[str]
    match_date: Optional[datetime]
    status: ProcessingStage
    duration_seconds: Optional[float]
    total_frames: Optional[int]
    fps: Optional[float]
    avg_detection_confidence: Optional[float]
    avg_tracking_accuracy: Optional[float]
    confirmed_tracks: Optional[int] = 0
    id_switches: Optional[int] = 0
    recovered_tracks: Optional[int] = 0
    rejected_false_detections: Optional[int] = 0
    created_at: datetime


class TrackingDebugEventOut(BaseModel):
    frame_number: int
    previous_track_id: int
    matched_detection_index: int
    iou_score: float
    appearance_score: float
    motion_score: float
    geometry_score: float
    final_cost: float
    decision: str


class ProcessingJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    match_id: str
    current_stage: ProcessingStage
    progress_percent: float
    error_message: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]


class TeamOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    match_id: str
    label: str
    display_name: Optional[str]
    jersey_color_hex: str


class PlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    team_id: str
    tracking_id: int
    jersey_number: Optional[int]
    estimated_role: Optional[str]
    thumbnail_path: Optional[str]


class AnalyticsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    scope: AnalyticsScope
    metrics: dict[str, Any]
    computed_at: datetime


class HeatmapOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    scope: AnalyticsScope
    image_path: str
    grid_data_path: Optional[str]


class TrajectoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    player_id: str
    points: list
    start_point: Optional[list]
    end_point: Optional[list]


class HomographyCalibration(BaseModel):
    """4 correspondences: pixel coords in the broadcast frame <-> pitch coords in meters."""

    image_points: list[tuple[float, float]]
    pitch_points: list[tuple[float, float]]
