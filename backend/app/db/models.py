"""
Normalized PostgreSQL schema via SQLAlchemy ORM models.

Relationships
-------------
User (future ready)  1---N  Match
Match                1---1  Video
Match                1---N  Team          (Team A / Team B for that match)
Team                 1---N  Player
Match                1---N  ProcessingJob
Player                1---N  TrackingData   (per-frame position samples)
Player                1---1  Heatmap        (per player, per match)
Player                1---1  Trajectory     (per player, per match)
Team                  1---1  Heatmap        (team-level, per match)
Match                 1---1  Heatmap        (overall match occupancy)
Player/Team/Match      ---   Analytics      (polymorphic scope, see AnalyticsScope)
"""

import enum
import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.database import Base


def UUID(as_uuid: bool = False):
    """Portable GUID representation compatible across SQLite and PostgreSQL."""
    return String(36)


def gen_uuid():
    return str(uuid.uuid4())


class ProcessingStage(str, enum.Enum):
    UPLOADING = "uploading"
    EXTRACTING_FRAMES = "extracting_frames"
    DETECTING_PLAYERS = "detecting_players"
    TRACKING_PLAYERS = "tracking_players"
    CLASSIFYING_TEAMS = "classifying_teams"
    MAPPING_PITCH = "mapping_pitch"
    GENERATING_HEATMAPS = "generating_heatmaps"
    GENERATING_TRAJECTORIES = "generating_trajectories"
    GENERATING_STATISTICS = "generating_statistics"
    PREPARING_DASHBOARD = "preparing_dashboard"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalyticsScope(str, enum.Enum):
    MATCH = "match"
    TEAM = "team"
    PLAYER = "player"


class User(Base):
    """Future-ready: platform accounts (coaches, analysts, scouts)."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    matches = relationship("Match", back_populates="owner")


class Match(Base):
    __tablename__ = "matches"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    owner_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)

    name = Column(String(255), nullable=False)
    team_a_name = Column(String(255), nullable=True)
    team_b_name = Column(String(255), nullable=True)
    tournament = Column(String(255), nullable=True)
    match_date = Column(DateTime(timezone=True), nullable=True)

    duration_seconds = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)

    avg_detection_confidence = Column(Float, nullable=True)
    avg_tracking_accuracy = Column(Float, nullable=True)
    confirmed_tracks = Column(Integer, default=0, nullable=True)
    id_switches = Column(Integer, default=0, nullable=True)
    recovered_tracks = Column(Integer, default=0, nullable=True)
    rejected_false_detections = Column(Integer, default=0, nullable=True)
    tracking_event_log = Column(JSON, nullable=True)

    status = Column(
        Enum(ProcessingStage), default=ProcessingStage.UPLOADING, nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    owner = relationship("User", back_populates="matches")
    video = relationship(
        "Video", back_populates="match", uselist=False, cascade="all, delete-orphan"
    )
    teams = relationship("Team", back_populates="match", cascade="all, delete-orphan")
    processing_jobs = relationship(
        "ProcessingJob", back_populates="match", cascade="all, delete-orphan"
    )
    analytics = relationship(
        "Analytics", back_populates="match", cascade="all, delete-orphan"
    )
    heatmaps = relationship(
        "Heatmap", back_populates="match", cascade="all, delete-orphan"
    )


class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    match_id = Column(
        UUID(as_uuid=False), ForeignKey("matches.id"), nullable=False, unique=True
    )

    original_filename = Column(String(500), nullable=False)
    storage_path = Column(String(1000), nullable=False)
    annotated_storage_path = Column(String(1000), nullable=True)
    thumbnail_path = Column(String(1000), nullable=True)

    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    codec = Column(String(50), nullable=True)
    size_bytes = Column(Integer, nullable=True)

    # Homography calibration: 4 image points <-> 4 pitch points (meters), JSON encoded
    homography_points = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    match = relationship("Match", back_populates="video")


class Team(Base):
    __tablename__ = "teams"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    match_id = Column(UUID(as_uuid=False), ForeignKey("matches.id"), nullable=False)

    label = Column(String(50), nullable=False)  # "Team A" / "Team B"
    display_name = Column(String(255), nullable=True)
    jersey_color_hex = Column(
        String(7), nullable=False
    )  # dominant color from KMeans cluster center

    match = relationship("Match", back_populates="teams")
    players = relationship(
        "Player", back_populates="team", cascade="all, delete-orphan"
    )
    heatmap = relationship(
        "Heatmap",
        back_populates="team",
        uselist=False,
        primaryjoin="and_(Team.id==Heatmap.team_id)",
    )
    analytics = relationship(
        "Analytics", back_populates="team", cascade="all, delete-orphan"
    )


class Player(Base):
    __tablename__ = "players"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    team_id = Column(UUID(as_uuid=False), ForeignKey("teams.id"), nullable=False)

    tracking_id = Column(Integer, nullable=False)  # persistent ID assigned by tracker
    jersey_number = Column(Integer, nullable=True)  # OCR future-ready, nullable for now
    estimated_role = Column(
        String(50), nullable=True
    )  # e.g. Defender/Midfielder/Forward (heuristic)
    thumbnail_path = Column(
        String(1000), nullable=True
    )  # best-confidence crop, not a "face photo"
    features = Column(JSON, nullable=True)

    team = relationship("Team", back_populates="players")
    tracking_points = relationship(
        "TrackingData", back_populates="player", cascade="all, delete-orphan"
    )
    trajectory = relationship(
        "Trajectory",
        back_populates="player",
        uselist=False,
        cascade="all, delete-orphan",
    )
    heatmap = relationship(
        "Heatmap", back_populates="player", uselist=False, cascade="all, delete-orphan"
    )
    analytics = relationship(
        "Analytics", back_populates="player", cascade="all, delete-orphan"
    )


class TrackingData(Base):
    """Per-frame position sample for a tracked player."""

    __tablename__ = "tracking_data"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    player_id = Column(
        UUID(as_uuid=False), ForeignKey("players.id"), nullable=False, index=True
    )

    frame_number = Column(Integer, nullable=False)
    timestamp_seconds = Column(Float, nullable=False)

    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    detection_confidence = Column(Float, nullable=False)

    pitch_x = Column(Float, nullable=True)  # meters, after homography mapping
    pitch_y = Column(Float, nullable=True)

    velocity_mps = Column(Float, nullable=True)
    acceleration_mps2 = Column(Float, nullable=True)
    direction_degrees = Column(Float, nullable=True)

    player = relationship("Player", back_populates="tracking_points")


class Heatmap(Base):
    """
    Polymorphic-ish: exactly one of match_id / team_id / player_id is set,
    identifying the scope this heatmap represents.
    """

    __tablename__ = "heatmaps"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    match_id = Column(UUID(as_uuid=False), ForeignKey("matches.id"), nullable=True)
    team_id = Column(UUID(as_uuid=False), ForeignKey("teams.id"), nullable=True)
    player_id = Column(UUID(as_uuid=False), ForeignKey("players.id"), nullable=True)

    scope = Column(Enum(AnalyticsScope), nullable=False)
    image_path = Column(String(1000), nullable=False)  # rendered PNG
    grid_data_path = Column(
        String(1000), nullable=True
    )  # raw density grid, JSON/NPY, for the interactive frontend
    generated_at = Column(DateTime(timezone=True), server_default=func.now())

    match = relationship("Match", back_populates="heatmaps")
    team = relationship("Team", back_populates="heatmap")
    player = relationship("Player", back_populates="heatmap")


class Trajectory(Base):
    __tablename__ = "trajectories"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    player_id = Column(
        UUID(as_uuid=False), ForeignKey("players.id"), nullable=False, unique=True
    )

    points = Column(JSON, nullable=False)  # smoothed [[t, x, y], ...] in pitch meters
    start_point = Column(JSON, nullable=True)
    end_point = Column(JSON, nullable=True)

    player = relationship("Player", back_populates="trajectory")


class Analytics(Base):
    """
    Computed statistics at match, team, or player scope.
    `metrics` holds the full structured payload (distance, speed, zone occupancy, etc.)
    so the schema stays stable as new metrics are added.
    """

    __tablename__ = "analytics"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    match_id = Column(UUID(as_uuid=False), ForeignKey("matches.id"), nullable=True)
    team_id = Column(UUID(as_uuid=False), ForeignKey("teams.id"), nullable=True)
    player_id = Column(UUID(as_uuid=False), ForeignKey("players.id"), nullable=True)

    scope = Column(Enum(AnalyticsScope), nullable=False)
    metrics = Column(JSON, nullable=False)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    match = relationship("Match", back_populates="analytics")
    team = relationship("Team", back_populates="analytics")
    player = relationship("Player", back_populates="analytics")


class ProcessingJob(Base):
    """Tracks the async analysis pipeline run for a match, driving the progress UI."""

    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    match_id = Column(UUID(as_uuid=False), ForeignKey("matches.id"), nullable=False)

    current_stage = Column(
        Enum(ProcessingStage), default=ProcessingStage.UPLOADING, nullable=False
    )
    progress_percent = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    match = relationship("Match", back_populates="processing_jobs")
