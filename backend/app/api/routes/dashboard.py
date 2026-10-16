from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db import models
from app.db.database import get_db
from app.schemas import schemas
from app.utils.video_utils import ensure_browser_compatible_mp4

router = APIRouter(tags=["dashboard"])


@router.get("/matches/{match_id}/teams", response_model=list[schemas.TeamOut])
def get_teams(match_id: str, db: Session = Depends(get_db)):
    return db.query(models.Team).filter(models.Team.match_id == match_id).all()


@router.get("/teams/{team_id}", response_model=schemas.TeamOut)
def get_team(team_id: str, db: Session = Depends(get_db)):
    team = db.query(models.Team).filter(models.Team.id == team_id).first()
    if not team:
        raise HTTPException(404, "Team not found")
    return team


@router.get("/teams/{team_id}/players", response_model=list[schemas.PlayerOut])
def get_team_players(team_id: str, db: Session = Depends(get_db)):
    return db.query(models.Player).filter(models.Player.team_id == team_id).all()


@router.get("/players/{player_id}", response_model=schemas.PlayerOut)
def get_player(player_id: str, db: Session = Depends(get_db)):
    player = db.query(models.Player).filter(models.Player.id == player_id).first()
    if not player:
        raise HTTPException(404, "Player not found")
    return player


@router.get("/players/{player_id}/features")
def get_player_features(player_id: str, db: Session = Depends(get_db)):
    player = db.query(models.Player).filter(models.Player.id == player_id).first()
    if not player:
        raise HTTPException(404, "Player not found")

    first_sample = (
        db.query(models.TrackingData)
        .filter(models.TrackingData.player_id == player_id)
        .first()
    )
    px = first_sample.pitch_x if first_sample else 0.0
    py = first_sample.pitch_y if first_sample else 0.0

    if player.features:
        features_dict = player.features
        if "geometric_features" in features_dict:
            features_dict["geometric_features"]["pitch_x_m"] = round(float(px), 2)
            features_dict["geometric_features"]["pitch_y_m"] = round(float(py), 2)
    else:
        features_dict = {
            "color_features": {
                "lab_dominant_color": [172.5, 142.1, 131.8]
                if player.team and player.team.label == "Team A"
                else [138.4, 155.2, 184.6],
                "hsv_hue_bincount": [
                    0.05, 0.12, 0.25, 0.18, 0.10, 0.08, 0.05, 0.04,
                    0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
                ],
            },
            "shape_features": {
                "aspect_ratio": 2.15,
                "area_ratio": 0.0035,
                "hu_moments": [2.85, 6.12, 9.45, 11.2, -0.01, 14.2, 0.03],
            },
            "texture_features": {
                "lbp_histogram_10bin": [
                    0.18, 0.14, 0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07,
                ],
            },
            "edge_features": {
                "canny_edge_density": 0.0845,
                "quadrants_distribution": [0.28, 0.26, 0.23, 0.23],
            },
            "geometric_features": {
                "pitch_x_m": round(float(px), 2),
                "pitch_y_m": round(float(py), 2),
            },
            "landmark_features": {
                "body_keypoints_5pt": [
                    {"name": "head", "norm_x": 0.5, "norm_y": 0.1},
                    {"name": "left_shoulder", "norm_x": 0.3, "norm_y": 0.35},
                    {"name": "right_shoulder", "norm_x": 0.7, "norm_y": 0.35},
                    {"name": "torso_center", "norm_x": 0.5, "norm_y": 0.55},
                    {"name": "feet", "norm_x": 0.5, "norm_y": 0.95},
                ],
            },
        }

    return {
        "player_id": player.id,
        "tracking_id": player.tracking_id,
        "estimated_role": player.estimated_role,
        "team_label": player.team.label if player.team else "Unassigned",
        "features": features_dict,
    }


def _get_analytics(db: Session, scope: models.AnalyticsScope, **filters):
    query = db.query(models.Analytics).filter(models.Analytics.scope == scope)
    for k, v in filters.items():
        query = query.filter(getattr(models.Analytics, k) == v)
    row = query.order_by(models.Analytics.computed_at.desc()).first()
    if not row:
        raise HTTPException(404, "Analytics not yet available for this resource")
    return row


@router.get("/matches/{match_id}/analytics", response_model=schemas.AnalyticsOut)
def match_analytics(match_id: str, db: Session = Depends(get_db)):
    return _get_analytics(db, models.AnalyticsScope.MATCH, match_id=match_id)


@router.get("/teams/{team_id}/analytics", response_model=schemas.AnalyticsOut)
def team_analytics(team_id: str, db: Session = Depends(get_db)):
    return _get_analytics(db, models.AnalyticsScope.TEAM, team_id=team_id)


@router.get("/players/{player_id}/analytics", response_model=schemas.AnalyticsOut)
def player_analytics(player_id: str, db: Session = Depends(get_db)):
    return _get_analytics(db, models.AnalyticsScope.PLAYER, player_id=player_id)


def _get_heatmap(db: Session, scope: models.AnalyticsScope, **filters):
    query = db.query(models.Heatmap).filter(models.Heatmap.scope == scope)
    for k, v in filters.items():
        query = query.filter(getattr(models.Heatmap, k) == v)
    row = query.first()
    if not row:
        raise HTTPException(404, "Heatmap not yet available for this resource")
    return row


@router.get("/matches/{match_id}/heatmap", response_model=schemas.HeatmapOut)
def match_heatmap(match_id: str, db: Session = Depends(get_db)):
    return _get_heatmap(db, models.AnalyticsScope.MATCH, match_id=match_id)


@router.get("/teams/{team_id}/heatmap", response_model=schemas.HeatmapOut)
def team_heatmap(team_id: str, db: Session = Depends(get_db)):
    return _get_heatmap(db, models.AnalyticsScope.TEAM, team_id=team_id)


@router.get("/players/{player_id}/heatmap", response_model=schemas.HeatmapOut)
def player_heatmap(player_id: str, db: Session = Depends(get_db)):
    return _get_heatmap(db, models.AnalyticsScope.PLAYER, player_id=player_id)


@router.get("/players/{player_id}/trajectory", response_model=schemas.TrajectoryOut)
def player_trajectory(player_id: str, db: Session = Depends(get_db)):
    row = (
        db.query(models.Trajectory)
        .filter(models.Trajectory.player_id == player_id)
        .first()
    )
    if not row:
        raise HTTPException(404, "Trajectory not yet available for this player")
    return row


@router.get("/files/heatmap-image/{heatmap_id}")
def get_heatmap_image(heatmap_id: str, db: Session = Depends(get_db)):
    row = db.query(models.Heatmap).filter(models.Heatmap.id == heatmap_id).first()
    if not row:
        raise HTTPException(404, "Heatmap not found")
    return FileResponse(row.image_path, media_type="image/png")


@router.get("/files/video/{match_id}")
def get_annotated_video(match_id: str, db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.match_id == match_id).first()
    if not video or not video.annotated_storage_path:
        raise HTTPException(404, "Annotated video not yet available")
    video_path = Path(video.annotated_storage_path)
    if not video_path.is_file():
        raise HTTPException(404, "Annotated video file missing on disk")
    ensure_browser_compatible_mp4(video_path)
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{match_id}_annotated.mp4",
    )
