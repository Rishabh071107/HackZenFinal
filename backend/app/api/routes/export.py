import csv
import io
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db import models
from app.db.database import get_db

router = APIRouter(prefix="/matches/{match_id}/export", tags=["export"])


def _match_payload(db: Session, match_id: str) -> dict:
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")

    teams = db.query(models.Team).filter(models.Team.match_id == match_id).all()
    match_analytics = (
        db.query(models.Analytics)
        .filter(
            models.Analytics.match_id == match_id,
            models.Analytics.scope == models.AnalyticsScope.MATCH,
        )
        .first()
    )

    payload = {
        "match": {
            "id": match.id,
            "name": match.name,
            "team_a_name": match.team_a_name,
            "team_b_name": match.team_b_name,
            "tournament": match.tournament,
            "duration_seconds": match.duration_seconds,
            "avg_detection_confidence": match.avg_detection_confidence,
            "avg_tracking_accuracy": match.avg_tracking_accuracy,
        },
        "match_metrics": match_analytics.metrics if match_analytics else {},
        "teams": [],
    }

    for team in teams:
        team_analytics = (
            db.query(models.Analytics)
            .filter(
                models.Analytics.team_id == team.id,
                models.Analytics.scope == models.AnalyticsScope.TEAM,
            )
            .first()
        )
        players_out = []
        for player in team.players:
            p_analytics = (
                db.query(models.Analytics)
                .filter(
                    models.Analytics.player_id == player.id,
                    models.Analytics.scope == models.AnalyticsScope.PLAYER,
                )
                .first()
            )
            players_out.append(
                {
                    "tracking_id": player.tracking_id,
                    "estimated_role": player.estimated_role,
                    "metrics": p_analytics.metrics if p_analytics else {},
                }
            )

        payload["teams"].append(
            {
                "label": team.label,
                "jersey_color": team.jersey_color_hex,
                "metrics": team_analytics.metrics if team_analytics else {},
                "players": players_out,
            }
        )

    return payload


@router.get("/json")
def export_json(match_id: str, db: Session = Depends(get_db)):
    payload = _match_payload(db, match_id)
    buf = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
    return StreamingResponse(
        buf,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=match_{match_id}.json"},
    )


@router.get("/csv")
def export_csv(match_id: str, db: Session = Depends(get_db)):
    payload = _match_payload(db, match_id)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "team",
            "tracking_id",
            "role",
            "distance_covered_m",
            "average_speed_mps",
            "max_speed_mps",
            "time_stationary_s",
        ]
    )
    for team in payload["teams"]:
        for p in team["players"]:
            m = p["metrics"]
            writer.writerow(
                [
                    team["label"],
                    p["tracking_id"],
                    p["estimated_role"],
                    m.get("distance_covered_m"),
                    m.get("average_speed_mps"),
                    m.get("max_speed_mps"),
                    m.get("time_stationary_s"),
                ]
            )
    byte_buf = io.BytesIO(buf.getvalue().encode("utf-8"))
    return StreamingResponse(
        byte_buf,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=match_{match_id}.csv"},
    )


@router.get("/pdf")
def export_pdf(match_id: str, db: Session = Depends(get_db)):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    payload = _match_payload(db, match_id)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, f"Match Report: {payload['match']['name']}")
    y -= 30
    c.setFont("Helvetica", 11)
    for key, val in payload["match_metrics"].items():
        c.drawString(40, y, f"{key.replace('_', ' ').title()}: {val}")
        y -= 16
        if y < 60:
            c.showPage()
            y = height - 50

    for team in payload["teams"]:
        y -= 20
        if y < 100:
            c.showPage()
            y = height - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, f"{team['label']} ({team['jersey_color']})")
        y -= 20
        c.setFont("Helvetica", 10)
        for p in team["players"]:
            m = p["metrics"]
            line = (
                f"#{p['tracking_id']} {p['estimated_role']}: "
                f"{m.get('distance_covered_m', 0)} m, avg {m.get('average_speed_mps', 0)} m/s"
            )
            c.drawString(50, y, line)
            y -= 14
            if y < 60:
                c.showPage()
                y = height - 50

    c.save()
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=match_{match_id}_report.pdf"
        },
    )
