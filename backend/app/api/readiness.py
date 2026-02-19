"""
GET /api/readiness/today — Today's composite readiness score.

Reads from the database (sub-500ms), no live Garmin API calls.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.readiness import compute_readiness_score
from app.db.engine import get_db
from app.db.queries import get_today_metrics
from app.models.schemas import ReadinessResponse, ReadinessFactors

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/readiness/today", response_model=ReadinessResponse)
def readiness_today(db: Session = Depends(get_db)):
    """
    Return today's readiness score (0-100) with a recommendation.

    Uses the most recent synced data. Falls back to up to 3 days ago
    if today's data hasn't been synced yet.
    """
    metrics = get_today_metrics(db)

    result = compute_readiness_score(
        hrv_value=metrics.get("hrv_7day_avg") or metrics.get("hrv_value"),
        hrv_baseline=metrics.get("hrv_baseline"),
        tsb=metrics.get("tsb"),
        sleep_score=metrics.get("sleep_score"),
        sleep_hours=metrics.get("sleep_hours"),
        body_battery=metrics.get("body_battery"),
        avg_stress=metrics.get("avg_stress"),
    )

    return ReadinessResponse(
        readiness_score=result["readiness_score"],
        recommendation=result["recommendation"],
        color=result["color"],
        factors=ReadinessFactors(
            hrv=metrics.get("hrv_value"),
            tsb=metrics.get("tsb"),
            sleep_hours=metrics.get("sleep_hours"),
            sleep_score=metrics.get("sleep_score"),
            body_battery=metrics.get("body_battery"),
            resting_hr=metrics.get("resting_hr"),
            avg_stress=metrics.get("avg_stress"),
        ),
        computed_at=datetime.utcnow(),
    )
