"""
GET /api/activities?limit=20 — Recent running activities with analysis.

Each activity is annotated with a predicted quality classification
based on pre-run HRV and Body Battery, plus contextual notes.
"""

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.engine import get_db
from app.db.queries import get_activities
from app.models.database import DailyHRV, DailyBodyBattery
from app.models.schemas import ActivitiesListResponse, ActivityResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/activities", response_model=ActivitiesListResponse)
def list_activities(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Return recent running activities with quality annotations."""
    rows, total = get_activities(db, limit=limit, offset=offset)

    activities = []
    for act in rows:
        # Look up pre-run HRV and Body Battery for quality prediction
        predicted_quality, notes = _classify_activity(db, act)

        activities.append(ActivityResponse(
            activity_id=act.activity_id,
            date=act.date,
            activity_name=act.activity_name,
            distance_km=act.distance_km,
            duration_min=act.duration_min,
            pace_min_per_km=act.pace_min_per_km,
            avg_hr=act.avg_hr,
            max_hr=act.max_hr,
            elevation_gain_m=act.elevation_gain_m,
            calories=act.calories,
            aerobic_te=act.aerobic_te,
            trimp=act.trimp,
            vo2max=act.vo2max,
            predicted_quality=predicted_quality,
            notes=notes,
        ))

    return ActivitiesListResponse(activities=activities, total=total)


def _classify_activity(db: Session, activity) -> tuple[str, str]:
    """
    Classify an activity's predicted quality based on pre-run metrics.

    Returns (quality_label, explanation_note).
    """
    act_date = activity.date
    notes_parts = []

    # Get HRV from that morning or previous day
    hrv = (db.query(DailyHRV)
           .filter(DailyHRV.date >= act_date - timedelta(days=1),
                   DailyHRV.date <= act_date)
           .order_by(DailyHRV.date.desc()).first())

    bb = (db.query(DailyBodyBattery)
          .filter(DailyBodyBattery.date >= act_date - timedelta(days=1),
                  DailyBodyBattery.date <= act_date)
          .order_by(DailyBodyBattery.date.desc()).first())

    hrv_val = hrv.hrv_7day_avg if hrv and hrv.hrv_7day_avg else None
    bb_val = bb.bb_morning if bb else None

    if hrv_val is not None and hrv.hrv_baseline:
        if hrv_val < hrv.hrv_baseline * 0.85:
            notes_parts.append("HRV was well below baseline pre-run")
        elif hrv_val < hrv.hrv_baseline:
            notes_parts.append("HRV was slightly below baseline")

    if bb_val is not None:
        if bb_val < 30:
            notes_parts.append("Body Battery was very low")
        elif bb_val < 50:
            notes_parts.append("Body Battery was moderate")

    # Simple quality classification
    score = 50  # Neutral baseline
    if hrv_val is not None and hrv.hrv_baseline and hrv.hrv_baseline > 0:
        ratio = hrv_val / hrv.hrv_baseline
        score += (ratio - 1.0) * 100  # +/- points based on HRV
    if bb_val is not None:
        score += (bb_val - 50) * 0.3  # Slight influence from BB

    if score >= 60:
        quality = "good"
    elif score >= 40:
        quality = "moderate"
    else:
        quality = "struggled"

    note = "; ".join(notes_parts) if notes_parts else None
    return quality, note
