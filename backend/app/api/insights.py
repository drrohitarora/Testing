"""
GET /api/insights — Personalized training patterns and recommendations.

Returns precomputed insights from the most recent sync.
These are cached in the DB during the sync background task.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.engine import get_db
from app.db.queries import get_cached_insight
from app.models.schemas import InsightsResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/insights", response_model=InsightsResponse)
def get_insights(db: Session = Depends(get_db)):
    """
    Return personalized training patterns and sweet spots.

    These are computed during the sync process and cached. If no
    insights are available, returns a prompt to sync first.
    """
    cached = get_cached_insight(db, "insights")

    if not cached:
        return InsightsResponse(
            sweet_spots=["Sync your Garmin data to discover your performance sweet spots."],
            warnings=["No data analyzed yet."],
            current_status="Run a sync first: POST /api/auth/sync",
        )

    return InsightsResponse(
        sweet_spots=cached.get("sweet_spots", []),
        warnings=cached.get("warnings", []),
        current_status=cached.get("current_status", "Unknown"),
        hrv_threshold=cached.get("hrv_threshold"),
        tsb_fatigue_limit=cached.get("tsb_fatigue_limit"),
        sleep_performance=cached.get("sleep_performance"),
    )
