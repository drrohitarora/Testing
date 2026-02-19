"""
GET /api/metrics?days=30 — Time-series data for charting.

Returns parallel arrays of dates and metric values. The frontend
renders these with a JS charting library (Chart.js, Recharts, etc.)
instead of server-generated PNGs.
"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.engine import get_db
from app.db.queries import get_metrics_series
from app.models.schemas import MetricsResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days of history"),
    db: Session = Depends(get_db),
):
    """Return time-series training and recovery metrics for the last N days."""
    data = get_metrics_series(db, days=days)
    return MetricsResponse(**data)
