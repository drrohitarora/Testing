"""
GET /api/predictions?days=7 — Forecasted readiness for upcoming days.

Uses linear extrapolation of training load trends and recent recovery
metrics to predict readiness and recommend workout types.
"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.predictions import predict_readiness
from app.db.engine import get_db
from app.db.queries import get_recent_daily_data
from app.models.schemas import PredictionsResponse, DayPrediction

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/predictions", response_model=PredictionsResponse)
def get_predictions(
    days: int = Query(default=7, ge=1, le=14, description="Days to forecast"),
    db: Session = Depends(get_db),
):
    """
    Predict readiness scores for the next N days.

    Confidence decreases linearly with forecast horizon
    (day 1 ≈ 85%, day 7 ≈ 35%).
    """
    recent = get_recent_daily_data(db, days=14)
    raw_predictions = predict_readiness(recent, forecast_days=days)

    return PredictionsResponse(
        predictions=[
            DayPrediction(
                date=p["date"],
                predicted_readiness=p["predicted_readiness"],
                confidence=p["confidence"],
                recommended_workout_type=p["recommended_workout_type"],
            )
            for p in raw_predictions
        ]
    )
