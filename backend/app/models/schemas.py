"""
Pydantic models for API request validation and response serialization.

Every endpoint has an explicit response model so FastAPI auto-generates
accurate OpenAPI docs and the frontend gets a predictable contract.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# POST /api/auth/sync
# ---------------------------------------------------------------------------

class SyncRequest(BaseModel):
    """Optional override credentials for a sync. If omitted, uses stored/.env creds."""
    email: Optional[str] = None
    password: Optional[str] = None
    months: Optional[int] = Field(None, ge=1, le=24)


class SyncResponse(BaseModel):
    status: str
    message: str
    sync_id: Optional[int] = None
    activities_synced: Optional[int] = None
    last_sync_time: Optional[datetime] = None


# ---------------------------------------------------------------------------
# GET /api/readiness/today
# ---------------------------------------------------------------------------

class ReadinessFactors(BaseModel):
    hrv: Optional[float] = None
    tsb: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_score: Optional[float] = None
    body_battery: Optional[float] = None
    resting_hr: Optional[float] = None
    avg_stress: Optional[float] = None


class ReadinessResponse(BaseModel):
    readiness_score: float = Field(..., ge=0, le=100)
    recommendation: str
    color: str  # "green" | "yellow" | "red"
    factors: ReadinessFactors
    computed_at: datetime


# ---------------------------------------------------------------------------
# GET /api/metrics?days=30
# ---------------------------------------------------------------------------

class MetricsResponse(BaseModel):
    dates: list[str]
    hrv: list[Optional[float]]
    resting_hr: list[Optional[float]]
    atl: list[Optional[float]]
    ctl: list[Optional[float]]
    tsb: list[Optional[float]]
    sleep_score: list[Optional[float]]
    sleep_hours: list[Optional[float]]
    body_battery: list[Optional[float]]
    avg_stress: list[Optional[float]]
    daily_trimp: list[Optional[float]]
    daily_distance_km: list[Optional[float]]
    weekly_distance_km: list[Optional[float]]


# ---------------------------------------------------------------------------
# GET /api/activities?limit=20
# ---------------------------------------------------------------------------

class ActivityResponse(BaseModel):
    activity_id: Optional[str] = None
    date: date
    activity_name: Optional[str] = None
    distance_km: Optional[float] = None
    duration_min: Optional[float] = None
    pace_min_per_km: Optional[float] = None
    avg_hr: Optional[float] = None
    max_hr: Optional[float] = None
    elevation_gain_m: Optional[float] = None
    calories: Optional[int] = None
    aerobic_te: Optional[float] = None
    trimp: Optional[float] = None
    vo2max: Optional[float] = None
    predicted_quality: Optional[str] = None
    notes: Optional[str] = None


class ActivitiesListResponse(BaseModel):
    activities: list[ActivityResponse]
    total: int


# ---------------------------------------------------------------------------
# GET /api/insights
# ---------------------------------------------------------------------------

class InsightsResponse(BaseModel):
    sweet_spots: list[str]
    warnings: list[str]
    current_status: str
    hrv_threshold: Optional[dict] = None
    tsb_fatigue_limit: Optional[dict] = None
    sleep_performance: Optional[dict] = None


# ---------------------------------------------------------------------------
# GET /api/predictions?days=7
# ---------------------------------------------------------------------------

class DayPrediction(BaseModel):
    date: date
    predicted_readiness: float
    confidence: float = Field(..., ge=0, le=1)
    recommended_workout_type: str


class PredictionsResponse(BaseModel):
    predictions: list[DayPrediction]
