"""
Forward-looking readiness predictions.

Uses linear extrapolation of ATL/CTL/TSB trends plus recent HRV and
sleep patterns to forecast readiness for the next N days.

This is deliberately simple (no ML model) because:
1. Personal training data is too small for meaningful ML training
2. Linear extrapolation of TSB is well-validated in sports science
3. Transparency matters — athletes need to understand the prediction

The key insight: TSB is predictable because CTL changes slowly (28-day EWM)
and future ATL can be estimated from your training plan or recent pattern.
"""

import logging
from datetime import date, timedelta

import numpy as np

from app.core.readiness import compute_readiness_score

logger = logging.getLogger(__name__)


def predict_readiness(
    recent_days: list[dict],
    forecast_days: int = 7,
) -> list[dict]:
    """
    Predict readiness scores for the next N days.

    Args:
        recent_days: List of dicts with recent daily metrics, ordered by date.
            Each dict should have: date, atl, ctl, tsb, hrv_value, hrv_baseline,
            sleep_score, sleep_hours, bb_morning, avg_stress, resting_hr.
        forecast_days: How many days forward to predict.

    Returns:
        List of prediction dicts with: date, predicted_readiness, confidence,
        recommended_workout_type.
    """
    if not recent_days:
        return _empty_predictions(forecast_days)

    # Extract recent trends (last 7 days) for extrapolation
    last_7 = recent_days[-7:] if len(recent_days) >= 7 else recent_days

    # Current values (most recent day with data)
    current = recent_days[-1]
    current_ctl = current.get("ctl") or 0
    current_atl = current.get("atl") or 0

    # Average daily TRIMP over last 7 days (predicts near-future load)
    recent_trimps = [d.get("daily_trimp", 0) or 0 for d in last_7]
    avg_daily_trimp = np.mean(recent_trimps) if recent_trimps else 0

    # HRV trend: is it rising, stable, or falling?
    hrv_values = [d.get("hrv_value") for d in last_7 if d.get("hrv_value") is not None]
    hrv_trend = _linear_slope(hrv_values) if len(hrv_values) >= 3 else 0
    current_hrv = hrv_values[-1] if hrv_values else None
    hrv_baseline = current.get("hrv_baseline")

    # Sleep trend
    sleep_scores = [d.get("sleep_score") for d in last_7 if d.get("sleep_score") is not None]
    sleep_hours_list = [d.get("sleep_hours") for d in last_7 if d.get("sleep_hours") is not None]
    avg_sleep_score = np.mean(sleep_scores) if sleep_scores else None
    avg_sleep_hours = np.mean(sleep_hours_list) if sleep_hours_list else None

    # Body Battery and stress (use recent average as forecast)
    bb_values = [d.get("bb_morning") for d in last_7 if d.get("bb_morning") is not None]
    avg_bb = np.mean(bb_values) if bb_values else None
    stress_values = [d.get("avg_stress") for d in last_7 if d.get("avg_stress") is not None]
    avg_stress = np.mean(stress_values) if stress_values else None

    predictions = []
    today = date.today()

    # EWM decay constants
    atl_decay = 2 / (7 + 1)
    ctl_decay = 2 / (28 + 1)

    proj_atl = current_atl
    proj_ctl = current_ctl

    for i in range(1, forecast_days + 1):
        pred_date = today + timedelta(days=i)

        # Project ATL and CTL forward assuming average training load continues.
        # EWM update: new_ewm = alpha * new_value + (1 - alpha) * old_ewm
        proj_atl = atl_decay * avg_daily_trimp + (1 - atl_decay) * proj_atl
        proj_ctl = ctl_decay * avg_daily_trimp + (1 - ctl_decay) * proj_ctl
        proj_tsb = proj_ctl - proj_atl

        # Project HRV with trend (clamped to reasonable bounds)
        proj_hrv = None
        if current_hrv is not None:
            proj_hrv = current_hrv + hrv_trend * i
            proj_hrv = max(10, min(200, proj_hrv))

        score_result = compute_readiness_score(
            hrv_value=proj_hrv,
            hrv_baseline=hrv_baseline,
            tsb=proj_tsb,
            sleep_score=avg_sleep_score,
            sleep_hours=avg_sleep_hours,
            body_battery=avg_bb,
            avg_stress=avg_stress,
        )

        # Confidence decreases with forecast horizon.
        # Day 1 = 0.85, Day 7 = 0.35 (linear decay).
        confidence = round(max(0.2, 0.9 - 0.1 * i), 2)

        readiness = score_result["readiness_score"]
        workout = _recommend_workout(readiness, proj_tsb, i)

        predictions.append({
            "date": pred_date,
            "predicted_readiness": readiness,
            "confidence": confidence,
            "recommended_workout_type": workout,
        })

    return predictions


def _linear_slope(values: list) -> float:
    """Compute the linear trend (slope) of a list of values."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    slope, _ = np.polyfit(x[mask], y[mask], 1)
    return float(slope)


def _recommend_workout(readiness: float, tsb: float | None, day_offset: int) -> str:
    """Suggest a workout type based on predicted readiness and training balance."""
    if readiness >= 80:
        return "Quality session — tempo, intervals, or race-pace long run"
    if readiness >= 65:
        return "Moderate effort — steady-state or progressive long run"
    if readiness >= 45:
        return "Easy run — zone 2 aerobic base building"
    if readiness >= 25:
        return "Recovery jog or cross-training"
    return "Rest day — active recovery, stretching, or complete rest"


def _empty_predictions(days: int) -> list[dict]:
    """Return placeholder predictions when no data is available."""
    today = date.today()
    return [
        {
            "date": today + timedelta(days=i),
            "predicted_readiness": 50.0,
            "confidence": 0.1,
            "recommended_workout_type": "Sync your Garmin data first",
        }
        for i in range(1, days + 1)
    ]
