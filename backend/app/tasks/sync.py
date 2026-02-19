"""
Background sync task.

Orchestrates the full Garmin data pipeline:
  authenticate → fetch all data types → process → write to DB → compute insights

This runs as a FastAPI BackgroundTask so the POST /api/auth/sync endpoint
returns immediately while the sync proceeds.
"""

import logging
import time
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from app.core import garmin, processing, analysis
from app.core.readiness import compute_readiness_score
from app.db.engine import get_session_factory
from app.db import queries
from app.models.database import (
    Activity, DailyBodyBattery, DailyHRV, DailySleep,
    DailyStress, DailyTrainingLoad, DailyTrainingStatus,
)

logger = logging.getLogger(__name__)


def run_sync(sync_id: int, months: int = 6, email: str | None = None, password: str | None = None):
    """
    Full sync pipeline. Designed to run in a background thread.

    Creates its own DB session (BackgroundTasks run outside the
    request lifecycle, so we can't reuse the request's session).
    """
    factory = get_session_factory()
    db = factory()

    try:
        logger.info("Sync %d: starting (%d months)...", sync_id, months)

        # --- Authenticate ---
        client = garmin.authenticate(db, email=email, password=password)
        days = months * 30

        # --- Fetch all data types ---
        raw = {}

        logger.info("Sync %d: fetching activities...", sync_id)
        raw["activities"] = garmin.fetch_activities(client, months=months)
        logger.info("Sync %d: found %d activities.", sync_id, len(raw["activities"]))

        # Fetch HR zones for up to 100 activities
        hr_zones = {}
        for act in raw["activities"][:100]:
            aid = act.get("activityId")
            if aid:
                zones = garmin.fetch_activity_hr_zones(client, aid)
                if zones:
                    hr_zones[aid] = zones
                time.sleep(0.3)
        raw["hr_zones"] = hr_zones

        time.sleep(1)
        logger.info("Sync %d: fetching HRV...", sync_id)
        raw["hrv"] = garmin.fetch_hrv_data(client, days=days)

        time.sleep(1)
        logger.info("Sync %d: fetching sleep...", sync_id)
        raw["sleep"] = garmin.fetch_sleep_data(client, days=days)

        time.sleep(1)
        logger.info("Sync %d: fetching stress...", sync_id)
        raw["stress"] = garmin.fetch_stress_data(client, days=days)

        time.sleep(1)
        logger.info("Sync %d: fetching Body Battery...", sync_id)
        raw["body_battery"] = garmin.fetch_body_battery(client, days=days)

        time.sleep(1)
        logger.info("Sync %d: fetching training status...", sync_id)
        raw["training_status"] = garmin.fetch_training_status(client, days=days)

        # --- Process ---
        logger.info("Sync %d: processing data...", sync_id)
        proc = {}
        proc["activities"] = processing.process_activities(raw["activities"], raw["hr_zones"])
        proc["training_load"] = processing.calculate_training_load(proc["activities"])
        proc["hrv"] = processing.process_hrv_data(raw.get("hrv", []))
        proc["sleep"] = processing.process_sleep_data(raw.get("sleep", []))
        proc["stress"] = processing.process_stress_data(raw.get("stress", []))
        proc["body_battery"] = processing.process_body_battery(raw.get("body_battery", []))
        proc["training_status"] = processing.process_training_status(raw.get("training_status", []))

        # --- Write to DB ---
        logger.info("Sync %d: writing to database...", sync_id)
        _upsert_activities(db, proc["activities"])
        _upsert_daily(db, proc["training_load"], DailyTrainingLoad, _training_load_cols())
        _upsert_daily(db, proc["hrv"], DailyHRV, _hrv_cols())
        _upsert_daily(db, proc["sleep"], DailySleep, _sleep_cols())
        _upsert_daily(db, proc["stress"], DailyStress, _stress_cols())
        _upsert_daily(db, proc["body_battery"], DailyBodyBattery, _bb_cols())
        _upsert_daily(db, proc["training_status"], DailyTrainingStatus, _ts_cols())

        # --- Compute and cache insights ---
        logger.info("Sync %d: computing insights...", sync_id)
        merged = processing.merge_all_data(
            proc["training_load"], proc["hrv"], proc["sleep"],
            proc["stress"], proc["body_battery"], proc["training_status"],
        )
        if not merged.empty:
            report = analysis.run_all_analyses(merged)
            insights = _build_insights(report, merged)
            queries.set_cached_insight(db, "insights", insights)

        # --- Done ---
        activities_count = len(proc["activities"]) if not proc["activities"].empty else 0
        queries.complete_sync(db, sync_id, activities_count, "Sync completed successfully.")
        logger.info("Sync %d: completed. %d activities.", sync_id, activities_count)

    except Exception as e:
        logger.error("Sync %d failed: %s", sync_id, e, exc_info=True)
        queries.fail_sync(db, sync_id, str(e))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# DB upsert helpers
# ---------------------------------------------------------------------------

def _upsert_activities(db: Session, df: pd.DataFrame):
    """Insert or update activities by activity_id."""
    if df.empty:
        return
    for _, row in df.iterrows():
        existing = db.query(Activity).filter_by(activity_id=str(row.get("activity_id"))).first()
        data = {
            "activity_id": str(row.get("activity_id")),
            "date": row["date"].date() if hasattr(row["date"], "date") else row["date"],
            "activity_name": row.get("activity_name"),
            "distance_km": row.get("distance_km"),
            "duration_min": row.get("duration_min"),
            "pace_min_per_km": row.get("pace_min_per_km"),
            "elevation_gain_m": row.get("elevation_gain_m"),
            "avg_hr": row.get("avg_hr"),
            "max_hr": row.get("max_hr"),
            "calories": int(row.get("calories", 0) or 0),
            "aerobic_te": row.get("aerobic_te"),
            "anaerobic_te": row.get("anaerobic_te"),
            "vo2max": row.get("vo2max"),
            "trimp": row.get("trimp"),
        }
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
        else:
            db.add(Activity(**data))
    db.commit()


def _upsert_daily(db: Session, df: pd.DataFrame, model, columns: list[str]):
    """Generic upsert for daily metric tables (keyed by date)."""
    if df.empty:
        return
    for _, row in df.iterrows():
        row_date = row["date"]
        if hasattr(row_date, "date"):
            row_date = row_date.date()

        existing = db.query(model).filter_by(date=row_date).first()
        data = {"date": row_date}
        for col in columns:
            val = row.get(col)
            # Convert numpy types to Python natives for SQLAlchemy
            if pd.notna(val) if not isinstance(val, str) else bool(val):
                data[col] = float(val) if isinstance(val, (int, float)) and col != "training_status" and col != "hrv_status" and col != "load_focus" else val
            else:
                data[col] = None

        if existing:
            for k, v in data.items():
                if k != "date":
                    setattr(existing, k, v)
        else:
            db.add(model(**data))
    db.commit()


def _training_load_cols():
    return ["daily_trimp", "daily_distance_km", "daily_duration_min",
            "num_activities", "atl", "ctl", "tsb", "weekly_distance_km"]

def _hrv_cols():
    return ["hrv_value", "hrv_status", "hrv_baseline", "weekly_avg",
            "resting_hr", "hrv_7day_avg", "rhr_7day_avg"]

def _sleep_cols():
    return ["sleep_score", "sleep_hours", "deep_sleep_hours", "light_sleep_hours",
            "rem_sleep_hours", "awake_hours", "deep_sleep_pct", "rem_sleep_pct",
            "sleep_score_7day", "sleep_hours_7day"]

def _stress_cols():
    return ["avg_stress", "max_stress", "rest_stress", "low_stress_pct",
            "medium_stress_pct", "high_stress_pct", "stress_7day_avg"]

def _bb_cols():
    return ["bb_morning", "bb_max", "bb_min", "bb_end_of_day", "bb_drain",
            "bb_morning_7day"]

def _ts_cols():
    return ["training_status", "training_load_7day", "vo2max_running", "load_focus"]


# ---------------------------------------------------------------------------
# Insight generation
# ---------------------------------------------------------------------------

def _build_insights(report: dict, merged_df: pd.DataFrame) -> dict:
    """Build the insights payload from the analysis report."""
    sweet_spots = []
    warnings = []

    hrv_analysis = report.get("hrv_threshold", {})
    if hrv_analysis.get("status") == "complete":
        thresh = hrv_analysis.get("hrv_threshold")
        if thresh:
            sweet_spots.append(f"HRV > {thresh:.0f} ms for best performance")
        good_avg = hrv_analysis.get("good_day_avg_hrv")
        if good_avg:
            sweet_spots.append(f"Good training days average HRV: {good_avg} ms")

    tsb_analysis = report.get("tsb_fatigue_limit", {})
    if tsb_analysis.get("status") == "complete":
        limit = tsb_analysis.get("fatigue_limit_days")
        if limit:
            warnings.append(f"Performance drops after {limit} days of negative TSB")
        baseline = tsb_analysis.get("baseline_pace")
        if baseline:
            sweet_spots.append(f"Baseline pace: {baseline} min/km")

    sleep_analysis = report.get("sleep_performance", {})
    if sleep_analysis.get("status") == "complete":
        predictor = sleep_analysis.get("strongest_predictor", {})
        if predictor.get("p_value", 1) < 0.05:
            sweet_spots.append(
                f"Sleep metric most linked to performance: {predictor['metric']}"
            )

    # Weekly distance sweet spot from recent data
    if "weekly_distance_km" in merged_df.columns:
        recent = merged_df.tail(90)
        if "pace_min_per_km" in recent.columns:
            good_runs = recent.dropna(subset=["pace_min_per_km", "weekly_distance_km"])
            if len(good_runs) > 10:
                fast_runs = good_runs[good_runs["pace_min_per_km"] <= good_runs["pace_min_per_km"].quantile(0.25)]
                if not fast_runs.empty:
                    low_km = fast_runs["weekly_distance_km"].quantile(0.25)
                    high_km = fast_runs["weekly_distance_km"].quantile(0.75)
                    sweet_spots.append(f"{low_km:.0f}-{high_km:.0f} km/week for best paces")

    # Current status
    if not merged_df.empty:
        latest = merged_df.iloc[-1]
        tsb_val = latest.get("tsb")
        if tsb_val is not None:
            if tsb_val > 5:
                current_status = "You're well rested — good window for quality training."
            elif tsb_val > -10:
                current_status = "You're in a balanced training zone."
            elif tsb_val > -30:
                current_status = "You're carrying moderate fatigue — normal for a build phase."
            else:
                current_status = "Heavy fatigue accumulated — consider a recovery block soon."
        else:
            current_status = "Sync more data for a status assessment."
    else:
        current_status = "No data available yet."

    if not warnings:
        warnings.append("Not enough data to identify warning patterns yet.")

    return {
        "sweet_spots": sweet_spots,
        "warnings": warnings,
        "current_status": current_status,
        "hrv_threshold": hrv_analysis if hrv_analysis.get("status") == "complete" else None,
        "tsb_fatigue_limit": tsb_analysis if tsb_analysis.get("status") == "complete" else None,
        "sleep_performance": sleep_analysis if sleep_analysis.get("status") == "complete" else None,
    }
