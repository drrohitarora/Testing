"""
Database query helpers.

All read operations go through this module. Each function takes a
SQLAlchemy Session and returns either ORM objects or plain dicts
suitable for API serialization.
"""

import json
import logging
from datetime import date, datetime, timedelta

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models.database import (
    Activity, CachedInsight, DailyBodyBattery, DailyHRV, DailySleep,
    DailyStress, DailyTrainingLoad, DailyTrainingStatus, SyncStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync status
# ---------------------------------------------------------------------------

def get_last_sync(db: Session) -> SyncStatus | None:
    return db.query(SyncStatus).order_by(desc(SyncStatus.id)).first()


def create_sync_record(db: Session) -> SyncStatus:
    record = SyncStatus(started_at=datetime.utcnow(), status="in_progress")
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def complete_sync(db: Session, sync_id: int, activities_count: int, message: str = ""):
    row = db.query(SyncStatus).get(sync_id)
    if row:
        row.status = "completed"
        row.completed_at = datetime.utcnow()
        row.activities_synced = activities_count
        row.message = message
        db.commit()


def fail_sync(db: Session, sync_id: int, error: str):
    row = db.query(SyncStatus).get(sync_id)
    if row:
        row.status = "failed"
        row.completed_at = datetime.utcnow()
        row.message = error
        db.commit()


# ---------------------------------------------------------------------------
# Today's readiness data
# ---------------------------------------------------------------------------

def get_today_metrics(db: Session, target_date: date | None = None) -> dict:
    """
    Gather the latest metrics needed to compute today's readiness score.

    Falls back to the most recent available data if today's data
    hasn't synced yet (common if the user checks early morning).
    """
    target = target_date or date.today()

    # Look back up to 3 days for the latest data
    hrv = _latest_before(db, DailyHRV, target, days_back=3)
    sleep = _latest_before(db, DailySleep, target, days_back=3)
    stress = _latest_before(db, DailyStress, target, days_back=3)
    bb = _latest_before(db, DailyBodyBattery, target, days_back=3)
    load = _latest_before(db, DailyTrainingLoad, target, days_back=3)

    return {
        "hrv_value": hrv.hrv_value if hrv else None,
        "hrv_7day_avg": hrv.hrv_7day_avg if hrv else None,
        "hrv_baseline": hrv.hrv_baseline if hrv else None,
        "resting_hr": hrv.resting_hr if hrv else None,
        "sleep_score": sleep.sleep_score if sleep else None,
        "sleep_hours": sleep.sleep_hours if sleep else None,
        "avg_stress": stress.avg_stress if stress else None,
        "body_battery": bb.bb_morning if bb else None,
        "tsb": load.tsb if load else None,
        "atl": load.atl if load else None,
        "ctl": load.ctl if load else None,
    }


def _latest_before(db: Session, model, target: date, days_back: int = 3):
    """Find the most recent row on or before the target date."""
    cutoff = target - timedelta(days=days_back)
    return (
        db.query(model)
        .filter(model.date >= cutoff, model.date <= target)
        .order_by(desc(model.date))
        .first()
    )


# ---------------------------------------------------------------------------
# Metrics time series
# ---------------------------------------------------------------------------

def get_metrics_series(db: Session, days: int = 30) -> dict:
    """
    Return time-series data for the last N days, suitable for charting.

    Returns a dict of parallel arrays keyed by metric name.
    """
    cutoff = date.today() - timedelta(days=days)

    loads = (db.query(DailyTrainingLoad)
             .filter(DailyTrainingLoad.date >= cutoff)
             .order_by(DailyTrainingLoad.date).all())

    hrvs = {r.date: r for r in
            db.query(DailyHRV).filter(DailyHRV.date >= cutoff).all()}
    sleeps = {r.date: r for r in
              db.query(DailySleep).filter(DailySleep.date >= cutoff).all()}
    stresses = {r.date: r for r in
                db.query(DailyStress).filter(DailyStress.date >= cutoff).all()}
    bbs = {r.date: r for r in
           db.query(DailyBodyBattery).filter(DailyBodyBattery.date >= cutoff).all()}

    result = {
        "dates": [], "hrv": [], "resting_hr": [],
        "atl": [], "ctl": [], "tsb": [],
        "sleep_score": [], "sleep_hours": [],
        "body_battery": [], "avg_stress": [],
        "daily_trimp": [], "daily_distance_km": [], "weekly_distance_km": [],
    }

    for load in loads:
        d = load.date
        result["dates"].append(d.isoformat())
        result["atl"].append(load.atl)
        result["ctl"].append(load.ctl)
        result["tsb"].append(load.tsb)
        result["daily_trimp"].append(load.daily_trimp)
        result["daily_distance_km"].append(load.daily_distance_km)
        result["weekly_distance_km"].append(load.weekly_distance_km)

        h = hrvs.get(d)
        result["hrv"].append(h.hrv_7day_avg if h else None)
        result["resting_hr"].append(h.resting_hr if h else None)

        s = sleeps.get(d)
        result["sleep_score"].append(s.sleep_score if s else None)
        result["sleep_hours"].append(s.sleep_hours if s else None)

        st = stresses.get(d)
        result["avg_stress"].append(st.avg_stress if st else None)

        bb = bbs.get(d)
        result["body_battery"].append(bb.bb_morning if bb else None)

    return result


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

def get_activities(db: Session, limit: int = 20, offset: int = 0) -> tuple[list[Activity], int]:
    """Return recent activities with total count."""
    total = db.query(Activity).count()
    rows = (db.query(Activity)
            .order_by(desc(Activity.date))
            .offset(offset).limit(limit).all())
    return rows, total


# ---------------------------------------------------------------------------
# Recent daily data for predictions
# ---------------------------------------------------------------------------

def get_recent_daily_data(db: Session, days: int = 14) -> list[dict]:
    """
    Gather recent daily data for readiness prediction.

    Returns a list of dicts ordered by date (oldest first).
    """
    cutoff = date.today() - timedelta(days=days)

    loads = {r.date: r for r in
             db.query(DailyTrainingLoad).filter(DailyTrainingLoad.date >= cutoff).all()}
    hrvs = {r.date: r for r in
            db.query(DailyHRV).filter(DailyHRV.date >= cutoff).all()}
    sleeps = {r.date: r for r in
              db.query(DailySleep).filter(DailySleep.date >= cutoff).all()}
    bbs = {r.date: r for r in
           db.query(DailyBodyBattery).filter(DailyBodyBattery.date >= cutoff).all()}
    stresses = {r.date: r for r in
                db.query(DailyStress).filter(DailyStress.date >= cutoff).all()}

    all_dates = sorted(set(
        list(loads.keys()) + list(hrvs.keys()) + list(sleeps.keys())
    ))

    result = []
    for d in all_dates:
        load = loads.get(d)
        hrv = hrvs.get(d)
        sleep = sleeps.get(d)
        bb = bbs.get(d)
        stress = stresses.get(d)

        result.append({
            "date": d,
            "daily_trimp": load.daily_trimp if load else 0,
            "atl": load.atl if load else None,
            "ctl": load.ctl if load else None,
            "tsb": load.tsb if load else None,
            "hrv_value": hrv.hrv_value if hrv else None,
            "hrv_baseline": hrv.hrv_baseline if hrv else None,
            "sleep_score": sleep.sleep_score if sleep else None,
            "sleep_hours": sleep.sleep_hours if sleep else None,
            "bb_morning": bb.bb_morning if bb else None,
            "avg_stress": stress.avg_stress if stress else None,
            "resting_hr": hrv.resting_hr if hrv else None,
        })

    return result


# ---------------------------------------------------------------------------
# Cached insights
# ---------------------------------------------------------------------------

def get_cached_insight(db: Session, key: str) -> dict | None:
    row = db.query(CachedInsight).filter_by(key=key).first()
    if row:
        try:
            return json.loads(row.value_json)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def set_cached_insight(db: Session, key: str, value: dict):
    row = db.query(CachedInsight).filter_by(key=key).first()
    serialized = json.dumps(value, default=str)
    if row:
        row.value_json = serialized
        row.computed_at = datetime.utcnow()
    else:
        db.add(CachedInsight(key=key, value_json=serialized, computed_at=datetime.utcnow()))
    db.commit()
