"""
SQLAlchemy ORM models for the Garmin training database.

Seven data tables mirror the CSV outputs from the CLI tool,
plus two metadata tables for sync status and cached insights.
"""

from datetime import date, datetime

from sqlalchemy import (
    Column, Integer, Float, String, Date, DateTime, Text, Boolean,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Activity(Base):
    """One row per running activity."""
    __tablename__ = "activities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    activity_id = Column(String, unique=True, index=True)
    date = Column(Date, index=True, nullable=False)
    activity_name = Column(String)
    distance_km = Column(Float)
    duration_min = Column(Float)
    pace_min_per_km = Column(Float)
    elevation_gain_m = Column(Float)
    avg_hr = Column(Float)
    max_hr = Column(Float)
    calories = Column(Integer)
    aerobic_te = Column(Float)
    anaerobic_te = Column(Float)
    vo2max = Column(Float)
    trimp = Column(Float)


class DailyTrainingLoad(Base):
    """Daily aggregated training load (ATL / CTL / TSB)."""
    __tablename__ = "daily_training_load"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, index=True, nullable=False)
    daily_trimp = Column(Float, default=0)
    daily_distance_km = Column(Float, default=0)
    daily_duration_min = Column(Float, default=0)
    num_activities = Column(Integer, default=0)
    atl = Column(Float)
    ctl = Column(Float)
    tsb = Column(Float)
    weekly_distance_km = Column(Float)


class DailyHRV(Base):
    """Daily HRV measurements from overnight readings."""
    __tablename__ = "daily_hrv"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, index=True, nullable=False)
    hrv_value = Column(Float)
    hrv_status = Column(String)
    hrv_baseline = Column(Float)
    weekly_avg = Column(Float)
    resting_hr = Column(Float)
    hrv_7day_avg = Column(Float)
    rhr_7day_avg = Column(Float)


class DailySleep(Base):
    """Nightly sleep data."""
    __tablename__ = "daily_sleep"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, index=True, nullable=False)
    sleep_score = Column(Float)
    sleep_hours = Column(Float)
    deep_sleep_hours = Column(Float)
    light_sleep_hours = Column(Float)
    rem_sleep_hours = Column(Float)
    awake_hours = Column(Float)
    deep_sleep_pct = Column(Float)
    rem_sleep_pct = Column(Float)
    sleep_score_7day = Column(Float)
    sleep_hours_7day = Column(Float)


class DailyStress(Base):
    """Daily stress level summaries."""
    __tablename__ = "daily_stress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, index=True, nullable=False)
    avg_stress = Column(Float)
    max_stress = Column(Float)
    rest_stress = Column(Float)
    low_stress_pct = Column(Float)
    medium_stress_pct = Column(Float)
    high_stress_pct = Column(Float)
    stress_7day_avg = Column(Float)


class DailyBodyBattery(Base):
    """Daily Body Battery readings."""
    __tablename__ = "daily_body_battery"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, index=True, nullable=False)
    bb_morning = Column(Float)
    bb_max = Column(Float)
    bb_min = Column(Float)
    bb_end_of_day = Column(Float)
    bb_drain = Column(Float)
    bb_morning_7day = Column(Float)


class DailyTrainingStatus(Base):
    """Garmin's own Training Status / Training Load classification."""
    __tablename__ = "daily_training_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, index=True, nullable=False)
    training_status = Column(String)
    training_load_7day = Column(Float)
    vo2max_running = Column(Float)
    load_focus = Column(String)


class SyncStatus(Base):
    """Tracks the state of Garmin data sync operations."""
    __tablename__ = "sync_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String, nullable=False, default="in_progress")  # in_progress | completed | failed
    message = Column(Text, default="")
    activities_synced = Column(Integer, default=0)


class CachedInsight(Base):
    """Stores precomputed analysis results (readiness, patterns, predictions)."""
    __tablename__ = "cached_insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, unique=True, index=True, nullable=False)
    value_json = Column(Text, nullable=False)
    computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
