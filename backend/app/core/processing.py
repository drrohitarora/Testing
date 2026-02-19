"""
Data processing and training metric calculations.

Ported directly from garmin_training_analysis/data_processing.py.
The logic is identical — only the module path changed.

Key metrics calculated:
- TRIMP (Banister's Training Impulse)
- ATL / CTL / TSB (Acute/Chronic Training Load, Training Stress Balance)
- Rolling averages for HRV, resting HR, sleep, stress, Body Battery
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activity processing
# ---------------------------------------------------------------------------

def process_activities(raw_activities: list[dict], hr_zones_map: dict | None = None) -> pd.DataFrame:
    """Convert raw Garmin activity dicts into a clean DataFrame."""
    records = []

    for act in raw_activities:
        activity_id = act.get("activityId")
        start_str = act.get("startTimeLocal", "")
        try:
            dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").date()
        except (ValueError, TypeError):
            continue

        distance_m = act.get("distance", 0) or 0
        distance_km = distance_m / 1000.0
        duration_s = act.get("duration", 0) or 0
        duration_min = duration_s / 60.0
        pace = duration_min / distance_km if distance_km > 0 else None
        elevation = act.get("elevationGain", 0) or 0
        avg_hr = act.get("averageHR")
        max_hr = act.get("maxHR")
        trimp = _calculate_trimp(avg_hr, max_hr, duration_min)

        records.append({
            "date": dt,
            "activity_id": activity_id,
            "distance_km": round(distance_km, 2),
            "duration_min": round(duration_min, 1),
            "pace_min_per_km": round(pace, 2) if pace else None,
            "elevation_gain_m": round(elevation, 0),
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "calories": act.get("calories", 0) or 0,
            "aerobic_te": act.get("aerobicTrainingEffect"),
            "anaerobic_te": act.get("anaerobicTrainingEffect"),
            "vo2max": act.get("vO2MaxValue"),
            "trimp": round(trimp, 1) if trimp else 0,
            "activity_name": act.get("activityName", ""),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _calculate_trimp(avg_hr, max_hr, duration_min, resting_hr=45):
    """
    Banister's TRIMP.

    TRIMP = duration * deltaHR_ratio * 0.64 * e^(1.92 * deltaHR_ratio)
    Resting HR defaults to 45 bpm (typical for trained ultra runners).
    """
    if not avg_hr or not duration_min:
        return None
    if not max_hr or max_hr <= avg_hr:
        max_hr = 190
    if max_hr <= resting_hr:
        return None
    ratio = (avg_hr - resting_hr) / (max_hr - resting_hr)
    ratio = max(0.0, min(1.0, ratio))
    return duration_min * ratio * 0.64 * np.exp(1.92 * ratio)


# ---------------------------------------------------------------------------
# Training load (ATL / CTL / TSB)
# ---------------------------------------------------------------------------

def calculate_training_load(activities_df: pd.DataFrame, atl_days: int = 7, ctl_days: int = 28) -> pd.DataFrame:
    """
    Compute daily ATL, CTL, TSB from activity TRIMP values.

    Uses 28-day CTL (not 42) for faster responsiveness with ultra training volume.
    """
    if activities_df.empty:
        return pd.DataFrame()

    daily = activities_df.groupby(activities_df["date"].dt.date).agg(
        daily_trimp=("trimp", "sum"),
        daily_distance_km=("distance_km", "sum"),
        daily_duration_min=("duration_min", "sum"),
        num_activities=("activity_id", "count"),
    ).reset_index()
    daily.columns = ["date", "daily_trimp", "daily_distance_km",
                     "daily_duration_min", "num_activities"]
    daily["date"] = pd.to_datetime(daily["date"])

    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(date_range).fillna(0)
    daily.index.name = "date"
    daily = daily.reset_index()

    daily["atl"] = daily["daily_trimp"].ewm(span=atl_days, adjust=False).mean()
    daily["ctl"] = daily["daily_trimp"].ewm(span=ctl_days, adjust=False).mean()
    daily["tsb"] = daily["ctl"] - daily["atl"]
    daily["weekly_distance_km"] = daily["daily_distance_km"].rolling(7, min_periods=1).sum()

    return daily


# ---------------------------------------------------------------------------
# HRV
# ---------------------------------------------------------------------------

def process_hrv_data(raw_hrv: list[dict]) -> pd.DataFrame:
    """Process raw HRV API responses into a clean DataFrame with rolling averages."""
    records = []
    for entry in raw_hrv:
        ds = entry.get("calendarDate", "")
        if not ds:
            continue
        summary = entry.get("hrvSummary", entry)
        records.append({
            "date": ds,
            "hrv_value": summary.get("lastNightAvg"),
            "hrv_status": summary.get("status"),
            "hrv_baseline": (summary.get("baseline", {}).get("lowUpper")
                             if isinstance(summary.get("baseline"), dict) else None),
            "weekly_avg": summary.get("weeklyAvg"),
            "resting_hr": summary.get("restingHeartRate"),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["hrv_7day_avg"] = df["hrv_value"].rolling(7, min_periods=3).mean()
    df["rhr_7day_avg"] = df["resting_hr"].rolling(7, min_periods=3).mean()
    return df


# ---------------------------------------------------------------------------
# Sleep
# ---------------------------------------------------------------------------

def process_sleep_data(raw_sleep: list[dict]) -> pd.DataFrame:
    """Process raw sleep API responses into a clean DataFrame."""
    records = []
    for entry in raw_sleep:
        daily = entry.get("dailySleepDTO", entry)
        ds = daily.get("calendarDate", "")
        if not ds:
            continue
        sleep_s = daily.get("sleepTimeSeconds", 0) or 0
        deep_s = daily.get("deepSleepSeconds", 0) or 0
        light_s = daily.get("lightSleepSeconds", 0) or 0
        rem_s = daily.get("remSleepSeconds", 0) or 0
        awake_s = daily.get("awakeSleepSeconds", 0) or 0

        records.append({
            "date": ds,
            "sleep_score": (daily.get("sleepScores", {}).get("overall", {}).get("value")
                            if isinstance(daily.get("sleepScores"), dict) else None),
            "sleep_hours": round(sleep_s / 3600, 2),
            "deep_sleep_hours": round(deep_s / 3600, 2),
            "light_sleep_hours": round(light_s / 3600, 2),
            "rem_sleep_hours": round(rem_s / 3600, 2),
            "awake_hours": round(awake_s / 3600, 2),
            "deep_sleep_pct": round(deep_s / sleep_s * 100, 1) if sleep_s > 0 else 0,
            "rem_sleep_pct": round(rem_s / sleep_s * 100, 1) if sleep_s > 0 else 0,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["sleep_score_7day"] = df["sleep_score"].rolling(7, min_periods=3).mean()
    df["sleep_hours_7day"] = df["sleep_hours"].rolling(7, min_periods=3).mean()
    return df


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------

def process_stress_data(raw_stress: list[dict]) -> pd.DataFrame:
    """Process daily stress data into a DataFrame."""
    records = []
    for entry in raw_stress:
        ds = entry.get("calendarDate", "")
        if not ds:
            continue
        records.append({
            "date": ds,
            "avg_stress": entry.get("overallStressLevel"),
            "max_stress": entry.get("maxStressLevel"),
            "rest_stress": entry.get("restStressDuration"),
            "low_stress_pct": entry.get("lowStressPercentage"),
            "medium_stress_pct": entry.get("mediumStressPercentage"),
            "high_stress_pct": entry.get("highStressPercentage"),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["stress_7day_avg"] = df["avg_stress"].rolling(7, min_periods=3).mean()
    return df


# ---------------------------------------------------------------------------
# Body Battery
# ---------------------------------------------------------------------------

def process_body_battery(raw_bb: list[dict]) -> pd.DataFrame:
    """Process Body Battery data into a DataFrame."""
    records = []
    for entry in raw_bb:
        ds = entry.get("calendarDate", "")
        data = entry.get("data", entry)
        if not ds:
            continue
        if isinstance(data, list) and data:
            values = [d.get("bodyBatteryLevel", 0) for d in data
                      if d.get("bodyBatteryLevel") is not None]
            if values:
                records.append({
                    "date": ds,
                    "bb_morning": values[0],
                    "bb_max": max(values),
                    "bb_min": min(values),
                    "bb_end_of_day": values[-1],
                    "bb_drain": values[0] - values[-1] if len(values) > 1 else None,
                })
        elif isinstance(data, dict):
            records.append({
                "date": ds,
                "bb_morning": data.get("bodyBatteryMorningValue"),
                "bb_max": data.get("bodyBatteryHighest"),
                "bb_min": data.get("bodyBatteryLowest"),
                "bb_end_of_day": None,
                "bb_drain": None,
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["bb_morning_7day"] = df["bb_morning"].rolling(7, min_periods=3).mean()
    return df


# ---------------------------------------------------------------------------
# Training Status
# ---------------------------------------------------------------------------

def process_training_status(raw_ts: list[dict]) -> pd.DataFrame:
    """Process Garmin Training Status data."""
    records = []
    for entry in raw_ts:
        ds = entry.get("calendarDate", "")
        if not ds:
            continue
        records.append({
            "date": ds,
            "training_status": entry.get("trainingStatusPhrase",
                                         entry.get("latestTrainingStatusPhrase")),
            "training_load_7day": entry.get("weeklyTrainingLoad",
                                            entry.get("latestTrainingLoad")),
            "vo2max_running": entry.get("mostRecentVO2Max",
                                        entry.get("latestVO2Max")),
            "load_focus": entry.get("trainingLoadBalancePhrase"),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Merge all sources
# ---------------------------------------------------------------------------

def merge_all_data(training_load_df, hrv_df, sleep_df, stress_df, bb_df, ts_df) -> pd.DataFrame:
    """Merge all processed DataFrames into a unified daily DataFrame."""
    merged = training_load_df.copy() if not training_load_df.empty else pd.DataFrame()
    if merged.empty:
        return merged

    for df, suffix in [
        (hrv_df, "_hrv"), (sleep_df, "_sleep"), (stress_df, "_stress"),
        (bb_df, "_bb"), (ts_df, "_ts"),
    ]:
        if df is not None and not df.empty:
            df_copy = df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["date"]).dt.normalize()
            merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
            merged = merged.merge(df_copy, on="date", how="left", suffixes=("", suffix))

    return merged
