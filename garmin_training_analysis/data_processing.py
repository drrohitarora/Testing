"""
Data Processing & Training Metrics
====================================
Transforms raw Garmin API data into structured DataFrames and calculates
endurance-specific training metrics.

Key Concepts for Ultramarathon Runners:
---------------------------------------
- TRIMP (Training Impulse): Quantifies training load by weighting duration
  by heart rate intensity. Higher TRIMP = more physiological stress.

- ATL (Acute Training Load): 7-day exponentially weighted TRIMP average.
  Represents recent fatigue / fitness stimulus.

- CTL (Chronic Training Load): 42-day exponentially weighted TRIMP average.
  Represents accumulated fitness. (We use 28-day for faster responsiveness
  given the high-volume nature of ultra training.)

- TSB (Training Stress Balance): CTL - ATL. Positive = fresh/rested,
  negative = fatigued. Ultra runners typically train at -10 to -30 TSB
  and taper to +5 to +15 before races.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activity data processing
# ---------------------------------------------------------------------------

def process_activities(raw_activities, hr_zones_map=None):
    """
    Convert raw Garmin activity dicts into a clean DataFrame.

    Extracts the fields most relevant for endurance training analysis:
    distance, duration, pace, elevation, heart rate, and TRIMP.

    Args:
        raw_activities: List of activity dicts from the Garmin API.
        hr_zones_map: Optional dict mapping activity_id -> HR zone data.

    Returns:
        pd.DataFrame with one row per activity, indexed by date.
    """
    records = []

    for act in raw_activities:
        activity_id = act.get("activityId")

        # Parse date
        start_str = act.get("startTimeLocal", "")
        try:
            date = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").date()
        except (ValueError, TypeError):
            continue

        # Distance in km (API returns meters)
        distance_m = act.get("distance", 0) or 0
        distance_km = distance_m / 1000.0

        # Duration in minutes (API returns seconds)
        duration_s = act.get("duration", 0) or 0
        duration_min = duration_s / 60.0

        # Pace: min/km
        pace_min_per_km = duration_min / distance_km if distance_km > 0 else None

        # Elevation gain in meters
        elevation_gain = act.get("elevationGain", 0) or 0

        # Heart rate stats
        avg_hr = act.get("averageHR", None)
        max_hr = act.get("maxHR", None)

        # Calories (used as a proxy for effort when HR data is missing)
        calories = act.get("calories", 0) or 0

        # Garmin's own training effect scores (aerobic and anaerobic)
        aerobic_te = act.get("aerobicTrainingEffect", None)
        anaerobic_te = act.get("anaerobicTrainingEffect", None)

        # VO2 Max estimate from Garmin
        vo2max = act.get("vO2MaxValue", None)

        # Calculate TRIMP (Training Impulse)
        # Banister's TRIMP = duration(min) * delta_HR_ratio * weighting
        # delta_HR_ratio = (avg_HR - resting_HR) / (max_HR - resting_HR)
        # We estimate resting HR as 45 bpm (typical for ultra runners)
        # and use the session max HR if individual max is unknown.
        trimp = _calculate_trimp(avg_hr, max_hr, duration_min)

        records.append({
            "date": date,
            "activity_id": activity_id,
            "distance_km": round(distance_km, 2),
            "duration_min": round(duration_min, 1),
            "pace_min_per_km": round(pace_min_per_km, 2) if pace_min_per_km else None,
            "elevation_gain_m": round(elevation_gain, 0),
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "calories": calories,
            "aerobic_te": aerobic_te,
            "anaerobic_te": anaerobic_te,
            "vo2max": vo2max,
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
    Calculate Banister's TRIMP (Training Impulse).

    TRIMP = duration * delta_HR_ratio * 0.64 * e^(1.92 * delta_HR_ratio)

    The exponential weighting means time spent at high intensity contributes
    disproportionately more to training load — which matters for ultra
    runners who do long easy runs punctuated by tempo efforts.

    Args:
        avg_hr: Average heart rate during activity.
        max_hr: Max heart rate during activity (or estimated max).
        duration_min: Activity duration in minutes.
        resting_hr: Assumed resting heart rate (45 bpm for trained ultra runners).

    Returns:
        float: TRIMP value, or None if HR data is missing.
    """
    if not avg_hr or not duration_min:
        return None

    # If we don't have a session max HR, estimate from age-generic formula
    if not max_hr or max_hr <= avg_hr:
        max_hr = 190  # Conservative estimate; user should calibrate

    if max_hr <= resting_hr:
        return None

    delta_hr_ratio = (avg_hr - resting_hr) / (max_hr - resting_hr)
    delta_hr_ratio = max(0.0, min(1.0, delta_hr_ratio))  # Clamp to [0, 1]

    # Banister's formula (male coefficients; the gender difference is small)
    trimp = duration_min * delta_hr_ratio * 0.64 * np.exp(1.92 * delta_hr_ratio)
    return trimp


# ---------------------------------------------------------------------------
# Training load metrics (ATL / CTL / TSB)
# ---------------------------------------------------------------------------

def calculate_training_load(activities_df, atl_days=7, ctl_days=28):
    """
    Calculate Acute Training Load, Chronic Training Load, and Training
    Stress Balance from daily TRIMP values.

    For ultramarathon training at ~100 km/week:
    - ATL (7-day): Captures recent training block fatigue. A sudden spike
      above 2x your CTL is a red flag for overreaching.
    - CTL (28-day): Your fitness baseline. For 100km/week runners, CTL
      typically ranges from 80-150 depending on intensity mix.
    - TSB = CTL - ATL: Target -10 to -30 during build phases, +5 to +15
      for race tapers. Deeply negative TSB (< -40) risks overtraining.

    We use 28 days for CTL instead of the traditional 42 because ultra
    runners' high weekly volume makes the 42-day window too sluggish
    to detect meaningful fitness changes.

    Args:
        activities_df: DataFrame with 'date' and 'trimp' columns.
        atl_days: ATL window (default 7).
        ctl_days: CTL window (default 28).

    Returns:
        pd.DataFrame: Daily training load metrics indexed by date.
    """
    if activities_df.empty:
        return pd.DataFrame()

    # Aggregate TRIMP by day (sum multiple activities on the same day)
    daily = activities_df.groupby(activities_df["date"].dt.date).agg(
        daily_trimp=("trimp", "sum"),
        daily_distance_km=("distance_km", "sum"),
        daily_duration_min=("duration_min", "sum"),
        num_activities=("activity_id", "count"),
    ).reset_index()
    daily.columns = ["date", "daily_trimp", "daily_distance_km",
                     "daily_duration_min", "num_activities"]
    daily["date"] = pd.to_datetime(daily["date"])

    # Create a continuous date range (fill rest days with 0 TRIMP)
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(date_range).fillna(0)
    daily.index.name = "date"
    daily = daily.reset_index()

    # Exponentially weighted moving averages
    # EWM decay factor: alpha = 2 / (N + 1)
    daily["atl"] = daily["daily_trimp"].ewm(span=atl_days, adjust=False).mean()
    daily["ctl"] = daily["daily_trimp"].ewm(span=ctl_days, adjust=False).mean()

    # Training Stress Balance = fitness (CTL) minus fatigue (ATL)
    daily["tsb"] = daily["ctl"] - daily["atl"]

    # Rolling weekly distance (useful context for ultra runners)
    daily["weekly_distance_km"] = daily["daily_distance_km"].rolling(7, min_periods=1).sum()

    return daily


# ---------------------------------------------------------------------------
# HRV data processing
# ---------------------------------------------------------------------------

def process_hrv_data(raw_hrv):
    """
    Process raw HRV API responses into a clean DataFrame.

    Key HRV metrics for endurance athletes:
    - HRV Status: Garmin's own classification (Balanced, Low, etc.)
    - Weekly Average: Smoothed HRV trend (more meaningful than daily values)
    - Baseline: Your personal HRV baseline calculated by Garmin

    A sustained drop in HRV below your baseline typically precedes
    illness, overtraining, or poor performance by 2-5 days.

    Args:
        raw_hrv: List of daily HRV dicts from the API.

    Returns:
        pd.DataFrame indexed by date.
    """
    records = []
    for entry in raw_hrv:
        date_str = entry.get("calendarDate", "")
        if not date_str:
            continue

        # The HRV summary sits in different locations depending on
        # the device and API version
        summary = entry.get("hrvSummary", entry)

        records.append({
            "date": date_str,
            "hrv_value": summary.get("lastNightAvg", None),
            "hrv_status": summary.get("status", None),
            "hrv_baseline": summary.get("baseline", {}).get("lowUpper", None)
                if isinstance(summary.get("baseline"), dict) else None,
            "weekly_avg": summary.get("weeklyAvg", None),
            "resting_hr": summary.get("restingHeartRate", None),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Calculate rolling averages for HRV and resting HR
    df["hrv_7day_avg"] = df["hrv_value"].rolling(7, min_periods=3).mean()
    df["rhr_7day_avg"] = df["resting_hr"].rolling(7, min_periods=3).mean()

    return df


# ---------------------------------------------------------------------------
# Sleep data processing
# ---------------------------------------------------------------------------

def process_sleep_data(raw_sleep):
    """
    Process raw sleep API responses into a clean DataFrame.

    For ultra runners doing 100+ km/week, sleep is the #1 recovery tool.
    Key metrics:
    - Sleep Score: Garmin's composite quality score (0-100)
    - Deep Sleep: Most restorative phase; critical for tissue repair
    - REM Sleep: Important for motor learning and cognitive recovery
    - Total Duration: Most ultra runners need 8-9h for adequate recovery

    Args:
        raw_sleep: List of nightly sleep dicts from the API.

    Returns:
        pd.DataFrame indexed by date.
    """
    records = []
    for entry in raw_sleep:
        daily = entry.get("dailySleepDTO", entry)
        date_str = daily.get("calendarDate", "")
        if not date_str:
            continue

        # Duration in hours (API returns seconds)
        sleep_seconds = daily.get("sleepTimeSeconds", 0) or 0
        sleep_hours = sleep_seconds / 3600.0

        # Stage durations in hours
        deep_seconds = daily.get("deepSleepSeconds", 0) or 0
        light_seconds = daily.get("lightSleepSeconds", 0) or 0
        rem_seconds = daily.get("remSleepSeconds", 0) or 0
        awake_seconds = daily.get("awakeSleepSeconds", 0) or 0

        records.append({
            "date": date_str,
            "sleep_score": daily.get("sleepScores", {}).get("overall", {}).get("value", None)
                if isinstance(daily.get("sleepScores"), dict) else None,
            "sleep_hours": round(sleep_hours, 2),
            "deep_sleep_hours": round(deep_seconds / 3600.0, 2),
            "light_sleep_hours": round(light_seconds / 3600.0, 2),
            "rem_sleep_hours": round(rem_seconds / 3600.0, 2),
            "awake_hours": round(awake_seconds / 3600.0, 2),
            "deep_sleep_pct": round(deep_seconds / sleep_seconds * 100, 1) if sleep_seconds > 0 else 0,
            "rem_sleep_pct": round(rem_seconds / sleep_seconds * 100, 1) if sleep_seconds > 0 else 0,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Rolling averages
    df["sleep_score_7day"] = df["sleep_score"].rolling(7, min_periods=3).mean()
    df["sleep_hours_7day"] = df["sleep_hours"].rolling(7, min_periods=3).mean()

    return df


# ---------------------------------------------------------------------------
# Stress data processing
# ---------------------------------------------------------------------------

def process_stress_data(raw_stress):
    """
    Process daily stress data into a DataFrame.

    Garmin stress scores range from 0 (rest) to 100 (high stress).
    For endurance athletes, average daily stress above 40-50 sustained
    over a week often correlates with inadequate recovery.

    Args:
        raw_stress: List of daily stress dicts.

    Returns:
        pd.DataFrame indexed by date.
    """
    records = []
    for entry in raw_stress:
        date_str = entry.get("calendarDate", "")
        if not date_str:
            continue

        records.append({
            "date": date_str,
            "avg_stress": entry.get("overallStressLevel", None),
            "max_stress": entry.get("maxStressLevel", None),
            "rest_stress": entry.get("restStressDuration", None),
            "low_stress_pct": entry.get("lowStressPercentage", None),
            "medium_stress_pct": entry.get("mediumStressPercentage", None),
            "high_stress_pct": entry.get("highStressPercentage", None),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["stress_7day_avg"] = df["avg_stress"].rolling(7, min_periods=3).mean()

    return df


# ---------------------------------------------------------------------------
# Body Battery processing
# ---------------------------------------------------------------------------

def process_body_battery(raw_bb):
    """
    Process Body Battery data into a DataFrame.

    Body Battery ranges from 0-100 and combines HRV, stress, sleep,
    and activity data. For ultra runners:
    - Morning Body Battery < 30: Consider an easy/rest day
    - Morning Body Battery > 70: Good to go for quality sessions
    - Consistently low (<40) Body Battery: Overtraining warning sign

    Args:
        raw_bb: List of daily Body Battery dicts.

    Returns:
        pd.DataFrame indexed by date.
    """
    records = []
    for entry in raw_bb:
        date_str = entry.get("calendarDate", "")
        data = entry.get("data", entry)

        if not date_str:
            continue

        # Body Battery data can be a list of timestamped readings
        # or a summary dict depending on the API endpoint
        if isinstance(data, list) and data:
            values = [d.get("bodyBatteryLevel", 0) for d in data
                      if d.get("bodyBatteryLevel") is not None]
            if values:
                records.append({
                    "date": date_str,
                    "bb_morning": values[0] if values else None,  # First reading ≈ wakeup
                    "bb_max": max(values),
                    "bb_min": min(values),
                    "bb_end_of_day": values[-1] if values else None,
                    "bb_drain": values[0] - values[-1] if len(values) > 1 else None,
                })
        elif isinstance(data, dict):
            records.append({
                "date": date_str,
                "bb_morning": data.get("bodyBatteryMorningValue", None),
                "bb_max": data.get("bodyBatteryHighest", None),
                "bb_min": data.get("bodyBatteryLowest", None),
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
# Training Status processing
# ---------------------------------------------------------------------------

def process_training_status(raw_ts):
    """
    Process Garmin Training Status / Training Load data.

    Training Status categories:
    - Peaking: Ideal race readiness
    - Productive: Fitness improving
    - Maintaining: Fitness stable
    - Recovery: Reduced load, recovering
    - Unproductive: Training hard but not improving (often from poor recovery)
    - Detraining: Fitness declining from insufficient stimulus
    - Overreaching: Excessive load relative to recovery

    For ultra runners, 'Unproductive' often signals that volume is adequate
    but recovery (sleep, nutrition, stress management) needs attention.

    Args:
        raw_ts: List of training status dicts.

    Returns:
        pd.DataFrame indexed by date.
    """
    records = []
    for entry in raw_ts:
        date_str = entry.get("calendarDate", "")
        if not date_str:
            continue

        records.append({
            "date": date_str,
            "training_status": entry.get("trainingStatusPhrase",
                                         entry.get("latestTrainingStatusPhrase", None)),
            "training_load_7day": entry.get("weeklyTrainingLoad",
                                            entry.get("latestTrainingLoad", None)),
            "vo2max_running": entry.get("mostRecentVO2Max",
                                        entry.get("latestVO2Max", None)),
            "load_focus": entry.get("trainingLoadBalancePhrase", None),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Merge all data sources
# ---------------------------------------------------------------------------

def merge_all_data(training_load_df, hrv_df, sleep_df, stress_df, bb_df, ts_df):
    """
    Merge all processed DataFrames into a single unified DataFrame,
    joined on date.

    This master DataFrame is the foundation for all visualizations
    and pattern analysis. Each row represents one day with all available
    metrics from every data source.

    Returns:
        pd.DataFrame: Unified daily metrics.
    """
    # Start with training load (has the continuous date range)
    merged = training_load_df.copy() if not training_load_df.empty else pd.DataFrame()

    if merged.empty:
        return merged

    merge_date_col = "date"

    for df, suffix in [
        (hrv_df, "_hrv"),
        (sleep_df, "_sleep"),
        (stress_df, "_stress"),
        (bb_df, "_bb"),
        (ts_df, "_ts"),
    ]:
        if df is not None and not df.empty:
            df_copy = df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["date"]).dt.normalize()
            merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
            merged = merged.merge(df_copy, on=merge_date_col, how="left",
                                  suffixes=("", suffix))

    return merged
