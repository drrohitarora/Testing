"""
Garmin Connect API Client
=========================
Handles authentication and raw data fetching from Garmin Connect.

Uses the 'garminconnect' library to interface with Garmin's API.
Credentials are loaded from environment variables or a .env config file
to avoid hardcoding sensitive information.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from garminconnect import Garmin, GarminConnectAuthenticationError

logger = logging.getLogger(__name__)


def load_credentials():
    """
    Load Garmin credentials from environment variables or .env file.

    Priority:
      1. GARMIN_EMAIL / GARMIN_PASSWORD environment variables
      2. .env file in project root
      3. config.json in project root

    Returns:
        tuple: (email, password)

    Raises:
        ValueError: If credentials cannot be found.
    """
    email = os.environ.get("GARMIN_EMAIL")
    password = os.environ.get("GARMIN_PASSWORD")

    if email and password:
        return email, password

    # Try .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if key == "GARMIN_EMAIL":
                    email = value
                elif key == "GARMIN_PASSWORD":
                    password = value
        if email and password:
            return email, password

    # Try config.json
    config_path = Path(__file__).parent.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            email = config.get("email")
            password = config.get("password")
        if email and password:
            return email, password

    raise ValueError(
        "Garmin credentials not found. Set GARMIN_EMAIL and GARMIN_PASSWORD "
        "environment variables, or create a .env or config.json file."
    )


# Path where we cache the Garmin session token so we don't re-authenticate
# on every run (Garmin rate-limits logins aggressively).
SESSION_CACHE = Path(__file__).parent.parent / ".garmin_session"


def authenticate():
    """
    Authenticate with Garmin Connect, reusing a cached session if available.

    Garmin enforces strict rate limits on login attempts. This function
    caches the session token to disk after a successful login and reuses
    it on subsequent calls.

    Returns:
        Garmin: An authenticated Garmin API client.
    """
    email, password = load_credentials()
    client = Garmin(email, password)

    # Attempt to restore a cached session first
    if SESSION_CACHE.exists():
        try:
            logger.info("Restoring cached Garmin session...")
            with open(SESSION_CACHE, "r") as f:
                session_data = json.load(f)
            client.login(session_data)
            logger.info("Session restored successfully.")
            return client
        except Exception:
            logger.info("Cached session expired, performing fresh login...")

    # Fresh login
    try:
        client.login()
        # Cache the session for next time
        session_data = client.session_data
        if session_data:
            with open(SESSION_CACHE, "w") as f:
                json.dump(session_data, f)
            logger.info("Session cached to %s", SESSION_CACHE)
        return client
    except GarminConnectAuthenticationError as e:
        raise RuntimeError(
            f"Garmin authentication failed: {e}. "
            "Check your email/password and ensure MFA is handled."
        ) from e


def fetch_activities(client, months=6, activity_type="running"):
    """
    Fetch running activities for the last N months.

    The Garmin API returns activities in pages. We request up to 200
    at a time (the max per call) and filter for running-type activities.

    Args:
        client: Authenticated Garmin client.
        months: Number of months of history to pull.
        activity_type: Filter string for activity type.

    Returns:
        list[dict]: Raw activity summaries from Garmin.
    """
    start_date = datetime.now() - timedelta(days=months * 30)
    all_activities = []
    page = 0
    page_size = 200

    while True:
        logger.info("Fetching activities page %d...", page)
        activities = client.get_activities(page * page_size, page_size)

        if not activities:
            break

        for act in activities:
            # Filter to running activities only
            act_type = (act.get("activityType", {}).get("typeKey", "") or "").lower()
            if activity_type.lower() not in act_type:
                continue

            act_start = act.get("startTimeLocal", "")
            if act_start:
                try:
                    act_date = datetime.strptime(act_start, "%Y-%m-%d %H:%M:%S")
                    if act_date < start_date:
                        # We've gone past our time window, stop
                        return all_activities
                except ValueError:
                    pass

            all_activities.append(act)

        page += 1

        # Safety valve: don't loop forever
        if page > 50:
            break

    return all_activities


def fetch_activity_details(client, activity_id):
    """
    Fetch detailed data for a single activity, including HR zones and splits.

    Args:
        client: Authenticated Garmin client.
        activity_id: The Garmin activity ID.

    Returns:
        dict: Detailed activity data.
    """
    try:
        return client.get_activity(activity_id)
    except Exception as e:
        logger.warning("Failed to fetch details for activity %s: %s", activity_id, e)
        return {}


def fetch_activity_hr_zones(client, activity_id):
    """
    Fetch heart rate zone data for a specific activity.

    HR zones are critical for calculating training impulse (TRIMP)
    and understanding intensity distribution.

    Returns:
        list[dict]: Heart rate zone breakdown with time-in-zone.
    """
    try:
        return client.get_activity_hr_in_timezones(activity_id)
    except Exception as e:
        logger.warning("Failed to fetch HR zones for activity %s: %s", activity_id, e)
        return []


def fetch_hrv_data(client, days=180):
    """
    Fetch daily HRV (Heart Rate Variability) status and measurements.

    HRV is measured overnight by Garmin devices. Higher HRV generally
    indicates better recovery and readiness to train. For ultramarathon
    runners, tracking HRV trends is essential for avoiding overtraining.

    Args:
        client: Authenticated Garmin client.
        days: Number of days of history.

    Returns:
        list[dict]: Daily HRV summaries.
    """
    results = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        try:
            hrv = client.get_hrv_data(date_str)
            if hrv:
                hrv["calendarDate"] = date_str
                results.append(hrv)
        except Exception as e:
            logger.debug("No HRV data for %s: %s", date_str, e)
        current += timedelta(days=1)

    return results


def fetch_sleep_data(client, days=180):
    """
    Fetch nightly sleep data including duration, stages, and quality scores.

    Sleep quality directly impacts recovery and next-day performance.
    Garmin tracks REM, deep, and light sleep stages along with
    a proprietary sleep score.

    Args:
        client: Authenticated Garmin client.
        days: Number of days of history.

    Returns:
        list[dict]: Nightly sleep summaries.
    """
    results = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        try:
            sleep = client.get_sleep_data(date_str)
            if sleep:
                results.append(sleep)
        except Exception as e:
            logger.debug("No sleep data for %s: %s", date_str, e)
        current += timedelta(days=1)

    return results


def fetch_stress_data(client, days=180):
    """
    Fetch daily stress level summaries.

    Garmin calculates stress from HRV data throughout the day.
    Chronic high stress scores correlate with reduced recovery
    and increased injury risk for high-mileage runners.

    Args:
        client: Authenticated Garmin client.
        days: Number of days of history.

    Returns:
        list[dict]: Daily stress summaries.
    """
    results = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        try:
            stress = client.get_stress_data(date_str)
            if stress:
                if isinstance(stress, dict):
                    stress["calendarDate"] = date_str
                results.append(stress)
        except Exception as e:
            logger.debug("No stress data for %s: %s", date_str, e)
        current += timedelta(days=1)

    return results


def fetch_body_battery(client, days=180):
    """
    Fetch Body Battery data.

    Body Battery is Garmin's proprietary metric combining HRV, stress,
    sleep, and activity data into a 0-100 energy score. For ultra
    runners, low morning Body Battery readings often predict poor
    training sessions.

    Args:
        client: Authenticated Garmin client.
        days: Number of days of history.

    Returns:
        list[dict]: Daily Body Battery summaries.
    """
    results = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        try:
            bb = client.get_body_battery(date_str)
            if bb:
                results.append({"calendarDate": date_str, "data": bb})
        except Exception as e:
            logger.debug("No Body Battery data for %s: %s", date_str, e)
        current += timedelta(days=1)

    return results


def fetch_training_status(client, days=180):
    """
    Fetch Garmin's Training Status and Training Load metrics.

    Training Status combines VO2 Max trends, training load, and HRV
    to classify your current state (Productive, Maintaining, Overreaching, etc.).
    Training Load is Garmin's 7-day EPOC-based load metric.

    Args:
        client: Authenticated Garmin client.
        days: Number of days of history.

    Returns:
        list[dict]: Training status/load records.
    """
    results = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        try:
            ts = client.get_training_status(date_str)
            if ts:
                if isinstance(ts, dict):
                    ts["calendarDate"] = date_str
                results.append(ts)
        except Exception as e:
            logger.debug("No training status for %s: %s", date_str, e)
        current += timedelta(days=1)

    return results
