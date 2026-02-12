"""
Garmin Connect API client — adapted for backend use.

Changes from the CLI version:
- Session token cached in DB instead of a file
- Credentials can be passed directly or read from settings
- Password encrypted at rest using Fernet symmetric encryption
"""

import json
import logging
import time
from datetime import datetime, timedelta

from cryptography.fernet import Fernet, InvalidToken
from garminconnect import Garmin, GarminConnectAuthenticationError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.database import CachedInsight

logger = logging.getLogger(__name__)

GARMIN_SESSION_KEY = "garmin_session_token"


def encrypt_password(password: str) -> str:
    """Encrypt a password using the app's SECRET_KEY."""
    key = get_settings().secret_key
    if not key:
        raise ValueError("SECRET_KEY is required for password encryption.")
    f = Fernet(key.encode() if isinstance(key, str) else key)
    return f.encrypt(password.encode()).decode()


def decrypt_password(token: str) -> str:
    """Decrypt a previously encrypted password."""
    key = get_settings().secret_key
    f = Fernet(key.encode() if isinstance(key, str) else key)
    try:
        return f.decrypt(token.encode()).decode()
    except InvalidToken:
        raise ValueError("Failed to decrypt password — SECRET_KEY may have changed.")


def _get_cached_session(db: Session) -> dict | None:
    """Load a cached Garmin session token from the database."""
    row = db.query(CachedInsight).filter_by(key=GARMIN_SESSION_KEY).first()
    if row:
        try:
            return json.loads(row.value_json)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _save_cached_session(db: Session, session_data: dict):
    """Persist a Garmin session token to the database."""
    row = db.query(CachedInsight).filter_by(key=GARMIN_SESSION_KEY).first()
    value = json.dumps(session_data, default=str)
    if row:
        row.value_json = value
        row.computed_at = datetime.utcnow()
    else:
        db.add(CachedInsight(
            key=GARMIN_SESSION_KEY,
            value_json=value,
            computed_at=datetime.utcnow(),
        ))
    db.commit()


def authenticate(db: Session, email: str | None = None, password: str | None = None) -> Garmin:
    """
    Authenticate with Garmin Connect.

    Tries cached session first, then falls back to fresh login.
    Accepts explicit credentials or reads from settings.
    """
    settings = get_settings()
    email = email or settings.garmin_email
    password = password or settings.garmin_password

    if not email or not password:
        raise ValueError(
            "Garmin credentials not configured. Set GARMIN_EMAIL and "
            "GARMIN_PASSWORD in your .env file."
        )

    client = Garmin(email, password)

    # Try cached session
    cached = _get_cached_session(db)
    if cached:
        try:
            client.login(cached)
            logger.info("Restored cached Garmin session.")
            return client
        except Exception:
            logger.info("Cached session expired, performing fresh login.")

    try:
        client.login()
        if client.session_data:
            _save_cached_session(db, client.session_data)
        logger.info("Fresh Garmin login successful.")
        return client
    except GarminConnectAuthenticationError as e:
        raise RuntimeError(f"Garmin authentication failed: {e}") from e


# ---------------------------------------------------------------------------
# Data fetching functions — unchanged logic from garmin_client.py
# ---------------------------------------------------------------------------

def fetch_activities(client: Garmin, months: int = 6) -> list[dict]:
    """Fetch running activities for the last N months."""
    start_date = datetime.now() - timedelta(days=months * 30)
    all_activities = []
    page = 0
    page_size = 200

    while True:
        activities = client.get_activities(page * page_size, page_size)
        if not activities:
            break

        for act in activities:
            act_type = (act.get("activityType", {}).get("typeKey", "") or "").lower()
            if "running" not in act_type:
                continue

            act_start = act.get("startTimeLocal", "")
            if act_start:
                try:
                    act_date = datetime.strptime(act_start, "%Y-%m-%d %H:%M:%S")
                    if act_date < start_date:
                        return all_activities
                except ValueError:
                    pass

            all_activities.append(act)

        page += 1
        if page > 50:
            break

    return all_activities


def fetch_activity_hr_zones(client: Garmin, activity_id) -> list:
    """Fetch HR zone breakdown for one activity."""
    try:
        return client.get_activity_hr_in_timezones(activity_id)
    except Exception as e:
        logger.warning("Failed to fetch HR zones for %s: %s", activity_id, e)
        return []


def fetch_hrv_data(client: Garmin, days: int = 180) -> list[dict]:
    """Fetch daily HRV status for the last N days."""
    results = []
    end = datetime.now()
    current = end - timedelta(days=days)

    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        try:
            hrv = client.get_hrv_data(ds)
            if hrv:
                hrv["calendarDate"] = ds
                results.append(hrv)
        except Exception:
            pass
        current += timedelta(days=1)

    return results


def fetch_sleep_data(client: Garmin, days: int = 180) -> list[dict]:
    """Fetch nightly sleep data for the last N days."""
    results = []
    end = datetime.now()
    current = end - timedelta(days=days)

    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        try:
            sleep = client.get_sleep_data(ds)
            if sleep:
                results.append(sleep)
        except Exception:
            pass
        current += timedelta(days=1)

    return results


def fetch_stress_data(client: Garmin, days: int = 180) -> list[dict]:
    """Fetch daily stress level summaries for the last N days."""
    results = []
    end = datetime.now()
    current = end - timedelta(days=days)

    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        try:
            stress = client.get_stress_data(ds)
            if stress:
                if isinstance(stress, dict):
                    stress["calendarDate"] = ds
                results.append(stress)
        except Exception:
            pass
        current += timedelta(days=1)

    return results


def fetch_body_battery(client: Garmin, days: int = 180) -> list[dict]:
    """Fetch daily Body Battery readings for the last N days."""
    results = []
    end = datetime.now()
    current = end - timedelta(days=days)

    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        try:
            bb = client.get_body_battery(ds)
            if bb:
                results.append({"calendarDate": ds, "data": bb})
        except Exception:
            pass
        current += timedelta(days=1)

    return results


def fetch_training_status(client: Garmin, days: int = 180) -> list[dict]:
    """Fetch Garmin Training Status / Training Load for the last N days."""
    results = []
    end = datetime.now()
    current = end - timedelta(days=days)

    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        try:
            ts = client.get_training_status(ds)
            if ts:
                if isinstance(ts, dict):
                    ts["calendarDate"] = ds
                results.append(ts)
        except Exception:
            pass
        current += timedelta(days=1)

    return results
