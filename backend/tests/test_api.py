"""
Integration tests for the Garmin Training Analysis API.

Tests all endpoints with an empty database (no Garmin credentials needed).
Verifies response shapes, status codes, and edge cases.

Run with:
    cd backend
    pytest tests/test_api.py -v
"""

import os
from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Override database to use an in-memory SQLite for tests
os.environ["DATABASE_URL"] = "sqlite:///./test_garmin.db"
os.environ["GARMIN_EMAIL"] = ""
os.environ["GARMIN_PASSWORD"] = ""
os.environ["SECRET_KEY"] = ""

from app.main import app
from app.db.engine import get_db, init_db
from app.models.database import (
    Base, Activity, DailyTrainingLoad, DailyHRV,
    DailySleep, DailyStress, DailyBodyBattery, CachedInsight,
)

# Test database setup
TEST_DB_URL = "sqlite:///./test_garmin.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def _seed_sample_data():
    """Insert realistic sample data for testing."""
    db = TestSession()
    today = date.today()

    # Activities
    for i in range(5):
        d = date(today.year, today.month, max(1, today.day - i * 2))
        db.add(Activity(
            activity_id=f"act_{i}",
            date=d,
            activity_name=f"Morning Run {i}",
            distance_km=12.0 + i,
            duration_min=60.0 + i * 5,
            pace_min_per_km=5.0 + i * 0.1,
            elevation_gain_m=100 + i * 20,
            avg_hr=145 + i,
            max_hr=170 + i,
            calories=700 + i * 50,
            aerobic_te=3.5,
            trimp=80 + i * 10,
        ))

    # Training load
    for i in range(30):
        d = date(today.year, today.month, max(1, today.day)) if i == 0 else \
            date.fromordinal(today.toordinal() - i)
        db.add(DailyTrainingLoad(
            date=d,
            daily_trimp=50 + (i % 7) * 20,
            daily_distance_km=10 + (i % 5) * 3,
            daily_duration_min=50 + (i % 5) * 10,
            num_activities=1 if (i % 7) != 0 else 0,
            atl=60 + i * 0.5,
            ctl=55 + i * 0.3,
            tsb=-5 + i * 0.2,
            weekly_distance_km=80 + i,
        ))

    # HRV
    for i in range(7):
        d = date.fromordinal(today.toordinal() - i)
        db.add(DailyHRV(
            date=d,
            hrv_value=55 + i,
            hrv_status="Balanced",
            hrv_baseline=52.0,
            weekly_avg=56.0,
            resting_hr=45 + (i % 3),
            hrv_7day_avg=57.0,
            rhr_7day_avg=46.0,
        ))

    # Sleep
    for i in range(7):
        d = date.fromordinal(today.toordinal() - i)
        db.add(DailySleep(
            date=d,
            sleep_score=75 + i,
            sleep_hours=7.5 + i * 0.1,
            deep_sleep_hours=1.5,
            light_sleep_hours=3.5,
            rem_sleep_hours=2.0,
            awake_hours=0.5,
            deep_sleep_pct=20.0,
            rem_sleep_pct=26.0,
            sleep_score_7day=78.0,
            sleep_hours_7day=7.8,
        ))

    # Stress
    for i in range(7):
        d = date.fromordinal(today.toordinal() - i)
        db.add(DailyStress(
            date=d,
            avg_stress=30 + i * 2,
            max_stress=60 + i,
            stress_7day_avg=35.0,
        ))

    # Body Battery
    for i in range(7):
        d = date.fromordinal(today.toordinal() - i)
        db.add(DailyBodyBattery(
            date=d,
            bb_morning=65 + i * 3,
            bb_max=90 - i,
            bb_min=30 + i,
            bb_morning_7day=70.0,
        ))

    db.commit()
    db.close()


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestHealth:
    def test_health_check(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestReadiness:
    def test_readiness_empty_db(self):
        """With no data, should still return a valid response."""
        r = client.get("/api/readiness/today")
        assert r.status_code == 200
        data = r.json()
        assert "readiness_score" in data
        assert "recommendation" in data
        assert "color" in data
        assert data["color"] in ("green", "yellow", "red")
        assert "factors" in data
        assert 0 <= data["readiness_score"] <= 100

    def test_readiness_with_data(self):
        _seed_sample_data()
        r = client.get("/api/readiness/today")
        assert r.status_code == 200
        data = r.json()
        assert data["readiness_score"] > 0
        assert data["factors"]["hrv"] is not None


class TestMetrics:
    def test_metrics_empty_db(self):
        r = client.get("/api/metrics?days=30")
        assert r.status_code == 200
        data = r.json()
        assert "dates" in data
        assert "atl" in data
        assert "tsb" in data
        assert isinstance(data["dates"], list)

    def test_metrics_with_data(self):
        _seed_sample_data()
        r = client.get("/api/metrics?days=30")
        assert r.status_code == 200
        data = r.json()
        assert len(data["dates"]) > 0
        assert len(data["atl"]) == len(data["dates"])
        assert len(data["hrv"]) == len(data["dates"])

    def test_metrics_custom_days(self):
        _seed_sample_data()
        r = client.get("/api/metrics?days=7")
        assert r.status_code == 200

    def test_metrics_invalid_days(self):
        r = client.get("/api/metrics?days=0")
        assert r.status_code == 422  # Validation error

    def test_metrics_too_many_days(self):
        r = client.get("/api/metrics?days=999")
        assert r.status_code == 422


class TestActivities:
    def test_activities_empty_db(self):
        r = client.get("/api/activities")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["activities"] == []

    def test_activities_with_data(self):
        _seed_sample_data()
        r = client.get("/api/activities?limit=3")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 5
        assert len(data["activities"]) == 3

        act = data["activities"][0]
        assert "activity_id" in act
        assert "date" in act
        assert "distance_km" in act
        assert "pace_min_per_km" in act
        assert "predicted_quality" in act
        assert act["predicted_quality"] in ("good", "moderate", "struggled")

    def test_activities_pagination(self):
        _seed_sample_data()
        r = client.get("/api/activities?limit=2&offset=2")
        assert r.status_code == 200
        data = r.json()
        assert len(data["activities"]) <= 2
        assert data["total"] == 5


class TestInsights:
    def test_insights_empty_db(self):
        r = client.get("/api/insights")
        assert r.status_code == 200
        data = r.json()
        assert "sweet_spots" in data
        assert "warnings" in data
        assert "current_status" in data

    def test_insights_with_cached_data(self):
        """Manually insert cached insights to simulate post-sync state."""
        import json
        db = TestSession()
        db.add(CachedInsight(
            key="insights",
            value_json=json.dumps({
                "sweet_spots": ["HRV > 55 for best performance"],
                "warnings": ["Performance drops after 10 days TSB < 0"],
                "current_status": "Balanced training zone",
            }),
            computed_at=datetime.utcnow(),
        ))
        db.commit()
        db.close()

        r = client.get("/api/insights")
        assert r.status_code == 200
        data = r.json()
        assert "HRV > 55" in data["sweet_spots"][0]
        assert "Balanced" in data["current_status"]


class TestPredictions:
    def test_predictions_empty_db(self):
        r = client.get("/api/predictions?days=3")
        assert r.status_code == 200
        data = r.json()
        assert len(data["predictions"]) == 3
        for p in data["predictions"]:
            assert "date" in p
            assert "predicted_readiness" in p
            assert "confidence" in p
            assert "recommended_workout_type" in p

    def test_predictions_with_data(self):
        _seed_sample_data()
        r = client.get("/api/predictions?days=7")
        assert r.status_code == 200
        data = r.json()
        assert len(data["predictions"]) == 7
        # Confidence should decrease over time
        confs = [p["confidence"] for p in data["predictions"]]
        assert confs[0] > confs[-1]

    def test_predictions_max_days(self):
        r = client.get("/api/predictions?days=14")
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 14

    def test_predictions_over_limit(self):
        r = client.get("/api/predictions?days=30")
        assert r.status_code == 422


class TestSyncEndpoint:
    def test_sync_status_never_synced(self):
        r = client.get("/api/auth/sync/status")
        assert r.status_code == 200
        assert r.json()["status"] == "never_synced"

    def test_sync_without_credentials(self):
        """Sync should fail with 400 when no credentials are configured."""
        r = client.post("/api/auth/sync", json={})
        assert r.status_code == 400


class TestResponseFormats:
    """Verify the exact shape of all response models."""

    def test_readiness_shape(self):
        _seed_sample_data()
        data = client.get("/api/readiness/today").json()
        assert isinstance(data["readiness_score"], (int, float))
        assert isinstance(data["recommendation"], str)
        assert isinstance(data["color"], str)
        assert isinstance(data["factors"], dict)
        for key in ["hrv", "tsb", "sleep_hours", "body_battery"]:
            assert key in data["factors"]

    def test_metrics_shape(self):
        _seed_sample_data()
        data = client.get("/api/metrics?days=7").json()
        array_keys = ["dates", "hrv", "atl", "ctl", "tsb", "sleep_score",
                       "sleep_hours", "body_battery", "avg_stress",
                       "daily_trimp", "daily_distance_km", "weekly_distance_km",
                       "resting_hr"]
        for key in array_keys:
            assert key in data, f"Missing key: {key}"
            assert isinstance(data[key], list), f"{key} should be a list"
        # All arrays same length
        lengths = [len(data[k]) for k in array_keys]
        assert len(set(lengths)) == 1, f"Mismatched array lengths: {dict(zip(array_keys, lengths))}"
