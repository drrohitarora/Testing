"""
POST /api/auth/sync — Trigger a Garmin data sync.

Launches the sync as a background task and returns immediately.
Rate-limited: only one sync can run at a time.
"""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.engine import get_db
from app.db.queries import create_sync_record, get_last_sync
from app.models.schemas import SyncRequest, SyncResponse
from app.tasks.sync import run_sync

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/auth/sync", response_model=SyncResponse)
def trigger_sync(
    body: SyncRequest | None = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """
    Start a Garmin data sync.

    If a sync is already running, returns 429. Otherwise queues a
    background task and returns immediately with the sync ID.
    """
    # Rate limit: block if a sync started within the last 5 minutes
    last = get_last_sync(db)
    if last and last.status == "in_progress":
        if last.started_at and (datetime.utcnow() - last.started_at) < timedelta(minutes=30):
            raise HTTPException(
                status_code=429,
                detail="A sync is already in progress. Please wait.",
            )

    settings = get_settings()
    email = (body.email if body and body.email else None) or settings.garmin_email
    password = (body.password if body and body.password else None) or settings.garmin_password
    months = (body.months if body and body.months else None) or settings.sync_months

    if not email or not password:
        raise HTTPException(
            status_code=400,
            detail="Garmin credentials not provided. Set GARMIN_EMAIL/GARMIN_PASSWORD or pass in request body.",
        )

    record = create_sync_record(db)

    background_tasks.add_task(run_sync, record.id, months, email, password)

    return SyncResponse(
        status="syncing",
        message="Sync started in the background. Poll this endpoint or check /api/readiness/today.",
        sync_id=record.id,
        last_sync_time=last.completed_at if last and last.status == "completed" else None,
    )


@router.get("/auth/sync/status", response_model=SyncResponse)
def sync_status(db: Session = Depends(get_db)):
    """Check the status of the most recent sync."""
    last = get_last_sync(db)
    if not last:
        return SyncResponse(status="never_synced", message="No sync has been run yet.")

    return SyncResponse(
        status=last.status,
        message=last.message or "",
        sync_id=last.id,
        activities_synced=last.activities_synced,
        last_sync_time=last.completed_at,
    )
