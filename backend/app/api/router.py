"""
Top-level API router that includes all sub-routers.
"""

from fastapi import APIRouter

from app.api.sync import router as sync_router
from app.api.readiness import router as readiness_router
from app.api.metrics import router as metrics_router
from app.api.activities import router as activities_router
from app.api.insights import router as insights_router
from app.api.predictions import router as predictions_router

api_router = APIRouter(prefix="/api")

api_router.include_router(sync_router, tags=["Sync"])
api_router.include_router(readiness_router, tags=["Readiness"])
api_router.include_router(metrics_router, tags=["Metrics"])
api_router.include_router(activities_router, tags=["Activities"])
api_router.include_router(insights_router, tags=["Insights"])
api_router.include_router(predictions_router, tags=["Predictions"])
