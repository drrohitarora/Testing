"""
SQLite database engine and session management.

Uses SQLAlchemy with a single-file SQLite database. WAL mode is enabled
for concurrent reads during background sync writes.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from app.config import get_settings

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for better concurrent read/write performance
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return _SessionLocal


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called once at application startup."""
    from app.models.database import Base
    Base.metadata.create_all(bind=get_engine())
