"""
SQLAlchemy engine, session factory and declarative base with SQLite fallback.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def create_app_engine():
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite"):
        return create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            future=True,
        )

    try:
        eng = create_engine(db_url, pool_pre_ping=True, future=True)
        # Verify connection
        with eng.connect():
            pass
        return eng
    except Exception as exc:
        if settings.USE_SQLITE_FALLBACK:
            fallback_url = f"sqlite:///{settings.SQLITE_DB_PATH}"
            logger.warning(
                "PostgreSQL not accessible (%s). Falling back to SQLite: %s",
                exc,
                fallback_url,
            )
            return create_engine(
                fallback_url,
                connect_args={"check_same_thread": False},
                future=True,
            )
        raise


engine = create_app_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and guarantees closure."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
