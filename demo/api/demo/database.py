"""Database connection and session management."""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base

from demo.config import settings


if settings.POSTGRES_URL:
    logging.info("Using PostgreSQL")
    engine = create_engine(
        settings.POSTGRES_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )
else:
    logging.info("Using SQLite for local development")
    engine = create_engine(
        "sqlite:///./sessions.db",
        connect_args={"check_same_thread": False},
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Get a database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
