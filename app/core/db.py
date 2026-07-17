"""Database foundation for SQLAlchemy models and sessions."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
	"""Base class for all SQLAlchemy models."""


_engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(
	autocommit=False,
	autoflush=False,
	bind=_engine,
)


def get_engine():
	"""Return the shared SQLAlchemy engine."""
	return _engine


def get_db() -> Generator[Session, None, None]:
	"""Yield a database session for FastAPI dependencies."""
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()


def create_tables() -> None:
	"""Create database tables for all registered models."""
	from app.infrastructure.auth.models import UserModel  # noqa: F401

	Base.metadata.create_all(bind=_engine)
