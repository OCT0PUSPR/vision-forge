"""Database engine/session management.

Creates a SQLAlchemy engine from ``DATABASE_URL`` (SQLite default). For the
default SQLite path the schema is created directly from the metadata; for other
backends use Alembic migrations (see ``alembic/``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker

    _HAS_SQLALCHEMY = True
except Exception:  # pragma: no cover
    _HAS_SQLALCHEMY = False


class Database:
    """Thin wrapper owning the engine + session factory."""

    def __init__(self, url: str = "sqlite:///./visionforge.db", echo: bool = False) -> None:
        if not _HAS_SQLALCHEMY:  # pragma: no cover
            raise RuntimeError(
                "SQLAlchemy is required for the persistence layer. Install it with: pip install 'sqlalchemy>=2.0'"
            )
        connect_args = {}
        if url.startswith("sqlite"):
            # Needed for SQLite when used across threads (FastAPI threadpool).
            connect_args = {"check_same_thread": False}
        self.url = url
        self.engine = create_engine(url, echo=echo, future=True, connect_args=connect_args)
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def create_all(self) -> None:
        """Create tables from the ORM metadata (dev / SQLite convenience)."""
        from visionforge.db.models import Base

        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        from visionforge.db.models import Base

        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Iterator["Session"]:
        """Context-managed session with commit/rollback handling."""
        session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()


_DB: Optional[Database] = None


def init_db(url: Optional[str] = None, create: bool = True) -> Database:
    """Initialize (or return) the process-wide database."""
    global _DB
    if _DB is None:
        from visionforge.config import get_settings

        resolved = url or get_settings().database_url
        _DB = Database(resolved)
        if create and resolved.startswith("sqlite"):
            _DB.create_all()
    return _DB


def get_db() -> Optional[Database]:
    return _DB


def reset_db() -> None:
    """Test helper: dispose and clear the global database."""
    global _DB
    if _DB is not None:
        _DB.dispose()
    _DB = None
