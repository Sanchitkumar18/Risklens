"""Generic repository base.

Repositories are the *only* code that issues database queries. Services depend on
repositories (not the ORM directly), which keeps SQL out of business logic and makes
the data layer swappable/mocked in tests.

Convention: repository methods ``flush`` (to assign primary keys / surface constraint
violations early) but do **not** ``commit``. Transaction boundaries are owned by the
caller (the request/service), so multiple repository calls compose into one atomic unit.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """CRUD helpers shared by all concrete repositories."""

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, id_: int) -> ModelT | None:
        """Return the row with the given primary key, or ``None``."""
        return self.session.get(self.model, id_)

    def list(self, *, limit: int | None = None, offset: int = 0) -> list[ModelT]:
        """Return rows in primary-key order, optionally paginated."""
        stmt = select(self.model).order_by(self.model.id).offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def add(self, obj: ModelT) -> ModelT:
        """Persist a new object and flush so its primary key is populated."""
        self.session.add(obj)
        self.session.flush()
        return obj

    def delete(self, obj: ModelT) -> None:
        """Mark an object for deletion and flush."""
        self.session.delete(obj)
        self.session.flush()
