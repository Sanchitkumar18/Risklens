"""FastAPI dependency providers.

Centralizes the wiring between the request layer and lower layers so routes stay
thin and services receive their collaborators via dependency injection. As services
are added in later phases they are provided here too.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.database import get_db as _get_db


def get_db_session() -> Iterator[Session]:
    """Yield a database session for the lifetime of a request."""
    yield from _get_db()


# Re-exported alias so routes can depend on the name they expect.
DbSession = Depends(get_db_session)
