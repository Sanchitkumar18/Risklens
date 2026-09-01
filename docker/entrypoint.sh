#!/usr/bin/env bash
# API container entrypoint: wait for the database, apply migrations, optionally seed
# demo data, then exec the container command (uvicorn).
set -euo pipefail

echo "[entrypoint] waiting for database..."
python - <<'PY'
import os, time
import sqlalchemy

url = os.environ["DATABASE_URL"]
for attempt in range(60):
    try:
        engine = sqlalchemy.create_engine(url)
        with engine.connect():
            pass
        print("[entrypoint] database is ready")
        break
    except Exception as exc:  # noqa: BLE001
        print(f"[entrypoint] db not ready ({attempt+1}/60): {exc}")
        time.sleep(2)
else:
    raise SystemExit("[entrypoint] database never became reachable")
PY

echo "[entrypoint] applying migrations..."
alembic upgrade head

if [ "${SEED_ON_START:-false}" = "true" ]; then
  echo "[entrypoint] seeding demo data..."
  python -m scripts.seed_demo || echo "[entrypoint] seed step failed (continuing)"
fi

echo "[entrypoint] starting: $*"
exec "$@"
