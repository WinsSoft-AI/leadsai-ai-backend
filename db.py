"""
Winssoft BMA — AI_Backend Database Utilities  v3.1
==================================================
Lightweight DB module for the AI Worker service.

The AI_Backend does NOT own or manage the database schema.
All tables are created and seeded by backend/db_init.py (the main orchestrator).

This file provides:
  - get_pool()   → shared asyncpg connection pool
  - close_pool() → cleanup on shutdown
  - init_db()    → verifies DB connectivity (no schema creation)

The only table the AI_Backend writes to is `ingest_jobs`
(tracked by rag_engine.py for background document ingestion).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

try:
    import asyncpg
except ImportError:
    logger.error("asyncpg not installed — run: pip install asyncpg")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
# DSN helper — strips SQLAlchemy prefix so asyncpg can use it directly
# ═════════════════════════════════════════════════════════════════════════════

def _dsn() -> str:
    raw = os.environ.get("DATABASE_URL", "")
    if not raw:
        logger.error("DATABASE_URL is not set in .env")
        sys.exit(1)
    return raw.replace("postgresql+asyncpg://", "postgresql://") \
               .replace("postgres+asyncpg://",   "postgresql://")


# ═════════════════════════════════════════════════════════════════════════════
# CONNECTION POOL (module-level singleton)
# ═════════════════════════════════════════════════════════════════════════════

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Return (or lazily create) the shared asyncpg connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=_dsn(),
            min_size=int(os.getenv("DB_POOL_MIN",     "2")),
            max_size=int(os.getenv("DB_POOL_MAX",     "10")),
            command_timeout=float(os.getenv("DB_POOL_TIMEOUT", "30")),
            init=_init_conn,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def _init_conn(conn: asyncpg.Connection) -> None:
    """Register JSONB codec so asyncpg auto-encodes/decodes Python dicts."""
    import json
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )

# ═════════════════════════════════════════════════════════════════════════════
# TENANT SCHEMA HELPERS
# ═════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def tenant_conn(tenant_id: str):
    """
    Acquire a connection with search_path set to the tenant's schema.
    Queries like 'SELECT * FROM sessions' resolve to t_{tenant_id}.sessions.
    Queries like 'SELECT * FROM tenants' resolve to public.tenants.
    search_path is reset on exit.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        schema = f"t_{tenant_id}"
        await conn.execute(f'SET search_path TO "{schema}", public')
        try:
            yield conn
        finally:
            await conn.execute("SET search_path TO public")


# ═════════════════════════════════════════════════════════════════════════════
# INIT — connectivity check only (schema managed by main backend)
# ═════════════════════════════════════════════════════════════════════════════

async def init_db(reset: bool = False, seed: bool = True) -> None:
    """
    Verify DB connectivity.
    The `reset` and `seed` params are kept for interface compatibility
    with main.py's lifespan call but are no-ops here.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        ver = await conn.fetchval("SELECT version()")
        logger.info(f"✅ AI_Backend DB connected — {ver[:40]}…")

        # Verify the one table we write to exists
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='ingest_jobs')"
        )
        if exists:
            logger.info("✅ ingest_jobs table found")
        else:
            logger.warning("⚠️  ingest_jobs table not found — run backend/db_init.py first")

    logger.info("🎉 AI_Backend DB ready!")
