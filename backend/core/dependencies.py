"""
FastAPI dependency injection container.
All shared resources (DB, cache, vector store) are created once and injected.
This is the Dependency Inversion Principle in practice.
"""

from functools import lru_cache
from typing import Annotated, AsyncGenerator

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import AppSettings, get_settings
from config.logging_config import get_logger

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


# ── Database ────────────────────────────────────────────────────────────────

def get_db_engine(settings: AppSettings):
    return create_async_engine(
        settings.postgres_dsn,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=settings.debug,
    )


def get_session_factory(settings: AppSettings):
    engine = get_db_engine(settings)
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db_session(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> AsyncGenerator[AsyncSession, None]:
    """Yields a DB session per request, auto-commits or rolls back."""
    factory = get_session_factory(settings)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Redis Cache ──────────────────────────────────────────────────────────────

@lru_cache
def get_redis_pool(redis_url: str) -> aioredis.Redis:
    return aioredis.from_url(redis_url, decode_responses=True)


async def get_cache(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> aioredis.Redis:
    return get_redis_pool(settings.redis_url)


# ── Auth ─────────────────────────────────────────────────────────────────────

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> dict:
    """
    Validates JWT or API key.
    Returns user payload dict on success, raises 401 on failure.
    """
    from security.auth import verify_token, verify_api_key

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    token = credentials.credentials
    user = verify_token(token, settings.secret_key, settings.jwt_algorithm)
    if user is None:
        user = verify_api_key(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired credentials",
        )

    return user


# ── Type Aliases (clean route signatures) ───────────────────────────────────

DBSession = Annotated[AsyncSession, Depends(get_db_session)]
Cache = Annotated[aioredis.Redis, Depends(get_cache)]
Settings = Annotated[AppSettings, Depends(get_settings)]
CurrentUser = Annotated[dict, Depends(get_current_user)]
