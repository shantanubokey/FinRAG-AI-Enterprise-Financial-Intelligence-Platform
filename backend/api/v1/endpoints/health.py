"""
Health and readiness endpoints for container orchestration.
Kubernetes uses /health/live for liveness and /health/ready for readiness.
"""

import time

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["Health"])

START_TIME = time.time()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]


@router.get("/live", response_model=HealthResponse, summary="Liveness probe")
async def liveness() -> HealthResponse:
    """Returns 200 if the service is running."""
    from config.settings import get_settings
    settings = get_settings()
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - START_TIME, 2),
        version=settings.app_version,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness() -> ReadinessResponse:
    """
    Checks all downstream dependencies.
    Returns 503 if any critical dependency is unavailable.
    """
    from fastapi import HTTPException

    checks: dict[str, str] = {}

    # Check Qdrant
    try:
        from qdrant_client import AsyncQdrantClient
        from config.settings import get_settings
        settings = get_settings()
        client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        await client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {str(e)}"

    # Check Redis
    try:
        import redis.asyncio as aioredis
        from config.settings import get_settings
        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())
    if not all_ok:
        raise HTTPException(status_code=503, detail={"status": "degraded", "checks": checks})

    return ReadinessResponse(status="ok", checks=checks)
