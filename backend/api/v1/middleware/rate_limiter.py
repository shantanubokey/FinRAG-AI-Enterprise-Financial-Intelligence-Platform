"""
Redis-backed sliding window rate limiter.
Per-user limits prevent one bad actor from saturating the LLM budget.
"""

import time

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter using Redis sorted sets.
    Key: rate_limit:{user_id_or_ip}
    Members: request timestamps (scored by timestamp)
    """

    def __init__(self, app, requests_per_window: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            settings = get_settings()
            self._redis = aioredis.from_url(settings.redis_url)
        return self._redis

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/api/v1/health"):
            return await call_next(request)

        identifier = self._get_identifier(request)
        redis = await self._get_redis()

        now = time.time()
        window_start = now - self.window_seconds
        key = f"rate_limit:{identifier}"

        pipe = redis.pipeline()
        # Remove timestamps outside the window
        pipe.zremrangebyscore(key, 0, window_start)
        # Count requests in window
        pipe.zcard(key)
        # Add this request
        pipe.zadd(key, {str(now): now})
        # Set expiry
        pipe.expire(key, self.window_seconds * 2)

        results = await pipe.execute()
        request_count = results[1]

        if request_count >= self.requests_per_window:
            logger.warning("rate_limit_exceeded", identifier=identifier, count=request_count)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please slow down.",
                headers={"Retry-After": str(self.window_seconds)},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_window - request_count - 1)
        )
        return response

    def _get_identifier(self, request: Request) -> str:
        """Use user ID from JWT if available, else fall back to IP."""
        # The auth middleware would have set this on request.state
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        forwarded = request.headers.get("X-Forwarded-For")
        ip = forwarded.split(",")[0] if forwarded else request.client.host
        return f"ip:{ip}"
