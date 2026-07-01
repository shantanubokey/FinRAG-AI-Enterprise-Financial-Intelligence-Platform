"""
Request/response logging middleware.
Injects a unique request_id into every log record for distributed tracing.
"""

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with timing and injects request_id into context."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Bind request_id to structlog context so all logs in this request include it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        logger.info("request_started")

        try:
            response = await call_next(request)
            latency_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "request_completed",
                status_code=response.status_code,
                latency_ms=round(latency_ms, 2),
            )

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{latency_ms:.2f}ms"
            return response

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception("request_failed", latency_ms=round(latency_ms, 2))
            raise
