"""
FastAPI application factory.
Uses the factory pattern so tests can spin up isolated app instances.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.api.v1.endpoints import health, ingest, query
from backend.api.v1.middleware.logging_middleware import RequestLoggingMiddleware
from backend.api.v1.middleware.rate_limiter import RateLimitMiddleware
from config.logging_config import get_logger, setup_logging
from config.settings import get_settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle hooks."""
    settings = get_settings()
    logger.info("application_starting", env=settings.environment)

    # Pre-load embedding model so first request isn't slow
    from embeddings.models.bge_embedder import BGEEmbedder
    app.state.embedder = BGEEmbedder(settings=settings)
    await app.state.embedder.initialize()

    logger.info("application_ready")
    yield

    # Cleanup
    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade Financial RAG System",
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )

    # Middleware (order matters — outermost runs first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )

    # Routers
    prefix = settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(ingest.router, prefix=prefix)
    app.include_router(query.router, prefix=prefix)

    # Prometheus metrics endpoint
    from monitoring.metrics.prometheus_metrics import setup_prometheus
    setup_prometheus(app)

    return app


app = create_app()
