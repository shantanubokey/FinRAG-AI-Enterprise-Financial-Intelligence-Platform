"""
Query endpoint — the main entry point for user questions.
Orchestrates: cache check → agent → evaluation → response.
"""

import time
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.core.dependencies import Cache, CurrentUser, DBSession, Settings
from backend.schemas.query import QueryRequest, QueryResponse
from config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Ask a financial question",
    description="Submit a natural language query against ingested financial documents.",
)
async def query_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: DBSession,
    cache: Cache,
    settings: Settings,
    current_user: CurrentUser,
) -> QueryResponse:
    start_time = time.perf_counter()
    query_id = uuid4()

    logger.info(
        "query_received",
        query_id=str(query_id),
        user_id=current_user.get("sub"),
        question_length=len(request.question),
    )

    try:
        # 1. Check cache
        cache_key = f"query:{hash(request.question + str(request.company_filter))}"
        cached = await cache.get(cache_key)
        if cached:
            logger.info("cache_hit", query_id=str(query_id))
            import json
            response = QueryResponse.model_validate_json(cached)
            response.metrics.cache_hit = True
            return response

        # 2. Import here to avoid circular deps at module load
        from agent.graph.financial_agent import FinancialAgent

        agent = FinancialAgent(settings=settings)

        # 3. Run agent
        result = await agent.arun(
            question=request.question,
            filters={
                "company": request.company_filter,
                "year": request.year_filter,
                "filing_type": request.filing_type_filter,
            },
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            session_id=request.session_id or str(query_id),
        )

        total_ms = (time.perf_counter() - start_time) * 1000
        result.metrics.total_latency_ms = total_ms
        result.query_id = query_id

        # 4. Cache the response
        background_tasks.add_task(
            cache.setex,
            cache_key,
            settings.redis_ttl_seconds,
            result.model_dump_json(),
        )

        logger.info(
            "query_completed",
            query_id=str(query_id),
            latency_ms=total_ms,
            citations_count=len(result.citations),
        )

        return result

    except Exception as exc:
        logger.exception("query_failed", query_id=str(query_id), error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(exc)}",
        )
