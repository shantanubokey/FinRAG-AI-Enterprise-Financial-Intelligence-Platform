"""
Prometheus metrics for the RAG system.
Tracks: request latency, token usage, cache hits, retrieval quality.
"""

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app


# Request metrics
REQUEST_COUNT = Counter(
    "financial_rag_requests_total",
    "Total number of query requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "financial_rag_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# LLM metrics
TOKEN_USAGE = Counter(
    "financial_rag_tokens_total",
    "Total tokens consumed",
    ["provider", "type"],  # type: prompt or completion
)

LLM_COST = Counter(
    "financial_rag_llm_cost_usd_total",
    "Total estimated LLM cost in USD",
    ["provider"],
)

# Retrieval metrics
RETRIEVAL_LATENCY = Histogram(
    "financial_rag_retrieval_duration_seconds",
    "Retrieval pipeline latency",
    ["mode"],  # dense, sparse, hybrid
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

CHUNKS_RETRIEVED = Histogram(
    "financial_rag_chunks_retrieved",
    "Number of chunks retrieved per query",
    buckets=[1, 2, 5, 10, 20, 50],
)

# Cache metrics
CACHE_HITS = Counter(
    "financial_rag_cache_hits_total",
    "Cache hit count",
    ["cache_type"],  # query, embedding
)

CACHE_MISSES = Counter(
    "financial_rag_cache_misses_total",
    "Cache miss count",
    ["cache_type"],
)

# Ingestion metrics
DOCUMENTS_INGESTED = Counter(
    "financial_rag_documents_ingested_total",
    "Total documents ingested",
    ["filing_type", "status"],
)

INGESTION_LATENCY = Histogram(
    "financial_rag_ingestion_duration_seconds",
    "Document ingestion pipeline latency",
    buckets=[1, 5, 10, 30, 60, 120],
)

# Evaluation metrics
FAITHFULNESS_SCORE = Histogram(
    "financial_rag_faithfulness_score",
    "Ragas faithfulness score distribution",
    buckets=[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
)

ANSWER_RELEVANCY_SCORE = Histogram(
    "financial_rag_answer_relevancy_score",
    "Ragas answer relevancy score distribution",
    buckets=[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
)


def setup_prometheus(app) -> None:
    """Mount Prometheus metrics endpoint on the FastAPI app."""
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
