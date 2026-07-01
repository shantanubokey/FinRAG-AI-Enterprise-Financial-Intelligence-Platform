"""
Schemas for query requests and responses.
The response schema includes citations and evaluation scores —
these are non-negotiable in production Financial RAG.
"""

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """Detected intent of the user's financial query."""

    FINANCIAL_RATIO = "financial_ratio"
    REVENUE_COMPARISON = "revenue_comparison"
    RISK_ANALYSIS = "risk_analysis"
    CASH_FLOW = "cash_flow"
    MULTI_COMPANY = "multi_company"
    MULTI_YEAR = "multi_year"
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    GENERIC_SEARCH = "generic_search"


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class QueryRequest(BaseModel):
    """Incoming query from the user."""

    question: str = Field(..., min_length=3, max_length=2000)
    company_filter: list[str] | None = None
    year_filter: list[int] | None = None
    filing_type_filter: list[str] | None = None
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    stream: bool = False
    session_id: str | None = None


class Citation(BaseModel):
    """Source citation for a piece of retrieved evidence."""

    document_id: UUID
    company: str | None
    filing_type: str | None
    year: int | None
    section: str | None
    page_number: int | None
    chunk_content: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_url: str | None = None


class EvaluationScores(BaseModel):
    """Auto-evaluation scores from Ragas/DeepEval."""

    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_relevancy: float | None = None
    hallucination_rate: float | None = None


class QueryMetrics(BaseModel):
    """Latency and cost breakdown for observability."""

    total_latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    llm_latency_ms: float
    embedding_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    cache_hit: bool = False


class QueryResponse(BaseModel):
    """Full response returned to the user."""

    query_id: UUID = Field(default_factory=uuid4)
    question: str
    answer: str
    citations: list[Citation]
    intent: QueryIntent
    evaluation: EvaluationScores | None = None
    metrics: QueryMetrics
    tools_used: list[str] = Field(default_factory=list)
    sql_query: str | None = None
    agent_steps: list[dict[str, Any]] = Field(default_factory=list)
