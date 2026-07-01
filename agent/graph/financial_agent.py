"""
LangGraph-based Financial Agent.
Uses a state machine to decide: retrieve → compute → SQL → summarize.
LangGraph is chosen over simple LangChain agents because:
1. Explicit state transitions (auditable)
2. Supports cycles (re-retrieve if context insufficient)
3. Better for multi-step financial reasoning
"""

from __future__ import annotations

from typing import Any, TypedDict

from config.logging_config import get_logger
from config.settings import AppSettings

logger = get_logger(__name__)


class AgentState(TypedDict):
    """Shared state passed between graph nodes."""
    question: str
    intent: str
    filters: dict[str, Any]
    retrieval_mode: str
    top_k: int
    use_reranker: bool
    session_id: str
    retrieved_chunks: list[dict[str, Any]]
    sql_result: dict[str, Any] | None
    tool_results: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    agent_steps: list[dict[str, Any]]
    tools_used: list[str]
    metrics: dict[str, Any]


class FinancialAgent:
    """
    LangGraph agent with nodes:
    1. classify_intent → decide routing
    2. retrieve_documents → hybrid RAG
    3. query_sql → structured metrics
    4. use_calculator → financial computations
    5. generate_answer → LLM with context + citations
    6. evaluate_response → auto-eval scores
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._graph = None

    def _build_graph(self):
        from langgraph.graph import StateGraph, END

        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("classify_intent", self._classify_intent)
        builder.add_node("retrieve_documents", self._retrieve_documents)
        builder.add_node("query_sql", self._query_sql)
        builder.add_node("use_calculator", self._use_calculator)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("evaluate_response", self._evaluate_response)

        # Entry point
        builder.set_entry_point("classify_intent")

        # Conditional routing from intent classifier
        builder.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "retrieve": "retrieve_documents",
                "sql": "query_sql",
                "calculator": "use_calculator",
            },
        )

        # After retrieval, always generate answer
        builder.add_edge("retrieve_documents", "generate_answer")
        builder.add_edge("query_sql", "generate_answer")
        builder.add_edge("use_calculator", "generate_answer")
        builder.add_edge("generate_answer", "evaluate_response")
        builder.add_edge("evaluate_response", END)

        return builder.compile()

    async def arun(
        self,
        question: str,
        filters: dict[str, Any],
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        use_reranker: bool = True,
        session_id: str = "",
    ):
        if self._graph is None:
            self._graph = self._build_graph()

        initial_state: AgentState = {
            "question": question,
            "intent": "",
            "filters": filters,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "use_reranker": use_reranker,
            "session_id": session_id,
            "retrieved_chunks": [],
            "sql_result": None,
            "tool_results": [],
            "answer": "",
            "citations": [],
            "agent_steps": [],
            "tools_used": [],
            "metrics": {
                "retrieval_latency_ms": 0,
                "rerank_latency_ms": 0,
                "llm_latency_ms": 0,
                "embedding_latency_ms": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "cache_hit": False,
            },
        }

        final_state = await self._graph.ainvoke(initial_state)
        return self._state_to_response(final_state)

    async def _classify_intent(self, state: AgentState) -> AgentState:
        """Classify query into one of the QueryIntent categories."""
        from agent.query_router import QueryRouter
        router = QueryRouter(settings=self.settings)
        intent = await router.classify(state["question"])
        state["intent"] = intent
        state["agent_steps"].append({"node": "classify_intent", "intent": intent})
        logger.info("intent_classified", intent=intent)
        return state

    def _route_by_intent(self, state: AgentState) -> str:
        """Routing logic based on classified intent."""
        sql_intents = {"financial_ratio", "revenue_comparison", "multi_year"}
        calc_intents = set()  # Calculator is triggered via LLM tool calls

        if state["intent"] in sql_intents:
            return "sql"
        return "retrieve"

    async def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Run hybrid retrieval pipeline."""
        import time
        from retrieval.hybrid.hybrid_retriever import HybridRetriever
        from reranker.cross_encoder import CrossEncoderReranker

        start = time.perf_counter()
        retriever = HybridRetriever(settings=self.settings)
        chunks = await retriever.retrieve(
            query=state["question"],
            filters=state["filters"],
            top_k=state["top_k"] * 4,  # Retrieve more, rerank down
            mode=state["retrieval_mode"],
        )
        state["metrics"]["retrieval_latency_ms"] = (time.perf_counter() - start) * 1000

        if state["use_reranker"] and chunks:
            rerank_start = time.perf_counter()
            reranker = CrossEncoderReranker(model_name=self.settings.reranker_model)
            chunks = await reranker.rerank(state["question"], chunks, top_k=state["top_k"])
            state["metrics"]["rerank_latency_ms"] = (time.perf_counter() - start) * 1000
            state["tools_used"].append("cross_encoder_reranker")

        state["retrieved_chunks"] = chunks
        state["tools_used"].append("hybrid_retriever")
        state["agent_steps"].append({
            "node": "retrieve_documents",
            "chunks_found": len(chunks),
        })
        return state

    async def _query_sql(self, state: AgentState) -> AgentState:
        """Generate and execute SQL query for structured financial data."""
        from agent.sql.sql_agent import SQLAgent
        sql_agent = SQLAgent(settings=self.settings)
        result = await sql_agent.query(state["question"], state["filters"])
        state["sql_result"] = result
        state["tools_used"].append("sql_agent")
        state["agent_steps"].append({"node": "query_sql", "result": result})
        # Also retrieve docs for context
        return await self._retrieve_documents(state)

    async def _use_calculator(self, state: AgentState) -> AgentState:
        """Retrieve first, then LLM decides to call calculator tools."""
        state = await self._retrieve_documents(state)
        state["tools_used"].append("financial_calculator")
        return state

    async def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate final answer using LLM with full context."""
        import time
        from llm.router.llm_router import LLMRouter
        from citations.citation_engine import CitationEngine
        from prompts.templates.financial_prompt import build_financial_prompt

        start = time.perf_counter()
        llm_router = LLMRouter(settings=self.settings)
        citation_engine = CitationEngine()

        prompt = build_financial_prompt(
            question=state["question"],
            chunks=state["retrieved_chunks"],
            sql_result=state["sql_result"],
            intent=state["intent"],
        )

        response = await llm_router.agenerate(prompt)
        state["metrics"]["llm_latency_ms"] = (time.perf_counter() - start) * 1000
        state["metrics"]["prompt_tokens"] = response.get("prompt_tokens", 0)
        state["metrics"]["completion_tokens"] = response.get("completion_tokens", 0)
        state["metrics"]["total_tokens"] = response.get("total_tokens", 0)
        state["metrics"]["estimated_cost_usd"] = response.get("cost_usd", 0.0)

        state["answer"] = response["content"]
        state["citations"] = citation_engine.extract_citations(
            answer=response["content"],
            chunks=state["retrieved_chunks"],
        )
        state["agent_steps"].append({"node": "generate_answer"})
        return state

    async def _evaluate_response(self, state: AgentState) -> AgentState:
        """Auto-evaluate every response with Ragas metrics."""
        try:
            from evaluation.ragas.evaluator import RagasEvaluator
            evaluator = RagasEvaluator()
            scores = await evaluator.evaluate(
                question=state["question"],
                answer=state["answer"],
                contexts=[c["content"] for c in state["retrieved_chunks"]],
            )
            state["metrics"]["evaluation"] = scores
        except Exception as exc:
            logger.warning("evaluation_failed", error=str(exc))
        return state

    def _state_to_response(self, state: AgentState):
        from backend.schemas.query import (
            QueryResponse, Citation, EvaluationScores, QueryMetrics, QueryIntent
        )
        from backend.schemas.documents import DocumentMetadata, FilingType

        citations = [
            Citation(
                document_id=__import__("uuid").uuid4(),
                company=c.get("metadata", {}).get("company"),
                filing_type=c.get("metadata", {}).get("filing_type"),
                year=c.get("metadata", {}).get("year"),
                section=c.get("metadata", {}).get("section"),
                page_number=c.get("metadata", {}).get("page"),
                chunk_content=c.get("raw_content", c.get("content", ""))[:500],
                confidence_score=min(1.0, max(0.0, c.get("rerank_score", c.get("score", 0.5)))),
            )
            for c in state["citations"]
        ]

        eval_data = state["metrics"].get("evaluation", {})
        m = state["metrics"]

        return QueryResponse(
            question=state["question"],
            answer=state["answer"],
            citations=citations,
            intent=state.get("intent", "generic_search"),
            evaluation=EvaluationScores(**eval_data) if eval_data else None,
            metrics=QueryMetrics(
                total_latency_ms=m.get("total_latency_ms", 0),
                retrieval_latency_ms=m.get("retrieval_latency_ms", 0),
                rerank_latency_ms=m.get("rerank_latency_ms", 0),
                llm_latency_ms=m.get("llm_latency_ms", 0),
                embedding_latency_ms=m.get("embedding_latency_ms", 0),
                prompt_tokens=m.get("prompt_tokens", 0),
                completion_tokens=m.get("completion_tokens", 0),
                total_tokens=m.get("total_tokens", 0),
                estimated_cost_usd=m.get("estimated_cost_usd", 0.0),
                cache_hit=m.get("cache_hit", False),
            ),
            tools_used=state.get("tools_used", []),
            agent_steps=state.get("agent_steps", []),
        )
