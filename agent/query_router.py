"""
Query intent classifier.
Uses a lightweight zero-shot classifier first, LLM fallback for ambiguous cases.
This avoids burning LLM tokens on every query just to classify it.
"""

import re
from typing import Literal

from config.logging_config import get_logger
from config.settings import AppSettings

logger = get_logger(__name__)

QueryIntentType = Literal[
    "financial_ratio", "revenue_comparison", "risk_analysis",
    "cash_flow", "multi_company", "multi_year", "balance_sheet",
    "income_statement", "generic_search",
]

# Keyword rules — fast, free, accurate for 90% of financial queries
INTENT_RULES: list[tuple[list[str], QueryIntentType]] = [
    (["roe", "roa", "ebitda", "eps", "p/e", "pe ratio", "debt to equity", "gross margin", "operating margin", "cagr"], "financial_ratio"),
    (["revenue", "sales", "income", "profit", "compare", "comparison", "vs", "versus"], "revenue_comparison"),
    (["risk", "risk factor", "mitigation", "threat", "uncertainty", "downside"], "risk_analysis"),
    (["cash flow", "free cash flow", "fcf", "operating cash", "capex", "cash from operations"], "cash_flow"),
    (["apple and microsoft", "compare companies", "multiple companies", "across companies"], "multi_company"),
    (["2020 vs 2021", "year over year", "yoy", "multiple years", "trend", "historical"], "multi_year"),
    (["balance sheet", "assets", "liabilities", "equity", "working capital"], "balance_sheet"),
    (["income statement", "revenue", "gross profit", "operating income", "net income", "ebit"], "income_statement"),
]


class QueryRouter:

    def __init__(self, settings: AppSettings):
        self.settings = settings

    async def classify(self, question: str) -> QueryIntentType:
        """Classify query intent. Fast rule-based first, LLM fallback."""
        lower = question.lower()

        for keywords, intent in INTENT_RULES:
            if any(kw in lower for kw in keywords):
                logger.debug("intent_classified_by_rules", intent=intent)
                return intent

        # Fallback: LLM classification (only for ambiguous queries)
        try:
            return await self._llm_classify(question)
        except Exception as exc:
            logger.warning("llm_classification_failed_using_generic", error=str(exc))
            return "generic_search"

    async def _llm_classify(self, question: str) -> QueryIntentType:
        from llm.router.llm_router import LLMRouter

        valid_intents = [
            "financial_ratio", "revenue_comparison", "risk_analysis",
            "cash_flow", "multi_company", "multi_year", "balance_sheet",
            "income_statement", "generic_search",
        ]

        prompt = f"""Classify this financial query into exactly one category.
Categories: {', '.join(valid_intents)}
Query: {question}
Reply with only the category name, nothing else."""

        router = LLMRouter(settings=self.settings)
        response = await router.agenerate(prompt)
        intent = response["content"].strip().lower()

        if intent in valid_intents:
            return intent  # type: ignore
        return "generic_search"
