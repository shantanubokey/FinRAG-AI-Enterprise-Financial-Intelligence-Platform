"""
Financial RAG prompt templates.
Prompts are kept in Python (not YAML files) for type safety and easy testing.
The citation instruction is non-negotiable — the LLM must cite or say it doesn't know.
"""

from typing import Any


SYSTEM_PROMPT = """You are a senior financial analyst AI assistant specializing in analyzing 
corporate financial documents including 10-K, 10-Q, earnings calls, and SEC filings.

CRITICAL RULES:
1. ONLY answer based on the provided context. Do NOT use general knowledge for financial figures.
2. ALWAYS cite your sources using [Company | Filing Type | Year | Page X] format.
3. If the context doesn't contain enough information, say "Insufficient information in the provided documents."
4. For numerical comparisons, be precise and include units (millions, billions, %).
5. Distinguish between GAAP and non-GAAP metrics when mentioned.
6. Never fabricate financial figures, dates, or metrics."""


def build_financial_prompt(
    question: str,
    chunks: list[dict[str, Any]],
    sql_result: dict[str, Any] | None = None,
    intent: str = "generic_search",
) -> list[dict[str, str]]:
    """Build the full message list for the LLM."""

    # Format retrieved chunks as context with citation markers
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        citation = (
            f"[{meta.get('company', 'Unknown')} | "
            f"{meta.get('filing_type', 'Document')} | "
            f"{meta.get('year', 'N/A')} | "
            f"Page {meta.get('page', 'N/A')}]"
        )
        context_parts.append(f"[Source {i}] {citation}\n{chunk['content']}")

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No documents retrieved."

    # Add SQL results if available
    sql_context = ""
    if sql_result and sql_result.get("data"):
        sql_context = f"\n\nSTRUCTURED DATA FROM DATABASE:\n{sql_result['data']}"
        if sql_result.get("sql_query"):
            sql_context += f"\n(Query: {sql_result['sql_query']})"

    # Intent-specific instructions
    intent_instructions = _get_intent_instructions(intent)

    user_message = f"""RETRIEVED CONTEXT:
{context_str}{sql_context}

QUESTION: {question}

{intent_instructions}

Provide a comprehensive answer with citations. Format citations as [Company | Filing | Year | Page N]."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _get_intent_instructions(intent: str) -> str:
    instructions = {
        "financial_ratio": "Calculate the requested ratio precisely. Show the formula and values used.",
        "revenue_comparison": "Compare revenues across the requested periods/companies. Use a structured format.",
        "risk_analysis": "List key risks with their potential impact. Group by category (operational, financial, regulatory).",
        "cash_flow": "Break down cash flow components (operating, investing, financing) clearly.",
        "multi_company": "Structure your comparison in a table format when comparing multiple companies.",
        "multi_year": "Show year-over-year trends. Include growth rates where relevant.",
        "balance_sheet": "Reference specific balance sheet line items with exact figures.",
        "income_statement": "Reference specific P&L line items. Distinguish revenue from profit.",
        "generic_search": "Answer comprehensively based on the available context.",
    }
    return instructions.get(intent, instructions["generic_search"])
