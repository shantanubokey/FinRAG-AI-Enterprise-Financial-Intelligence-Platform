"""
SQL Agent for structured financial metrics.
LLM generates SQL → validated → executed → results returned.
Stored metrics: revenue, net_income, eps, total_assets, etc. per company/year/quarter.
"""

from typing import Any

from config.logging_config import get_logger
from config.settings import AppSettings

logger = get_logger(__name__)

# Schema description sent to LLM for SQL generation
DB_SCHEMA = """
Table: financial_metrics
Columns:
  - id: UUID (primary key)
  - company: VARCHAR (company name, e.g. 'Apple', 'Microsoft')
  - ticker: VARCHAR (e.g. 'AAPL', 'MSFT')
  - year: INTEGER (fiscal year, e.g. 2023)
  - quarter: VARCHAR (e.g. 'Q1', 'Q2', NULL for annual)
  - filing_type: VARCHAR (e.g. '10-K', '10-Q')
  - revenue: NUMERIC (in millions USD)
  - gross_profit: NUMERIC
  - operating_income: NUMERIC
  - net_income: NUMERIC
  - ebitda: NUMERIC
  - total_assets: NUMERIC
  - total_liabilities: NUMERIC
  - shareholders_equity: NUMERIC
  - operating_cash_flow: NUMERIC
  - free_cash_flow: NUMERIC
  - eps_basic: NUMERIC
  - eps_diluted: NUMERIC
  - shares_outstanding: NUMERIC

Only use SELECT statements. Never modify data.
"""


class SQLAgent:

    def __init__(self, settings: AppSettings):
        self.settings = settings

    async def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate SQL from natural language, execute it, return results."""
        try:
            sql = await self._generate_sql(question, filters)
            logger.info("sql_generated", sql=sql)

            # Safety check: only allow SELECT
            if not sql.strip().upper().startswith("SELECT"):
                return {"error": "Only SELECT queries are allowed", "sql_query": sql}

            data = await self._execute_sql(sql)
            return {"data": data, "sql_query": sql, "row_count": len(data)}

        except Exception as exc:
            logger.exception("sql_agent_failed", error=str(exc))
            return {"error": str(exc), "data": [], "sql_query": ""}

    async def _generate_sql(self, question: str, filters: dict | None) -> str:
        from llm.router.llm_router import LLMRouter

        filter_hint = ""
        if filters:
            if filters.get("company"):
                filter_hint += f"\nFocus on companies: {filters['company']}"
            if filters.get("year"):
                filter_hint += f"\nFocus on years: {filters['year']}"

        prompt = f"""Given this database schema:
{DB_SCHEMA}

Generate a SQL SELECT query to answer:
"{question}"{filter_hint}

Return ONLY the SQL query, no explanation, no markdown.
Limit results to 50 rows max."""

        router = LLMRouter(settings=self.settings)
        response = await router.agenerate(prompt)
        sql = response["content"].strip()

        # Strip markdown code blocks if present
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        return sql

    async def _execute_sql(self, sql: str) -> list[dict[str, Any]]:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(self.settings.postgres_dsn)
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]
