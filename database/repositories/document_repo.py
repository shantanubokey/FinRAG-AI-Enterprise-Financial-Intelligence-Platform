"""
Repository pattern for document persistence.
Keeps DB queries out of route handlers — testable, swappable.
"""

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config.logging_config import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    """All document CRUD operations in one place."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, doc_id: UUID) -> dict[str, Any] | None:
        result = await self.session.execute(
            text("SELECT * FROM documents WHERE id = :id"),
            {"id": str(doc_id)},
        )
        row = result.fetchone()
        return dict(row._mapping) if row else None

    async def create(self, doc_data: dict[str, Any]) -> dict[str, Any]:
        await self.session.execute(
            text("""
                INSERT INTO documents (id, filename, status, company, ticker, year, filing_type, chunk_count)
                VALUES (:id, :filename, :status, :company, :ticker, :year, :filing_type, :chunk_count)
            """),
            doc_data,
        )
        return doc_data

    async def update_status(
        self,
        doc_id: UUID,
        status: str,
        chunk_count: int = 0,
        error: str | None = None,
    ) -> None:
        await self.session.execute(
            text("""
                UPDATE documents
                SET status = :status, chunk_count = :chunk_count, error = :error
                WHERE id = :id
            """),
            {"id": str(doc_id), "status": status, "chunk_count": chunk_count, "error": error},
        )

    async def delete(self, doc_id: UUID) -> None:
        await self.session.execute(
            text("DELETE FROM documents WHERE id = :id"),
            {"id": str(doc_id)},
        )

    async def list_documents(
        self,
        company: str | None = None,
        year: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        filters = []
        params: dict[str, Any] = {"limit": limit}

        if company:
            filters.append("company = :company")
            params["company"] = company
        if year:
            filters.append("year = :year")
            params["year"] = year

        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        result = await self.session.execute(
            text(f"SELECT * FROM documents {where} ORDER BY created_at DESC LIMIT :limit"),
            params,
        )
        return [dict(row._mapping) for row in result.fetchall()]
