"""
Qdrant client wrapper.
Encapsulates all vector DB operations behind a clean interface.
If we ever switch from Qdrant to Weaviate/Pinecone, only this file changes.
"""

import time
from typing import Any
from uuid import uuid4

from config.logging_config import get_logger
from config.settings import AppSettings
from ingestion.chunkers.base_chunker import DocumentChunk

logger = get_logger(__name__)

PAYLOAD_FIELDS = [
    "doc_id", "company", "ticker", "year", "quarter",
    "filing_type", "chunk_type", "page", "section",
    "chunk_index", "parent_id", "chunk_strategy",
]


class QdrantClientWrapper:

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.collection = settings.qdrant_collection_name
        self._client = None

    async def _get_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            if self.settings.qdrant_use_cloud:
                self._client = AsyncQdrantClient(
                    url=f"https://{self.settings.qdrant_host}",
                    api_key=self.settings.qdrant_api_key,
                )
            else:
                self._client = AsyncQdrantClient(
                    host=self.settings.qdrant_host,
                    port=self.settings.qdrant_port,
                )
        return self._client

    async def ensure_collection(self) -> None:
        """Create collection if it doesn't exist. Idempotent."""
        from qdrant_client.models import Distance, VectorParams

        client = await self._get_client()
        collections = await client.get_collections()
        existing = {c.name for c in collections.collections}

        if self.collection not in existing:
            await client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload indexes for fast metadata filtering
            from qdrant_client.models import PayloadSchemaType
            for field in ["company", "year", "filing_type", "doc_id", "chunk_type"]:
                await client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            logger.info("qdrant_collection_created", collection=self.collection)

    async def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        parent_chunks: dict[str, DocumentChunk] | None = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        client = await self._get_client()
        parent_chunks = parent_chunks or {}

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = {k: v for k, v in chunk.metadata.items() if k in PAYLOAD_FIELDS}
            payload["content"] = chunk.content
            payload["chunk_id"] = chunk.chunk_id

            # Store parent content inline so we can retrieve it without a second query
            if chunk.parent_id and chunk.parent_id in parent_chunks:
                payload["parent_content"] = parent_chunks[chunk.parent_id].content

            points.append(
                PointStruct(
                    id=str(uuid4()),  # Qdrant point ID (separate from chunk_id)
                    vector=embedding,
                    payload=payload,
                )
            )

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            await client.upsert(
                collection_name=self.collection,
                points=points[i: i + batch_size],
            )

        logger.info("qdrant_upsert_complete", count=len(points))

    async def search(
        self,
        query_vector: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

        client = await self._get_client()
        qdrant_filter = self._build_filter(filters) if filters else None

        start = time.perf_counter()
        results = await client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("qdrant_search_complete", results=len(results), elapsed_ms=round(elapsed, 2))

        return [
            {
                "chunk_id": r.payload.get("chunk_id"),
                "content": r.payload.get("parent_content") or r.payload.get("content"),
                "raw_content": r.payload.get("content"),
                "score": r.score,
                "metadata": {k: r.payload.get(k) for k in PAYLOAD_FIELDS},
            }
            for r in results
        ]

    async def delete_by_document_id(self, doc_id: str) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = await self._get_client()
        await client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        logger.info("qdrant_delete_complete", doc_id=doc_id)

    def _build_filter(self, filters: dict[str, Any]):
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

        conditions = []
        if filters.get("company"):
            conditions.append(
                FieldCondition(key="company", match=MatchAny(any=filters["company"]))
            )
        if filters.get("year"):
            years = [str(y) for y in filters["year"]]
            conditions.append(
                FieldCondition(key="year", match=MatchAny(any=years))
            )
        if filters.get("filing_type"):
            conditions.append(
                FieldCondition(key="filing_type", match=MatchAny(any=filters["filing_type"]))
            )
        # Only retrieve child/table chunks for embedding search (not raw parents)
        conditions.append(
            FieldCondition(key="chunk_type", match=MatchAny(any=["child", "table", None]))
        )

        return Filter(must=conditions) if conditions else None
