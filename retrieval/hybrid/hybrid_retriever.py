"""Hybrid retriever combining dense (Qdrant) + sparse (BM25) via RRF."""

import time
from typing import Any, Literal

from config.logging_config import get_logger
from config.settings import AppSettings
from retrieval.hybrid.rrf_fusion import reciprocal_rank_fusion

logger = get_logger(__name__)


class HybridRetriever:

    def __init__(self, settings: AppSettings):
        self.settings = settings

    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
        mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
    ) -> list[dict[str, Any]]:
        dense_results, sparse_results = [], []

        if mode in ("dense", "hybrid"):
            dense_results = await self._dense_search(query, filters, top_k)

        if mode in ("sparse", "hybrid"):
            sparse_results = self._sparse_search(query, filters, top_k)

        if mode == "dense":
            return dense_results
        if mode == "sparse":
            return sparse_results

        # Hybrid: merge with RRF
        merged = reciprocal_rank_fusion(
            result_lists=[dense_results, sparse_results],
            weights=[0.6, 0.4],  # Dense weighted slightly higher for semantic queries
        )
        return merged[:top_k]

    async def _dense_search(self, query: str, filters, top_k: int) -> list[dict[str, Any]]:
        from embeddings.models.bge_embedder import BGEEmbedder
        from vectordb.qdrant.client import QdrantClientWrapper

        embedder = BGEEmbedder(settings=self.settings)
        await embedder.initialize()
        query_vec = await embedder.aembed_query(query)

        qdrant = QdrantClientWrapper(settings=self.settings)
        return await qdrant.search(query_vector=query_vec, filters=filters, top_k=top_k)

    def _sparse_search(self, query: str, filters, top_k: int) -> list[dict[str, Any]]:
        from retrieval.sparse.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever(persist_path="./data/bm25_index.pkl")
        if not bm25.load():
            logger.warning("bm25_index_not_found_skipping_sparse_search")
            return []
        return bm25.search(query=query, top_k=top_k, filters=filters)
