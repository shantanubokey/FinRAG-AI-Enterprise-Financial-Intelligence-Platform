"""
Cross-encoder re-ranker.

Why cross-encoders beat bi-encoders for re-ranking:
- Bi-encoder: encode(query) · encode(doc) — fast but independent encoding
- Cross-encoder: encode(query + doc) together — slower but sees interaction

Applied only on top-k from hybrid search (k=20 → rerank → k=5).
Latency cost: ~200ms for 20 pairs on CPU, acceptable for final ranking.
"""

import asyncio
import time
from typing import Any

from config.logging_config import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks retrieval results using a cross-encoder model.
    Default: ms-marco-MiniLM-L-6-v2 (good balance of speed/accuracy).
    For higher accuracy: ms-marco-electra-base (2x slower, ~5% better).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    async def initialize(self) -> None:
        if self._model is not None:
            return
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        logger.info("reranker_loaded", model=self.model_name)

    def _load_model(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(self.model_name)

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Re-rank a list of retrieval results.
        Returns top_k results sorted by cross-encoder score.
        """
        if not results:
            return results

        if self._model is None:
            await self.initialize()

        start = time.perf_counter()

        # Build (query, passage) pairs
        pairs = [(query, r["content"]) for r in results]

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, show_progress_bar=False).tolist(),
        )

        # Attach scores and sort
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
            result["pre_rerank_score"] = result.get("score", 0.0)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(
            "reranking_complete",
            input_count=len(results),
            output_count=len(reranked),
            elapsed_ms=round(elapsed, 2),
        )

        return reranked
