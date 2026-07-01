"""
BM25 sparse retriever using rank_bm25.
Critical for financial queries with exact terms: ticker symbols, GAAP line items,
specific years/quarters that semantic search might miss.
"""

import pickle
from pathlib import Path
from typing import Any

from config.logging_config import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """
    In-memory BM25 over stored chunks.
    For production scale, swap with Elasticsearch/OpenSearch BM25.
    This implementation is suitable for up to ~100K chunks.
    """

    def __init__(self, persist_path: str | None = None):
        self.persist_path = Path(persist_path) if persist_path else None
        self._bm25 = None
        self._chunk_index: list[dict[str, Any]] = []  # chunk_id → chunk data
        self._tokenized_corpus: list[list[str]] = []

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """Build BM25 index from chunk dicts with 'content' and 'chunk_id' keys."""
        from rank_bm25 import BM25Okapi

        self._chunk_index = chunks
        self._tokenized_corpus = [
            self._tokenize(c["content"]) for c in chunks
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("bm25_index_built", chunk_count=len(chunks))

        if self.persist_path:
            self._save()

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top_k chunks ranked by BM25 score."""
        if self._bm25 is None:
            logger.warning("bm25_not_indexed_returning_empty")
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Apply metadata filters
        candidates = list(enumerate(self._chunk_index))
        if filters:
            candidates = [
                (i, c) for i, c in candidates
                if self._matches_filters(c["metadata"], filters)
            ]

        # Sort by score
        ranked = sorted(candidates, key=lambda x: scores[x[0]], reverse=True)[:top_k]

        results = []
        max_score = scores[ranked[0][0]] if ranked else 1.0
        for idx, chunk in ranked:
            score = scores[idx]
            if score <= 0:
                continue
            results.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
                "score": float(score / max_score),  # Normalize to [0, 1]
                "raw_bm25_score": float(score),
            })

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer. Fine for BM25."""
        import re
        # Keep financial terms like "10-K", "$100M", "Q3" intact
        tokens = re.findall(r"[\w\$\-\.]+", text.lower())
        return tokens

    def _matches_filters(self, metadata: dict, filters: dict) -> bool:
        for key, value in filters.items():
            if not value:
                continue
            meta_val = metadata.get(key)
            if isinstance(value, list):
                if meta_val not in value:
                    return False
            elif meta_val != value:
                return False
        return True

    def _save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump((self._bm25, self._chunk_index), f)

    def load(self) -> bool:
        if not self.persist_path or not self.persist_path.exists():
            return False
        with open(self.persist_path, "rb") as f:
            self._bm25, self._chunk_index = pickle.load(f)
        logger.info("bm25_index_loaded", chunk_count=len(self._chunk_index))
        return True
