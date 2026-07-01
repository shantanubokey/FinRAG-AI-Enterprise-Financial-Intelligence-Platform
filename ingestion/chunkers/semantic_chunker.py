"""
Semantic chunker — splits on embedding similarity drops.
More expensive but produces semantically coherent chunks.
Best for financial narratives where topics shift mid-paragraph.
"""

from typing import Any

import numpy as np

from config.logging_config import get_logger
from ingestion.chunkers.base_chunker import BaseChunker, DocumentChunk

logger = get_logger(__name__)


class SemanticChunker(BaseChunker):
    """
    Algorithm:
    1. Split into sentences
    2. Embed each sentence
    3. Compute cosine similarity between adjacent sentences
    4. Split at similarity valleys (topic changes)
    """

    def __init__(
        self,
        embedder=None,
        breakpoint_threshold: float = 0.75,
        min_chunk_size: int = 100,
    ):
        self.embedder = embedder
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def chunk(self, text: str, metadata: dict[str, Any], doc_id: str) -> list[DocumentChunk]:
        import hashlib

        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            # Too short to split semantically
            import hashlib as h
            cid = h.md5(f"{doc_id}:0".encode()).hexdigest()
            return [DocumentChunk(
                chunk_id=cid,
                content=text,
                metadata={**metadata, "chunk_strategy": self.strategy_name},
            )]

        if self.embedder is None:
            logger.warning("no_embedder_for_semantic_chunker_falling_back_to_sentence_split")
            return self._fallback_chunk(sentences, metadata, doc_id)

        # Embed sentences
        embeddings = self.embedder.embed_documents(sentences)
        similarities = self._compute_similarities(embeddings)

        # Find split points where similarity drops below threshold
        split_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < self.breakpoint_threshold:
                split_indices.append(i + 1)
        split_indices.append(len(sentences))

        # Build chunks from split points
        chunks = []
        for idx, (start, end) in enumerate(zip(split_indices, split_indices[1:])):
            chunk_text = " ".join(sentences[start:end])
            if len(chunk_text) < self.min_chunk_size and chunks:
                # Merge tiny chunks with previous
                chunks[-1] = DocumentChunk(
                    chunk_id=chunks[-1].chunk_id,
                    content=chunks[-1].content + " " + chunk_text,
                    metadata=chunks[-1].metadata,
                    chunk_index=chunks[-1].chunk_index,
                )
                continue

            chunk_id = hashlib.md5(f"{doc_id}:{idx}:{chunk_text[:30]}".encode()).hexdigest()
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata={**metadata, "chunk_strategy": self.strategy_name},
                chunk_index=idx,
                token_count=len(chunk_text.split()),
            ))

        # Update total_chunks
        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarities(self, embeddings: list[list[float]]) -> list[float]:
        sims = []
        for i in range(len(embeddings) - 1):
            a = np.array(embeddings[i])
            b = np.array(embeddings[i + 1])
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            sims.append(sim)
        return sims

    def _fallback_chunk(
        self, sentences: list[str], metadata: dict[str, Any], doc_id: str
    ) -> list[DocumentChunk]:
        import hashlib
        # Group sentences into fixed windows
        window = 5
        chunks = []
        for i in range(0, len(sentences), window):
            text = " ".join(sentences[i: i + window])
            cid = hashlib.md5(f"{doc_id}:{i}".encode()).hexdigest()
            chunks.append(DocumentChunk(
                chunk_id=cid,
                content=text,
                metadata={**metadata, "chunk_strategy": self.strategy_name},
                chunk_index=i // window,
            ))
        return chunks
