"""
Parent-Child chunker.

Why: Small chunks = precise retrieval. Large chunks = coherent LLM context.
Solution: Index small child chunks, but retrieve and send the parent chunk to the LLM.

This solves the precision/context tradeoff that plagues naive RAG systems.
"""

import hashlib
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.chunkers.base_chunker import BaseChunker, DocumentChunk


class ParentChildChunker(BaseChunker):
    """
    Creates two sets of chunks:
    - Parent chunks (large, ~2048 chars): sent to LLM as context
    - Child chunks (small, ~256 chars): used for retrieval/embedding

    Child chunks store parent_id so we can fetch the parent after retrieval.
    """

    def __init__(
        self,
        parent_chunk_size: int = 2048,
        child_chunk_size: int = 256,
        overlap: int = 32,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=overlap * 2,
            separators=["\n\n", "\n", ". ", " "],
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    @property
    def strategy_name(self) -> str:
        return "parent_child"

    def chunk(self, text: str, metadata: dict[str, Any], doc_id: str) -> list[DocumentChunk]:
        parent_texts = self._parent_splitter.split_text(text)
        all_chunks: list[DocumentChunk] = []

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = hashlib.md5(f"{doc_id}:parent:{p_idx}".encode()).hexdigest()

            # Store parent chunk (not embedded, used for context retrieval)
            parent_chunk = DocumentChunk(
                chunk_id=parent_id,
                content=parent_text,
                metadata={
                    **metadata,
                    "chunk_strategy": self.strategy_name,
                    "chunk_type": "parent",
                    "parent_index": p_idx,
                },
                chunk_index=p_idx,
                total_chunks=len(parent_texts),
                token_count=len(parent_text.split()),
            )
            all_chunks.append(parent_chunk)

            # Create child chunks, each pointing to their parent
            child_texts = self._child_splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(child_texts):
                child_id = hashlib.md5(
                    f"{doc_id}:child:{p_idx}:{c_idx}".encode()
                ).hexdigest()
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=child_id,
                        content=child_text,
                        metadata={
                            **metadata,
                            "chunk_strategy": self.strategy_name,
                            "chunk_type": "child",
                            "parent_index": p_idx,
                        },
                        parent_id=parent_id,
                        chunk_index=c_idx,
                        total_chunks=len(child_texts),
                        token_count=len(child_text.split()),
                    )
                )

        return all_chunks
