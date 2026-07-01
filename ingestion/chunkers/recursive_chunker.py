"""
Recursive character chunker — the workhorse for most text.
Splits on paragraphs → sentences → words, preserving semantic boundaries.
"""

import hashlib
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.chunkers.base_chunker import BaseChunker, DocumentChunk


class RecursiveChunker(BaseChunker):
    """
    Uses LangChain's RecursiveCharacterTextSplitter.
    Preferred over fixed-size chunking because it respects sentence boundaries.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    @property
    def strategy_name(self) -> str:
        return "recursive"

    def chunk(self, text: str, metadata: dict[str, Any], doc_id: str) -> list[DocumentChunk]:
        raw_chunks = self._splitter.split_text(text)
        total = len(raw_chunks)
        chunks = []

        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = hashlib.md5(f"{doc_id}:{i}:{chunk_text[:50]}".encode()).hexdigest()
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    metadata={**metadata, "chunk_strategy": self.strategy_name},
                    chunk_index=i,
                    total_chunks=total,
                    token_count=len(chunk_text.split()),
                )
            )

        return chunks
