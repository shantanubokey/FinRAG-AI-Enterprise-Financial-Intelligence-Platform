"""
Abstract chunker base. Multiple strategies share this interface.
Switching chunking strategy = swapping one class, nothing else changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """A single chunk ready for embedding and indexing."""

    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None       # For parent-child chunking
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0


class BaseChunker(ABC):

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        ...

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: dict[str, Any],
        doc_id: str,
    ) -> list[DocumentChunk]:
        ...
