"""
Abstract base loader. Every format-specific loader implements this interface.
This is the Open/Closed Principle — add new formats without touching existing code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDocument:
    """Unified output from any document loader."""

    content: str                          # Full text content
    tables: list[dict[str, Any]] = field(default_factory=list)   # Extracted tables
    images: list[dict[str, Any]] = field(default_factory=list)   # Extracted images
    metadata: dict[str, Any] = field(default_factory=dict)       # File-level metadata
    page_contents: list[str] = field(default_factory=list)       # Per-page text


class BaseDocumentLoader(ABC):
    """All loaders must implement load() and report supported types."""

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """e.g. ['.pdf', '.PDF']"""
        ...

    @abstractmethod
    async def load(self, content: bytes, filename: str) -> RawDocument:
        """
        Parse raw bytes into a RawDocument.
        Must be async to support async I/O (e.g., cloud storage reads).
        """
        ...

    def can_handle(self, filename: str) -> bool:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()
        return suffix in [ext.lower() for ext in self.supported_extensions]
