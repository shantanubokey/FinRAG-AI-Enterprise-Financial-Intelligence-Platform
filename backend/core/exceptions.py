"""
Custom exception hierarchy.
All domain exceptions inherit from FinancialRAGError so callers
can catch broadly or narrowly.
"""

from typing import Any


class FinancialRAGError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ── Ingestion ────────────────────────────────────────────────────────────────

class DocumentLoadError(FinancialRAGError):
    """Failed to load or parse a document."""


class UnsupportedFileTypeError(FinancialRAGError):
    """File type not supported by any loader."""


class ChunkingError(FinancialRAGError):
    """Error during document chunking."""


class MetadataExtractionError(FinancialRAGError):
    """Failed to extract metadata from document."""


# ── Embeddings ───────────────────────────────────────────────────────────────

class EmbeddingError(FinancialRAGError):
    """Failed to generate embeddings."""


class EmbeddingModelNotFoundError(FinancialRAGError):
    """Requested embedding model is not available."""


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievalError(FinancialRAGError):
    """Failed to retrieve documents."""


class VectorDBError(FinancialRAGError):
    """Vector database operation failed."""


class CollectionNotFoundError(VectorDBError):
    """Qdrant collection does not exist."""


# ── LLM ──────────────────────────────────────────────────────────────────────

class LLMError(FinancialRAGError):
    """LLM call failed."""


class LLMProviderNotFoundError(LLMError):
    """Requested LLM provider is not configured."""


class LLMRateLimitError(LLMError):
    """LLM provider rate limit hit."""


class LLMContextLengthError(LLMError):
    """Input exceeds model context window."""


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentError(FinancialRAGError):
    """Agent execution failed."""


class ToolExecutionError(AgentError):
    """A financial tool raised an error."""


class SQLGenerationError(AgentError):
    """Failed to generate valid SQL."""


# ── Auth ──────────────────────────────────────────────────────────────────────

class AuthenticationError(FinancialRAGError):
    """Authentication failed."""


class AuthorizationError(FinancialRAGError):
    """Insufficient permissions."""


class RateLimitExceededError(FinancialRAGError):
    """Rate limit exceeded."""
