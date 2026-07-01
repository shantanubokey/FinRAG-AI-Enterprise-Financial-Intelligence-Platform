"""
Pydantic schemas for document ingestion API.
Separating schemas from ORM models keeps the API contract stable
even when the DB schema changes.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class FilingType(str, Enum):
    ANNUAL_REPORT = "10-K"
    QUARTERLY_REPORT = "10-Q"
    EARNINGS_CALL = "earnings_call"
    INVESTOR_PRESENTATION = "investor_presentation"
    FINANCIAL_STATEMENT = "financial_statement"
    SEC_FILING = "sec_filing"
    RISK_REPORT = "risk_report"
    OTHER = "other"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Metadata extracted from a financial document."""

    company: str | None = None
    ticker: str | None = None
    year: int | None = None
    quarter: str | None = None  # Q1, Q2, Q3, Q4
    filing_type: FilingType = FilingType.OTHER
    section: str | None = None
    page_number: int | None = None
    source_url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: int | None) -> int | None:
        if v is not None and not (1900 <= v <= 2100):
            raise ValueError(f"Invalid year: {v}")
        return v


class DocumentUploadRequest(BaseModel):
    """Request body for document upload endpoint."""

    company: str | None = None
    ticker: str | None = None
    filing_type: FilingType = FilingType.OTHER
    year: int | None = None
    auto_detect_metadata: bool = True


class DocumentResponse(BaseModel):
    """Response after document upload."""

    id: UUID = Field(default_factory=uuid4)
    filename: str
    status: DocumentStatus
    metadata: DocumentMetadata
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error: str | None = None

    model_config = {"from_attributes": True}


class ChunkResponse(BaseModel):
    """A single retrieved document chunk with citation info."""

    chunk_id: str
    content: str
    metadata: DocumentMetadata
    score: float = Field(ge=0.0, le=1.0)
    rerank_score: float | None = None

    model_config = {"from_attributes": True}
