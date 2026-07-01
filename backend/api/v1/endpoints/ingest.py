"""
Document ingestion endpoint.
Accepts file uploads, runs the ingestion pipeline async,
and returns job status that can be polled.
"""

from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from backend.core.dependencies import Cache, CurrentUser, DBSession, Settings
from backend.schemas.documents import (
    DocumentResponse,
    DocumentStatus,
    DocumentUploadRequest,
    FilingType,
)
from config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

SUPPORTED_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/csv": ".csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/html": ".html",
}


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a financial document",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company: str | None = Form(None),
    ticker: str | None = Form(None),
    filing_type: FilingType = Form(FilingType.OTHER),
    year: int | None = Form(None),
    auto_detect_metadata: bool = Form(True),
    db: DBSession = None,
    cache: Cache = None,
    settings: Settings = None,
    current_user: CurrentUser = None,
) -> DocumentResponse:
    # Validate file type
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Supported: {list(SUPPORTED_TYPES.keys())}",
        )

    # Validate file size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb}MB limit",
        )

    doc_id = uuid4()
    logger.info(
        "document_upload_received",
        doc_id=str(doc_id),
        filename=file.filename,
        size_bytes=len(content),
        filing_type=filing_type,
    )

    # Queue background ingestion
    from ingestion.pipeline.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline(settings=settings)

    background_tasks.add_task(
        pipeline.process_document,
        doc_id=doc_id,
        content=content,
        filename=file.filename,
        content_type=file.content_type,
        metadata_hints={
            "company": company,
            "ticker": ticker,
            "filing_type": filing_type,
            "year": year,
            "auto_detect": auto_detect_metadata,
        },
    )

    from backend.schemas.documents import DocumentMetadata

    return DocumentResponse(
        id=doc_id,
        filename=file.filename,
        status=DocumentStatus.PENDING,
        metadata=DocumentMetadata(
            company=company,
            ticker=ticker,
            filing_type=filing_type,
            year=year,
        ),
    )


@router.get(
    "/{doc_id}/status",
    response_model=DocumentResponse,
    summary="Check ingestion status",
)
async def get_ingestion_status(
    doc_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
) -> DocumentResponse:
    from database.repositories.document_repo import DocumentRepository

    repo = DocumentRepository(db)
    doc = await repo.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse.model_validate(doc)


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its vectors",
)
async def delete_document(
    doc_id: UUID,
    db: DBSession,
    settings: Settings,
    current_user: CurrentUser,
) -> None:
    from database.repositories.document_repo import DocumentRepository
    from vectordb.qdrant.client import QdrantClientWrapper

    repo = DocumentRepository(db)
    qdrant = QdrantClientWrapper(settings=settings)

    await repo.delete(doc_id)
    await qdrant.delete_by_document_id(str(doc_id))

    logger.info("document_deleted", doc_id=str(doc_id))
