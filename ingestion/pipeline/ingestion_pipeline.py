"""
Ingestion pipeline — orchestrates loader → extractor → chunker → embedder → vectordb.
This is the Single Responsibility version: each step is swappable independently.
"""

import time
from typing import Any
from uuid import UUID

from config.logging_config import get_logger
from config.settings import AppSettings
from ingestion.loaders.csv_loader import CSVLoader
from ingestion.loaders.docx_loader import DOCXLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.base_loader import BaseDocumentLoader
from ingestion.chunkers.parent_child_chunker import ParentChildChunker
from ingestion.chunkers.recursive_chunker import RecursiveChunker
from ingestion.metadata.extractor import MetadataExtractor

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Full ingestion pipeline.
    Supports pluggable loaders, chunkers, and embedders.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.loaders: list[BaseDocumentLoader] = [
            PDFLoader(),
            DOCXLoader(),
            CSVLoader(),
        ]
        self.metadata_extractor = MetadataExtractor()

        # Default chunker — configurable per document
        self.chunker = ParentChildChunker(
            parent_chunk_size=settings.parent_chunk_size,
            child_chunk_size=settings.chunk_size,
        )

    async def process_document(
        self,
        doc_id: UUID,
        content: bytes,
        filename: str,
        content_type: str,
        metadata_hints: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Full pipeline: load → extract metadata → chunk → embed → index.
        Returns summary stats. Updates DB status throughout.
        """
        start = time.perf_counter()
        logger.info("ingestion_pipeline_start", doc_id=str(doc_id), filename=filename)

        try:
            # Step 1: Load document
            loader = self._get_loader(filename)
            raw_doc = await loader.load(content, filename)

            # Step 2: Extract metadata
            meta = self.metadata_extractor.extract(
                text=raw_doc.content,
                filename=filename,
                hints=metadata_hints,
            )

            metadata_dict = {
                "doc_id": str(doc_id),
                "filename": filename,
                "company": meta.company,
                "ticker": meta.ticker,
                "year": meta.year,
                "quarter": meta.quarter,
                "filing_type": meta.filing_type,
            }

            # Step 3: Chunk
            chunks = self.chunker.chunk(
                text=raw_doc.content,
                metadata=metadata_dict,
                doc_id=str(doc_id),
            )

            # Also chunk tables separately for better table retrieval
            table_chunks = self._chunk_tables(raw_doc.tables, metadata_dict, str(doc_id))
            chunks.extend(table_chunks)

            # Step 4: Filter to child chunks only for embedding
            embed_chunks = [c for c in chunks if c.metadata.get("chunk_type") != "parent"]
            parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]

            # Step 5: Embed and index
            from embeddings.models.bge_embedder import BGEEmbedder
            from vectordb.qdrant.client import QdrantClientWrapper

            embedder = BGEEmbedder(settings=self.settings)
            await embedder.initialize()

            qdrant = QdrantClientWrapper(settings=self.settings)
            await qdrant.ensure_collection()

            texts = [c.content for c in embed_chunks]
            embeddings = await embedder.aembed_documents(texts)

            await qdrant.upsert_chunks(
                chunks=embed_chunks,
                embeddings=embeddings,
                parent_chunks={p.chunk_id: p for p in parent_chunks},
            )

            elapsed = (time.perf_counter() - start) * 1000
            result = {
                "doc_id": str(doc_id),
                "total_chunks": len(chunks),
                "embedded_chunks": len(embed_chunks),
                "tables_chunked": len(table_chunks),
                "elapsed_ms": elapsed,
                "company": meta.company,
                "year": meta.year,
            }

            logger.info("ingestion_pipeline_complete", **result)
            return result

        except Exception as exc:
            logger.exception("ingestion_pipeline_failed", doc_id=str(doc_id), error=str(exc))
            raise

    def _get_loader(self, filename: str) -> BaseDocumentLoader:
        for loader in self.loaders:
            if loader.can_handle(filename):
                return loader
        from backend.core.exceptions import UnsupportedFileTypeError
        raise UnsupportedFileTypeError(f"No loader available for: {filename}")

    def _chunk_tables(
        self, tables: list[dict], metadata: dict[str, Any], doc_id: str
    ) -> list:
        """Convert tables to text chunks with table-specific metadata."""
        import hashlib
        from ingestion.chunkers.base_chunker import DocumentChunk

        table_chunks = []
        for i, table in enumerate(tables):
            rows = table.get("data", [])
            if not rows:
                continue
            # Convert table to markdown for better LLM readability
            md_table = self._table_to_markdown(rows)
            cid = hashlib.md5(f"{doc_id}:table:{i}".encode()).hexdigest()
            table_chunks.append(DocumentChunk(
                chunk_id=cid,
                content=md_table,
                metadata={
                    **metadata,
                    "chunk_type": "table",
                    "chunk_strategy": "table",
                    "table_index": i,
                    "page": table.get("page", 0),
                },
                chunk_index=i,
            ))
        return table_chunks

    def _table_to_markdown(self, rows: list[list]) -> str:
        if not rows:
            return ""
        header = "| " + " | ".join(str(c) for c in rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
        body = "\n".join(
            "| " + " | ".join(str(c) for c in row) + " |"
            for row in rows[1:]
        )
        return f"{header}\n{separator}\n{body}"
