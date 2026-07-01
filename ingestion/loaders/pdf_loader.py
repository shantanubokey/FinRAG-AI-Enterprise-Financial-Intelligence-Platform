"""
PDF loader using PyMuPDF (fitz) for text + pdfplumber for tables.
PyMuPDF is ~10x faster than pypdf for text extraction.
pdfplumber handles complex table layouts better than camelot for financials.
"""

import io
from typing import Any

from config.logging_config import get_logger
from ingestion.loaders.base_loader import BaseDocumentLoader, RawDocument

logger = get_logger(__name__)


class PDFLoader(BaseDocumentLoader):
    """
    Two-pass PDF extraction:
    Pass 1 (PyMuPDF): fast text + page metadata
    Pass 2 (pdfplumber): accurate table extraction
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    async def load(self, content: bytes, filename: str) -> RawDocument:
        logger.info("pdf_loader_start", filename=filename, size_bytes=len(content))

        page_contents: list[str] = []
        tables: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        file_metadata: dict[str, Any] = {}

        try:
            # Pass 1: PyMuPDF for text
            import fitz  # PyMuPDF

            doc = fitz.open(stream=content, filetype="pdf")
            file_metadata = dict(doc.metadata)
            file_metadata["page_count"] = len(doc)

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                page_contents.append(text)

                # Extract embedded images
                for img in page.get_images(full=True):
                    images.append({
                        "page": page_num,
                        "xref": img[0],
                        "width": img[2],
                        "height": img[3],
                    })

            doc.close()

        except ImportError:
            logger.warning("pymupdf_not_installed_falling_back_to_pypdf")
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(content))
            file_metadata["page_count"] = len(reader.pages)
            for page in reader.pages:
                page_contents.append(page.extract_text() or "")

        # Pass 2: pdfplumber for tables
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    for table in page.extract_tables():
                        if table:
                            tables.append({
                                "page": page_num,
                                "data": table,
                                "headers": table[0] if table else [],
                                "rows": table[1:] if len(table) > 1 else [],
                            })

        except ImportError:
            logger.warning("pdfplumber_not_installed_skipping_table_extraction")
        except Exception as exc:
            logger.warning("table_extraction_failed", error=str(exc))

        full_text = "\n\n".join(page_contents)
        logger.info(
            "pdf_loader_complete",
            filename=filename,
            pages=len(page_contents),
            tables_found=len(tables),
            char_count=len(full_text),
        )

        return RawDocument(
            content=full_text,
            tables=tables,
            images=images,
            metadata=file_metadata,
            page_contents=page_contents,
        )
