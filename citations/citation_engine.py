"""
Citation engine — maps answer text back to source chunks.
Every claim in the answer should traceable to a specific document.
"""

import re
from typing import Any

from config.logging_config import get_logger

logger = get_logger(__name__)

# Pattern matching [Company | Filing | Year | Page N]
CITATION_PATTERN = re.compile(
    r"\[([^\]]+)\s*\|\s*([^\]]+)\s*\|\s*(\d{4}|N/A)\s*\|\s*Page\s*(\d+|N/A)\]",
    re.IGNORECASE,
)


class CitationEngine:
    """
    Extracts and enriches citations from generated answers.
    Links LLM-generated citations back to actual retrieved chunks.
    """

    def extract_citations(
        self,
        answer: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Find all [Company | Filing | Year | Page N] citations in the answer
        and enrich them with chunk metadata and confidence scores.
        """
        # Build lookup from metadata
        chunk_lookup = self._build_lookup(chunks)

        citations = []
        seen = set()

        for match in CITATION_PATTERN.finditer(answer):
            company = match.group(1).strip()
            filing = match.group(2).strip()
            year_str = match.group(3).strip()
            page_str = match.group(4).strip()

            key = (company.lower(), filing.lower(), year_str, page_str)
            if key in seen:
                continue
            seen.add(key)

            year = int(year_str) if year_str.isdigit() else None
            page = int(page_str) if page_str.isdigit() else None

            # Try to find matching chunk
            matched_chunk = self._find_matching_chunk(chunk_lookup, company, filing, year, page)

            citation = {
                "company": company,
                "filing_type": filing,
                "year": year,
                "page_number": page,
                "chunk_content": matched_chunk.get("raw_content", "")[:300] if matched_chunk else "",
                "confidence_score": min(1.0, matched_chunk.get("rerank_score", matched_chunk.get("score", 0.5))) if matched_chunk else 0.3,
                "metadata": matched_chunk.get("metadata", {}) if matched_chunk else {},
            }
            citations.append(citation)

        # If no citations found in answer but chunks exist, add top chunks as citations
        if not citations and chunks:
            for chunk in chunks[:3]:
                meta = chunk.get("metadata", {})
                citations.append({
                    "company": meta.get("company", "Unknown"),
                    "filing_type": meta.get("filing_type", "Document"),
                    "year": meta.get("year"),
                    "page_number": meta.get("page"),
                    "chunk_content": chunk.get("raw_content", chunk.get("content", ""))[:300],
                    "confidence_score": min(1.0, chunk.get("rerank_score", chunk.get("score", 0.5))),
                    "metadata": meta,
                })

        logger.debug("citations_extracted", count=len(citations))
        return citations

    def _build_lookup(self, chunks: list[dict[str, Any]]) -> dict:
        lookup = {}
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            key = (
                str(meta.get("company", "")).lower(),
                str(meta.get("filing_type", "")).lower(),
                meta.get("year"),
                meta.get("page"),
            )
            if key not in lookup:
                lookup[key] = chunk
        return lookup

    def _find_matching_chunk(
        self, lookup: dict, company: str, filing: str, year: int | None, page: int | None
    ) -> dict | None:
        key = (company.lower(), filing.lower(), year, page)
        return lookup.get(key)
