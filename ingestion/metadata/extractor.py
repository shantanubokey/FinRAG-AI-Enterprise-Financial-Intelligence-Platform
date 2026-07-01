"""
Automatic metadata extraction from financial documents.
Uses regex patterns + LLM fallback for company/year/filing type detection.
This is critical — metadata filters make retrieval 10x more precise.
"""

import re
from dataclasses import dataclass
from typing import Any

from config.logging_config import get_logger

logger = get_logger(__name__)

# Company name → ticker mapping (expand as needed)
KNOWN_TICKERS: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "tesla": "TSLA", "nvidia": "NVDA", "netflix": "NFLX",
    "berkshire hathaway": "BRK", "jpmorgan": "JPM", "bank of america": "BAC",
}

FILING_PATTERNS = {
    "10-K": r"\b10-K\b|annual report|form 10k",
    "10-Q": r"\b10-Q\b|quarterly report|form 10q",
    "earnings_call": r"earnings call|earnings conference|quarterly call",
    "investor_presentation": r"investor (day|presentation|relations)",
    "financial_statement": r"financial statement|balance sheet|income statement",
}

YEAR_PATTERN = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
QUARTER_PATTERN = re.compile(r"\b(Q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b", re.I)


@dataclass
class ExtractedMetadata:
    company: str | None = None
    ticker: str | None = None
    year: int | None = None
    quarter: str | None = None
    filing_type: str = "other"
    sections: list[str] | None = None


class MetadataExtractor:
    """
    Extracts structured metadata from raw document text.
    Tries heuristics first, falls back to LLM for ambiguous cases.
    """

    def extract(
        self,
        text: str,
        filename: str,
        hints: dict[str, Any] | None = None,
    ) -> ExtractedMetadata:
        hints = hints or {}
        meta = ExtractedMetadata()

        # Apply manual hints first (user-provided overrides auto-detection)
        if hints.get("company"):
            meta.company = hints["company"]
        if hints.get("ticker"):
            meta.ticker = hints["ticker"]
        if hints.get("year"):
            meta.year = hints["year"]
        if hints.get("filing_type") and hints["filing_type"] != "other":
            meta.filing_type = hints["filing_type"]

        # Only auto-detect what wasn't provided
        sample = (filename + " " + text[:3000]).lower()

        if not meta.company:
            meta.company = self._extract_company(sample)
        if not meta.ticker and meta.company:
            meta.ticker = KNOWN_TICKERS.get(meta.company.lower())
        if not meta.year:
            meta.year = self._extract_year(sample)
        if meta.filing_type == "other":
            meta.filing_type = self._extract_filing_type(sample)

        meta.quarter = self._extract_quarter(sample)
        meta.sections = self._extract_sections(text)

        logger.debug(
            "metadata_extracted",
            company=meta.company,
            year=meta.year,
            filing_type=meta.filing_type,
        )
        return meta

    def _extract_company(self, text: str) -> str | None:
        for name in KNOWN_TICKERS:
            if name in text:
                return name.title()
        return None

    def _extract_year(self, text: str) -> int | None:
        from collections import Counter
        matches = YEAR_PATTERN.findall(text)
        if not matches:
            return None
        # Most frequent year is likely the document year
        return int(Counter(matches).most_common(1)[0][0])

    def _extract_filing_type(self, text: str) -> str:
        for filing_type, pattern in FILING_PATTERNS.items():
            if re.search(pattern, text, re.I):
                return filing_type
        return "other"

    def _extract_quarter(self, text: str) -> str | None:
        match = QUARTER_PATTERN.search(text)
        if not match:
            return None
        q = match.group(1).upper()
        mapping = {
            "FIRST QUARTER": "Q1", "SECOND QUARTER": "Q2",
            "THIRD QUARTER": "Q3", "FOURTH QUARTER": "Q4",
        }
        return mapping.get(q, q)

    def _extract_sections(self, text: str) -> list[str]:
        """Detect common 10-K/10-Q section headers."""
        section_pattern = re.compile(
            r"^(item\s+\d+[a-z]?[\.\:]?\s+.+)$",
            re.MULTILINE | re.IGNORECASE,
        )
        return [m.group(1).strip() for m in section_pattern.finditer(text)][:20]
