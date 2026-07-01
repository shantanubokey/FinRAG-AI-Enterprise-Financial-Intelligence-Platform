"""Unit tests for metadata extraction."""

from ingestion.metadata.extractor import MetadataExtractor


def test_extract_company_apple():
    ext = MetadataExtractor()
    meta = ext.extract("Apple Inc. annual report 2023 10-K filing", "apple_10k.pdf")
    assert meta.company is not None
    assert "apple" in meta.company.lower()


def test_extract_year():
    ext = MetadataExtractor()
    meta = ext.extract("Fiscal year 2023 revenue was $383 billion", "report.pdf")
    assert meta.year == 2023


def test_extract_filing_type_10k():
    ext = MetadataExtractor()
    meta = ext.extract("Form 10-K Annual Report for fiscal year ended September 2023", "file.pdf")
    assert meta.filing_type == "10-K"


def test_extract_filing_type_earnings():
    ext = MetadataExtractor()
    meta = ext.extract("Q3 2023 earnings call transcript", "transcript.pdf")
    assert meta.filing_type == "earnings_call"


def test_manual_hints_override_detection():
    ext = MetadataExtractor()
    meta = ext.extract(
        "Some document content about Apple",
        "doc.pdf",
        hints={"company": "Microsoft", "year": 2022},
    )
    assert meta.company == "Microsoft"
    assert meta.year == 2022


def test_extract_quarter():
    ext = MetadataExtractor()
    meta = ext.extract("Q3 2023 quarterly report 10-Q", "q3.pdf")
    assert meta.quarter == "Q3"
