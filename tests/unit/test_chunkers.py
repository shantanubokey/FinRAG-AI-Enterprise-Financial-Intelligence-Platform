"""Unit tests for chunking strategies."""

import pytest
from ingestion.chunkers.recursive_chunker import RecursiveChunker
from ingestion.chunkers.parent_child_chunker import ParentChildChunker


SAMPLE_TEXT = """Apple Inc. reported revenue of $383.3 billion for fiscal year 2023.
The company's gross margin was 44.1%, up from 43.3% in the prior year.

iPhone revenue was $200.6 billion, representing 52% of total revenue.
Services revenue grew 16% year-over-year to $85.2 billion.

Risk factors include macroeconomic headwinds, supply chain concentration in China,
and increasing competition in key product categories."""


class TestRecursiveChunker:

    def test_basic_chunking(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT, {"doc_id": "test"}, "doc1")
        assert len(chunks) > 0
        assert all(c.content for c in chunks)

    def test_chunk_ids_unique(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT, {}, "doc1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_strategy_name(self):
        assert RecursiveChunker().strategy_name == "recursive"

    def test_metadata_preserved(self):
        meta = {"company": "Apple", "year": 2023}
        chunker = RecursiveChunker(chunk_size=200)
        chunks = chunker.chunk(SAMPLE_TEXT, meta, "doc1")
        for chunk in chunks:
            assert chunk.metadata["company"] == "Apple"


class TestParentChildChunker:

    def test_creates_parent_and_child_chunks(self):
        chunker = ParentChildChunker(parent_chunk_size=500, child_chunk_size=100)
        chunks = chunker.chunk(SAMPLE_TEXT, {}, "doc1")

        parents = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        children = [c for c in chunks if c.metadata.get("chunk_type") == "child"]

        assert len(parents) > 0
        assert len(children) > 0
        assert len(children) > len(parents)  # More children than parents

    def test_children_reference_parents(self):
        chunker = ParentChildChunker(parent_chunk_size=500, child_chunk_size=100)
        chunks = chunker.chunk(SAMPLE_TEXT, {}, "doc1")

        parent_ids = {c.chunk_id for c in chunks if c.metadata.get("chunk_type") == "parent"}
        children = [c for c in chunks if c.metadata.get("chunk_type") == "child"]

        for child in children:
            assert child.parent_id in parent_ids
