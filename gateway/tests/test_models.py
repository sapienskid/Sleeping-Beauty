"""Unit tests for Pydantic models."""

from __future__ import annotations

import pytest

from gateway.models import (
    RetrievedChunk,
    RetrievalResult,
    build_low_confidence_response,
    IngestRequest,
    MemoryCreateRequest,
    MemoryItem,
    CollectionInfo,
)


class TestRetrievedChunk:
    def test_pdf_citation_anchor(self):
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="hello",
            source_path="/data/docs/report.pdf",
            file_type="pdf",
            page_number=5,
        )
        assert chunk.citation_anchor() == "file:///data/docs/report.pdf#page=5"

    def test_txt_citation_with_lines(self):
        chunk = RetrievedChunk(
            chunk_id="c2",
            text="hello",
            source_path="/data/docs/notes.txt",
            file_type="txt",
            line_start=10,
            line_end=20,
        )
        assert chunk.citation_anchor() == "file:///data/docs/notes.txt#L10-L20"

    def test_md_citation_with_heading(self):
        chunk = RetrievedChunk(
            chunk_id="c3",
            text="hello",
            source_path="/data/docs/guide.md",
            file_type="md",
            section_heading="Getting Started",
        )
        assert chunk.citation_anchor() == "file:///data/docs/guide.md#getting-started"

    def test_image_citation(self):
        chunk = RetrievedChunk(
            chunk_id="c4",
            text="OCR text",
            source_path="/data/docs/scan.png",
            file_type="png",
        )
        assert chunk.citation_anchor() == "file:///data/docs/scan.png#page=1"

    def test_fallback_citation(self):
        chunk = RetrievedChunk(
            chunk_id="c5",
            text="hello",
            source_path="/data/docs/file.docx",
            file_type="docx",
        )
        assert chunk.citation_anchor() == "file:///data/docs/file.docx"

    def test_citation_link_format(self):
        chunk = RetrievedChunk(
            chunk_id="chunk_abc123",
            text="hello",
            source_path="/data/docs/test.pdf",
            file_type="pdf",
            page_number=3,
        )
        assert chunk.citation_link() == "[chunk_abc123](file:///data/docs/test.pdf#page=3)"


class TestRetrievalResult:
    def test_has_results_true(self):
        result = RetrievalResult(
            chunks=[RetrievedChunk(chunk_id="c1", text="x", similarity=0.8)],
            collections_searched=["test"],
            max_similarity=0.8,
        )
        assert result.has_results is True

    def test_has_results_false(self):
        result = RetrievalResult()
        assert result.has_results is False


class TestBuildLowConfidenceResponse:
    def test_basic_structure(self):
        result = RetrievalResult(
            chunks=[
                RetrievedChunk(
                    chunk_id="c1",
                    text="Some irrelevant text here for testing.",
                    source_path="/data/docs/test.txt",
                    file_type="txt",
                    similarity=0.15,
                )
            ],
            collections_searched=["project-docs"],
            max_similarity=0.15,
        )
        resp = build_low_confidence_response("sleeping-beauty", result, 0.35)

        assert resp["object"] == "chat.completion"
        assert len(resp["choices"]) == 1
        content = resp["choices"][0]["message"]["content"]
        assert "sufficient evidence" in content.lower()
        assert "project-docs" in content
        assert "0.15" in content
        assert "0.35" in content

    def test_empty_retrieval(self):
        result = RetrievalResult(collections_searched=["test"], max_similarity=0.0)
        resp = build_low_confidence_response("sleeping-beauty", result, 0.35)
        content = resp["choices"][0]["message"]["content"]
        assert "sufficient evidence" in content.lower()


class TestPydanticModels:
    def test_ingest_request_defaults(self):
        req = IngestRequest(source_path="/data/test.txt")
        assert req.collection == "project-docs"
        assert req.force_reindex is False
        assert req.recursive is True

    def test_memory_create_request_defaults(self):
        req = MemoryCreateRequest(content="hello")
        assert req.type == "preference"
        assert req.tags == []

    def test_memory_item_serialization(self):
        item = MemoryItem(
            id="mem_123",
            content="Test memory",
            type="fact",
            tags=["tag1", "tag2"],
            created_at="2026-01-01T00:00:00Z",
        )
        d = item.model_dump()
        assert d["id"] == "mem_123"
        assert d["tags"] == ["tag1", "tag2"]

    def test_collection_info(self):
        info = CollectionInfo(name="test", count=42)
        assert info.model_dump() == {"name": "test", "count": 42}
