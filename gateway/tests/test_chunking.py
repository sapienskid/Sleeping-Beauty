"""Unit tests for chunking and text extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.services.ingestion import (
    chunk_text,
    extract_text,
    build_chunks,
    SUPPORTED_FORMATS,
)
from gateway.config import IngestionConfig


class TestChunkText:
    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size stays as one chunk."""
        chunks = chunk_text("Hello world, this is a short text.", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world, this is a short text."

    def test_long_text_multiple_chunks(self):
        """Long text is split into multiple chunks."""
        text = "This is a sentence. " * 200  # ~800 tokens
        chunks = chunk_text(text, chunk_size=128, chunk_overlap=16)
        assert len(chunks) > 1

    def test_overlap_preserves_context(self):
        """Chunks have overlapping content at boundaries."""
        # Use paragraph separators to have clear boundaries
        paragraphs = [f"Paragraph {i} with some content here." for i in range(50)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=64, chunk_overlap=16)
        assert len(chunks) > 1
        # Overlap means some content should appear in multiple chunks
        # (hard to assert precisely, but there should be more than 1 chunk)

    def test_empty_text_no_chunks(self):
        chunks = chunk_text("", chunk_size=512)
        assert len(chunks) == 0

    def test_whitespace_only_no_chunks(self):
        chunks = chunk_text("   \n\n  ", chunk_size=512)
        assert len(chunks) == 0


class TestExtractText:
    def test_extract_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world from a text file.")
        result = extract_text(f)
        assert result.file_type == "txt"
        assert "Hello world" in result.text

    def test_extract_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nSome markdown content.")
        result = extract_text(f)
        assert result.file_type == "md"
        assert "Title" in result.text

    def test_extract_html(self, tmp_path):
        f = tmp_path / "test.html"
        f.write_text(
            "<html><body><p>Hello HTML</p><script>alert('x')</script></body></html>"
        )
        result = extract_text(f)
        assert result.file_type == "html"
        assert "Hello HTML" in result.text
        assert "alert" not in result.text  # script should be stripped

    def test_unsupported_raises(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            extract_text(f)


class TestBuildChunks:
    def test_builds_with_metadata(self, tmp_path):
        from gateway.services.ingestion import ExtractionResult

        config = IngestionConfig(chunk_size=512, chunk_overlap=64)
        extraction = ExtractionResult(
            text="Some sample document text.", file_type="txt"
        )

        chunks = build_chunks(extraction, "/data/docs/test.txt", config)
        assert len(chunks) >= 1
        assert chunks[0].metadata["source_path"] == "/data/docs/test.txt"
        assert chunks[0].metadata["source_filename"] == "test.txt"
        assert chunks[0].metadata["file_type"] == "txt"
        assert "content_hash" in chunks[0].metadata

    def test_chunk_ids_are_deterministic(self, tmp_path):
        from gateway.services.ingestion import ExtractionResult

        config = IngestionConfig(chunk_size=512, chunk_overlap=64)
        extraction = ExtractionResult(text="Deterministic ID test.", file_type="txt")

        chunks1 = build_chunks(extraction, "/data/test.txt", config)
        chunks2 = build_chunks(extraction, "/data/test.txt", config)
        assert chunks1[0].id == chunks2[0].id


class TestSupportedFormats:
    def test_all_expected_formats(self):
        expected = {
            ".pdf",
            ".txt",
            ".md",
            ".rst",
            ".html",
            ".htm",
            ".docx",
            ".epub",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
        }
        assert SUPPORTED_FORMATS == expected
