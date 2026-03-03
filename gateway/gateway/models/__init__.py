"""Pydantic models for gateway requests and responses."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


# ─── Ingestion ───────────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    source_path: str
    collection: str = "project-docs"
    force_reindex: bool = False
    recursive: bool = True


class IngestResponse(BaseModel):
    status: str = "accepted"
    job_id: str
    source_path: str
    collection: str


class IngestJobStatus(BaseModel):
    job_id: str
    status: str  # "running", "completed", "failed"
    chunks_created: int = 0
    source_path: str = ""
    collection: str = ""
    duration_seconds: float = 0.0
    error: str | None = None


# ─── Memory ──────────────────────────────────────────────────────────────────


class MemoryCreateRequest(BaseModel):
    content: str
    type: str = "preference"  # "preference" | "fact" | "writing-style"
    tags: list[str] = Field(default_factory=list)


class MemoryItem(BaseModel):
    id: str
    content: str
    type: str
    tags: list[str] = Field(default_factory=list)
    created_at: str = ""


class MemoryCreateResponse(BaseModel):
    id: str
    status: str = "stored"


class MemoryListResponse(BaseModel):
    memories: list[MemoryItem]


# ─── Collections ─────────────────────────────────────────────────────────────


class CollectionInfo(BaseModel):
    name: str
    count: int


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]


# ─── Retrieval (internal) ───────────────────────────────────────────────────


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with metadata for prompt injection."""

    chunk_id: str
    text: str
    source_path: str = ""
    source_filename: str = ""
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    section_heading: str | None = None
    file_type: str = ""
    similarity: float = 0.0

    def citation_anchor(self) -> str:
        """Build the file://path#anchor citation string."""
        base = f"file://{self.source_path}"
        if self.file_type == "pdf" and self.page_number is not None:
            return f"{base}#page={self.page_number}"
        if (
            self.file_type in ("txt", "md", "rst", "py", "js", "ts")
            and self.line_start is not None
        ):
            anchor = f"#L{self.line_start}"
            if self.line_end is not None:
                anchor += f"-L{self.line_end}"
            return f"{base}{anchor}"
        if self.file_type in ("md", "html") and self.section_heading:
            slug = self.section_heading.lower().replace(" ", "-")
            return f"{base}#{slug}"
        if self.file_type in ("png", "jpg", "jpeg", "tiff"):
            return f"{base}#page=1"
        return base

    def citation_link(self) -> str:
        """Full markdown citation: [chunk_id](file://path#anchor)."""
        return f"[{self.chunk_id}]({self.citation_anchor()})"


class RetrievalResult(BaseModel):
    """Aggregated retrieval results."""

    chunks: list[RetrievedChunk] = Field(default_factory=list)
    collections_searched: list[str] = Field(default_factory=list)
    max_similarity: float = 0.0

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0


# ─── Chat (low-confidence response builder) ─────────────────────────────────


def build_low_confidence_response(
    model: str,
    retrieval: RetrievalResult,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Build a canned response when retrieval confidence is below threshold."""
    best_chunk_info = ""
    if retrieval.chunks:
        best = retrieval.chunks[0]
        snippet = best.text[:120].replace("\n", " ")
        best_chunk_info = (
            f"\n**Best matching chunk:** {best.citation_link()} "
            f'— "{snippet}..." (similarity: {best.similarity:.2f})'
        )

    content = (
        "I don't have sufficient evidence to answer this question.\n\n"
        f"**Collections searched:** {', '.join(retrieval.collections_searched)}\n"
        f"**Top result similarity:** {retrieval.max_similarity:.2f} "
        f"(threshold: {confidence_threshold})"
        f"{best_chunk_info}\n\n"
        "Try rephrasing your question or check if the relevant documents "
        "have been ingested."
    )

    return {
        "id": f"chatcmpl-gateway-noconf-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
