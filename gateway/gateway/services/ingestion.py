"""Ingestion service — extract text, chunk, embed, upsert to Chroma."""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import chromadb

    from gateway.config import IngestionConfig
    from gateway.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

# ─── Data structures ─────────────────────────────────────────────────────────


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    text: str
    pages: list[PageInfo] | None = None  # for PDFs
    file_type: str = ""


@dataclass
class PageInfo:
    page_number: int
    text: str
    ocr_applied: bool = False


# ─── Supported formats ──────────────────────────────────────────────────────

SUPPORTED_FORMATS: set[str] = {
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


# ─── Text extraction ────────────────────────────────────────────────────────


def extract_text(file_path: Path) -> ExtractionResult:
    """Extract text from a file based on its extension."""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(file_path)
    if ext in (".txt", ".md", ".rst"):
        return _extract_plaintext(file_path, ext.lstrip("."))
    if ext in (".html", ".htm"):
        return _extract_html(file_path)
    if ext == ".docx":
        return _extract_docx(file_path)
    if ext == ".epub":
        return _extract_epub(file_path)
    if ext in (".png", ".jpg", ".jpeg", ".tiff"):
        return _extract_image_ocr(file_path, ext.lstrip("."))

    raise ValueError(f"Unsupported format: {ext}")


def _extract_pdf(file_path: Path) -> ExtractionResult:
    import fitz  # pymupdf

    doc = fitz.open(str(file_path))
    pages: list[PageInfo] = []
    full_text_parts: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        ocr_applied = False
        if not text:
            # Image-only page — attempt OCR
            text = _ocr_pdf_page(page)
            ocr_applied = bool(text)

        if text:
            pages.append(
                PageInfo(
                    page_number=page_num + 1,  # 1-indexed
                    text=text,
                    ocr_applied=ocr_applied,
                )
            )
            full_text_parts.append(text)

    doc.close()
    return ExtractionResult(
        text="\n\n".join(full_text_parts),
        pages=pages,
        file_type="pdf",
    )


def _ocr_pdf_page(page: Any) -> str:
    """OCR a single PDF page by rendering to image then running tesseract."""
    try:
        import pytesseract
        from PIL import Image
        import io

        # Render page at 300 DPI
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Grayscale for better OCR
        img = img.convert("L")

        text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()
    except Exception as e:
        logger.warning("OCR failed for page: %s", e)
        return ""


def _extract_plaintext(file_path: Path, file_type: str) -> ExtractionResult:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return ExtractionResult(text=text, file_type=file_type)


def _extract_html(file_path: Path) -> ExtractionResult:
    from bs4 import BeautifulSoup

    raw = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    # Remove script/style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return ExtractionResult(text=text, file_type="html")


def _extract_docx(file_path: Path) -> ExtractionResult:
    from docx import Document

    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return ExtractionResult(text="\n\n".join(paragraphs), file_type="docx")


def _extract_epub(file_path: Path) -> ExtractionResult:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(file_path))
    texts: list[str] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text:
            texts.append(text)

    return ExtractionResult(text="\n\n".join(texts), file_type="epub")


def _extract_image_ocr(file_path: Path, file_type: str) -> ExtractionResult:
    import pytesseract
    from PIL import Image

    img = Image.open(str(file_path)).convert("L")
    text = pytesseract.image_to_string(img, lang="eng").strip()
    return ExtractionResult(text=text, file_type=file_type)


# ─── Chunking ───────────────────────────────────────────────────────────────

# Ordered coarse → fine split separators.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Split text into overlapping chunks using LangChain token-aware splitting."""
    if not text.strip():
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        # Fallback for constrained environments where LangChain splitter
        # is unavailable at runtime.
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def _token_len(s: str) -> int:
            return len(enc.encode(s))

        return _recursive_split(text, chunk_size, chunk_overlap, SEPARATORS, _token_len)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _recursive_split(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: list[str],
    token_len: Any,
) -> list[str]:
    """Recursively split text by separators until chunks fit within token limit."""
    if not text.strip():
        return []

    if token_len(text) <= chunk_size:
        return [text.strip()]

    # Find the best separator
    sep = separators[0] if separators else " "
    for s in separators:
        if s in text:
            sep = s
            break

    parts = text.split(sep)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for part in parts:
        part_len = token_len(part)
        if current and current_len + part_len > chunk_size:
            chunk_text_str = sep.join(current).strip()
            if chunk_text_str:
                chunks.append(chunk_text_str)

            # Overlap: keep tail tokens
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                p_len = token_len(p)
                if overlap_len + p_len > overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_len += p_len

            current = overlap_parts + [part]
            current_len = overlap_len + part_len
        else:
            current.append(part)
            current_len += part_len

    # Flush remainder
    if current:
        remainder = sep.join(current).strip()
        if remainder:
            chunks.append(remainder)

    return chunks


def build_chunks(
    extraction: ExtractionResult,
    source_path: str,
    config: IngestionConfig,
) -> list[Chunk]:
    """Build Chunk objects with metadata from extraction result."""
    source_filename = os.path.basename(source_path)
    content_hash = hashlib.sha256(extraction.text.encode()).hexdigest()

    # For PDFs with page info, chunk per-page for accurate page metadata
    if extraction.pages:
        return _chunk_pdf_pages(
            extraction.pages,
            source_path,
            source_filename,
            content_hash,
            extraction.file_type,
            config,
        )

    # For everything else, chunk the full text
    raw_chunks = chunk_text(
        extraction.text,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    chunks: list[Chunk] = []
    for i, text in enumerate(raw_chunks):
        chunk_id = _make_chunk_id(source_path, i)
        meta: dict[str, Any] = {
            "source_path": source_path,
            "source_filename": source_filename,
            "file_type": extraction.file_type,
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "content_hash": content_hash,
            "ocr_applied": False,
        }

        # Add line numbers for plain text / markdown
        if extraction.file_type in ("txt", "md", "rst"):
            # Approximate line numbers from position in full text
            start_pos = extraction.text.find(text[:80])
            if start_pos >= 0:
                line_start = extraction.text[:start_pos].count("\n") + 1
                line_end = line_start + text.count("\n")
                meta["line_start"] = line_start
                meta["line_end"] = line_end

        chunks.append(Chunk(id=chunk_id, text=text, metadata=meta))

    return chunks


def _chunk_pdf_pages(
    pages: list[PageInfo],
    source_path: str,
    source_filename: str,
    content_hash: str,
    file_type: str,
    config: IngestionConfig,
) -> list[Chunk]:
    """Chunk PDF text page-by-page, preserving page number metadata."""
    chunks: list[Chunk] = []
    global_idx = 0

    for page in pages:
        page_chunks = chunk_text(
            page.text,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        for text in page_chunks:
            chunk_id = _make_chunk_id(source_path, global_idx)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    metadata={
                        "source_path": source_path,
                        "source_filename": source_filename,
                        "file_type": file_type,
                        "page_number": page.page_number,
                        "chunk_index": global_idx,
                        "content_hash": content_hash,
                        "ocr_applied": page.ocr_applied,
                    },
                )
            )
            global_idx += 1

    # Backfill total_chunks
    for chunk in chunks:
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


def _make_chunk_id(source_path: str, index: int) -> str:
    """Deterministic chunk ID from source path and index."""
    raw = f"{source_path}::{index}"
    return f"chunk_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


# ─── Ingestion orchestrator ─────────────────────────────────────────────────


class IngestionService:
    """Orchestrates document ingestion: extract → chunk → embed → upsert."""

    def __init__(
        self,
        chroma_client: chromadb.HttpClient,
        embedder: EmbeddingService,
        config: IngestionConfig,
    ) -> None:
        self._chroma = chroma_client
        self._embedder = embedder
        self._config = config
        self._jobs: dict[str, dict[str, Any]] = {}

    def ingest_file(
        self,
        file_path: str,
        collection_name: str,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Ingest a single file into a Chroma collection.

        Returns job status dict.
        """
        job_id = f"ingest-{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        path = Path(file_path)
        if not path.exists():
            return self._fail(
                job_id, file_path, collection_name, f"File not found: {file_path}"
            )

        if path.suffix.lower() not in SUPPORTED_FORMATS:
            return self._fail(
                job_id, file_path, collection_name, f"Unsupported format: {path.suffix}"
            )

        collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Deduplication check
        if not force_reindex:
            existing = collection.get(where={"source_path": file_path})
            if existing and existing["ids"]:
                # Check content hash
                file_content = path.read_bytes()
                current_hash = hashlib.sha256(file_content).hexdigest()
                old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
                if old_meta.get("content_hash") == current_hash:
                    logger.info("File unchanged, skipping: %s", file_path)
                    return {
                        "job_id": job_id,
                        "status": "skipped",
                        "chunks_created": 0,
                        "source_path": file_path,
                        "collection": collection_name,
                        "duration_seconds": round(time.time() - start_time, 2),
                    }
                # File changed — delete old chunks
                logger.info("File changed, re-indexing: %s", file_path)
                collection.delete(where={"source_path": file_path})

        # Extract
        try:
            extraction = extract_text(path)
        except Exception as e:
            return self._fail(
                job_id, file_path, collection_name, f"Extraction failed: {e}"
            )

        if not extraction.text.strip():
            return self._fail(job_id, file_path, collection_name, "No text extracted")

        # Chunk
        chunks = build_chunks(extraction, file_path, self._config)
        if not chunks:
            return self._fail(job_id, file_path, collection_name, "No chunks produced")

        # Embed
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)

        # Upsert to Chroma
        collection.upsert(
            ids=[c.id for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
        )

        duration = round(time.time() - start_time, 2)
        logger.info(
            "Ingested %s → %d chunks into '%s' (%.1fs)",
            file_path,
            len(chunks),
            collection_name,
            duration,
        )

        result = {
            "job_id": job_id,
            "status": "completed",
            "chunks_created": len(chunks),
            "source_path": file_path,
            "collection": collection_name,
            "duration_seconds": duration,
        }
        self._jobs[job_id] = result
        return result

    def ingest_directory(
        self,
        dir_path: str,
        collection_name: str,
        recursive: bool = True,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Ingest all supported files in a directory."""
        job_id = f"ingest-{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        path = Path(dir_path)
        if not path.is_dir():
            return self._fail(
                job_id, dir_path, collection_name, f"Not a directory: {dir_path}"
            )

        pattern = "**/*" if recursive else "*"
        files = [
            f
            for f in path.glob(pattern)
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
        ]

        if not files:
            return self._fail(
                job_id, dir_path, collection_name, "No supported files found"
            )

        total_chunks = 0
        errors: list[str] = []

        for file in sorted(files):
            try:
                result = self.ingest_file(str(file), collection_name, force_reindex)
                total_chunks += result.get("chunks_created", 0)
                if result["status"] == "failed":
                    errors.append(f"{file.name}: {result.get('error', 'unknown')}")
            except Exception as e:
                errors.append(f"{file.name}: {e}")
                logger.error("Failed to ingest %s: %s", file, e)

        duration = round(time.time() - start_time, 2)
        status = "completed" if not errors else "completed_with_errors"

        result = {
            "job_id": job_id,
            "status": status,
            "chunks_created": total_chunks,
            "files_processed": len(files),
            "source_path": dir_path,
            "collection": collection_name,
            "duration_seconds": duration,
            "errors": errors if errors else None,
        }
        self._jobs[job_id] = result
        return result

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def _fail(
        self, job_id: str, source_path: str, collection: str, error: str
    ) -> dict[str, Any]:
        logger.error("Ingestion failed: %s", error)
        result = {
            "job_id": job_id,
            "status": "failed",
            "chunks_created": 0,
            "source_path": source_path,
            "collection": collection,
            "duration_seconds": 0.0,
            "error": error,
        }
        self._jobs[job_id] = result
        return result
