"""Semantic and header-aware chunker designed for micro-LLM context budgets."""

from dataclasses import dataclass
from typing import List
import re
from src.ingest.parser import ParsedDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    file_path: str
    vault_name: str
    relative_path: str
    title: str
    section: str
    text: str
    chunk_index: int


def split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits plain text into overlapping chunks, respecting paragraphs and sentences."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a newline or paragraph
        if end < text_len:
            newline_pos = text.rfind("\n", start, end)
            period_pos = text.rfind(". ", start, end)
            space_pos = text.rfind(" ", start, end)

            if newline_pos > start + (chunk_size // 2):
                end = newline_pos + 1
            elif period_pos > start + (chunk_size // 2):
                end = period_pos + 2
            elif space_pos > start + (chunk_size // 2):
                end = space_pos + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= text_len:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_document(doc: ParsedDocument, chunk_size: int = 400, chunk_overlap: int = 80) -> List[DocumentChunk]:
    """Break a ParsedDocument into header-aware chunks suitable for 1-2B model RAG."""
    chunks: List[DocumentChunk] = []

    # Check if document has markdown headings
    if doc.extension in [".md", ".markdown"]:
        # Split by markdown headers (#, ##, ###, ####)
        header_pattern = re.compile(r"^(#{1,4}\s+.+)$", re.MULTILINE)
        sections = header_pattern.split(doc.content)

        current_header = doc.title
        chunk_idx = 0

        # sections alternates between [pre-header-text, header-1, text-1, header-2, text-2...]
        for i in range(len(sections)):
            part = sections[i].strip()
            if not part:
                continue

            if header_pattern.match(sections[i]):
                current_header = part.lstrip("#").strip()
            else:
                raw_chunks = split_text_with_overlap(part, chunk_size, chunk_overlap)
                for sub_chunk in raw_chunks:
                    # Prepend context breadcrumb for micro LLMs
                    context_prefixed_text = f"[{doc.title} > {current_header}]\n{sub_chunk}"
                    chunk_id = f"{doc.doc_id}::c{chunk_idx}"

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc.doc_id,
                            file_path=doc.file_path,
                            vault_name=doc.vault_name,
                            relative_path=doc.relative_path,
                            title=doc.title,
                            section=current_header,
                            text=context_prefixed_text,
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1
    else:
        # Non-markdown file: straightforward overlap chunking
        raw_chunks = split_text_with_overlap(doc.content, chunk_size, chunk_overlap)
        for idx, sub_chunk in enumerate(raw_chunks):
            context_prefixed_text = f"[{doc.title}]\n{sub_chunk}"
            chunk_id = f"{doc.doc_id}::c{idx}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    file_path=doc.file_path,
                    vault_name=doc.vault_name,
                    relative_path=doc.relative_path,
                    title=doc.title,
                    section=doc.title,
                    text=context_prefixed_text,
                    chunk_index=idx,
                )
            )

    return chunks
