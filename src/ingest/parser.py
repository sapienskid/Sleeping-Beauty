"""Document parser supporting Markdown, code, plain text, and PDF files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import re
import pymupdf


@dataclass
class ParsedDocument:
    doc_id: str
    file_path: str
    vault_name: str
    relative_path: str
    title: str
    content: str
    extension: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter from markdown files if present."""
    metadata: Dict[str, Any] = {}
    content = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            yaml_block = parts[1]
            content = parts[2].strip()
            # Basic YAML key: value extractor without heavy parser dependency
            for line in yaml_block.splitlines():
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if v:
                        metadata[k] = v

    return metadata, content


def parse_pdf(file_path: Path) -> str:
    """Extract text from a PDF file with page markers."""
    doc = pymupdf.open(file_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append(f"--- [Page {page_num + 1}] ---\n{text}")
    return "\n\n".join(pages)


def parse_file(file_path: Path, vault_root: Path) -> Optional[ParsedDocument]:
    """Parse a single file from a given vault root."""
    try:
        try:
            relative_path = str(file_path.relative_to(vault_root))
        except ValueError:
            relative_path = file_path.name

        resolved_vault = vault_root.resolve()
        vault_name = resolved_vault.name
        ext = file_path.suffix.lower()

        title = file_path.stem
        metadata: Dict[str, Any] = {}

        if ext == ".pdf":
            content = parse_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()

            if ext in [".md", ".markdown"]:
                metadata, content = parse_frontmatter(raw_text)
                if "title" in metadata:
                    title = metadata["title"]
                elif "# " in content:
                    first_header = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                    if first_header:
                        title = first_header.group(1).strip()
            else:
                content = raw_text

        if not content.strip():
            return None

        # Clean document id
        doc_id = f"{vault_name}::{relative_path}"

        return ParsedDocument(
            doc_id=doc_id,
            file_path=str(file_path.resolve()),
            vault_name=vault_name,
            relative_path=relative_path,
            title=title,
            content=content,
            extension=ext,
            metadata=metadata,
        )

    except Exception as e:
        print(f"Warning: Failed to parse {file_path}: {e}")
        return None
