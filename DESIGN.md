# Sleeping Beauty — RAG + Memory Gateway Design Document

> A fully local, portable RAG and memory gateway for Arch Linux that sits
> between Open WebUI and llama.cpp's `llama-server`. No Ollama. No cloud.

**Hardware target**: 24 GB RAM, GTX 1650 (4 GB VRAM), GPU-first inference (CPU spillover for layers that don't fit VRAM)  
**Date**: 2026-03-03  
**Status**: Draft v1

---

## Table of Contents

1. [Architecture Decision + Rationale](#1-architecture-decision--rationale)
2. [Research Checklist](#2-research-checklist)
3. [API Contract](#3-api-contract)
4. [Citation Standard](#4-citation-standard)
5. [Ingestion Pipeline](#5-ingestion-pipeline)
6. [Memory System](#6-memory-system)
7. [Agents Plan](#7-agents-plan)
8. [Deployment Plan](#8-deployment-plan)
9. [Repository Structure](#9-repository-structure)
10. [Testing and Evaluation](#10-testing-and-evaluation)

---

## 1. Architecture Decision + Rationale

### 1.1 High-Level Topology

```
┌─────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Open WebUI  │────▶│   Gateway (Python)   │────▶│ llama-server     │
│  :3000       │◀────│   :8800              │◀────│ (chat) :8080     │
│  (Docker)    │     │   (Docker)           │     │ (host, systemd)  │
└─────────────┘     └──────────┬────────────┘     └──────────────────┘
                               │
                    ┌──────────┴────────────┐
                    │                       │
              ┌─────▼─────┐          ┌──────▼──────────┐
              │  Chroma    │          │ llama-server     │
              │  :8000     │          │ (embed) :8085    │
              │  (Docker)  │          │ (host, systemd)  │
              └────────────┘          └─────────────────┘
```

**Four processes, three containers:**

| Component | Runs as | Why |
|-----------|---------|-----|
| **llama-server (chat)** | systemd user unit on host | Direct GPU/CPU access, no Docker overhead. Single-model mode, serves `/v1/chat/completions` and `/v1/models`. |
| **llama-server (embed)** | systemd user unit on host | Separate instance required — llama-server's `--embedding` flag restricts to embedding-only mode. Serves `/v1/embeddings`. |
| **Chroma** | Docker container | Official `chromadb/chroma` image. HTTP server mode required for fork-safety with multi-worker gateway. |
| **Gateway** | Docker container | Python (FastAPI). Intercepts all OpenAI-compatible requests from Open WebUI. Performs retrieval from Chroma, injects context + citation instructions, proxies to llama-server. |
| **Open WebUI** | Docker container | Points `OPENAI_API_BASE_URL` at the gateway, not directly at llama-server. Built-in RAG bypassed. |

### 1.2 Key Decisions

**Decision 1: Gateway owns all retrieval and citation logic.**  
*Rationale*: Open WebUI's built-in RAG is a black box — no control over retrieval strategy, prompt injection format, or citation rendering. By setting `BYPASS_EMBEDDING_AND_RETRIEVAL=True` and routing through the gateway, we get full control over what context is injected and how citations are formatted. The gateway is UI-agnostic: swap Open WebUI for anything that speaks OpenAI-compatible API.

**Decision 2: Two separate llama-server instances (chat + embed).**  
*Rationale*: llama-server's `--embedding` flag locks the instance to embedding-only mode. There is no way to serve both chat completions and embeddings from a single instance. The embed instance runs a small model (e.g., `nomic-embed-text-v1.5` at ~280 MB) so the memory cost is negligible.

**Decision 3: llama-server on host, everything else in Docker.**  
*Rationale*: GPU passthrough in Docker adds complexity (nvidia-container-toolkit, device mapping, driver version coupling). Running llama-server directly on the host with systemd user units gives reliable GPU access and simpler debugging. The gateway, Chroma, and Open WebUI have no GPU needs — Docker gives us portability and isolation.

**Decision 4: Chroma as HTTP server, not embedded client.**  
*Rationale*: Chroma's `PersistentClient` is not fork-safe. The gateway will run with multiple workers (uvicorn). Running Chroma as a separate HTTP server container avoids corruption and is the recommended production setup.

**Decision 5: Python + FastAPI for the gateway.**  
*Rationale*: First-class Chroma client library, first-class async HTTP (httpx), SSE streaming support, minimal boilerplate. Sentence-transformers as fallback embedding option. The team (you) knows Python.

**Decision 6: Sentence-transformers as the default embedding engine.**  
*Rationale*: Running a dedicated llama-server embed instance is the alternative, but sentence-transformers (`all-MiniLM-L6-v2`, 80 MB) runs in-process, has zero startup latency, and is battle-tested for RAG. The gateway config will support both backends (in-process sentence-transformers or external llama-server `/v1/embeddings`), but sentence-transformers is the default to keep moving parts minimal. If you later want to use a GGUF embedding model for quality reasons, switch to the llama-server embed backend via config.

**Decision 7: No forking or patching of upstream projects.**  
*Rationale*: Hard constraint. All integration is via documented APIs and environment variables.

### 1.3 Data Flow: Chat with RAG

```
1. User sends message in Open WebUI
2. Open WebUI POST /v1/chat/completions → Gateway :8800
3. Gateway extracts user query from messages[-1]
4. Gateway embeds query (sentence-transformers or llama-server embed)
5. Gateway queries Chroma: top-k similar chunks + metadata
6. Gateway evaluates retrieval confidence (max similarity score vs threshold)
7a. IF confidence >= threshold:
      - Build augmented system prompt with retrieved chunks + citation instructions
      - Forward augmented request to llama-server (chat) :8080
      - Stream response back to Open WebUI
7b. IF confidence < threshold:
      - Return "insufficient evidence" response with list of collections searched
      - Do NOT hallucinate an answer
8. Open WebUI renders response (citations appear as markdown links)
```

### 1.4 Constraints Summary

| Constraint | Value |
|-----------|-------|
| Max RAM budget for inference | ~18 GB (leaving 6 GB for OS + containers) |
| Max VRAM | 4 GB (GTX 1650) |
| Chat model format | GGUF, Q4_K_M or Q5_K_M quantization |
| Embedding model | `all-MiniLM-L6-v2` (default) or GGUF embed model |
| Vector DB | Chroma (HTTP server mode) |
| Container runtime | Docker Compose v2 |
| UI | Open WebUI (only UI, not the RAG engine) |
| Cloud dependencies | Zero |

---

## 2. Research Checklist

All items researched and verified against official documentation as of 2026-03-03.

### 2.1 llama.cpp server

| Item | Status | Finding |
|------|--------|---------|
| OpenAI-compatible routes | Verified | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings` |
| `/v1/models` response shape | Verified | Returns `{data: [{id, object: "model", ...}]}` — always exactly one element |
| `--alias` flag | Verified | Overrides model `id` field. Comma-separated for multiple aliases. Env: `LLAMA_ARG_ALIAS` |
| Streaming support | Verified | SSE on both chat and completions endpoints |
| Embedding mode | Verified | `--embedding` flag restricts to embed-only. Requires pooling type != none. **Separate instance required.** |
| Function calling | Verified | Supported via `--jinja` (default on). Requires compatible chat template. |
| Default port/host | Verified | `:8080` / `127.0.0.1` |
| Docker images | Verified | `ghcr.io/ggml-org/llama.cpp:server` and `:server-cuda` |

### 2.2 Open WebUI (v0.8.5)

| Item | Status | Finding |
|------|--------|---------|
| Connecting to OpenAI-compatible backend | Verified | `OPENAI_API_BASE_URL=http://host:port/v1`, `OPENAI_API_KEY=none` |
| Disabling built-in RAG | Verified | `BYPASS_EMBEDDING_AND_RETRIEVAL=True`. No single "disable RAG" toggle. |
| Disabling Ollama | Verified | `ENABLE_OLLAMA_API=False` |
| PersistentConfig behavior | Verified | After first launch, DB values override env vars. Use `ENABLE_PERSISTENT_CONFIG=False` to force env. |
| Single-user mode | Verified | `WEBUI_AUTH=False` |
| Task model for OpenAI mode | Verified | `TASK_MODEL_EXTERNAL` sets model for title gen etc. |
| Memory feature | Verified | `ENABLE_MEMORIES=True` (default). Uses same vector DB + embedding engine as RAG. |

### 2.3 Chroma

| Item | Status | Finding |
|------|--------|---------|
| Client types | Verified | `Client()` (ephemeral), `PersistentClient(path)` (SQLite), `HttpClient(host, port)` (server mode) |
| Server mode | Verified | `chroma run --path /db` or Docker `chromadb/chroma`. Port 8000. |
| Fork safety | Verified | `PersistentClient` is NOT fork-safe. Must use HTTP server for multi-worker. |
| HNSW config | Verified | `space`: l2/cosine/ip. `ef_construction`, `ef_search`, `max_neighbors` tunable. Default space: l2. |
| Metadata + filtering | Verified | Per-document metadata stored alongside embeddings. Filter at query time. |
| Idempotent ops | Verified | `get_or_create_collection`, `upsert` supported. |
| Docker env vars | Verified | `IS_PERSISTENT=TRUE`, `PERSIST_DIRECTORY=/chroma/chroma` |

### 2.4 Open Questions (deferred to implementation)

| Question | Notes |
|----------|-------|
| Optimal chunk size for mixed doc types | Start with 512 tokens, 64-token overlap. Tune via eval. |
| Best GGUF embedding model vs sentence-transformers | Default to sentence-transformers. Benchmark later. |
| Chroma `ef_search` tuning | Default (10) may be too low for large collections. Tune when collection exceeds ~50k chunks. |

---

## 3. API Contract

The gateway exposes a strict subset of the OpenAI API. Open WebUI connects to it as if it were any OpenAI-compatible provider.

### 3.1 Endpoints

#### `GET /v1/models`

Proxied directly to llama-server (chat). Returns the model list unchanged.

```json
// Response (proxied from llama-server)
{
  "object": "list",
  "data": [
    {
      "id": "sleeping-beauty",
      "object": "model",
      "created": 1709424000,
      "owned_by": "llamacpp",
      "meta": {
        "vocab_type": "SPM",
        "n_vocab": 32000,
        "n_ctx_train": 8192,
        "n_embd": 4096,
        "n_params": 7000000000,
        "size": 4370000000
      }
    }
  ]
}
```

The `--alias sleeping-beauty` flag on llama-server sets the model `id`.

#### `POST /v1/chat/completions`

**This is the core gateway endpoint.** It intercepts the request, performs RAG, and proxies to llama-server.

**Request** (from Open WebUI — standard OpenAI shape):

```json
{
  "model": "sleeping-beauty",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What does section 4.2 of the project charter say about timelines?"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Gateway processing pipeline:**

```
1. Extract query = messages[-1].content (user's latest message)
2. Embed query → vector
3. Query Chroma (top-k=5, with metadata filters if applicable)
4. Evaluate confidence: max_score = max(result.distances) mapped to similarity
5. IF max_similarity < CONFIDENCE_THRESHOLD (default 0.35 for cosine):
     → Return low-confidence response (see 3.2)
6. Build augmented messages:
     - Prepend retrieval context block to system message
     - Include citation instructions in system message
     - Preserve all user/assistant messages unchanged
7. Forward augmented request to llama-server :8080/v1/chat/completions
8. Stream response back to caller (SSE passthrough)
```

**Augmented system message (injected by gateway):**

```
You have access to the following retrieved document excerpts. Use ONLY these
excerpts to answer the user's question. Cite every claim using the format
[source_id](file_path#page) where source_id is the chunk identifier and
file_path is the local file path. If the excerpts do not contain sufficient
information to answer, say so explicitly.

---RETRIEVED CONTEXT---
[chunk_abc123](file:///home/user/docs/charter.pdf#page=12)
"The project timeline specifies a 6-month delivery window with quarterly
milestones. Phase 1 covers requirements gathering (weeks 1-4)..."

[chunk_def456](file:///home/user/docs/charter.pdf#page=13)
"Section 4.2 explicitly states that timeline extensions require board
approval and must be documented in the change log..."
---END CONTEXT---

{original_system_message}
```

**Response** (streamed SSE, standard OpenAI shape):

```json
{
  "id": "chatcmpl-gateway-xxx",
  "object": "chat.completion.chunk",
  "created": 1709424000,
  "model": "sleeping-beauty",
  "choices": [
    {
      "index": 0,
      "delta": {"content": "According to section 4.2 of the project charter"},
      "finish_reason": null
    }
  ]
}
```

The `id` is generated by the gateway (prefixed `chatcmpl-gateway-`) to distinguish from raw llama-server responses. The `model` field is preserved from the original request.

#### `POST /v1/embeddings` (internal use only)

If using llama-server embed backend, the gateway proxies embedding requests to the embed instance. This endpoint is NOT exposed to Open WebUI — it's used internally by the ingestion pipeline and query embedding.

#### `GET /health`

Gateway health check. Returns status of all dependencies.

```json
{
  "status": "ok",
  "components": {
    "llama_chat": {"status": "ok", "url": "http://127.0.0.1:8080"},
    "llama_embed": {"status": "ok", "url": "http://127.0.0.1:8085"},
    "chroma": {"status": "ok", "url": "http://chroma:8000"},
    "gateway": {"status": "ok", "version": "0.1.0"}
  }
}
```

If `llama_embed` is not configured (using in-process sentence-transformers), that component is omitted.

#### `POST /ingest`

Trigger document ingestion. Not part of the OpenAI API — this is a gateway-specific management endpoint.

```json
// Request
{
  "source_path": "/data/docs/charter.pdf",
  "collection": "project-docs",
  "force_reindex": false
}

// Response
{
  "status": "accepted",
  "job_id": "ingest-xxxx",
  "source_path": "/data/docs/charter.pdf",
  "collection": "project-docs"
}
```

#### `GET /ingest/{job_id}`

Check ingestion job status.

```json
{
  "job_id": "ingest-xxxx",
  "status": "completed",
  "chunks_created": 47,
  "source_path": "/data/docs/charter.pdf",
  "collection": "project-docs",
  "duration_seconds": 12.3
}
```

#### `GET /collections`

List all Chroma collections with document counts.

```json
{
  "collections": [
    {"name": "project-docs", "count": 1247},
    {"name": "emails", "count": 389},
    {"name": "writing-samples", "count": 52}
  ]
}
```

### 3.2 Low-Confidence Response

When retrieval confidence is below threshold, the gateway does NOT forward to llama-server. It returns a canned response directly:

```json
{
  "id": "chatcmpl-gateway-noconf-xxx",
  "object": "chat.completion",
  "created": 1709424000,
  "model": "sleeping-beauty",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I don't have sufficient evidence to answer this question.\n\n**Collections searched:** project-docs, emails\n**Top result similarity:** 0.21 (threshold: 0.35)\n**Best matching chunk:** [chunk_xyz](file:///home/user/docs/notes.txt) — \"quarterly budget review process...\" (similarity: 0.21)\n\nTry rephrasing your question or check if the relevant documents have been ingested."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

This is a non-streamed response even if `stream: true` was requested — the gateway closes the SSE stream immediately after sending this single message.

### 3.3 Gateway Configuration Schema

```yaml
# gateway-config.yaml
gateway:
  host: "0.0.0.0"
  port: 8800
  workers: 2
  log_level: "info"

llama_chat:
  base_url: "http://host.docker.internal:8080"
  # No API key needed for local llama-server

llama_embed:
  # Option A: in-process sentence-transformers (default)
  backend: "sentence-transformers"
  model: "all-MiniLM-L6-v2"
  # Option B: external llama-server
  # backend: "llama-server"
  # base_url: "http://host.docker.internal:8085"

chroma:
  host: "chroma"  # Docker service name
  port: 8000

retrieval:
  top_k: 5
  confidence_threshold: 0.35  # cosine similarity minimum
  collections: ["project-docs"]  # default collections to search
  # "all" to search every collection
  max_context_tokens: 3000  # max tokens of retrieved context to inject

ingestion:
  chunk_size: 512  # tokens
  chunk_overlap: 64  # tokens
  ocr:
    enabled: true
    engine: "tesseract"  # only supported engine for now
    languages: ["eng"]
  supported_formats:
    - ".pdf"
    - ".txt"
    - ".md"
    - ".html"
    - ".docx"
    - ".epub"
    - ".png"
    - ".jpg"
    - ".jpeg"
    - ".tiff"

memory:
  enabled: true
  collection: "user-memory"
  # See Section 6 for full memory config
```

### 3.4 Request Routing Logic (Pseudocode)

```python
async def handle_chat_completions(request: ChatCompletionRequest):
    # 1. Check if this is a RAG-eligible request
    if not request.messages or request.messages[-1].role != "user":
        # No user message to search on — proxy directly
        return await proxy_to_llama(request)

    query = request.messages[-1].content

    # 2. Check for explicit bypass (e.g., prefix "!norag")
    if query.startswith("!norag"):
        request.messages[-1].content = query.removeprefix("!norag").strip()
        return await proxy_to_llama(request)

    # 3. Determine collections to search
    collections = resolve_collections(request, config)

    # 4. Embed query
    query_embedding = await embed(query)

    # 5. Retrieve from Chroma
    results = await chroma_query(
        collections=collections,
        query_embedding=query_embedding,
        top_k=config.retrieval.top_k
    )

    # 6. Evaluate confidence
    if results.max_similarity < config.retrieval.confidence_threshold:
        return build_low_confidence_response(request, results)

    # 7. Build augmented prompt
    augmented = inject_context(request, results)

    # 8. Proxy to llama-server and stream back
    return await proxy_to_llama_streaming(augmented)
```

---

## 4. Citation Standard

### 4.1 Citation Format

Every claim derived from retrieved context MUST be cited. The format is:

```
[source_id](file_path#anchor)
```

Where:
- **`source_id`**: The Chroma chunk ID (e.g., `chunk_abc123`). Short, unique, traceable.
- **`file_path`**: Absolute `file://` URI to the source document on the local filesystem.
- **`anchor`**: Page number (`#page=N`) for PDFs, line range (`#L10-L20`) for text files, section heading (`#section-name`) for markdown/HTML.

### 4.2 Examples

```markdown
The project timeline is 6 months [chunk_a1b2](file:///home/user/docs/charter.pdf#page=12).

According to the style guide, passive voice should be avoided
[chunk_c3d4](file:///home/user/docs/style-guide.md#L45-L52).

The API rate limit is 100 requests per minute
[chunk_e5f6](file:///home/user/docs/api-spec.html#rate-limiting).
```

### 4.3 Citation Metadata in Chroma

Each chunk stored in Chroma carries metadata that enables citation construction:

```json
{
  "id": "chunk_a1b2c3d4",
  "document": "The project timeline specifies a 6-month delivery window...",
  "metadata": {
    "source_path": "/home/user/docs/charter.pdf",
    "source_filename": "charter.pdf",
    "collection": "project-docs",
    "page_number": 12,
    "line_start": null,
    "line_end": null,
    "section_heading": "4.2 Timelines",
    "chunk_index": 5,
    "total_chunks": 47,
    "ingested_at": "2026-03-03T10:00:00Z",
    "content_hash": "sha256:abcdef...",
    "ocr_applied": false,
    "file_type": "pdf"
  },
  "embedding": [0.012, -0.034, ...]
}
```

### 4.4 Citation Instruction Template

This is injected into the system prompt by the gateway when retrieved context is present:

```
CITATION RULES:
1. Cite every factual claim using [chunk_id](file://path#anchor) format.
2. Place citations immediately after the sentence they support.
3. If multiple chunks support one claim, cite all of them.
4. NEVER fabricate a citation. Only use chunk IDs provided in the context above.
5. If the provided context does not answer the question, state "The retrieved
   documents do not contain sufficient information to answer this question."
6. Do not refer to chunks by their ID in prose — use natural language and
   place the citation link at the end of the relevant sentence.
```

### 4.5 Anchor Construction Logic

| File type | Anchor format | Example |
|-----------|--------------|---------|
| PDF | `#page=N` | `file:///docs/report.pdf#page=7` |
| Text, code | `#L{start}-L{end}` | `file:///docs/notes.txt#L10-L20` |
| Markdown | `#section-slug` | `file:///docs/guide.md#installation` |
| HTML | `#element-id` or `#section-slug` | `file:///docs/spec.html#rate-limiting` |
| Image (OCR'd) | `#page=1` (always page 1 for single images) | `file:///docs/scan.png#page=1` |
| EPUB | `#chapter-N` | `file:///docs/book.epub#chapter-3` |

### 4.6 Clickability

`file://` URIs are clickable in most terminal emulators and some markdown renderers. Open WebUI renders markdown, so `[text](file://...)` will appear as a link. Whether the link opens the file depends on the browser/OS — on a local machine with the files present, most browsers will open the file. This is the best we can do without a custom file server.

**Future enhancement**: If `file://` links don't work reliably in the browser context (Open WebUI runs in Docker, user's browser is on the host), consider adding a static file server to the gateway that serves `/files/...` routes mapped to the ingested document paths. This is deferred to implementation.

---

## 5. Ingestion Pipeline

### 5.1 Overview

```
Source file → Format detection → Text extraction (+ OCR if needed)
           → Chunking → Embedding → Chroma upsert
```

### 5.2 Format Detection + Text Extraction

| Format | Library | Notes |
|--------|---------|-------|
| PDF (text) | `pymupdf` (fitz) | Extracts text with page numbers. Fast, pure-text extraction. |
| PDF (scanned/image) | `pymupdf` + `tesseract` | pymupdf detects image-only pages (no extractable text). Falls back to OCR per page. |
| Plain text (.txt, .md, .rst) | Built-in | Read as UTF-8. Line numbers tracked for citation anchors. |
| HTML | `beautifulsoup4` | Extract text content, preserve heading structure for section anchors. |
| DOCX | `python-docx` | Paragraph-level extraction. |
| EPUB | `ebooklib` + `beautifulsoup4` | Chapter-level extraction. |
| Images (.png, .jpg, .tiff) | `tesseract` (via `pytesseract`) | Full OCR. Single page. |

### 5.3 OCR Pipeline Detail

```
1. Input: PDF page image or standalone image file
2. Pre-processing (optional, via Pillow):
   - Grayscale conversion
   - Contrast normalization
   - Deskew (if skew angle > 2 degrees)
3. Tesseract OCR:
   - Language: configurable (default: eng)
   - PSM: 3 (fully automatic page segmentation)
   - OEM: 3 (default, LSTM neural net)
4. Output: extracted text + confidence score per page
5. If confidence < 60%: log warning, still index but mark metadata.ocr_confidence
```

**Tesseract installation** (Arch Linux):
```
pacman -S tesseract tesseract-data-eng
```

The gateway Docker image must include tesseract. Use `apt-get install tesseract-ocr` in Dockerfile (Debian-based Python image).

### 5.4 Chunking Strategy

**Method**: Recursive character/token splitting with overlap.

```python
# Pseudocode
class ChunkingConfig:
    chunk_size: int = 512      # tokens
    chunk_overlap: int = 64    # tokens
    separators: list = ["\n\n", "\n", ". ", " "]  # split priority order

def chunk_document(text: str, metadata: dict, config: ChunkingConfig) -> list[Chunk]:
    tokenizer = get_tokenizer()  # tiktoken or sentence-transformers tokenizer
    chunks = recursive_split(text, config.chunk_size, config.chunk_overlap, config.separators)

    result = []
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"chunk_{hash(metadata['source_path'] + str(i))[:12]}"
        result.append(Chunk(
            id=chunk_id,
            text=chunk_text,
            metadata={
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
        ))
    return result
```

**Why 512 tokens**: Balances context density with retrieval precision. Smaller chunks (256) risk losing context; larger chunks (1024) dilute relevance scores. 512 is a well-tested default for embedding models like `all-MiniLM-L6-v2`.

**Overlap of 64 tokens**: Prevents information loss at chunk boundaries. ~12% overlap ratio.

### 5.5 Embedding

```python
# Default: in-process sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([chunk.text for chunk in chunks], normalize_embeddings=True)

# Alternative: llama-server embed instance
# POST http://host:8085/v1/embeddings
# {"input": ["chunk text..."], "model": "nomic-embed-text"}
```

**Dimension**: `all-MiniLM-L6-v2` produces 384-dimensional vectors. Chroma collection must be configured with `space: "cosine"` to match normalized embeddings.

### 5.6 Chroma Upsert

```python
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

collection.upsert(
    ids=[chunk.id for chunk in chunks],
    documents=[chunk.text for chunk in chunks],
    embeddings=[emb.tolist() for emb in embeddings],
    metadatas=[chunk.metadata for chunk in chunks]
)
```

**Idempotency**: `upsert` ensures re-ingesting the same document updates existing chunks rather than creating duplicates. Chunk IDs are deterministic (derived from source path + chunk index).

### 5.7 Ingestion Deduplication

Before ingesting, the gateway checks if the document has already been ingested:

```python
# Check content hash
existing = collection.get(
    where={"source_path": source_path}
)
if existing and not force_reindex:
    current_hash = sha256(file_content)
    if existing_metadata["content_hash"] == current_hash:
        return "already indexed, no changes"
    else:
        # File changed — delete old chunks, re-ingest
        collection.delete(where={"source_path": source_path})
```

### 5.8 Batch Ingestion

For ingesting a directory of documents:

```
POST /ingest
{
  "source_path": "/data/docs/",     // directory path
  "collection": "project-docs",
  "recursive": true,
  "force_reindex": false
}
```

The gateway walks the directory, filters by `supported_formats`, and processes each file. A background task queue (Python `asyncio.Queue` or `concurrent.futures.ProcessPoolExecutor`) handles files in parallel, bounded by available RAM.

### 5.9 Acceptance Criteria

| # | Criterion |
|---|-----------|
| 1 | PDF with extractable text is chunked and indexed with page-number metadata |
| 2 | Scanned PDF with no extractable text is OCR'd, chunked, and indexed with page-number metadata and `ocr_applied: true` |
| 3 | Re-ingesting an unchanged file is a no-op (deduplication by content hash) |
| 4 | Re-ingesting a changed file deletes old chunks and creates new ones |
| 5 | Mixed PDF (some text pages, some scanned) correctly OCRs only the scanned pages |
| 6 | Standalone images (.png, .jpg) are OCR'd and indexed |
| 7 | Markdown files are chunked with section-heading anchors in metadata |
| 8 | Ingesting a directory processes all supported files recursively |
| 9 | Ingestion job status is queryable via `GET /ingest/{job_id}` |
| 10 | Unsupported file formats are skipped with a warning log, not a crash |

---

## 6. Memory System

### 6.1 Purpose

The memory system enables two capabilities:

1. **Writing style learning**: The gateway analyzes the user's writing samples and stores stylistic patterns (tone, vocabulary, sentence structure, formatting preferences) for use when drafting emails/posts.
2. **Persistent user preferences**: Facts the user explicitly tells the system to remember ("My name is X", "I prefer formal tone", "Always use metric units").

### 6.2 Architecture

Memory is a **separate Chroma collection** (`user-memory`) with a different schema than document collections:

```json
{
  "id": "mem_abc123",
  "document": "User prefers formal business English with short paragraphs.",
  "metadata": {
    "type": "preference",       // "preference" | "writing-style" | "fact"
    "source": "explicit",       // "explicit" (user told us) | "inferred" (from writing samples)
    "created_at": "2026-03-03T10:00:00Z",
    "updated_at": "2026-03-03T10:00:00Z",
    "confidence": 0.9,
    "tags": ["writing", "tone"]
  }
}
```

### 6.3 Memory Types

| Type | Source | Example |
|------|--------|---------|
| `preference` | Explicit user instruction | "Remember: I sign emails as 'Best, Alex'" |
| `fact` | Explicit user instruction | "Remember: My team uses Jira for project tracking" |
| `writing-style` | Inferred from writing samples | "Tends to use active voice, short sentences (avg 12 words), semicolons for lists" |

### 6.4 Writing Style Analysis

When the user ingests writing samples into a `writing-samples` collection:

```
1. Ingest writing samples (emails, posts, documents) into "writing-samples" collection
2. Gateway runs style analysis (triggered manually or on ingestion):
   a. Retrieve all chunks from "writing-samples"
   b. Send to llama-server with a style-analysis prompt:
      "Analyze the following writing samples and extract:
       - Average sentence length
       - Vocabulary level (simple/moderate/advanced)
       - Tone (formal/casual/mixed)
       - Common phrases and patterns
       - Formatting preferences (bullet points, paragraphs, etc.)
       - Sign-off patterns
       Output as structured JSON."
   c. Store extracted style profile in "user-memory" collection with type: "writing-style"
3. When user requests draft (detected via intent classification or explicit prefix):
   a. Retrieve style memories from "user-memory" where type = "writing-style"
   b. Inject style instructions into system prompt alongside any RAG context
```

### 6.5 Memory Retrieval Integration

During chat, the gateway ALWAYS queries the `user-memory` collection in addition to document collections:

```python
async def handle_chat_completions(request):
    query = request.messages[-1].content
    query_embedding = await embed(query)

    # Retrieve from document collections (RAG)
    doc_results = await chroma_query(collections=config.retrieval.collections, ...)

    # Retrieve from memory collection (always)
    memory_results = await chroma_query(
        collections=["user-memory"],
        query_embedding=query_embedding,
        top_k=3
    )

    # Inject both into prompt
    augmented = inject_context(request, doc_results, memory_results)
```

Memory results are injected into a separate section of the system prompt:

```
---USER PREFERENCES---
- User signs emails as "Best, Alex" [mem_abc123]
- User prefers formal business English [mem_def456]
---END PREFERENCES---
```

### 6.6 Memory Management Endpoints

#### `POST /memory`

Add an explicit memory.

```json
// Request
{
  "content": "I sign emails as 'Best, Alex'",
  "type": "preference",
  "tags": ["email", "sign-off"]
}

// Response
{
  "id": "mem_abc123",
  "status": "stored"
}
```

#### `GET /memory`

List all memories, optionally filtered.

```json
// GET /memory?type=preference&tag=email
{
  "memories": [
    {
      "id": "mem_abc123",
      "content": "I sign emails as 'Best, Alex'",
      "type": "preference",
      "created_at": "2026-03-03T10:00:00Z"
    }
  ]
}
```

#### `DELETE /memory/{id}`

Delete a specific memory.

### 6.7 Acceptance Criteria

| # | Criterion |
|---|-----------|
| 1 | Explicit memories (via `POST /memory`) are stored and retrievable |
| 2 | Memories are injected into chat context automatically |
| 3 | Writing style analysis extracts at least: tone, sentence length, vocabulary level |
| 4 | Drafting requests use stored style profile to match user's voice |
| 5 | Memories can be listed, filtered, and deleted |
| 6 | Memory retrieval does not block or significantly delay chat responses (< 100ms added latency) |

---

## 7. Agents Plan

### 7.1 Overview

Agents are **prompt-specialized behaviors** of the gateway, not separate processes. The gateway detects user intent and selects the appropriate agent, which determines:
- Which collections to search
- What system prompt to use
- Whether to apply memory/style context
- Post-processing of the response

### 7.2 Agent Definitions

#### Agent: `search`

**Trigger**: Default for most queries. Any question about ingested documents.  
**Collections**: All document collections (or configured subset).  
**System prompt emphasis**: Cite everything. "Insufficient evidence" if low confidence.  
**Memory**: Inject user preferences (not writing style).

#### Agent: `draft`

**Trigger**: Queries containing "draft", "write", "compose", "email", "reply to", "blog post".  
**Collections**: Relevant document collections (for content) + `writing-samples` (for reference).  
**System prompt emphasis**: Match user's writing style. Use style profile from memory. Produce ready-to-send output.  
**Memory**: Inject both preferences AND writing style.

#### Agent: `remember`

**Trigger**: Queries containing "remember", "note that", "keep in mind", "from now on".  
**Action**: Extract the fact/preference from the message, store in `user-memory` collection.  
**Response**: Confirm what was stored.  
**No RAG**: Does not search document collections.

#### Agent: `passthrough`

**Trigger**: `!norag` prefix, or queries that are clearly general knowledge / coding help.  
**Action**: Forward directly to llama-server with no RAG injection.  
**Memory**: Still inject user preferences (not writing style).

### 7.3 Intent Detection

**Phase 1 (rule-based)**: Keyword matching and prefix detection. Fast, no LLM call.

```python
def detect_intent(query: str) -> str:
    query_lower = query.lower().strip()

    if query_lower.startswith("!norag"):
        return "passthrough"

    if any(kw in query_lower for kw in ["remember", "note that", "keep in mind", "from now on"]):
        return "remember"

    if any(kw in query_lower for kw in ["draft", "write me", "compose", "email to", "reply to", "blog post"]):
        return "draft"

    return "search"  # default
```

**Phase 2 (future, LLM-based)**: Use a fast, small model or the chat model itself with a classification prompt. Only implement when rule-based detection proves insufficient.

### 7.4 Agent Configuration

```yaml
# In gateway-config.yaml
agents:
  search:
    system_prompt_template: "prompts/search.txt"
    collections: "all"           # or list of specific collections
    memory_types: ["preference", "fact"]
    confidence_threshold: 0.35

  draft:
    system_prompt_template: "prompts/draft.txt"
    collections: "all"
    memory_types: ["preference", "fact", "writing-style"]
    confidence_threshold: 0.25   # lower threshold — style matters more than exact retrieval

  remember:
    system_prompt_template: null  # no LLM call needed
    collections: []
    memory_types: []

  passthrough:
    system_prompt_template: "prompts/passthrough.txt"
    collections: []
    memory_types: ["preference", "fact"]
```

### 7.5 Acceptance Criteria

| # | Criterion |
|---|-----------|
| 1 | "What does the charter say about X?" routes to `search` agent |
| 2 | "Draft an email to Bob about the Q3 results" routes to `draft` agent |
| 3 | "Remember that I prefer bullet points" routes to `remember` agent and stores a memory |
| 4 | "!norag Explain Python decorators" routes to `passthrough` agent |
| 5 | `draft` agent produces output matching user's stored writing style |
| 6 | `search` agent returns "insufficient evidence" when confidence is low |
| 7 | Agent routing adds < 5ms latency (rule-based phase 1) |

---

## 8. Deployment Plan

### 8.1 Deployment Topology

```
Host (Arch Linux)
├── systemd user units
│   ├── llama-chat.service    → llama-server :8080 (chat model)
│   └── llama-embed.service   → llama-server :8085 (embed model, optional)
│
├── Docker Compose
│   ├── gateway       → :8800  (Python/FastAPI)
│   ├── chroma        → :8000  (chromadb/chroma)
│   └── open-webui    → :3000  (ghcr.io/open-webui/open-webui)
│
├── ~/models/          → GGUF model files
└── ~/docs/            → Documents to ingest
```

### 8.2 CLI Model Manager

A shell script (`sb-models`) for managing GGUF models:

```bash
# Usage
sb-models list                    # list downloaded models
sb-models download <url>          # download model to ~/models/
sb-models set-chat <model-file>   # update llama-chat.service to use this model
sb-models set-embed <model-file>  # update llama-embed.service to use this model
sb-models status                  # show running llama-server instances
```

**Implementation**: Bash script that modifies systemd unit override files (`~/.config/systemd/user/llama-chat.service.d/model.conf`) and reloads/restarts the service.

### 8.3 systemd User Units

#### `~/.config/systemd/user/llama-chat.service`

```ini
[Unit]
Description=llama.cpp server (chat)
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/llama-server \
    --model %h/models/current-chat.gguf \
    --alias sleeping-beauty \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --threads 8 \
    --jinja
Restart=on-failure
RestartSec=5
Environment=LLAMA_ARG_N_GPU_LAYERS=99

[Install]
WantedBy=default.target
```

#### `~/.config/systemd/user/llama-embed.service`

```ini
[Unit]
Description=llama.cpp server (embedding)
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/llama-server \
    --model %h/models/current-embed.gguf \
    --alias sleeping-beauty-embed \
    --host 127.0.0.1 \
    --port 8085 \
    --embedding \
    --ctx-size 512 \
    --threads 4
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

**Note**: The embed service is optional — only needed if using llama-server for embeddings instead of in-process sentence-transformers.

### 8.4 Docker Compose

```yaml
# docker-compose.yaml
version: "3.8"

services:
  chroma:
    image: chromadb/chroma:latest
    container_name: sb-chroma
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/chroma
      - ANONYMIZED_TELEMETRY=FALSE
    restart: unless-stopped

  gateway:
    build:
      context: ./gateway
      dockerfile: Dockerfile
    container_name: sb-gateway
    ports:
      - "8800:8800"
    volumes:
      - ./gateway-config.yaml:/app/config.yaml:ro
      - ./prompts:/app/prompts:ro
      - ${DOCS_PATH:-~/docs}:/data/docs:ro
    environment:
      - CONFIG_PATH=/app/config.yaml
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      - chroma
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: sb-open-webui
    ports:
      - "3000:8080"
    volumes:
      - open-webui-data:/app/backend/data
    environment:
      - OPENAI_API_BASE_URL=http://gateway:8800/v1
      - OPENAI_API_KEY=none
      - ENABLE_OPENAI_API=True
      - ENABLE_OLLAMA_API=False
      - BYPASS_EMBEDDING_AND_RETRIEVAL=True
      - ENABLE_PERSISTENT_CONFIG=False
      - WEBUI_AUTH=False
      - TASK_MODEL_EXTERNAL=sleeping-beauty
    depends_on:
      - gateway
    restart: unless-stopped

volumes:
  chroma-data:
  open-webui-data:
```

### 8.5 Gateway Dockerfile

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8800", "--workers", "2"]
```

### 8.6 Startup Order

```
1. systemctl --user start llama-chat.service
2. systemctl --user start llama-embed.service  (optional)
3. docker compose up -d
   → chroma starts first (no deps)
   → gateway starts after chroma
   → open-webui starts after gateway
4. Open browser to http://localhost:3000
```

Convenience script: `sb-start`

```bash
#!/bin/bash
set -euo pipefail
systemctl --user start llama-chat.service
# systemctl --user start llama-embed.service  # uncomment if using llama-server embeddings
docker compose up -d
echo "Sleeping Beauty is awake. Open http://localhost:3000"
```

### 8.7 Volume Mounts Summary

| Container | Mount | Purpose |
|-----------|-------|---------|
| chroma | `chroma-data:/chroma/chroma` | Persistent vector DB storage |
| gateway | `./gateway-config.yaml:/app/config.yaml` | Gateway configuration |
| gateway | `./prompts:/app/prompts` | System prompt templates |
| gateway | `~/docs:/data/docs` (read-only) | Documents for ingestion. Must match paths used in citations. |
| open-webui | `open-webui-data:/app/backend/data` | Open WebUI persistent state |

### 8.8 Acceptance Criteria

| # | Criterion |
|---|-----------|
| 1 | `docker compose up -d` starts all three containers without errors |
| 2 | `systemctl --user start llama-chat` starts llama-server and it responds on :8080 |
| 3 | Open WebUI at :3000 shows the `sleeping-beauty` model in the model selector |
| 4 | A chat message in Open WebUI flows through gateway to llama-server and back |
| 5 | Gateway health endpoint shows all components as "ok" |
| 6 | `sb-models set-chat <model>` switches the model and restarts the service |
| 7 | Stopping and restarting Docker Compose preserves Chroma data and Open WebUI settings |
| 8 | Gateway logs show retrieval queries and confidence scores |

---

## 9. Repository Structure

```
sleeping-beauty/
├── DESIGN.md                      # This document
├── docker-compose.yaml            # Docker Compose for gateway + chroma + open-webui
├── .env.example                   # Template for user-specific paths
│
├── gateway/                       # Python FastAPI application
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app, route registration
│   │   ├── config.py              # Config loader (YAML → Pydantic models)
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py            # POST /v1/chat/completions (core RAG logic)
│   │   │   ├── models.py          # GET /v1/models (proxy)
│   │   │   ├── ingest.py          # POST /ingest, GET /ingest/{job_id}
│   │   │   ├── memory.py          # POST/GET/DELETE /memory
│   │   │   ├── collections.py     # GET /collections
│   │   │   └── health.py          # GET /health
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── retrieval.py       # Chroma query, confidence evaluation
│   │   │   ├── embedding.py       # Embedding backend abstraction
│   │   │   ├── proxy.py           # HTTP proxy to llama-server (streaming)
│   │   │   ├── ingestion.py       # Document processing + chunking
│   │   │   ├── ocr.py             # Tesseract OCR wrapper
│   │   │   ├── memory.py          # Memory storage + retrieval
│   │   │   └── agents.py          # Intent detection + agent routing
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── openai.py          # Pydantic models for OpenAI API shapes
│   │   │   ├── config.py          # Pydantic models for gateway config
│   │   │   └── chunks.py          # Chunk, IngestionJob models
│   │   └── prompts/
│   │       ├── context_injection.py   # Build augmented system prompts
│   │       └── templates.py           # Load prompt templates from files
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py            # Fixtures: mock Chroma, mock llama-server
│       ├── test_chat.py           # Chat endpoint tests
│       ├── test_ingestion.py      # Ingestion pipeline tests
│       ├── test_retrieval.py      # Retrieval + confidence tests
│       ├── test_memory.py         # Memory CRUD tests
│       ├── test_agents.py         # Intent detection tests
│       ├── test_ocr.py            # OCR pipeline tests
│       └── test_citations.py      # Citation format validation
│
├── prompts/                       # System prompt template files
│   ├── search.txt                 # Search agent system prompt
│   ├── draft.txt                  # Draft agent system prompt
│   ├── passthrough.txt            # Passthrough agent system prompt
│   └── citation_rules.txt         # Citation instruction block
│
├── scripts/                       # Host-side scripts
│   ├── sb-start                   # Start everything
│   ├── sb-stop                    # Stop everything
│   ├── sb-models                  # CLI model manager
│   └── sb-ingest                  # CLI ingestion trigger
│
├── systemd/                       # systemd unit templates
│   ├── llama-chat.service
│   └── llama-embed.service
│
├── gateway-config.yaml            # Default gateway configuration
│
└── eval/                          # Evaluation datasets and scripts
    ├── datasets/
    │   ├── retrieval_qa.jsonl     # Question-answer pairs with expected citations
    │   └── style_samples/         # Writing style test samples
    ├── run_eval.py                # Evaluation runner
    └── metrics.py                 # Precision, recall, citation accuracy
```

### 9.1 Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `routes/chat.py` | Receives OpenAI chat request, orchestrates agent selection → retrieval → prompt augmentation → proxy → response streaming |
| `services/retrieval.py` | Embeds query, queries Chroma, computes confidence, returns ranked results with metadata |
| `services/embedding.py` | Abstracts over sentence-transformers (in-process) and llama-server (HTTP). Single `embed(texts)` interface. |
| `services/proxy.py` | Async HTTP proxy to llama-server. Handles both streaming (SSE) and non-streaming responses. |
| `services/ingestion.py` | File type detection, text extraction dispatch, chunking, embedding, Chroma upsert. Manages background job queue. |
| `services/ocr.py` | Tesseract wrapper. Pre-processing, OCR, confidence reporting. |
| `services/memory.py` | CRUD for user-memory collection. Style analysis orchestration. |
| `services/agents.py` | Intent detection (rule-based). Returns agent config for the current query. |
| `models/openai.py` | Pydantic models matching OpenAI API request/response shapes. Used for validation and serialization. |

---

## 10. Testing and Evaluation

### 10.1 Testing Strategy

Three tiers:

| Tier | Scope | Tools | What it tests |
|------|-------|-------|---------------|
| **Unit** | Individual functions | pytest, unittest.mock | Chunking logic, citation formatting, intent detection, config parsing |
| **Integration** | Service interactions | pytest, Docker (Chroma), httpx mock | Retrieval pipeline, ingestion pipeline, memory CRUD, prompt augmentation |
| **End-to-end** | Full stack | pytest, real llama-server (tiny model), real Chroma | Complete flow: ingest doc → ask question → get cited answer |

### 10.2 Unit Tests

#### Chunking (`test_ingestion.py`)

```python
def test_chunk_size_respected():
    """Chunks should not exceed configured token limit."""

def test_chunk_overlap():
    """Adjacent chunks should share overlap tokens."""

def test_chunk_metadata_page_numbers():
    """PDF chunks carry correct page_number metadata."""

def test_chunk_ids_deterministic():
    """Same input produces same chunk IDs."""

def test_chunk_ids_unique():
    """No duplicate IDs within a document."""
```

#### Citation formatting (`test_citations.py`)

```python
def test_pdf_citation_format():
    """PDF citation produces [id](file://path#page=N)."""

def test_text_citation_format():
    """Text file citation produces [id](file://path#L{start}-L{end})."""

def test_markdown_citation_format():
    """Markdown citation produces [id](file://path#section-slug)."""
```

#### Intent detection (`test_agents.py`)

```python
def test_search_intent_default():
    assert detect_intent("What does the charter say?") == "search"

def test_draft_intent():
    assert detect_intent("Draft an email to Bob") == "draft"

def test_remember_intent():
    assert detect_intent("Remember that I prefer bullet points") == "remember"

def test_passthrough_intent():
    assert detect_intent("!norag Explain Python decorators") == "passthrough"
```

### 10.3 Integration Tests

#### Retrieval pipeline (`test_retrieval.py`)

```python
async def test_retrieval_returns_relevant_chunks():
    """Given indexed documents, querying a known topic returns matching chunks."""
    # Setup: ingest a test document into Chroma
    # Query: ask about a topic in the document
    # Assert: returned chunks contain relevant text, correct metadata

async def test_low_confidence_detection():
    """Querying a topic NOT in the index returns low confidence."""
    # Assert: max_similarity < threshold

async def test_retrieval_metadata_filtering():
    """Collection-specific queries only return chunks from that collection."""
```

#### Ingestion pipeline (`test_ingestion.py`)

```python
async def test_pdf_ingestion():
    """PDF file is processed into chunks with page metadata."""

async def test_ocr_fallback():
    """Scanned PDF page triggers OCR."""

async def test_deduplication():
    """Re-ingesting same file is a no-op."""

async def test_changed_file_reindex():
    """Re-ingesting changed file replaces old chunks."""
```

### 10.4 End-to-End Tests

Require a running stack (llama-server with a tiny test model, Chroma, gateway).

```python
async def test_e2e_rag_flow():
    """
    1. Ingest a test PDF
    2. Ask a question about its content
    3. Verify response contains correct information
    4. Verify response contains citation in expected format
    5. Verify citation's source_path matches the ingested file
    """

async def test_e2e_low_confidence():
    """
    1. Ask about a topic not in any ingested document
    2. Verify response says "insufficient evidence"
    3. Verify response lists searched collections
    """

async def test_e2e_memory_flow():
    """
    1. Store a preference via POST /memory
    2. Ask a question
    3. Verify the preference influenced the response
    """

async def test_e2e_draft_with_style():
    """
    1. Ingest writing samples
    2. Trigger style analysis
    3. Request "Draft an email about X"
    4. Verify output matches stored style characteristics
    """
```

### 10.5 Evaluation Dataset

File: `eval/datasets/retrieval_qa.jsonl`

Each line is a JSON object:

```json
{
  "question": "What is the timeline for Phase 1?",
  "expected_answer_contains": ["6 months", "weeks 1-4", "requirements gathering"],
  "expected_source_file": "charter.pdf",
  "expected_page": 12,
  "collection": "project-docs"
}
```

Build this dataset manually from your actual documents. Start with 20-30 questions covering:
- Questions answerable from a single chunk
- Questions requiring information from multiple chunks
- Questions NOT answerable from the corpus (should trigger "insufficient evidence")
- Questions about OCR'd documents (tests OCR quality)

### 10.6 Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Retrieval Precision@5** | (relevant chunks in top-5) / 5 | >= 0.6 |
| **Retrieval Recall@5** | (relevant chunks in top-5) / (total relevant chunks) | >= 0.8 |
| **Answer Accuracy** | (correct answers) / (total questions) | >= 0.7 |
| **Citation Accuracy** | (correct citations) / (total citations in response) | >= 0.9 |
| **False Positive Rate** | (confident answers to unanswerable questions) / (unanswerable questions) | <= 0.1 |
| **OCR Accuracy** | (correctly OCR'd and retrievable pages) / (total scanned pages) | >= 0.85 |

### 10.7 Evaluation Runner

```python
# eval/run_eval.py (pseudocode)
async def run_evaluation(dataset_path: str, gateway_url: str):
    results = []
    for item in load_jsonl(dataset_path):
        response = await post(f"{gateway_url}/v1/chat/completions", {
            "model": "sleeping-beauty",
            "messages": [{"role": "user", "content": item["question"]}],
            "stream": False
        })

        answer = response["choices"][0]["message"]["content"]
        results.append({
            "question": item["question"],
            "answer": answer,
            "contains_expected": all(kw in answer for kw in item["expected_answer_contains"]),
            "has_citation": bool(re.search(r'\[chunk_\w+\]\(file://', answer)),
            "correct_source": item["expected_source_file"] in answer if item.get("expected_source_file") else None,
        })

    # Compute aggregate metrics
    print_metrics(results)
```

### 10.8 CI / Local Test Runner

No CI server needed (fully local). Convenience script:

```bash
# scripts/sb-test
#!/bin/bash
set -euo pipefail

echo "=== Unit tests ==="
docker compose exec gateway pytest tests/ -m "not integration and not e2e" -v

echo "=== Integration tests ==="
docker compose exec gateway pytest tests/ -m "integration" -v

echo "=== E2E tests (requires running stack) ==="
docker compose exec gateway pytest tests/ -m "e2e" -v
```

### 10.9 Acceptance Criteria

| # | Criterion |
|---|-----------|
| 1 | All unit tests pass with `pytest` |
| 2 | Integration tests pass with Chroma container running |
| 3 | E2E test: ingest PDF → ask question → get cited answer |
| 4 | E2E test: unanswerable question → "insufficient evidence" response |
| 5 | Evaluation runner produces metrics report from JSONL dataset |
| 6 | Retrieval Precision@5 >= 0.6 on evaluation dataset |
| 7 | False positive rate <= 0.1 (confident answers to unanswerable questions) |
| 8 | Citation accuracy >= 0.9 (cited sources are correct) |

---

## Appendix A: Key Dependencies

### Gateway Python Packages (`requirements.txt`)

```
fastapi>=0.109
uvicorn[standard]>=0.27
httpx>=0.27
chromadb>=0.5
sentence-transformers>=3.0
pydantic>=2.6
pyyaml>=6.0
pymupdf>=1.24
pytesseract>=0.3
Pillow>=10.0
python-docx>=1.1
ebooklib>=0.18
beautifulsoup4>=4.12
tiktoken>=0.7
```

### System Dependencies (Host)

```
# Arch Linux
pacman -S llama.cpp tesseract tesseract-data-eng docker docker-compose
```

### System Dependencies (Gateway Docker image)

```
tesseract-ocr tesseract-ocr-eng libgl1-mesa-glx libglib2.0-0
```

---

## Appendix B: Open WebUI Environment Summary

```bash
# All env vars for Open WebUI in docker-compose.yaml
OPENAI_API_BASE_URL=http://gateway:8800/v1
OPENAI_API_KEY=none
ENABLE_OPENAI_API=True
ENABLE_OLLAMA_API=False
BYPASS_EMBEDDING_AND_RETRIEVAL=True
ENABLE_PERSISTENT_CONFIG=False
WEBUI_AUTH=False
TASK_MODEL_EXTERNAL=sleeping-beauty
```

---

## Appendix C: Configuration Reference

See `gateway-config.yaml` (Section 3.3) for the complete gateway configuration schema with defaults and explanations.
