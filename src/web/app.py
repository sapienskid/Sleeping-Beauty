"""FastAPI Application providing Web Dashboard, REST API, and OpenAI-compatible proxy."""

from pathlib import Path
from typing import List, Optional
import os
import asyncio
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.ingest.parser import parse_file
from src.ingest.chunker import chunk_document
from src.embeddings.embedder import GemmaEmbedder
from src.db.store import KnowledgeStore
from src.rag.engine import RAGEngine

app = FastAPI(
    title="Sleeping Beauty - Local Knowledge Hub",
    description="Zero-Docker, high-performance local knowledge search & RAG engine.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Models ---
class IndexRequest(BaseModel):
    vault_path: str
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    vault_name: Optional[str] = None


class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    vault_name: Optional[str] = None
    temperature: Optional[float] = None


# --- Ingestion Helper ---
def run_indexing(vault_dir: Path, chunk_size: int, chunk_overlap: int) -> dict:
    """Scan and index all supported files in vault_dir."""
    store = KnowledgeStore.get_instance()
    embedder = GemmaEmbedder.get_instance()

    supported_exts = set(settings.ALLOWED_EXTENSIONS)
    ignored_dirs = {
        ".git", ".obsidian", ".trash", ".cache", ".vscode", ".idea",
        "node_modules", "target", "dist", "build", "__pycache__",
        ".venv", "venv", "env"
    }
    file_paths = []
    for root, dirs, files in os.walk(vault_dir):
        # Skip hidden and dependency/build directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in ignored_dirs]
        for f in files:
            if f.startswith("."):
                continue
            p = Path(root) / f
            if p.suffix.lower() in supported_exts:
                file_paths.append(p)

    total_files = len(file_paths)
    all_chunks = []

    for fp in file_paths:
        doc = parse_file(fp, vault_dir)
        if doc:
            chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)

    if all_chunks:
        texts_to_embed = [c.text for c in all_chunks]
        vectors = embedder.embed_documents(texts_to_embed)
        store.add_chunks(all_chunks, vectors)

    return {
        "vault_name": vault_dir.name,
        "vault_path": str(vault_dir.resolve()),
        "files_scanned": total_files,
        "chunks_indexed": len(all_chunks),
    }


# --- Endpoints ---

@app.get("/api/health")
def health():
    return {"status": "ok", "project": settings.PROJECT_NAME, "version": "0.2.0"}


@app.get("/api/stats")
def get_stats():
    store = KnowledgeStore.get_instance()
    return store.get_stats()


@app.get("/api/vaults")
def list_vaults():
    store = KnowledgeStore.get_instance()
    return store.list_vaults()


@app.delete("/api/vaults/{vault_name}")
def delete_vault(vault_name: str):
    store = KnowledgeStore.get_instance()
    success = store.delete_vault(vault_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Vault {vault_name} not found or error deleting")
    return {"status": "deleted", "vault_name": vault_name}


@app.post("/api/index")
def index_vault(req: IndexRequest):
    p = Path(req.vault_path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {req.vault_path}")

    cs = req.chunk_size or settings.CHUNK_SIZE
    co = req.chunk_overlap or settings.CHUNK_OVERLAP
    result = run_indexing(p, cs, co)
    return result


@app.post("/api/search")
def search(req: SearchRequest):
    engine = RAGEngine()
    results = engine.retrieve(req.query, top_k=req.top_k, vault_name=req.vault_name)
    return {"query": req.query, "results": results}


@app.post("/api/ask")
async def ask(req: AskRequest):
    engine = RAGEngine()
    result = await engine.ask(
        query=req.query,
        top_k=req.top_k,
        vault_name=req.vault_name,
        temperature=req.temperature,
    )
    return result


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    engine = RAGEngine()

    async def event_generator():
        import json
        async for item in engine.ask_stream(
            query=req.query,
            top_k=req.top_k,
            vault_name=req.vault_name,
            temperature=req.temperature,
        ):
            yield f"data: {json.dumps(item)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- OpenAI-Compatible Chat Completions Proxy ---
@app.post("/v1/chat/completions")
async def chat_completions(payload: dict):
    """Augments standard OpenAI chat requests with local hybrid retrieval."""
    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Messages array required")

    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    engine = RAGEngine()
    chunks = engine.retrieve(last_user_msg, top_k=settings.RAG_TOP_K)
    from src.rag.prompt import format_rag_prompt, SYSTEM_PROMPT

    augmented_user_content = format_rag_prompt(last_user_msg, chunks)
    new_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages[:-1]:
        if m.get("role") != "system":
            new_messages.append(m)
    new_messages.append({"role": "user", "content": augmented_user_content})

    payload["messages"] = new_messages

    import httpx
    endpoint = f"{settings.LLAMA_SERVER_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.LLAMA_API_KEY}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)
        return resp.json()


# --- Built-In Modern Web Dashboard ---
@app.get("/", response_class=HTMLResponse)
def index_page():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Sleeping Beauty Local Knowledge Hub is running.</h1>"
