"""FastAPI Application providing Web Dashboard, REST API, and OpenAI-compatible proxy."""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Optional
import httpx
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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


# --- llama.cpp Server Proxy Endpoints (Required by llama.cpp UI) ---
@app.get("/props")
async def proxy_props():
    url = f"{settings.LLAMA_SERVER_URL.rstrip('/v1')}/props"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.get("/models")
@app.get("/v1/models")
async def proxy_models():
    url = f"{settings.LLAMA_SERVER_URL.rstrip('/')}/models"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.get("/slots")
async def proxy_slots():
    url = f"{settings.LLAMA_SERVER_URL.rstrip('/v1')}/slots"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


# --- OpenAI-Compatible Chat Completions with Streaming RAG ---
@app.post("/v1/chat/completions")
async def chat_completions(payload: dict):
    """Augments standard OpenAI chat requests with local hybrid retrieval and streaming citations."""
    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Messages array required")

    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                last_user_msg = content
            elif isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                last_user_msg = " ".join(parts)
            break

    engine = RAGEngine()
    chunks = []
    if last_user_msg.strip():
        try:
            chunks = engine.retrieve(last_user_msg, top_k=settings.RAG_TOP_K)
        except Exception as e:
            print(f"Retrieval notice: {e}")

    from src.rag.prompt import format_rag_prompt, SYSTEM_PROMPT

    if chunks:
        augmented_user_content = format_rag_prompt(last_user_msg, chunks)
        new_messages = []
        has_system = False
        for m in messages[:-1]:
            if m.get("role") == "system":
                new_messages.append({"role": "system", "content": f"{m.get('content', '')}\n\n{SYSTEM_PROMPT}"})
                has_system = True
            else:
                new_messages.append(m)
        if not has_system:
            new_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        new_messages.append({"role": "user", "content": augmented_user_content})
        payload["messages"] = new_messages

    endpoint = f"{settings.LLAMA_SERVER_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.LLAMA_API_KEY}"}

    if payload.get("stream", False):
        async def stream_generator():
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: [DONE]"):
                            if chunks:
                                citations = "\n\n---\n**📚 Grounding Sources:**\n"
                                for idx, c in enumerate(chunks, 1):
                                    rel = c.get("relative_path", "")
                                    sec = f" > {c.get('section')}" if c.get("section") else ""
                                    citations += f"- [{idx}] `{rel}{sec}`\n"
                                citation_chunk = {
                                    "choices": [{"delta": {"content": citations}, "index": 0}],
                                    "object": "chat.completion.chunk",
                                }
                                yield f"data: {json.dumps(citation_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                        yield f"{line}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)
        data = resp.json()
        if chunks and "choices" in data and len(data["choices"]) > 0:
            citations = "\n\n---\n**📚 Grounding Sources:**\n"
            for idx, c in enumerate(chunks, 1):
                rel = c.get("relative_path", "")
                sec = f" > {c.get('section')}" if c.get("section") else ""
                citations += f"- [{idx}] `{rel}{sec}`\n"
            data["choices"][0]["message"]["content"] += citations
        return data


# --- Dedicated Vault & RAG Dashboard ---
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Sleeping Beauty Local Knowledge Hub is running.</h1>"


# --- Serve llama.cpp Official Web UI at Root ---
LLAMA_UI_DIR = Path(__file__).parent / "llama_ui"
if LLAMA_UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(LLAMA_UI_DIR), html=True), name="llama_ui")

