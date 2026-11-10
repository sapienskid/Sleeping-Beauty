"""FastAPI Application providing Web Dashboard, REST API, and OpenAI-compatible proxy."""

import os
import re
import json
import asyncio
from pathlib import Path
from typing import List, Optional
import httpx
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Response, Request
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


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in ("/", "/index.html", "/sw.js"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


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
def _llama_v1_url() -> str:
    return settings.LLAMA_SERVER_URL.rstrip("/")


def _llama_root_url() -> str:
    base = _llama_v1_url()
    return base[:-3] if base.endswith("/v1") else base


def _forward_headers(request: Request) -> dict:
    headers = {"Authorization": f"Bearer {settings.LLAMA_API_KEY}"}
    for h in ("x-conversation-id", "content-type"):
        if h in request.headers:
            headers[h] = request.headers[h]
    return headers


@app.get("/props")
async def proxy_props(request: Request):
    qs = f"?{request.url.query}" if request.url.query else ""
    url = f"{_llama_root_url()}/props{qs}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, headers=_forward_headers(request))
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.get("/models")
@app.get("/v1/models")
async def proxy_models(request: Request):
    url = f"{_llama_v1_url()}/models"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers=_forward_headers(request))
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.post("/models/load")
async def proxy_models_load(request: Request):
    url = f"{_llama_root_url()}/models/load"
    body = await request.body()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, content=body, headers=_forward_headers(request))
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.post("/models/unload")
async def proxy_models_unload(request: Request):
    url = f"{_llama_root_url()}/models/unload"
    body = await request.body()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, content=body, headers=_forward_headers(request))
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.get("/models/sse")
async def proxy_models_sse(request: Request):
    url = f"{_llama_root_url()}/models/sse"
    client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
    try:
        req = client.build_request("GET", url, headers=_forward_headers(request))
        r = await client.send(req, stream=True)
    except Exception:
        await client.aclose()
        raise

    if r.status_code != 200:
        body = await r.aread()
        ctype = r.headers.get("content-type", "application/json")
        await r.aclose()
        await client.aclose()
        return Response(content=body, status_code=r.status_code, media_type=ctype)

    async def _stream_sse():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    return StreamingResponse(_stream_sse(), status_code=r.status_code, media_type="text/event-stream")


@app.get("/slots")
async def proxy_slots(request: Request):
    qs = f"?{request.url.query}" if request.url.query else ""
    url = f"{_llama_root_url()}/slots{qs}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.api_route("/v1/streams/lookup", methods=["GET", "POST", "OPTIONS"])
async def proxy_streams_lookup(request: Request):
    url = f"{_llama_root_url()}/v1/streams/lookup"
    body = await request.body()
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.request(request.method, url, content=body, headers=_forward_headers(request))
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.api_route("/v1/stream", methods=["GET", "DELETE", "OPTIONS"])
async def proxy_stream(request: Request):
    qs = f"?{request.url.query}" if request.url.query else ""
    url = f"{_llama_root_url()}/v1/stream{qs}"
    headers = _forward_headers(request)

    if request.method == "DELETE":
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.delete(url, headers=headers)
            if r.status_code == 204:
                return Response(status_code=204)
            return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

    client = httpx.AsyncClient(timeout=120.0)
    try:
        req = client.build_request(request.method, url, headers=headers)
        r = await client.send(req, stream=True)
    except Exception:
        await client.aclose()
        raise

    ctype = r.headers.get("content-type", "")
    if r.status_code != 200 or "text/event-stream" not in ctype:
        body = await r.aread()
        await r.aclose()
        await client.aclose()
        if r.status_code == 204:
            return Response(status_code=204)
        return Response(content=body, status_code=r.status_code, media_type=ctype or "application/json")

    async def _stream_sse():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    return StreamingResponse(_stream_sse(), status_code=r.status_code, media_type="text/event-stream")


@app.api_route("/v1/chat/completions/control", methods=["POST", "OPTIONS"])
async def proxy_completions_control(request: Request):
    url = f"{_llama_root_url()}/v1/chat/completions/control"
    body = await request.body()
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, content=body, headers=_forward_headers(request))
        if r.status_code == 204:
            return Response(status_code=204)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


@app.api_route("/tools", methods=["GET", "POST", "OPTIONS"])
async def proxy_tools(request: Request):
    return Response(
        content='{"error":{"message":"Server tools are disabled"}}',
        status_code=403,
        media_type="application/json",
    )


# --- OpenAI-Compatible Chat Completions with Streaming RAG ---
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, payload: dict):
    """Augments standard OpenAI chat requests with local hybrid retrieval and streaming citations."""
    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Messages array required")

    # Options from client (injected UI control or body parameter)
    rag_enabled = payload.get("rag_enabled", True)
    selected_vault = payload.get("selected_vault", None)
    disabled_vaults = payload.get("disabled_vaults", [])
    top_k = payload.get("top_k", settings.RAG_TOP_K)

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

    # Parse inline flags from prompt (e.g., /norag, /direct, /vault:handbook)
    cleaned_user_msg = last_user_msg
    if "/norag" in last_user_msg.lower() or "/direct" in last_user_msg.lower():
        rag_enabled = False
        cleaned_user_msg = re.sub(r"/(norag|direct)", "", cleaned_user_msg, flags=re.IGNORECASE).strip()
    elif "/rag" in last_user_msg.lower():
        rag_enabled = True
        cleaned_user_msg = re.sub(r"/rag", "", cleaned_user_msg, flags=re.IGNORECASE).strip()

    vault_match = re.search(r"/vault:(\w+)", cleaned_user_msg, flags=re.IGNORECASE)
    if vault_match:
        selected_vault = vault_match.group(1)
        cleaned_user_msg = re.sub(r"/vault:\w+", "", cleaned_user_msg, flags=re.IGNORECASE).strip()

    chunks = []
    if rag_enabled and cleaned_user_msg.strip() and payload.get("n_predict") != 0:
        def _retrieve_sync():
            engine = RAGEngine()
            return engine.retrieve(
                cleaned_user_msg,
                top_k=top_k,
                vault_name=selected_vault if selected_vault != "all" else None,
                disabled_vaults=disabled_vaults,
            )

        try:
            chunks = await asyncio.to_thread(_retrieve_sync)
        except Exception as e:
            print(f"Retrieval notice: {e}")

    from src.rag.prompt import format_rag_prompt, SYSTEM_PROMPT

    if chunks:
        augmented_user_content = format_rag_prompt(cleaned_user_msg, chunks)
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
    else:
        if cleaned_user_msg != last_user_msg:
            new_messages = list(messages[:-1])
            new_messages.append({"role": "user", "content": cleaned_user_msg})
            payload["messages"] = new_messages

    endpoint = f"{_llama_v1_url()}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.LLAMA_API_KEY}"}
    for h in ("x-conversation-id", "content-type"):
        if h in request.headers:
            headers[h] = request.headers[h]

    clean_payload = {
        k: v for k, v in payload.items()
        if k not in ("rag_enabled", "selected_vault", "disabled_vaults", "top_k", "include_sources")
    }
    clean_payload.pop("tools", None)
    clean_payload.pop("tool_choice", None)
    if clean_payload.get("model") in (None, "", "default"):
        clean_payload["model"] = settings.LLM_MODEL_NAME

    if payload.get("stream", False):
        client = httpx.AsyncClient(timeout=120.0)
        try:
            req = client.build_request("POST", endpoint, json=clean_payload, headers=headers)
            response = await client.send(req, stream=True)
        except Exception:
            await client.aclose()
            raise

        if response.status_code != 200:
            err_body = await response.aread()
            err_ctype = response.headers.get("content-type", "application/json")
            await response.aclose()
            await client.aclose()
            return Response(content=err_body, status_code=response.status_code, media_type=err_ctype)

        async def stream_generator():
            try:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: [DONE]"):
                        if chunks and payload.get("include_sources", True):
                            citations = "\n\n---\n**Sources & Citations:**\n"
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
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(endpoint, json=clean_payload, headers=headers)
        if resp.status_code != 200:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        data = resp.json()
        if chunks and payload.get("include_sources", True) and "choices" in data and len(data["choices"]) > 0:
            citations = "\n\n---\n**Sources & Citations:**\n"
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


# --- Serve llama.cpp Official Web UI at Root with Cache-Busting ---
LLAMA_UI_DIR = Path(__file__).parent / "llama_ui"

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_root():
    index_path = LLAMA_UI_DIR / "index.html"
    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        return Response(
            content=content,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return HTMLResponse("<h1>UI not found</h1>", status_code=404)


@app.get("/sw.js")
async def serve_sw():
    sw_path = LLAMA_UI_DIR / "sw.js"
    if sw_path.exists():
        content = sw_path.read_text(encoding="utf-8")
        return Response(
            content=content,
            media_type="application/javascript",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return Response(content="", media_type="application/javascript")


if LLAMA_UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(LLAMA_UI_DIR), html=True), name="llama_ui")

