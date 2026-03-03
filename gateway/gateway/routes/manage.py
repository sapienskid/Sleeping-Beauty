"""Management routes: /ingest, /collections, /memory."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from gateway.models import (
    CollectionInfo,
    CollectionsResponse,
    IngestRequest,
    IngestResponse,
    MemoryCreateRequest,
    MemoryCreateResponse,
    MemoryListResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ─── Ingestion ───────────────────────────────────────────────────────────────


@router.post("/ingest")
async def ingest(request: Request, body: IngestRequest) -> JSONResponse:
    """Trigger document ingestion."""
    ingestion_service = getattr(request.app.state, "ingestion", None)
    if ingestion_service is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Ingestion service not available"},
        )

    path = Path(body.source_path)
    if path.is_dir():
        result = ingestion_service.ingest_directory(
            dir_path=body.source_path,
            collection_name=body.collection,
            recursive=body.recursive,
            force_reindex=body.force_reindex,
        )
    elif path.is_file():
        result = ingestion_service.ingest_file(
            file_path=body.source_path,
            collection_name=body.collection,
            force_reindex=body.force_reindex,
        )
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"Path not found: {body.source_path}"},
        )

    status_code = 200 if result["status"] != "failed" else 500
    return JSONResponse(content=result, status_code=status_code)


@router.get("/ingest/{job_id}")
async def ingest_status(request: Request, job_id: str) -> JSONResponse:
    """Check ingestion job status."""
    ingestion_service = getattr(request.app.state, "ingestion", None)
    if ingestion_service is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Ingestion service not available"},
        )

    job = ingestion_service.get_job(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Job not found: {job_id}"},
        )
    return JSONResponse(content=job)


# ─── Collections ─────────────────────────────────────────────────────────────


@router.get("/collections")
async def list_collections(request: Request) -> JSONResponse:
    """List all Chroma collections with document counts."""
    chroma = getattr(request.app.state, "chroma", None)
    if chroma is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Chroma not available"},
        )

    try:
        collections = chroma.list_collections()
        items = []
        for coll in collections:
            items.append(CollectionInfo(name=coll.name, count=coll.count()))
        return JSONResponse(content=CollectionsResponse(collections=items).model_dump())
    except Exception as e:
        logger.error("Failed to list collections: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


# ─── Memory ──────────────────────────────────────────────────────────────────


@router.post("/memory")
async def create_memory(request: Request, body: MemoryCreateRequest) -> JSONResponse:
    """Store a new memory."""
    memory_service = getattr(request.app.state, "memory", None)
    if memory_service is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Memory service not available"},
        )

    mem_id = memory_service.store(
        content=body.content,
        memory_type=body.type,
        tags=body.tags,
    )
    return JSONResponse(
        content=MemoryCreateResponse(id=mem_id, status="stored").model_dump(),
        status_code=201,
    )


@router.get("/memory")
async def list_memories(
    request: Request,
    type: str | None = Query(None),
    tag: str | None = Query(None),
) -> JSONResponse:
    """List all memories, optionally filtered."""
    memory_service = getattr(request.app.state, "memory", None)
    if memory_service is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Memory service not available"},
        )

    items = memory_service.list_all(memory_type=type, tag=tag)
    return JSONResponse(content=MemoryListResponse(memories=items).model_dump())


@router.delete("/memory/{memory_id}")
async def delete_memory(request: Request, memory_id: str) -> JSONResponse:
    """Delete a specific memory."""
    memory_service = getattr(request.app.state, "memory", None)
    if memory_service is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Memory service not available"},
        )

    deleted = memory_service.delete(memory_id)
    if deleted:
        return JSONResponse(content={"status": "deleted", "id": memory_id})
    return JSONResponse(
        status_code=404,
        content={"error": f"Memory not found: {memory_id}"},
    )
