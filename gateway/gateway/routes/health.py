"""GET /health — gateway and dependency health checks."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """Return health status of the gateway and its dependencies."""
    config = request.app.state.config
    llama_chat = request.app.state.llama_chat

    components: dict = {
        "gateway": {"status": "ok", "version": "0.1.0"},
    }

    # Check llama-server (chat)
    if await llama_chat.health_check():
        components["llama_chat"] = {
            "status": "ok",
            "url": config.llama_chat.base_url,
        }
    else:
        components["llama_chat"] = {
            "status": "unavailable",
            "url": config.llama_chat.base_url,
        }

    # Chroma health check (deferred — not wired up yet in phase 1)
    # TODO: add chroma health check when retrieval is implemented

    all_ok = all(c.get("status") == "ok" for c in components.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        content={"status": "ok" if all_ok else "degraded", "components": components},
        status_code=status_code,
    )
