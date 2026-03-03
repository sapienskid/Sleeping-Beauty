"""GET /health — gateway and dependency health checks."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Check health of all gateway dependencies."""
    components: dict = {}
    overall = "ok"

    # Gateway itself
    from gateway import __version__

    components["gateway"] = {"status": "ok", "version": __version__}

    # llama-server (chat)
    llama_chat = getattr(request.app.state, "llama_chat", None)
    if llama_chat is not None:
        try:
            ok = await llama_chat.health_check()
        except Exception as exc:
            logger.warning("llama_chat health check failed: %s", exc)
            ok = False
        components["llama_chat"] = {
            "status": "ok" if ok else "unavailable",
            "url": llama_chat.base_url,
        }
        if not ok:
            overall = "degraded"
    else:
        components["llama_chat"] = {"status": "not_configured"}
        overall = "degraded"

    # Chroma
    chroma = getattr(request.app.state, "chroma", None)
    if chroma is not None:
        try:
            chroma.heartbeat()
            components["chroma"] = {"status": "ok"}
        except Exception:
            components["chroma"] = {"status": "unavailable"}
            overall = "degraded"
    else:
        components["chroma"] = {"status": "not_configured"}

    # Embedding model
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is not None:
        dimension: int | None
        status = "ok"
        try:
            dimension = embedder.dimension
        except AssertionError:
            # llama-server backend does not know dimension until first embed call.
            dimension = None
            status = "warming_up"

        components["embedder"] = {
            "status": status,
            "dimension": dimension,
        }
    else:
        components["embedder"] = {"status": "not_configured"}

    return JSONResponse(content={"status": overall, "components": components})
