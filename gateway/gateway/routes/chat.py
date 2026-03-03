"""POST /v1/chat/completions — proxy to llama-server.

Phase 1: Pure proxy (no RAG). The gateway forwards the request as-is
to llama-server and streams the response back. RAG injection will be
added in a later phase.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    """Forward chat completion requests to llama-server.

    Supports both streaming (SSE) and non-streaming modes.
    """
    body = await request.json()
    proxy = request.app.state.llama_chat
    stream = body.get("stream", False)

    logger.info(
        "chat request: model=%s stream=%s messages=%d",
        body.get("model", "?"),
        stream,
        len(body.get("messages", [])),
    )

    if stream:
        return StreamingResponse(
            proxy.proxy_stream("/v1/chat/completions", json_body=body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: proxy and return JSON
    response = await proxy.proxy_json("POST", "/v1/chat/completions", json_body=body)
    return JSONResponse(content=response.json(), status_code=response.status_code)
