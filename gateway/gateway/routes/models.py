"""GET /v1/models — proxy to llama-server."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """Proxy the models list from llama-server."""
    proxy = request.app.state.llama_chat
    response = await proxy.proxy_json("GET", "/v1/models")
    return JSONResponse(content=response.json(), status_code=response.status_code)
