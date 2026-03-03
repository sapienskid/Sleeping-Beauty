"""Async HTTP proxy to llama-server with SSE streaming support."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LlamaProxy:
    """Proxies requests to a llama-server instance."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LlamaProxy not started — call start() first")
        return self._client

    async def proxy_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Proxy a non-streaming request and return the full response."""
        response = await self.client.request(
            method,
            path,
            json=json_body,
        )
        response.raise_for_status()
        return response

    async def proxy_stream(
        self,
        path: str,
        *,
        json_body: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Proxy a streaming (SSE) POST request.

        Yields raw bytes from the upstream SSE stream.
        """
        async with self.client.stream(
            "POST",
            path,
            json=json_body,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk

    async def health_check(self) -> bool:
        """Check if llama-server is reachable."""
        try:
            resp = await self.client.get("/health")
            return resp.status_code == 200
        except (httpx.HTTPError, RuntimeError):
            return False
