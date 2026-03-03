"""Tests for /health endpoint."""

from __future__ import annotations

import pytest
import httpx


pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_health_all_ok(client: httpx.AsyncClient):
    """All components healthy → status=ok."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["components"]["gateway"]["status"] == "ok"
    assert data["components"]["gateway"]["version"] == "0.2.0"
    assert data["components"]["llama_chat"]["status"] == "ok"
    assert data["components"]["chroma"]["status"] == "ok"
    assert data["components"]["embedder"]["status"] == "ok"
    assert data["components"]["embedder"]["dimension"] == 384


async def test_health_llama_down(client: httpx.AsyncClient, app):
    """When llama-server is unreachable → status=degraded."""
    app.state.llama_chat.health_check.return_value = False
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["components"]["llama_chat"]["status"] == "unavailable"
    # Restore
    app.state.llama_chat.health_check.return_value = True
