"""Unit tests for embedding backend configuration and behavior."""

from __future__ import annotations

import numpy as np
import pytest

from gateway.config import LlamaEmbedConfig
from gateway.services.embedding import EmbeddingService


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def post(self, path: str, *, json: dict) -> _FakeResponse:
        self.calls.append((path, json))
        return _FakeResponse(
            {
                "object": "list",
                "data": [
                    {"index": 1, "embedding": [0.0, 3.0, 4.0]},
                    {"index": 0, "embedding": [3.0, 4.0, 0.0]},
                ],
            }
        )

    def close(self) -> None:
        self.closed = True


def test_llama_server_backend_requires_base_url():
    service = EmbeddingService(
        LlamaEmbedConfig(backend="llama-server", model="nomic-embed-text")
    )
    with pytest.raises(ValueError, match="base_url"):
        service.load()


def test_llama_server_backend_embeds_and_normalizes(monkeypatch: pytest.MonkeyPatch):
    fake_client = _FakeClient()

    def _build_client(*args, **kwargs):
        return fake_client

    monkeypatch.setattr("gateway.services.embedding.httpx.Client", _build_client)

    service = EmbeddingService(
        LlamaEmbedConfig(
            backend="llama-server",
            base_url="http://localhost:8085",
            model="nomic-embed-text",
        )
    )
    service.load()

    vectors = service.embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert service.dimension == 3

    # Input order should be preserved even if API returns shuffled indices.
    assert vectors[0][0] > 0.0
    assert vectors[1][2] > 0.0

    # Vectors should be unit-normalized.
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(vectors[1]), 1.0, atol=1e-6)

    path, payload = fake_client.calls[0]
    assert path == "/v1/embeddings"
    assert payload["input"] == ["hello", "world"]
    assert payload["model"] == "nomic-embed-text"

    service.close()
    assert fake_client.closed is True
