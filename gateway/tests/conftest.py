"""Shared test fixtures for gateway tests.

Uses httpx ASGITransport to test the FastAPI app in-process.
Mocks external llama-server; uses Chroma in ephemeral mode for integration.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import chromadb
import httpx
import numpy as np
import pytest
import pytest_asyncio

from gateway.config import (
    AppConfig,
    ChromaConfig,
    GatewayConfig,
    IngestionConfig,
    LlamaChatConfig,
    LlamaEmbedConfig,
    MemoryConfig,
    RetrievalConfig,
)
from gateway.main import create_app
from gateway.services.embedding import EmbeddingService


# ─── Fake embedding service ─────────────────────────────────────────────────


class FakeEmbeddingService(EmbeddingService):
    """Deterministic lexical embedding service for tests — no model download.

    Uses stable hashed term vectors so scores are reproducible across runs.
    The goal is not semantic quality; it is predictable confidence behavior
    for retrieval-path tests.
    """

    _STOPWORDS = {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "does",
        "for",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "tell",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }

    def __init__(self) -> None:
        # Skip parent __init__ which requires a config
        self._model = None
        self._dim = 384

    def load(self) -> None:
        pass  # no-op

    @property
    def dimension(self) -> int:
        return self._dim

    @staticmethod
    def _normalize_token(token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        for suffix in ("ments", "ment", "ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[: -len(suffix)]
        return token

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
        normalized = [self._normalize_token(token) for token in tokens]
        return [
            token
            for token in normalized
            if token not in self._STOPWORDS and not token.isdigit()
        ]

    def _word_to_dims(self, word: str) -> list[int]:
        """Map a word to deterministic dimensions in [0, self._dim)."""
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=16).digest()
        return [
            int.from_bytes(digest[0:4], "little") % self._dim,
            int.from_bytes(digest[4:8], "little") % self._dim,
        ]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic normalized vectors."""
        result = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            tokens = self._tokenize(text)
            for word in tokens:
                for dim in self._word_to_dims(word):
                    vec[dim] += 1.0
            # Normalize to unit vector (cosine similarity = dot product)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            else:
                vec[0] = 1.0
            result.append(vec.tolist())
        return result

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


# ─── Fake llama-server responses ────────────────────────────────────────────


def make_chat_response(content: str = "Test response", model: str = "sleeping-beauty") -> dict:
    """Build a minimal OpenAI-compatible chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def make_sse_stream(content: str = "Test response", model: str = "sleeping-beauty") -> bytes:
    """Build a minimal SSE stream as raw bytes."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    done_chunk = {
        "id": chunk["id"],
        "object": "chat.completion.chunk",
        "created": chunk["created"],
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }

    data = f"data: {json.dumps(chunk)}\n\n"
    data += f"data: {json.dumps(done_chunk)}\n\n"
    data += "data: [DONE]\n\n"
    return data.encode()


def make_models_response() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": "sleeping-beauty",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llama.cpp",
            }
        ],
    }


# ─── App + client fixtures ──────────────────────────────────────────────────


@pytest.fixture
def test_config(tmp_path) -> AppConfig:
    """Create an AppConfig with test-friendly settings."""
    return AppConfig(
        gateway=GatewayConfig(host="0.0.0.0", port=8800, workers=1),
        llama_chat=LlamaChatConfig(base_url="http://fake-llama:8080"),
        llama_embed=LlamaEmbedConfig(backend="sentence-transformers", model="fake"),
        chroma=ChromaConfig(host="localhost", port=0),  # unused — we inject ephemeral
        retrieval=RetrievalConfig(
            top_k=5,
            confidence_threshold=0.30,
            collections=["project-docs"],
        ),
        ingestion=IngestionConfig(chunk_size=512, chunk_overlap=64),
        memory=MemoryConfig(enabled=True, collection="test-memory"),
    )


@pytest.fixture
def chroma_client() -> chromadb.ClientAPI:
    """Ephemeral in-memory Chroma for tests."""
    return chromadb.EphemeralClient()


@pytest.fixture
def fake_embedder() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest_asyncio.fixture
async def app(test_config, chroma_client, fake_embedder, tmp_path):
    """Create a fully wired FastAPI app with mocked externals."""
    from gateway.services.ingestion import IngestionService
    from gateway.services.memory import MemoryService
    from gateway.services.retrieval import RetrievalService

    application = create_app(config=test_config)

    # Mock the LlamaProxy so we don't need a real llama-server
    mock_proxy = AsyncMock()
    mock_proxy.base_url = "http://fake-llama:8080"
    mock_proxy.health_check = AsyncMock(return_value=True)

    # Default proxy_json: return a chat completion
    mock_proxy.proxy_json = AsyncMock(
        return_value=MagicMock(
            json=lambda: make_chat_response(),
            status_code=200,
        )
    )

    # Default proxy_stream: yield SSE bytes
    async def _fake_stream(path, *, json_body):
        yield make_sse_stream()

    mock_proxy.proxy_stream = _fake_stream

    # Wire everything into app.state
    application.state.llama_chat = mock_proxy
    application.state.embedder = fake_embedder
    application.state.chroma = chroma_client
    application.state.retrieval = RetrievalService(
        chroma_client, fake_embedder, test_config.retrieval
    )
    application.state.ingestion = IngestionService(
        chroma_client, fake_embedder, test_config.ingestion
    )
    application.state.memory = MemoryService(chroma_client, fake_embedder, test_config.memory)
    application.state.config = test_config

    # Create prompts dir with templates for agent tests
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "search.txt").write_text(
        "Context:\n{context}\n\n{memory}\n\n{original_system_message}"
    )
    (prompts_dir / "draft.txt").write_text(
        "Draft mode.\n{context}\n\n{memory}\n\n{original_system_message}"
    )
    (prompts_dir / "passthrough.txt").write_text("You are a helpful assistant.")

    # Monkey-patch load_prompt to use our test prompts directory instead of /app/prompts
    from gateway.services.agents import AgentConfig

    _original_load_prompt = AgentConfig.load_prompt
    test_prompts_dir = str(prompts_dir)

    def _patched_load_prompt(self, prompts_dir: str = test_prompts_dir) -> str | None:
        if self.system_prompt_template is None:
            return None
        from pathlib import Path

        path = Path(prompts_dir) / self.system_prompt_template
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return None

    AgentConfig.load_prompt = _patched_load_prompt
    try:
        yield application
    finally:
        AgentConfig.load_prompt = _original_load_prompt


@pytest_asyncio.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    """httpx AsyncClient wired to the test app via ASGITransport."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
