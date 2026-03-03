"""Embedding service — wraps sentence-transformers or external llama-server."""

from __future__ import annotations

import logging
from typing import Any
from typing import TYPE_CHECKING

import httpx
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from gateway.config import LlamaEmbedConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Produces embeddings using sentence-transformers or llama-server."""

    def __init__(self, config: LlamaEmbedConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None
        self._http_client: httpx.Client | None = None
        self._llama_dimension: int | None = None

    def load(self) -> None:
        """Initialize the configured embedding backend."""
        if self._config.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._config.model)
            self._model = SentenceTransformer(self._config.model)
            logger.info(
                "Embedding model loaded — dimension=%d",
                self._model.get_sentence_embedding_dimension(),
            )
            return

        if self._config.backend == "llama-server":
            if not self._config.base_url:
                raise ValueError(
                    "llama_embed.base_url is required when backend='llama-server'"
                )
            base_url = self._config.base_url.rstrip("/")
            self._http_client = httpx.Client(base_url=base_url, timeout=60.0)
            logger.info(
                "Embedding backend ready: llama-server at %s", base_url
            )
            return

        raise ValueError(f"Unsupported embedding backend: {self._config.backend}")

    @property
    def dimension(self) -> int:
        if self._config.backend == "sentence-transformers":
            assert self._model is not None, "Call load() first"
            dim = self._model.get_sentence_embedding_dimension()
            assert dim is not None
            return int(dim)
        assert self._llama_dimension is not None, (
            "Embedding dimension is not available yet for llama-server backend. "
            "Call embed() first."
        )
        return self._llama_dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        if not texts:
            return []

        if self._config.backend == "sentence-transformers":
            assert self._model is not None, "Call load() first"
            embeddings: np.ndarray = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return embeddings.tolist()

        return self._embed_with_llama_server(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([query])[0]

    def close(self) -> None:
        """Release embedding backend resources."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def _embed_with_llama_server(self, texts: list[str]) -> list[list[float]]:
        assert self._http_client is not None, "Call load() first"

        payload: dict[str, Any] = {"input": texts}
        if self._config.model:
            payload["model"] = self._config.model

        response = self._http_client.post("/v1/embeddings", json=payload)
        response.raise_for_status()
        body = response.json()
        raw_data = body.get("data", [])
        if not isinstance(raw_data, list):
            raise RuntimeError("Invalid llama-server embeddings response: missing 'data'")

        # Preserve original input ordering.
        data_sorted = sorted(raw_data, key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in data_sorted]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "llama-server returned unexpected number of embeddings "
                f"(expected {len(texts)}, got {len(embeddings)})"
            )

        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] == 0:
            raise RuntimeError("Invalid llama-server embeddings response shape")

        # Keep embedding norms consistent with sentence-transformers path.
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        array = array / norms

        self._llama_dimension = int(array.shape[1])
        return array.tolist()
