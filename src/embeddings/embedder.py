"""Embedding engine supporting EmbeddingGemma-2 via llama-server /v1/embeddings or local SentenceTransformers."""

import os
from typing import List, Optional
import httpx
import numpy as np
from src.config import settings


class GemmaEmbedder:
    _instance: Optional["GemmaEmbedder"] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        mrl_dim: Optional[int] = None,
        server_url: Optional[str] = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.mrl_dim = mrl_dim or settings.EMBEDDING_MRL_DIM
        self.server_url = (server_url or settings.EMBEDDING_SERVER_URL or "").rstrip("/")
        self.use_server = False
        self.model = None

        # Check if llama.cpp embedding server is active
        if self.server_url:
            try:
                r = httpx.get(f"{self.server_url}/models", timeout=1.5)
                if r.status_code == 200:
                    self.use_server = True
                    self.raw_dim = settings.EMBEDDING_DIM
                    self.effective_dim = self.mrl_dim if self.mrl_dim else self.raw_dim
                    print(f"GemmaEmbedder: connected to llama-server at {self.server_url} (dim: {self.effective_dim}, ~260MB VRAM).")
                    return
            except Exception:
                pass

        # Fallback to local SentenceTransformer
        import torch
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from sentence_transformers import SentenceTransformer

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() and settings.EMBEDDING_DEVICE == "cuda" else "cpu"
        else:
            self.device = device

        self.dtype_str = dtype or settings.EMBEDDING_DTYPE
        torch_dtype = torch.float16 if self.dtype_str == "float16" and self.device == "cuda" else torch.float32

        print(f"Loading local embedding model '{self.model_name}' on {self.device} ({self.dtype_str})...")
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            model_kwargs={"torch_dtype": torch_dtype},
            trust_remote_code=True,
        )

        dummy = self.model.encode(["test"], normalize_embeddings=True)
        self.raw_dim = dummy.shape[1]
        self.effective_dim = self.mrl_dim if self.mrl_dim else self.raw_dim
        print(f"Local embedder ready. Raw dim: {self.raw_dim}, Effective dim: {self.effective_dim}")

    @classmethod
    def get_instance(cls) -> "GemmaEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _apply_mrl(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply Matryoshka Representation Learning dimension truncation and re-normalization."""
        if self.mrl_dim and self.mrl_dim < embeddings.shape[1]:
            truncated = embeddings[:, :self.mrl_dim]
            norms = np.linalg.norm(truncated, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            return truncated / norms
        return embeddings

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Embed a batch of document chunks with the document search result task prefix."""
        bs = batch_size or settings.EMBEDDING_BATCH_SIZE
        prefixed = [f"{settings.EMBED_DOC_PREFIX}{text}" for text in texts]

        if self.use_server:
            all_vecs: List[List[float]] = []
            with httpx.Client(timeout=120.0) as client:
                for i in range(0, len(prefixed), bs):
                    batch = prefixed[i : i + bs]
                    payload = {"input": batch, "model": "default"}
                    resp = client.post(f"{self.server_url}/embeddings", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    # Sort by index to maintain ordering
                    sorted_items = sorted(data["data"], key=lambda x: x.get("index", 0))
                    for item in sorted_items:
                        all_vecs.append(item["embedding"])
            
            arr = np.array(all_vecs, dtype=np.float32)
            arr = self._apply_mrl(arr)
            return arr.tolist()

        # In-process fallback
        import torch
        with torch.inference_mode():
            embeddings = self.model.encode(
                prefixed,
                batch_size=bs,
                show_progress_bar=len(texts) > 50,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        embeddings = self._apply_mrl(embeddings)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query with the search query task prefix."""
        prefixed = f"{settings.EMBED_QUERY_PREFIX}{query}"

        if self.use_server:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self.server_url}/embeddings",
                    json={"input": [prefixed], "model": "default"},
                )
                resp.raise_for_status()
                vec = resp.json()["data"][0]["embedding"]
                arr = np.array([vec], dtype=np.float32)
                arr = self._apply_mrl(arr)
                return arr[0].tolist()

        import torch
        with torch.inference_mode():
            embedding = self.model.encode(
                [prefixed],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        embedding = self._apply_mrl(embedding)
        return embedding[0].tolist()
