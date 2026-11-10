"""Configuration settings for Sleeping Beauty."""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    PROJECT_NAME: str = "Sleeping Beauty"

    # LanceDB Vector Database
    LANCEDB_DIR: Path = Path("./data/lancedb")
    TABLE_NAME: str = "knowledge_chunks"

    # Embedding settings (EmbeddingGemma-2 default)
    # Supports llama-server /v1/embeddings endpoint (ultra fast, ~260MB VRAM) or local SentenceTransformers
    EMBEDDING_SERVER_URL: Optional[str] = "http://127.0.0.1:10001/v1"
    EMBEDDING_GGUF_PATH: Path = Path("/home/sapiens/models/embeddinggemma-2-Q8_0.gguf")
    EMBEDDING_MODEL: str = "google/embeddinggemma-2"
    EMBEDDING_DEVICE: str = "cuda"  # "cuda" or "cpu" for local fallback
    EMBEDDING_DTYPE: str = "float16"  # "float16", "bfloat16", "float32"
    EMBEDDING_DIM: int = 768
    EMBEDDING_MRL_DIM: Optional[int] = None  # None for full 768, or e.g. 512, 256
    EMBEDDING_BATCH_SIZE: int = 64

    # Task prefixes (critical for EmbeddingGemma-2)
    EMBED_DOC_PREFIX: str = "task: search result | text: "
    EMBED_QUERY_PREFIX: str = "task: search query | text: "

    # Chunking settings (balanced budget for 1-2B micro LLMs)
    CHUNK_SIZE: int = 800  # characters/subwords
    CHUNK_OVERLAP: int = 120
    ALLOWED_EXTENSIONS: List[str] = [
        ".md", ".markdown", ".txt", ".pdf",
        ".py", ".rs", ".js", ".ts", ".html",
        ".css", ".json", ".yaml", ".yml", ".toml",
        ".sh", ".bash", ".csv"
    ]

    # Local LLM Backend (llama-server running on host GPU :10000)
    MODELS_DIR: Path = Path("/home/sapiens/models")
    LLAMA_SERVER_BIN: Path = Path("/home/sapiens/llama.cpp/build-cuda/bin/llama-server")
    LLM_GGUF_PATH: Path = Path("/home/sapiens/models/qwen2.5-1.5b-instruct-q8_0.gguf")
    LLAMA_SERVER_URL: str = "http://127.0.0.1:10000/v1"
    LLAMA_API_KEY: str = "sk-local"
    LLM_MODEL_NAME: str = "default"
    LLM_TEMPERATURE: float = 0.05  # low temp for strict factual adherence
    LLM_MAX_TOKENS: int = 1024

    # Retrieval tuning (Hybrid search: BM25 + Dense vector)
    RAG_TOP_K: int = 4  # micro LLMs thrive on top 3-4 high-precision snippets
    RAG_MIN_SCORE: float = 0.10


settings = Settings()
