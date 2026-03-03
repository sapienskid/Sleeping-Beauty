"""Configuration loader — YAML → Pydantic models."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8800
    workers: int = 2
    log_level: str = "info"


class LlamaChatConfig(BaseModel):
    base_url: str = "http://host.docker.internal:8080"


class LlamaEmbedConfig(BaseModel):
    backend: str = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"
    base_url: str | None = None  # only used when backend == "llama-server"


class ChromaConfig(BaseModel):
    host: str = "chroma"
    port: int = 8000


class RetrievalConfig(BaseModel):
    top_k: int = 5
    confidence_threshold: float = 0.35
    collections: list[str] = Field(default_factory=lambda: ["project-docs"])
    max_context_tokens: int = 3000


class IngestionConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64


class MemoryConfig(BaseModel):
    enabled: bool = True
    collection: str = "user-memory"


class AppConfig(BaseModel):
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    llama_chat: LlamaChatConfig = Field(default_factory=LlamaChatConfig)
    llama_embed: LlamaEmbedConfig = Field(default_factory=LlamaEmbedConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file.

    Falls back to CONFIG_PATH env var, then defaults.
    """
    if path is None:
        path = os.environ.get("CONFIG_PATH", "/app/config.yaml")

    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig(**raw)

    return AppConfig()
