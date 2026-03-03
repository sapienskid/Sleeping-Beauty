"""Sleeping Beauty Gateway — FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from gateway.config import AppConfig, load_config
from gateway.routes import chat, health, manage, models
from gateway.services.proxy import LlamaProxy

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of shared resources."""
    config: AppConfig = app.state.config

    # ── llama-server proxy ───────────────────────────────────────────────
    llama_chat = LlamaProxy(config.llama_chat.base_url)
    await llama_chat.start()
    app.state.llama_chat = llama_chat

    logger.info(
        "Gateway started — proxying to llama-server at %s",
        config.llama_chat.base_url,
    )

    # ── Embedding model ──────────────────────────────────────────────────
    from gateway.services.embedding import EmbeddingService

    embedder = EmbeddingService(config.llama_embed)
    embedder.load()
    app.state.embedder = embedder
    try:
        dim = embedder.dimension
        logger.info(
            "Embedding model ready (backend=%s, dim=%d)",
            config.llama_embed.backend,
            dim,
        )
    except AssertionError:
        logger.info(
            "Embedding model ready (backend=%s, dim=unknown until first embedding call)",
            config.llama_embed.backend,
        )

    # ── Chroma client ────────────────────────────────────────────────────
    import chromadb

    chroma_client = chromadb.HttpClient(
        host=config.chroma.host,
        port=config.chroma.port,
    )
    app.state.chroma = chroma_client
    logger.info(
        "Chroma client connected: %s:%d", config.chroma.host, config.chroma.port
    )

    # ── Retrieval service ────────────────────────────────────────────────
    from gateway.services.retrieval import RetrievalService

    retrieval = RetrievalService(chroma_client, embedder, config.retrieval)
    app.state.retrieval = retrieval
    logger.info(
        "Retrieval service ready (top_k=%d, threshold=%.2f)",
        config.retrieval.top_k,
        config.retrieval.confidence_threshold,
    )

    # ── Ingestion service ────────────────────────────────────────────────
    from gateway.services.ingestion import IngestionService

    ingestion = IngestionService(chroma_client, embedder, config.ingestion)
    app.state.ingestion = ingestion
    logger.info("Ingestion service ready")

    # ── Memory service ───────────────────────────────────────────────────
    if config.memory.enabled:
        from gateway.services.memory import MemoryService

        memory = MemoryService(chroma_client, embedder, config.memory)
        app.state.memory = memory
        logger.info("Memory service ready (collection=%s)", config.memory.collection)
    else:
        app.state.memory = None
        logger.info("Memory service disabled")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    embedder.close()
    await llama_chat.stop()
    logger.info("Gateway stopped")


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = load_config()

    app = FastAPI(
        title="Sleeping Beauty Gateway",
        version="0.2.0",
        description="Local RAG + Memory gateway for llama.cpp",
        lifespan=lifespan,
    )
    app.state.config = config

    # Register routes
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(manage.router)

    return app


# Default app instance for uvicorn
app = create_app()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
