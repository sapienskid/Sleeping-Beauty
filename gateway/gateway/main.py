"""Sleeping Beauty Gateway — FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from gateway.config import AppConfig, load_config
from gateway.routes import chat, health, models
from gateway.services.proxy import LlamaProxy

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of shared resources."""
    config: AppConfig = app.state.config

    # Start llama-server proxy
    llama_chat = LlamaProxy(config.llama_chat.base_url)
    await llama_chat.start()
    app.state.llama_chat = llama_chat

    logger.info(
        "Gateway started — proxying to llama-server at %s",
        config.llama_chat.base_url,
    )

    yield

    # Shutdown
    await llama_chat.stop()
    logger.info("Gateway stopped")


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = load_config()

    app = FastAPI(
        title="Sleeping Beauty Gateway",
        version="0.1.0",
        description="Local RAG + Memory gateway for llama.cpp",
        lifespan=lifespan,
    )
    app.state.config = config

    # Register routes
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(health.router)

    return app


# Default app instance for uvicorn
app = create_app()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
