"""Retrieval service — query Chroma and return ranked chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import chromadb

    from gateway.config import RetrievalConfig
    from gateway.services.embedding import EmbeddingService

from gateway.models import RetrievedChunk, RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalService:
    """Queries Chroma collections and returns ranked RetrievedChunks."""

    def __init__(
        self,
        chroma_client: chromadb.HttpClient,
        embedder: EmbeddingService,
        config: RetrievalConfig,
    ) -> None:
        self._chroma = chroma_client
        self._embedder = embedder
        self._config = config

    def query(
        self,
        query: str,
        collections: list[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Embed query and search across specified Chroma collections.

        Args:
            query: The user's question text.
            collections: Collection names to search. None = use config default.
            top_k: Number of results per collection. None = use config default.

        Returns:
            RetrievalResult with ranked chunks and similarity scores.
        """
        if collections is None:
            collections = self._config.collections
        if top_k is None:
            top_k = self._config.top_k

        query_embedding = self._embedder.embed_query(query)

        all_chunks: list[RetrievedChunk] = []
        searched: list[str] = []

        for coll_name in collections:
            try:
                collection = self._chroma.get_collection(name=coll_name)
            except Exception:
                logger.warning("Collection '%s' not found, skipping", coll_name)
                continue

            searched.append(coll_name)

            if collection.count() == 0:
                logger.debug("Collection '%s' is empty, skipping", coll_name)
                continue

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            # Chroma returns distances; for cosine space, distance = 1 - similarity
            ids = results["ids"][0] if results["ids"] else []
            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []
            dists = results["distances"][0] if results["distances"] else []

            for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
                similarity = 1.0 - float(dist)  # cosine distance → similarity
                meta = meta or {}
                all_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=doc or "",
                        source_path=meta.get("source_path", ""),
                        source_filename=meta.get("source_filename", ""),
                        page_number=meta.get("page_number"),
                        line_start=meta.get("line_start"),
                        line_end=meta.get("line_end"),
                        section_heading=meta.get("section_heading"),
                        file_type=meta.get("file_type", ""),
                        similarity=round(similarity, 4),
                    )
                )

        # Sort by similarity descending, take top_k overall
        all_chunks.sort(key=lambda c: c.similarity, reverse=True)
        all_chunks = all_chunks[:top_k]

        max_sim = all_chunks[0].similarity if all_chunks else 0.0

        return RetrievalResult(
            chunks=all_chunks,
            collections_searched=searched,
            max_similarity=round(max_sim, 4),
        )
