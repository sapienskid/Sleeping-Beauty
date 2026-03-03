"""Memory service — persistent user preferences and writing style."""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import chromadb

    from gateway.config import MemoryConfig
    from gateway.services.embedding import EmbeddingService

from gateway.models import MemoryItem

logger = logging.getLogger(__name__)


class MemoryService:
    """CRUD operations for the user-memory Chroma collection."""

    def __init__(
        self,
        chroma_client: chromadb.HttpClient,
        embedder: EmbeddingService,
        config: MemoryConfig,
    ) -> None:
        self._chroma = chroma_client
        self._embedder = embedder
        self._config = config
        self._collection_name = config.collection

    def _get_collection(self) -> Any:
        return self._chroma.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def store(
        self,
        content: str,
        memory_type: str = "preference",
        tags: list[str] | None = None,
        source: str = "explicit",
    ) -> str:
        """Store a new memory. Returns the memory ID."""
        mem_id = f"mem_{uuid.uuid4().hex[:12]}"
        embedding = self._embedder.embed_query(content)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        metadata: dict[str, Any] = {
            "type": memory_type,
            "source": source,
            "created_at": now,
            "updated_at": now,
        }
        if tags:
            metadata["tags"] = ",".join(tags)  # Chroma doesn't support list values

        collection = self._get_collection()
        collection.upsert(
            ids=[mem_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
        )

        logger.info("Stored memory %s: type=%s", mem_id, memory_type)
        return mem_id

    def retrieve_relevant(
        self,
        query: str,
        memory_types: list[str] | None = None,
        top_k: int = 3,
    ) -> list[MemoryItem]:
        """Retrieve memories relevant to a query.

        Args:
            query: The user's message text.
            memory_types: Filter by type (e.g., ["preference", "fact"]).
            top_k: Max number of memories to return.

        Returns:
            List of MemoryItem sorted by relevance.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        query_embedding = self._embedder.embed_query(query)

        where_filter = None
        if memory_types and len(memory_types) == 1:
            where_filter = {"type": memory_types[0]}
        elif memory_types and len(memory_types) > 1:
            where_filter = {"type": {"$in": memory_types}}

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                where=where_filter,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning("Memory query failed: %s", e)
            return []

        items: list[MemoryItem] = []
        ids = results["ids"][0] if results["ids"] else []
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []

        for mem_id, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            tags_str = meta.get("tags", "")
            items.append(
                MemoryItem(
                    id=mem_id,
                    content=doc or "",
                    type=meta.get("type", "unknown"),
                    tags=tags_str.split(",") if tags_str else [],
                    created_at=meta.get("created_at", ""),
                )
            )

        return items

    def list_all(
        self,
        memory_type: str | None = None,
        tag: str | None = None,
    ) -> list[MemoryItem]:
        """List all memories, optionally filtered by type and/or tag."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        where_filter = None
        if memory_type and tag:
            where_filter = {
                "$and": [
                    {"type": memory_type},
                    {"tags": {"$contains": tag}},
                ]
            }
        elif memory_type:
            where_filter = {"type": memory_type}
        elif tag:
            where_filter = {"tags": {"$contains": tag}}

        try:
            results = collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning("Memory list failed: %s", e)
            return []

        items: list[MemoryItem] = []
        ids = results["ids"] if results["ids"] else []
        docs = results["documents"] if results["documents"] else []
        metas = results["metadatas"] if results["metadatas"] else []

        for mem_id, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            tags_str = meta.get("tags", "")
            items.append(
                MemoryItem(
                    id=mem_id,
                    content=doc or "",
                    type=meta.get("type", "unknown"),
                    tags=tags_str.split(",") if tags_str else [],
                    created_at=meta.get("created_at", ""),
                )
            )

        return items

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        collection = self._get_collection()
        try:
            collection.delete(ids=[memory_id])
            logger.info("Deleted memory: %s", memory_id)
            return True
        except Exception as e:
            logger.warning("Failed to delete memory %s: %s", memory_id, e)
            return False
