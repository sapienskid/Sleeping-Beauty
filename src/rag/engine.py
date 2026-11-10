"""RAG Engine orchestrating hybrid retrieval and streaming LLM generation."""

from typing import List, Dict, Any, AsyncGenerator, Optional
import httpx
from src.config import settings
from src.embeddings.embedder import GemmaEmbedder
from src.db.store import KnowledgeStore
from src.rag.prompt import build_chat_messages


class RAGEngine:
    def __init__(
        self,
        embedder: Optional[GemmaEmbedder] = None,
        store: Optional[KnowledgeStore] = None,
    ):
        self.embedder = embedder or GemmaEmbedder.get_instance()
        self.store = store or KnowledgeStore.get_instance()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        vault_name: Optional[str] = None,
        disabled_vaults: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run hybrid semantic & keyword search for the query."""
        k = top_k or settings.RAG_TOP_K
        query_vector = self.embedder.embed_query(query)
        results = self.store.hybrid_search(
            query=query,
            query_vector=query_vector,
            top_k=k,
            vault_name=vault_name,
            disabled_vaults=disabled_vaults,
        )
        return results

    async def ask_stream(
        self,
        query: str,
        top_k: Optional[int] = None,
        vault_name: Optional[str] = None,
        disabled_vaults: Optional[List[str]] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response tokens from llama-server alongside retrieved sources."""
        # 1. Retrieve relevant chunks
        chunks = self.retrieve(query, top_k=top_k, vault_name=vault_name, disabled_vaults=disabled_vaults)

        # First yield the retrieved sources metadata
        yield {
            "type": "sources",
            "sources": chunks,
        }

        # 2. Build messages payload
        messages = build_chat_messages(query, chunks)
        payload = {
            "model": settings.LLM_MODEL_NAME,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {settings.LLAMA_API_KEY}",
            "Content-Type": "application/json",
        }

        # 3. Stream from llama-server
        endpoint = f"{settings.LLAMA_SERVER_URL.rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        yield {
                            "type": "error",
                            "error": f"LLM server error ({response.status_code}): {err_text.decode('utf-8', errors='ignore')}",
                        }
                        return

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            import json
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content") or ""
                            if content:
                                yield {
                                    "type": "token",
                                    "token": content,
                                }
                        except Exception:
                            continue
        except httpx.ConnectError:
            yield {
                "type": "error",
                "error": f"Could not connect to llama-server at {settings.LLAMA_SERVER_URL}. Start llama-server first with: ./sb start-llm",
            }

    async def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        vault_name: Optional[str] = None,
        disabled_vaults: Optional[List[str]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Synchronous/blocking complete answer generation with sources."""
        chunks = self.retrieve(query, top_k=top_k, vault_name=vault_name, disabled_vaults=disabled_vaults)
        messages = build_chat_messages(query, chunks)

        payload = {
            "model": settings.LLM_MODEL_NAME,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {settings.LLAMA_API_KEY}",
            "Content-Type": "application/json",
        }

        endpoint = f"{settings.LLAMA_SERVER_URL.rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(endpoint, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return {
                        "query": query,
                        "answer": answer,
                        "sources": chunks,
                    }
                else:
                    return {
                        "query": query,
                        "answer": f"LLM error: {resp.text}",
                        "sources": chunks,
                    }
        except httpx.ConnectError:
            return {
                "query": query,
                "answer": f"[Notice: llama-server not running at {settings.LLAMA_SERVER_URL}. Retrieved {len(chunks)} relevant snippets above.]",
                "sources": chunks,
            }
