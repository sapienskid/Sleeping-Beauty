"""POST /v1/chat/completions — RAG-augmented proxy to llama-server.

Implements the full pipeline:
1. Intent detection (agent routing)
2. Retrieval from Chroma (for search/draft agents)
3. Memory injection (preferences, facts, writing style)
4. Prompt augmentation with retrieved context + citations
5. Proxy to llama-server (streaming or non-streaming)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.models import (
    RetrievalResult,
    RetrievedChunk,
    build_low_confidence_response,
)
from gateway.services.agents import detect_intent, get_agent, strip_norag_prefix

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    """Handle chat completion requests with RAG augmentation."""
    body = await request.json()
    proxy = request.app.state.llama_chat
    config = request.app.state.config

    messages = body.get("messages", [])
    model = body.get("model", "sleeping-beauty")
    stream = body.get("stream", False)

    # ── Guard: no messages or last message isn't from user → pure proxy ──
    if not messages or messages[-1].get("role") != "user":
        return await _proxy_response(proxy, body, stream)

    query = messages[-1]["content"]

    # ── 1. Intent detection ──────────────────────────────────────────────
    intent = detect_intent(query)
    agent = get_agent(intent)

    logger.info(
        "chat request: model=%s intent=%s agent=%s stream=%s messages=%d",
        model,
        intent,
        agent.name,
        stream,
        len(messages),
    )

    # ── 2. Handle 'remember' agent — store memory, no LLM call ──────────
    if intent == "remember":
        return await _handle_remember(request, query, model)

    # ── 3. Handle 'passthrough' — strip prefix, proxy directly ───────────
    if intent == "passthrough":
        messages[-1]["content"] = strip_norag_prefix(query)
        # Still inject memory preferences if available
        memory_text = _get_memory_text(request, query, agent.memory_types)
        if memory_text:
            messages = _inject_memory_only(messages, memory_text)
        body["messages"] = messages
        return await _proxy_response(proxy, body, stream)

    # ── 4. Retrieval (search / draft agents) ─────────────────────────────
    retrieval_service = getattr(request.app.state, "retrieval", None)
    retrieval = RetrievalResult()

    if retrieval_service is not None:
        # Determine which collections to search
        search_collections = None  # None = use config defaults
        if isinstance(agent.collections, list) and agent.collections:
            search_collections = agent.collections
        elif agent.collections == "all":
            search_collections = None  # retrieval service uses config defaults

        retrieval = retrieval_service.query(
            query=query,
            collections=search_collections,
        )

    # ── 5. Confidence check ──────────────────────────────────────────────
    retrieval_cfg = getattr(config, "retrieval", None)
    global_threshold = (
        retrieval_cfg.confidence_threshold
        if retrieval_cfg is not None
        else agent.confidence_threshold
    )
    # Per-agent threshold remains the lower bound for specialized agents
    # (e.g., draft can be more permissive than default search).
    threshold = min(agent.confidence_threshold, global_threshold)

    if retrieval.has_results and retrieval.max_similarity < threshold:
        logger.info(
            "Low confidence: max_similarity=%.4f threshold=%.2f",
            retrieval.max_similarity,
            threshold,
        )
        low_conf = build_low_confidence_response(
            model=model,
            retrieval=retrieval,
            confidence_threshold=threshold,
        )
        if stream:
            # Return as single SSE event then close
            return _stream_single_response(low_conf)
        return JSONResponse(content=low_conf)

    # If no retrieval results at all (empty collections), proxy directly
    if not retrieval.has_results:
        # Still inject memory if available
        memory_text = _get_memory_text(request, query, agent.memory_types)
        if memory_text:
            messages = _inject_memory_only(messages, memory_text)
        body["messages"] = messages
        return await _proxy_response(proxy, body, stream)

    # ── 6. Build augmented prompt ────────────────────────────────────────
    memory_text = _get_memory_text(request, query, agent.memory_types)
    context_block = _build_context_block(retrieval.chunks)

    # Load agent's prompt template
    prompt_template = agent.load_prompt()
    original_system = _extract_system_message(messages)

    if prompt_template:
        augmented_system = prompt_template.format(
            context=context_block,
            memory=memory_text or "",
            original_system_message=original_system,
        )
    else:
        # Fallback: inline context injection
        augmented_system = (
            f"---RETRIEVED CONTEXT---\n{context_block}\n---END CONTEXT---\n\n"
            f"{memory_text or ''}\n\n{original_system}"
        )

    # Replace or prepend system message
    messages = _set_system_message(messages, augmented_system)
    body["messages"] = messages

    # ── 7. Proxy to llama-server ─────────────────────────────────────────
    return await _proxy_response(proxy, body, stream)


# ─── Helper functions ────────────────────────────────────────────────────────


async def _proxy_response(
    proxy, body: dict, stream: bool
) -> StreamingResponse | JSONResponse:
    """Forward request to llama-server."""
    if stream:
        return StreamingResponse(
            proxy.proxy_stream("/v1/chat/completions", json_body=body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    response = await proxy.proxy_json("POST", "/v1/chat/completions", json_body=body)
    return JSONResponse(content=response.json(), status_code=response.status_code)


async def _handle_remember(request: Request, query: str, model: str) -> JSONResponse:
    """Extract and store a memory from the user's message."""
    memory_service = getattr(request.app.state, "memory", None)

    # Strip trigger keywords to get the actual content
    content = query
    for prefix in [
        "remember that ",
        "remember ",
        "note that ",
        "keep in mind that ",
        "keep in mind ",
        "from now on ",
        "from now on, ",
    ]:
        if content.lower().startswith(prefix):
            content = content[len(prefix) :]
            break

    if memory_service is not None:
        mem_id = memory_service.store(content=content, memory_type="preference")
        response_text = f'Got it — I\'ll remember that. (Stored as `{mem_id}`)\n\n**Memorized:** "{content}"'
    else:
        response_text = (
            "Memory system is not available. Could not store this preference."
        )

    import time, uuid

    return JSONResponse(
        content={
            "id": f"chatcmpl-gateway-mem-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


def _get_memory_text(
    request: Request, query: str, memory_types: list[str]
) -> str | None:
    """Retrieve relevant memories and format as text block."""
    memory_service = getattr(request.app.state, "memory", None)
    if memory_service is None or not memory_types:
        return None

    items = memory_service.retrieve_relevant(
        query=query,
        memory_types=memory_types,
        top_k=3,
    )
    if not items:
        return None

    lines = [f"- {item.content} [{item.id}]" for item in items]
    return "---USER PREFERENCES---\n" + "\n".join(lines) + "\n---END PREFERENCES---"


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as a context block for prompt injection."""
    parts: list[str] = []
    for chunk in chunks:
        citation = chunk.citation_link()
        # Truncate very long chunks for prompt space
        text = chunk.text[:2000] if len(chunk.text) > 2000 else chunk.text
        parts.append(f'{citation}\n"{text}"')
    return "\n\n".join(parts)


def _extract_system_message(messages: list[dict]) -> str:
    """Extract the system message content, or return empty string."""
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    return ""


def _set_system_message(messages: list[dict], system_content: str) -> list[dict]:
    """Replace or prepend the system message in the message list."""
    result = []
    found_system = False
    for msg in messages:
        if msg.get("role") == "system":
            result.append({"role": "system", "content": system_content})
            found_system = True
        else:
            result.append(msg)
    if not found_system:
        result.insert(0, {"role": "system", "content": system_content})
    return result


def _inject_memory_only(messages: list[dict], memory_text: str) -> list[dict]:
    """Inject memory text into the system message without full RAG template."""
    original_system = _extract_system_message(messages)
    new_system = (
        f"{memory_text}\n\n{original_system}" if original_system else memory_text
    )
    return _set_system_message(messages, new_system)


def _stream_single_response(response_dict: dict) -> StreamingResponse:
    """Wrap a non-streaming response as a single SSE event."""
    import json

    async def _generate():
        # Convert to a streaming chunk format
        chunk = {
            "id": response_dict["id"],
            "object": "chat.completion.chunk",
            "created": response_dict["created"],
            "model": response_dict["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": response_dict["choices"][0]["message"]["content"]
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
