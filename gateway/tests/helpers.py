"""Shared test helpers — importable by test modules."""

from __future__ import annotations

import json
import time
import uuid


def make_chat_response(content: str = "Test response", model: str = "sleeping-beauty") -> dict:
    """Build a minimal OpenAI-compatible chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def make_sse_stream(content: str = "Test response", model: str = "sleeping-beauty") -> bytes:
    """Build a minimal SSE stream as raw bytes."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    done_chunk = {
        "id": chunk["id"],
        "object": "chat.completion.chunk",
        "created": chunk["created"],
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }

    data = f"data: {json.dumps(chunk)}\n\n"
    data += f"data: {json.dumps(done_chunk)}\n\n"
    data += "data: [DONE]\n\n"
    return data.encode()


def make_models_response() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": "sleeping-beauty",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llama.cpp",
            }
        ],
    }
