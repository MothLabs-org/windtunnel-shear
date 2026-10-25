"""Re-serialize a complete response into SSE chunks for the client.

Converts a fully assembled InterceptedResponse back into the OpenAI-compatible
SSE stream format so the client receives proper streaming output.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from windtunnel_shear.core.models import InterceptedResponse

# How many words per content chunk when re-streaming.
_WORDS_PER_CHUNK = 4


def _make_chunk_dict(
    response_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an SSE chunk dict matching the OpenAI streaming format.

    Args:
        response_id: The response ID.
        model: The model name.
        created: The creation timestamp.
        delta: The delta object for this chunk.
        finish_reason: The finish reason (None for intermediate chunks).
        usage: Usage dict (included in final chunk if present).

    Returns:
        A dict matching the chat.completion.chunk format.
    """
    chunk: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            },
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _encode_sse_frame(data: dict[str, Any]) -> bytes:
    """Encode a data dict as an SSE frame.

    Args:
        data: The JSON-serializable data.

    Returns:
        Encoded SSE frame bytes (``data: <json>\\n\\n``).
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def restream_response(response: InterceptedResponse) -> AsyncIterator[bytes]:
    """Convert a complete response back into SSE chunks.

    Splits content into word-group segments for realistic streaming
    behavior, then yields properly formatted SSE frames.

    Args:
        response: The complete response to re-stream.

    Yields:
        Raw SSE chunk bytes for the client.
    """
    response_id = response.raw.get("id", "")
    model = response.raw.get("model", "")
    created = response.raw.get("created", 0)
    usage = response.usage if response.usage else None

    content = response.content
    tool_calls = response.tool_calls
    finish_reason = response.finish_reason or "stop"

    # 1. Role chunk
    yield _encode_sse_frame(_make_chunk_dict(response_id, model, created, {"role": "assistant"}))

    # 2. Content chunks (split into word groups)
    if content:
        words = content.split(" ")
        for i in range(0, len(words), _WORDS_PER_CHUNK):
            segment = " ".join(words[i : i + _WORDS_PER_CHUNK])
            # Add leading space for chunks after the first
            if i > 0:
                segment = " " + segment
            yield _encode_sse_frame(
                _make_chunk_dict(response_id, model, created, {"content": segment})
            )

    # 3. Tool calls chunk (if any)
    if tool_calls:
        # Send all tool calls in a single chunk with index info
        delta_tool_calls = []
        for idx, tc in enumerate(tool_calls):
            delta_tool_calls.append(
                {
                    "index": idx,
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": tc.get("function", {}),
                }
            )
        yield _encode_sse_frame(
            _make_chunk_dict(
                response_id,
                model,
                created,
                {"tool_calls": delta_tool_calls},
            )
        )

    # 4. Final chunk with finish_reason and usage
    yield _encode_sse_frame(
        _make_chunk_dict(
            response_id,
            model,
            created,
            {},
            finish_reason=finish_reason,
            usage=usage,
        )
    )

    # 5. Done sentinel
    yield b"data: [DONE]\n\n"
