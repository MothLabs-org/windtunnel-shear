"""Collect SSE chunks into a complete response for hook processing.

Parses the OpenAI-compatible SSE stream format, accumulates deltas,
and builds a fully assembled InterceptedResponse that matches the
non-streaming response structure.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from windtunnel_shear.core.models import InterceptedResponse


def _parse_sse_frames(raw: str) -> tuple[list[str], bool]:
    """Parse raw SSE text into individual data payloads.

    Handles the SSE line protocol: skips comments (lines starting with ':'),
    extracts 'data:' fields, and splits on blank-line boundaries.

    Args:
        raw: Raw SSE text (may contain multiple frames).

    Returns:
        Tuple of (data payload strings, done flag). The done flag is True
        if a ``[DONE]`` sentinel was encountered.
    """
    payloads: list[str] = []
    done = False
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                done = True
                break
            if data:
                payloads.append(data)
    return payloads, done


def _merge_tool_call_delta(
    accumulated: list[dict[str, Any]],
    delta_tool_calls: list[dict[str, Any]],
) -> None:
    """Merge incremental tool call deltas into the accumulated list.

    Tool calls arrive across multiple chunks, identified by index.
    Each chunk may provide the function name, partial arguments, or both.

    Args:
        accumulated: The list being built up (mutated in place).
        delta_tool_calls: The tool_calls from the current chunk's delta.
    """
    for tc in delta_tool_calls:
        idx = tc.get("index", 0)
        # Extend list if needed
        while len(accumulated) <= idx:
            accumulated.append(
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            )

        entry = accumulated[idx]
        if tc.get("id"):
            entry["id"] = tc["id"]
        if tc.get("type"):
            entry["type"] = tc["type"]

        fn = tc.get("function", {})
        if fn.get("name"):
            entry["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            entry["function"]["arguments"] += fn["arguments"]


def _process_chunk_data(
    data: dict[str, Any],
    state: dict[str, Any],
    content_parts: list[str],
    tool_calls: list[dict[str, Any]],
) -> None:
    """Process a single parsed SSE chunk JSON object.

    Extracts and accumulates fields into the running state.

    Args:
        data: Parsed JSON from one SSE data line.
        state: Mutable dict tracking response_id, model, created, role,
            finish_reason, usage.
        content_parts: List of content strings being accumulated.
        tool_calls: List of tool call dicts being accumulated.
    """
    # Capture top-level fields from first chunk
    if not state["response_id"] and data.get("id"):
        state["response_id"] = data["id"]
    if not state["model"] and data.get("model"):
        state["model"] = data["model"]
    if not state["created"] and data.get("created"):
        state["created"] = data["created"]

    # Process choices
    for choice in data.get("choices", []):
        delta = choice.get("delta", {})

        if "role" in delta:
            state["role"] = delta["role"]

        if "content" in delta and delta["content"] is not None:
            content_parts.append(delta["content"])

        if "tool_calls" in delta:
            _merge_tool_call_delta(tool_calls, delta["tool_calls"])

        if choice.get("finish_reason"):
            state["finish_reason"] = choice["finish_reason"]

    # Capture usage (usually in final chunk)
    if data.get("usage"):
        state["usage"] = data["usage"]


async def reassemble_stream(
    chunks: AsyncIterator[bytes],
    status_code: int = 200,
) -> InterceptedResponse:
    """Collect SSE chunks and build a complete InterceptedResponse.

    Buffers incoming bytes, parses SSE frames, and accumulates deltas
    into a response structure identical to non-streaming responses.

    Args:
        chunks: Async iterator of raw SSE chunk bytes from upstream.
        status_code: HTTP status code from upstream.

    Returns:
        A fully assembled InterceptedResponse.
    """
    buffer = b""
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    done = False
    state: dict[str, Any] = {
        "response_id": "",
        "model": "",
        "created": 0,
        "role": "assistant",
        "finish_reason": "",
        "usage": {},
    }

    async for chunk in chunks:
        if done:
            break
        buffer += chunk
        # Process complete frames (separated by \n\n)
        while b"\n\n" in buffer:
            frame_bytes, buffer = buffer.split(b"\n\n", 1)
            frame_text = frame_bytes.decode("utf-8", errors="replace")

            payloads, frame_done = _parse_sse_frames(frame_text)
            if frame_done:
                done = True

            for data_str in payloads:
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                _process_chunk_data(data, state, content_parts, tool_calls)

            if done:
                break

    # Process any remaining buffer content
    if not done and buffer.strip():
        remaining = buffer.decode("utf-8", errors="replace")
        payloads, _ = _parse_sse_frames(remaining)
        for data_str in payloads:
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            _process_chunk_data(data, state, content_parts, tool_calls)

    # Build the complete response dict (matching non-streaming format)
    content = "".join(content_parts)
    message: dict[str, Any] = {"role": state["role"]}
    if content:
        message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls

    assembled: dict[str, Any] = {
        "id": state["response_id"],
        "object": "chat.completion",
        "created": state["created"],
        "model": state["model"],
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": state["finish_reason"] or "stop",
            },
        ],
    }
    if state["usage"]:
        assembled["usage"] = state["usage"]

    return InterceptedResponse(
        raw=assembled,
        status_code=status_code,
        timestamp=time.monotonic(),
    )
