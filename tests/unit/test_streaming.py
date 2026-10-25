"""Tests for SSE reassembly and re-streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from windtunnel_shear.core.models import InterceptedResponse
from windtunnel_shear.streaming.reassembly import reassemble_stream
from windtunnel_shear.streaming.restream import restream_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_line(data: dict | str) -> str:
    """Build a single SSE data line."""
    if isinstance(data, str):
        return f"data: {data}"
    return f"data: {json.dumps(data)}"


def _make_chunk(
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    tool_calls: list | None = None,
    usage: dict | None = None,
    chunk_id: str = "chatcmpl-test123",
    model: str = "gpt-4o",
) -> dict:
    """Build an OpenAI-format SSE chunk dict."""
    delta: dict = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls

    chunk: dict = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": 1700000000,
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


async def _chunks_from_lines(lines: list[str]) -> AsyncIterator[bytes]:
    """Create an async iterator of bytes from SSE lines.

    Each line is yielded as a separate byte chunk followed by double newline.
    """
    for line in lines:
        yield (line + "\n\n").encode("utf-8")


async def _chunks_as_single_blob(lines: list[str]) -> AsyncIterator[bytes]:
    """Yield all SSE lines as a single byte blob (tests buffer splitting)."""
    blob = "\n\n".join(lines) + "\n\n"
    yield blob.encode("utf-8")


# ---------------------------------------------------------------------------
# Reassembly tests
# ---------------------------------------------------------------------------


class TestReassembly:
    """Test SSE chunk reassembly into complete responses."""

    async def test_basic_content(self) -> None:
        """Reassemble a simple text response."""
        lines = [
            _sse_line(_make_chunk(role="assistant")),
            _sse_line(_make_chunk(content="Hello")),
            _sse_line(_make_chunk(content=" world")),
            _sse_line(_make_chunk(content="!")),
            _sse_line(_make_chunk(finish_reason="stop")),
            _sse_line("[DONE]"),
        ]
        resp = await reassemble_stream(_chunks_from_lines(lines))

        assert resp.content == "Hello world!"
        assert resp.finish_reason == "stop"
        assert resp.raw["model"] == "gpt-4o"
        assert resp.raw["id"] == "chatcmpl-test123"
        assert resp.status_code == 200

    async def test_content_as_single_blob(self) -> None:
        """Reassembly works when all chunks arrive in one byte blob."""
        lines = [
            _sse_line(_make_chunk(role="assistant")),
            _sse_line(_make_chunk(content="Hello world!")),
            _sse_line(_make_chunk(finish_reason="stop")),
            _sse_line("[DONE]"),
        ]
        resp = await reassemble_stream(_chunks_as_single_blob(lines))

        assert resp.content == "Hello world!"
        assert resp.finish_reason == "stop"

    async def test_tool_calls(self) -> None:
        """Reassemble a response with tool calls spanning multiple chunks."""
        lines = [
            _sse_line(_make_chunk(role="assistant")),
            _sse_line(
                _make_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_abc123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                )
            ),
            _sse_line(
                _make_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "function": {"arguments": '{"city":'},
                        }
                    ]
                )
            ),
            _sse_line(
                _make_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "function": {"arguments": ' "Paris"}'},
                        }
                    ]
                )
            ),
            _sse_line(_make_chunk(finish_reason="tool_calls")),
            _sse_line("[DONE]"),
        ]
        resp = await reassemble_stream(_chunks_from_lines(lines))

        assert resp.has_tool_calls
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "Paris"}'
        assert resp.finish_reason == "tool_calls"

    async def test_multiple_tool_calls(self) -> None:
        """Reassemble response with multiple tool calls by index."""
        lines = [
            _sse_line(_make_chunk(role="assistant")),
            _sse_line(
                _make_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "fn1", "arguments": ""},
                        },
                        {
                            "index": 1,
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "fn2", "arguments": ""},
                        },
                    ]
                )
            ),
            _sse_line(_make_chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"a":1}'}}])),
            _sse_line(_make_chunk(tool_calls=[{"index": 1, "function": {"arguments": '{"b":2}'}}])),
            _sse_line(_make_chunk(finish_reason="tool_calls")),
            _sse_line("[DONE]"),
        ]
        resp = await reassemble_stream(_chunks_from_lines(lines))

        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0]["function"]["name"] == "fn1"
        assert resp.tool_calls[0]["function"]["arguments"] == '{"a":1}'
        assert resp.tool_calls[1]["function"]["name"] == "fn2"
        assert resp.tool_calls[1]["function"]["arguments"] == '{"b":2}'

    async def test_usage_captured(self) -> None:
        """Usage dict is captured from the final chunk."""
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        lines = [
            _sse_line(_make_chunk(role="assistant")),
            _sse_line(_make_chunk(content="Hi")),
            _sse_line(_make_chunk(finish_reason="stop", usage=usage)),
            _sse_line("[DONE]"),
        ]
        resp = await reassemble_stream(_chunks_from_lines(lines))

        assert resp.usage == usage
        assert resp.usage["total_tokens"] == 15

    async def test_done_terminates(self) -> None:
        """[DONE] sentinel terminates reassembly."""
        lines = [
            _sse_line(_make_chunk(content="before")),
            _sse_line("[DONE]"),
            # This should not be processed
            _sse_line(_make_chunk(content=" after")),
        ]
        resp = await reassemble_stream(_chunks_from_lines(lines))
        assert resp.content == "before"

    async def test_empty_stream(self) -> None:
        """Empty stream produces empty response."""
        lines = [_sse_line("[DONE]")]
        resp = await reassemble_stream(_chunks_from_lines(lines))

        assert resp.content == ""
        assert resp.finish_reason == "stop"

    async def test_comment_lines_skipped(self) -> None:
        """SSE comment lines (starting with :) are ignored."""

        async def _chunks() -> AsyncIterator[bytes]:
            yield b": this is a comment\n\n"
            yield (_sse_line(_make_chunk(content="Hello")) + "\n\n").encode()
            yield b": another comment\n\n"
            yield (_sse_line(_make_chunk(finish_reason="stop")) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

        resp = await reassemble_stream(_chunks())
        assert resp.content == "Hello"

    async def test_custom_status_code(self) -> None:
        """Status code is passed through."""
        lines = [
            _sse_line(_make_chunk(content="ok")),
            _sse_line("[DONE]"),
        ]
        resp = await reassemble_stream(_chunks_from_lines(lines), status_code=201)
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Restream tests
# ---------------------------------------------------------------------------


class TestRestream:
    """Test re-serialization of complete responses to SSE format."""

    async def test_basic_restream(self) -> None:
        """Restream a simple text response to valid SSE."""
        resp = InterceptedResponse(
            raw={
                "id": "chatcmpl-test",
                "model": "gpt-4o",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello world"},
                        "finish_reason": "stop",
                    },
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

        frames: list[bytes] = []
        async for chunk in restream_response(resp):
            frames.append(chunk)

        # Should have: role chunk, content chunk(s), final chunk, [DONE]
        assert len(frames) >= 3

        # All frames should be valid SSE format
        for frame in frames:
            text = frame.decode("utf-8")
            assert text.startswith("data: ")
            assert text.endswith("\n\n")

        # Last frame should be [DONE]
        assert frames[-1] == b"data: [DONE]\n\n"

        # First frame should have role
        first_data = json.loads(frames[0].decode().removeprefix("data: ").strip())
        assert first_data["choices"][0]["delta"]["role"] == "assistant"

        # Second-to-last frame should have finish_reason
        final_data = json.loads(frames[-2].decode().removeprefix("data: ").strip())
        assert final_data["choices"][0]["finish_reason"] == "stop"

    async def test_restream_with_tool_calls(self) -> None:
        """Restream a tool-call response."""
        resp = InterceptedResponse(
            raw={
                "id": "chatcmpl-tc",
                "model": "gpt-4o",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": '{"q":"test"}'},
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                ],
            },
        )

        frames: list[bytes] = []
        async for chunk in restream_response(resp):
            frames.append(chunk)

        # Should have tool_calls in one of the frames
        all_text = b"".join(frames).decode()
        assert "tool_calls" in all_text
        assert "search" in all_text

    async def test_restream_ends_with_done(self) -> None:
        """Every restream ends with [DONE] sentinel."""
        resp = InterceptedResponse(
            raw={
                "id": "test",
                "model": "gpt-4o",
                "created": 0,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "x"},
                        "finish_reason": "stop",
                    },
                ],
            },
        )

        frames: list[bytes] = []
        async for chunk in restream_response(resp):
            frames.append(chunk)

        assert frames[-1] == b"data: [DONE]\n\n"

    async def test_restream_preserves_metadata(self) -> None:
        """Restream preserves id, model, created in chunks."""
        resp = InterceptedResponse(
            raw={
                "id": "chatcmpl-xyz",
                "model": "gpt-4o-mini",
                "created": 1234567890,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    },
                ],
            },
        )

        frames: list[bytes] = []
        async for chunk in restream_response(resp):
            frames.append(chunk)

        # Check a content frame (not [DONE])
        content_frame = json.loads(frames[1].decode().removeprefix("data: ").strip())
        assert content_frame["id"] == "chatcmpl-xyz"
        assert content_frame["model"] == "gpt-4o-mini"
        assert content_frame["created"] == 1234567890


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """Test that reassemble → restream → reassemble produces consistent results."""

    async def test_content_roundtrip(self) -> None:
        """Content survives a reassemble → restream → reassemble cycle."""
        # Start with SSE chunks
        original_lines = [
            _sse_line(_make_chunk(role="assistant")),
            _sse_line(_make_chunk(content="The capital ")),
            _sse_line(_make_chunk(content="of France ")),
            _sse_line(_make_chunk(content="is Paris.")),
            _sse_line(
                _make_chunk(
                    finish_reason="stop",
                    usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                )
            ),
            _sse_line("[DONE]"),
        ]

        # Reassemble
        resp1 = await reassemble_stream(_chunks_from_lines(original_lines))
        assert resp1.content == "The capital of France is Paris."

        # Restream
        restreamed_chunks: list[bytes] = []
        async for chunk in restream_response(resp1):
            restreamed_chunks.append(chunk)

        # Reassemble again from restreamed output
        async def _from_list() -> AsyncIterator[bytes]:
            for c in restreamed_chunks:
                yield c

        resp2 = await reassemble_stream(_from_list())

        # Content should match
        assert resp2.content == resp1.content
        assert resp2.finish_reason == resp1.finish_reason
        assert resp2.raw["model"] == resp1.raw["model"]

    async def test_tool_calls_roundtrip(self) -> None:
        """Tool calls survive a roundtrip."""
        original_lines = [
            _sse_line(_make_chunk(role="assistant")),
            _sse_line(
                _make_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_rt1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":"val"}'},
                        }
                    ]
                )
            ),
            _sse_line(_make_chunk(finish_reason="tool_calls")),
            _sse_line("[DONE]"),
        ]

        resp1 = await reassemble_stream(_chunks_from_lines(original_lines))
        assert resp1.has_tool_calls

        # Restream and reassemble
        restreamed: list[bytes] = []
        async for chunk in restream_response(resp1):
            restreamed.append(chunk)

        async def _from_list() -> AsyncIterator[bytes]:
            for c in restreamed:
                yield c

        resp2 = await reassemble_stream(_from_list())

        assert resp2.has_tool_calls
        assert resp2.tool_calls[0]["function"]["name"] == "lookup"
        assert resp2.tool_calls[0]["function"]["arguments"] == '{"key":"val"}'
