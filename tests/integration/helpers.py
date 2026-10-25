"""Helpers for running Shear's servers in-process against a fake upstream.

The proxy app runs through ``httpx.ASGITransport`` and the upstream LLM API
is an ``httpx.MockTransport``, so there are no network calls and coverage
measures the server code.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from starlette.applications import Starlette

from windtunnel_shear.transport import proxy
from windtunnel_shear.transport.proxy import ProxyConfig, create_app

UPSTREAM = "https://upstream.test/v1"

UpstreamHandler = Callable[[httpx.Request], httpx.Response]


def chat_request(content: str = "Hello", **extra: Any) -> dict[str, Any]:
    """Build an OpenAI-style chat completion request body."""
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


def completion(content: str = "Hi there", total_tokens: int = 12) -> dict[str, Any]:
    """Build an OpenAI-style chat completion response body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": total_tokens},
    }


def sse_body(chunks: list[str]) -> bytes:
    """Build an upstream SSE stream whose deltas spell out ``chunks``."""
    frames = []
    for i, text in enumerate(chunks):
        delta: dict[str, str] = {"content": text}
        if i == 0:
            delta["role"] = "assistant"
        chunk = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        frames.append(f"data: {json.dumps(chunk)}\n\n")
    final = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    frames.append(f"data: {json.dumps(final)}\n\n")
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode()


def sse_content(body: str) -> str:
    """Join the content deltas of an SSE response body back into text."""
    parts = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for choice in json.loads(line[len("data: ") :]).get("choices", []):
            parts.append(choice.get("delta", {}).get("content") or "")
    return "".join(parts)


def drive_app(asgi_app: Starlette, send: Callable[[httpx.AsyncClient], Awaitable[None]]) -> None:
    """From sync code, run ``send`` against ``asgi_app`` inside its lifespan."""

    async def main() -> None:
        async with asgi_app.router.lifespan_context(asgi_app):
            transport = httpx.ASGITransport(app=asgi_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://shear.test") as c:
                await send(c)

    asyncio.run(main())


def use_fake_upstream(handler: UpstreamHandler) -> None:
    """Make the proxy send upstream requests to ``handler`` instead of the network."""
    proxy._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


@asynccontextmanager
async def run_proxy(handler: UpstreamHandler, **config: Any) -> AsyncIterator[httpx.AsyncClient]:
    """Run the proxy app with ``handler`` standing in for the upstream API.

    ``config`` overrides ``ProxyConfig`` fields. By default the upstream is
    ``UPSTREAM`` and console logging is off.

    Yields:
        A client whose requests go to the proxy app.
    """
    settings: dict[str, Any] = {"upstream": UPSTREAM, "quiet": True, **config}
    app = create_app(ProxyConfig(**settings))
    use_fake_upstream(handler)
    # The lifespan closes the fake upstream client on exit, as on server shutdown.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://shear.test") as client:
            yield client
