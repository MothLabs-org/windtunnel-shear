"""End-to-end proxy round-trip tests.

Each test runs the proxy app in-process with a fake upstream API (see
``helpers.run_proxy``), so no network calls are made.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any

import httpx
import pytest

from tests.integration.helpers import (
    UPSTREAM,
    chat_request,
    completion,
    run_proxy,
    sse_body,
    sse_content,
)
from windtunnel_shear.core.hooks import HookRegistry
from windtunnel_shear.core.models import InterceptedRequest, InterceptedResponse
from windtunnel_shear.faults.engine import FaultEngine, parse_fault_flag
from windtunnel_shear.transport import proxy
from windtunnel_shear.transport.proxy import ProxyConfig, create_app

AUTH = {"authorization": "Bearer sk-test-key"}


def _ok(content: str = "Hi there") -> httpx.Response:
    return httpx.Response(200, json=completion(content))


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


def test_create_app_uses_no_removed_or_deprecated_starlette_api() -> None:
    """Regression: Starlette 1.0 removed ``on_shutdown``, which crashed ``shear proxy``.

    Starlette 0.x still accepts it but warns, so failing on the warning makes
    this test catch the problem on every supported Starlette version.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        app = create_app(ProxyConfig(quiet=True))

    assert len(app.routes) == 1


async def test_shutdown_closes_shared_http_client() -> None:
    app = create_app(ProxyConfig(quiet=True))

    async with app.router.lifespan_context(app):
        client = proxy._get_client()
        assert not client.is_closed

    assert client.is_closed
    assert proxy._http_client is None


async def test_shutdown_without_client_is_a_no_op() -> None:
    proxy._http_client = None
    app = create_app(ProxyConfig(quiet=True))

    async with app.router.lifespan_context(app):
        pass

    assert proxy._http_client is None


async def test_shared_client_uses_configured_timeout_and_is_reused() -> None:
    proxy._http_client = None
    create_app(ProxyConfig(timeout_s=5.0, quiet=True))

    client = proxy._get_client()
    try:
        assert client.timeout.read == 5.0
        assert proxy._get_client() is client
    finally:
        await client.aclose()


def test_create_app_without_config_keeps_current_config() -> None:
    config = ProxyConfig(upstream=UPSTREAM, quiet=True)
    create_app(config)

    create_app()

    assert proxy._config is config


# ---------------------------------------------------------------------------
# Forwarding
# ---------------------------------------------------------------------------


async def test_forwards_request_and_returns_upstream_response() -> None:
    seen: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok("Paris")

    async with run_proxy(upstream) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=chat_request("Capital of France?"),
            headers={**AUTH, "x-not-forwarded": "1"},
        )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Paris"
    [sent] = seen
    # The /v1 prefix shared with the upstream URL is not doubled.
    assert str(sent.url) == f"{UPSTREAM}/chat/completions"
    assert sent.headers["authorization"] == "Bearer sk-test-key"
    assert "x-not-forwarded" not in sent.headers
    assert json.loads(sent.content) == chat_request("Capital of France?")


async def test_request_to_root_goes_to_upstream_base_url() -> None:
    seen: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return _ok()

    async with run_proxy(upstream) as client:
        await client.post("/", json=chat_request())

    assert seen == [UPSTREAM]


async def test_auto_detects_upstream_from_api_key() -> None:
    seen: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return _ok()

    async with run_proxy(upstream, upstream="") as client:
        resp = await client.post("/v1/chat/completions", json=chat_request(), headers=AUTH)

    assert resp.status_code == 200
    assert seen == ["https://api.openai.com/v1/chat/completions"]


async def test_unrecognized_key_without_upstream_returns_502() -> None:
    async with run_proxy(lambda _: _ok(), upstream="") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=chat_request(),
            headers={"authorization": "Bearer not-a-known-key"},
        )

    assert resp.status_code == 502
    assert "Could not auto-detect upstream" in resp.json()["error"]["message"]


async def test_invalid_json_body_returns_400() -> None:
    async with run_proxy(lambda _: _ok()) as client:
        resp = await client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 400
    assert resp.json()["error"]["source"] == "shear"


# ---------------------------------------------------------------------------
# Hooks and faults
# ---------------------------------------------------------------------------


async def test_before_request_hook_changes_what_upstream_receives() -> None:
    seen: list[dict[str, Any]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _ok()

    def swap_model(req: InterceptedRequest) -> InterceptedRequest:
        req.raw["model"] = "gpt-4o-mini"
        return req

    registry = HookRegistry()
    registry.register(swap_model, "before_request")

    async with run_proxy(upstream, hook_registry=registry) as client:
        await client.post("/v1/chat/completions", json=chat_request())

    assert seen[0]["model"] == "gpt-4o-mini"


async def test_failing_before_request_hook_returns_500_without_calling_upstream() -> None:
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _ok()

    def broken(req: InterceptedRequest) -> InterceptedRequest:
        raise RuntimeError("boom")

    registry = HookRegistry()
    registry.register(broken, "before_request")

    async with run_proxy(upstream, hook_registry=registry) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request())

    assert resp.status_code == 500
    assert "Hook error" in resp.json()["error"]["message"]
    assert calls == []


async def test_after_response_hook_changes_what_client_receives() -> None:
    def redact(req: InterceptedRequest, resp: InterceptedResponse) -> InterceptedResponse:
        resp.content = "[redacted]"
        return resp

    registry = HookRegistry()
    registry.register(redact, "after_response")

    async with run_proxy(lambda _: _ok("secret"), hook_registry=registry) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request())

    assert resp.json()["choices"][0]["message"]["content"] == "[redacted]"


@pytest.mark.parametrize("stream", [False, True])
async def test_failing_after_response_hook_is_logged_and_response_kept(
    stream: bool, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(req: InterceptedRequest, resp: InterceptedResponse) -> InterceptedResponse:
        raise RuntimeError("after boom")

    def upstream(request: httpx.Request) -> httpx.Response:
        if stream:
            return httpx.Response(200, content=sse_body(["Hi ", "there"]))
        return _ok("Hi there")

    registry = HookRegistry()
    registry.register(broken, "after_response")

    with caplog.at_level(logging.ERROR, logger="windtunnel_shear.transport.proxy"):
        async with run_proxy(upstream, hook_registry=registry) as client:
            resp = await client.post("/v1/chat/completions", json=chat_request(stream=stream))

    assert resp.status_code == 200
    content = sse_content(resp.text) if stream else resp.json()["choices"][0]["message"]["content"]
    assert content == "Hi there"
    assert "after_response hook error" in caplog.text


async def test_injected_fault_is_logged(capsys: pytest.CaptureFixture[str]) -> None:
    engine = FaultEngine([parse_fault_flag("error:503")])

    async with run_proxy(lambda _: _ok(), fault_engine=engine, quiet=False) as client:
        await client.post("/v1/chat/completions", json=chat_request())

    assert "← 503 | [shear: shear:fault:error:503]" in capsys.readouterr().err


async def test_fault_injection_short_circuits_upstream() -> None:
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _ok()

    engine = FaultEngine([parse_fault_flag("error:503")])

    async with run_proxy(upstream, fault_engine=engine) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request())

    assert resp.status_code == 503
    assert calls == []


async def test_fault_engine_that_does_not_fire_lets_request_through() -> None:
    engine = FaultEngine([parse_fault_flag("rate-limit:0")])

    async with run_proxy(lambda _: _ok(), fault_engine=engine) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request())

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Upstream failures (streaming and non-streaming)
# ---------------------------------------------------------------------------


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("too slow", request=request)


def _unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


@pytest.mark.parametrize("stream", [False, True])
async def test_upstream_timeout_returns_504(stream: bool) -> None:
    async with run_proxy(_timeout) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request(stream=stream))

    assert resp.status_code == 504
    assert resp.json()["error"]["message"] == "Upstream request timed out"


@pytest.mark.parametrize("stream", [False, True])
async def test_upstream_connection_error_returns_502(stream: bool) -> None:
    async with run_proxy(_unreachable) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request(stream=stream))

    assert resp.status_code == 502
    assert "connection refused" in resp.json()["error"]["message"]


@pytest.mark.parametrize("stream", [False, True])
async def test_on_error_hook_supplies_fallback_response(stream: bool) -> None:
    def fallback(req: InterceptedRequest, exc: Exception) -> InterceptedResponse:
        return InterceptedResponse(raw=completion("fallback"), status_code=200, reason="fallback")

    registry = HookRegistry()
    registry.register(fallback, "on_error")

    async with run_proxy(_unreachable, hook_registry=registry) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request(stream=stream))

    assert resp.status_code == 200
    content = sse_content(resp.text) if stream else resp.json()["choices"][0]["message"]["content"]
    assert content == "fallback"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_streaming_response_runs_hooks_then_restreams() -> None:
    def shout(req: InterceptedRequest, resp: InterceptedResponse) -> InterceptedResponse:
        resp.content = resp.content.upper()
        return resp

    registry = HookRegistry()
    registry.register(shout, "after_response")

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body(["The capital ", "is Paris."]),
            headers={"content-type": "text/event-stream"},
        )

    async with run_proxy(upstream, hook_registry=registry) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request(stream=True))

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert sse_content(resp.text) == "THE CAPITAL IS PARIS."
    assert resp.text.rstrip().endswith("data: [DONE]")


async def test_streaming_upstream_json_error_is_returned_as_json() -> None:
    error = {"error": {"message": "Rate limit reached", "type": "rate_limit"}}

    async with run_proxy(lambda _: httpx.Response(429, json=error)) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request(stream=True))

    assert resp.status_code == 429
    assert resp.json() == error


async def test_streaming_upstream_non_json_error_is_wrapped() -> None:
    async with run_proxy(lambda _: httpx.Response(500, text="Internal oops")) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request(stream=True))

    assert resp.status_code == 500
    assert resp.json()["error"] == {"message": "Internal oops", "source": "upstream"}


# ---------------------------------------------------------------------------
# Console logging
# ---------------------------------------------------------------------------


async def test_exchange_is_logged_to_stderr_unless_quiet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async with run_proxy(lambda _: _ok("Paris"), quiet=False) as client:
        await client.post("/v1/chat/completions", json=chat_request())

    err = capsys.readouterr().err
    assert "→ gpt-4o | 1 messages" in err
    assert "← 200 | 12 tokens | " in err
    assert '"Paris"' in err


async def test_quiet_mode_logs_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    async with run_proxy(lambda _: _ok()) as client:
        await client.post("/v1/chat/completions", json=chat_request())

    assert capsys.readouterr().err == ""
