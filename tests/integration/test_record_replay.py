"""Record then replay cycle tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import uvicorn

from tests.integration.helpers import chat_request, completion, run_proxy
from windtunnel_shear.core.hooks import HookRegistry
from windtunnel_shear.core.models import (
    Episode,
    InterceptedRequest,
    InterceptedResponse,
    Session,
    Turn,
    TurnType,
)
from windtunnel_shear.core.session import SessionRecorder
from windtunnel_shear.transport.replay import ReplayServer


def _session(*pairs: tuple[str, str]) -> Session:
    """Build a session with one turn per (question, answer) pair."""
    turns = [
        Turn(
            turn_type=TurnType.FINAL_RESPONSE,
            request=InterceptedRequest(raw=chat_request(question)),
            response=InterceptedResponse(raw=completion(answer)),
        )
        for question, answer in pairs
    ]
    return Session(model="gpt-4o", episodes=[Episode(turns=turns)])


def _replay_client(server: ReplayServer) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=server.create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://replay.test")


def _answer(resp: httpx.Response) -> str:
    content: str = resp.json()["choices"][0]["message"]["content"]
    return content


# ---------------------------------------------------------------------------
# Replay server
# ---------------------------------------------------------------------------


async def test_sequential_replay_returns_recorded_responses_in_order() -> None:
    server = ReplayServer(_session(("q1", "a1"), ("q2", "a2")))

    async with _replay_client(server) as client:
        # Sequential mode ignores the request content.
        first = await client.post("/v1/chat/completions", json=chat_request("anything"))
        second = await client.post("/v1/chat/completions", json=chat_request("else"))

    assert [_answer(first), _answer(second)] == ["a1", "a2"]


async def test_sequential_replay_miss_returns_502_with_request_hash() -> None:
    server = ReplayServer(_session(("q1", "a1")))

    async with _replay_client(server) as client:
        await client.post("/v1/chat/completions", json=chat_request("q1"))
        resp = await client.post("/v1/chat/completions", json=chat_request("q2"))

    assert resp.status_code == 502
    error = resp.json()["error"]
    assert error["type"] == "replay_miss"
    assert "exhausted" in error["message"]
    assert error["request_hash"]


async def test_exact_replay_matches_on_request_content() -> None:
    server = ReplayServer(_session(("q1", "a1"), ("q2", "a2")), match_mode="exact")

    async with _replay_client(server) as client:
        second = await client.post("/v1/chat/completions", json=chat_request("q2"))
        first = await client.post("/v1/chat/completions", json=chat_request("q1"))
        unknown = await client.post("/v1/chat/completions", json=chat_request("q3"))

    assert [_answer(second), _answer(first)] == ["a2", "a1"]
    assert unknown.status_code == 502
    assert unknown.json()["error"]["type"] == "replay_miss"


async def test_fallback_on_miss_returns_placeholder_completion() -> None:
    server = ReplayServer(_session(), on_miss="fallback")

    async with _replay_client(server) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request())

    assert resp.status_code == 200
    assert _answer(resp) == "[shear:replay:fallback]"


async def test_passthrough_on_miss_is_not_supported_yet() -> None:
    server = ReplayServer(_session(), on_miss="passthrough")

    async with _replay_client(server) as client:
        resp = await client.post("/v1/chat/completions", json=chat_request())

    assert resp.status_code == 501
    assert "Passthrough" in resp.json()["error"]["message"]


async def test_invalid_json_body_returns_400() -> None:
    server = ReplayServer(_session(("q1", "a1")))

    async with _replay_client(server) as client:
        resp = await client.post(
            "/v1/chat/completions",
            content=b"{broken",
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 400


def test_unknown_match_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown match mode"):
        ReplayServer(_session(), match_mode="fuzzy")


async def test_start_serves_the_replay_app_on_the_given_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[uvicorn.Config] = []

    async def fake_serve(self: uvicorn.Server, *args: Any, **kwargs: Any) -> None:
        served.append(self.config)

    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)

    await ReplayServer(_session(("q1", "a1"))).start(port=9911)

    [config] = served
    assert config.port == 9911
    assert config.host == "0.0.0.0"


# ---------------------------------------------------------------------------
# Record -> replay cycle
# ---------------------------------------------------------------------------


async def test_recorded_session_replays_without_upstream() -> None:
    answers = {"What is 2+2?": "4", "And 3+3?": "6"}
    upstream_calls: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        question = json.loads(request.content)["messages"][-1]["content"]
        upstream_calls.append(question)
        return httpx.Response(200, json=completion(answers[question]))

    recorder = SessionRecorder()

    def record(req: InterceptedRequest, resp: InterceptedResponse) -> InterceptedResponse:
        recorder.record_turn(req, resp)
        return resp

    registry = HookRegistry()
    registry.register(record, "after_response", stage="observe")

    async with run_proxy(upstream, hook_registry=registry) as client:
        live = [
            (await client.post("/v1/chat/completions", json=chat_request(q))).json()
            for q in answers
        ]

    # Save and reload through JSON, as `shear record` / `shear replay` do.
    saved = json.loads(json.dumps(recorder.finalize().to_json()))
    session = Session.from_json(saved)
    assert session.total_turns == 2

    for mode in ("sequential", "exact"):
        async with _replay_client(ReplayServer(session, match_mode=mode)) as client:
            replayed = [
                (await client.post("/v1/chat/completions", json=chat_request(q))).json()
                for q in answers
            ]
        assert replayed == live, mode

    assert upstream_calls == list(answers)
