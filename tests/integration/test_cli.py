"""CLI command tests.

``uvicorn.run`` is replaced with a fake server that records how it was
called and can send requests through the app, so no port is opened.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from typer.testing import CliRunner

from tests.integration.helpers import chat_request, completion, drive_app, use_fake_upstream
from windtunnel_shear.cli.main import app
from windtunnel_shear.core.models import (
    Episode,
    InterceptedRequest,
    InterceptedResponse,
    Session,
    Turn,
    TurnType,
)
from windtunnel_shear.faults.engine import FaultEngine
from windtunnel_shear.transport import proxy

runner = CliRunner()

AUTH = {"authorization": "Bearer sk-test-key"}


def _fake_server(
    monkeypatch: pytest.MonkeyPatch,
    on_start: Callable[[Starlette], None] | None = None,
) -> list[dict[str, Any]]:
    """Replace ``uvicorn.run`` and return the list of calls it receives."""
    calls: list[dict[str, Any]] = []

    def fake_run(asgi_app: Starlette, **kwargs: Any) -> None:
        calls.append({"app": asgi_app, **kwargs})
        if on_start is not None:
            on_start(asgi_app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return calls


def _write_session(path: Path, *pairs: tuple[str, str]) -> None:
    turns = [
        Turn(
            turn_type=TurnType.FINAL_RESPONSE,
            request=InterceptedRequest(raw=chat_request(question)),
            response=InterceptedResponse(raw=completion(answer)),
        )
        for question, answer in pairs
    ]
    session = Session(model="gpt-4o", episodes=[Episode(turns=turns)])
    path.write_text(json.dumps(session.to_json()))


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("proxy", "record", "replay", "simulate", "inspect"):
        assert command in result.output


# ---------------------------------------------------------------------------
# shear proxy
# ---------------------------------------------------------------------------


def test_proxy_defaults_to_port_9800_and_auto_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_server(monkeypatch)

    result = runner.invoke(app, ["proxy"])

    assert result.exit_code == 0, (result.output, result.exception)
    [call] = calls
    assert (call["host"], call["port"]) == ("0.0.0.0", 9800)
    assert "Shear proxy listening on http://0.0.0.0:9800" in result.output
    assert "Upstream: auto-detect from Authorization header" in result.output
    assert proxy._config.upstream == ""
    assert proxy._config.timeout_s == 120.0
    assert proxy._config.fault_engine is None
    assert proxy._config.hook_registry.hooks == []


def test_proxy_wires_flags_into_the_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _fake_server(monkeypatch)
    hooks_file = tmp_path / "my_hooks.py"
    hooks_file.write_text("LOADED = True\n")
    monkeypatch.setitem(sys.modules, "user_hooks", None)  # removed again after the test

    result = runner.invoke(
        app,
        [
            "proxy",
            "--port=9900",
            "--upstream=https://llm.example/v1",
            f"--hooks={hooks_file}",
            "--fault=error:503:0.5",
            "--jitter=contradict",
            "--timeout=30s",
            "--verbose",
            "--debug",
            "--json",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls[0]["port"] == 9900
    for line in (
        "Upstream: https://llm.example/v1",
        f"Hooks: {hooks_file}",
        "Fault: error:503:0.5",
        "Jitter: contradict",
        "Timeout: 30s",
    ):
        assert line in result.output

    config = proxy._config
    assert config.upstream == "https://llm.example/v1"
    assert config.timeout_s == 30.0
    assert (config.verbose, config.debug, config.json_output) == (True, True, True)
    assert config.quiet is False
    assert isinstance(config.fault_engine, FaultEngine)
    assert sys.modules["user_hooks"].LOADED is True

    [jitter] = config.hook_registry.hooks
    assert (jitter.hook_type, jitter.metadata.name, jitter.metadata.stage) == (
        "before_request",
        "jitter",
        "mutate",
    )
    jittered = jitter.fn(InterceptedRequest(raw=chat_request("Hi")))
    assert len(jittered.messages) == 2
    assert jittered.jitter_log


def test_proxy_quiet_prints_no_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_server(monkeypatch)

    result = runner.invoke(app, ["proxy", "--quiet"])

    assert result.exit_code == 0
    assert len(calls) == 1
    assert result.output == ""
    # Quiet also turns off per-request logging, not just the banner.
    assert proxy._config.quiet is True


def test_proxy_missing_hook_file_exits_with_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_server(monkeypatch)

    result = runner.invoke(app, ["proxy", f"--hooks={tmp_path / 'nope.py'}"])

    assert result.exit_code == 1
    assert "Hook file not found" in result.output
    assert calls == []


def test_proxy_non_python_hook_file_exits_with_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_server(monkeypatch)
    hooks_file = tmp_path / "hooks.txt"
    hooks_file.write_text("not python")

    result = runner.invoke(app, ["proxy", f"--hooks={hooks_file}"])

    assert result.exit_code == 1
    assert "Could not load hook file" in result.output
    assert calls == []


# ---------------------------------------------------------------------------
# shear record / shear replay
# ---------------------------------------------------------------------------


def test_record_saves_traffic_when_stopped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "session.json"
    seen_hosts: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        question = json.loads(request.content)["messages"][-1]["content"]
        return httpx.Response(200, json=completion(f"Answer to {question}"))

    async def send(client: httpx.AsyncClient) -> None:
        for question in ("one", "two"):
            resp = await client.post(
                "/v1/chat/completions", json=chat_request(question), headers=AUTH
            )
            assert resp.status_code == 200

    def send_traffic_then_ctrl_c(asgi_app: Starlette) -> None:
        use_fake_upstream(upstream)
        drive_app(asgi_app, send)
        raise KeyboardInterrupt

    calls = _fake_server(monkeypatch, send_traffic_then_ctrl_c)

    try:
        result = runner.invoke(app, ["record", "-o", str(output)])
    except KeyboardInterrupt:  # CliRunner does not catch it; keep pytest running
        pytest.fail("shear record did not handle Ctrl+C")

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls[0]["port"] == 9800
    assert proxy._config.quiet is False
    assert f"Shear recording to {output}" in result.output
    assert "Upstream: auto-detect from Authorization header" in result.output
    assert f"Saved 2 turns to {output}" in result.output
    assert seen_hosts == ["api.openai.com", "api.openai.com"]
    session = Session.from_json(json.loads(output.read_text()))
    turns = [turn for episode in session.episodes for turn in episode.turns]
    assert [t.request.messages[-1]["content"] for t in turns] == ["one", "two"]
    assert [t.response.content for t in turns] == ["Answer to one", "Answer to two"]


def test_record_passes_flags_to_the_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _fake_server(monkeypatch)
    output = tmp_path / "session.json"

    result = runner.invoke(
        app,
        [
            "record",
            "-o",
            str(output),
            "--port=9902",
            "--upstream=https://llm.example/v1",
            "--verbose",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls[0]["port"] == 9902
    assert "Upstream: https://llm.example/v1" in result.output
    assert "Saved 0 turns" in result.output
    assert proxy._config.upstream == "https://llm.example/v1"
    assert proxy._config.verbose is True


def test_record_quiet_still_saves_the_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_server(monkeypatch)
    output = tmp_path / "session.json"

    result = runner.invoke(app, ["record", "-o", str(output), "--quiet"])

    assert result.exit_code == 0
    assert result.output == ""
    assert proxy._config.quiet is True
    assert Session.from_json(json.loads(output.read_text())).total_turns == 0


def test_replay_serves_recorded_responses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session_file = tmp_path / "session.json"
    _write_session(session_file, ("What is 2+2?", "4"), ("And 3+3?", "6"))
    answers: list[tuple[int, str]] = []

    async def ask(client: httpx.AsyncClient) -> None:
        # Out of order (only exact mode answers "6"), then an unknown question
        # (only on_miss=fallback answers 200).
        for question in ("And 3+3?", "What is 2+2?", "Unknown?"):
            resp = await client.post("/v1/chat/completions", json=chat_request(question))
            answers.append((resp.status_code, resp.json()["choices"][0]["message"]["content"]))

    calls = _fake_server(monkeypatch, lambda asgi_app: drive_app(asgi_app, ask))

    result = runner.invoke(
        app,
        [
            "replay",
            "-i",
            str(session_file),
            "--port=9901",
            "--match-mode=exact",
            "--on-miss=fallback",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert "2 turns, mode=exact, on_miss=fallback" in result.output
    assert calls[0]["port"] == 9901
    assert answers == [(200, "6"), (200, "4"), (200, "[shear:replay:fallback]")]


def test_replay_missing_session_file_exits_with_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_server(monkeypatch)

    result = runner.invoke(app, ["replay", "-i", str(tmp_path / "missing.json")])

    assert result.exit_code == 1
    assert "Session file not found" in result.output
    assert calls == []


# ---------------------------------------------------------------------------
# Commands that are not implemented yet
# ---------------------------------------------------------------------------


def test_inspect_is_not_implemented_yet(tmp_path: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tmp_path / "session.json")])

    assert result.exit_code == 1
    assert isinstance(result.exception, NotImplementedError)
    assert "Inspecting" in result.output


def test_simulate_is_not_implemented_yet(tmp_path: Path) -> None:
    result = runner.invoke(app, ["simulate", "--responses", str(tmp_path / "r.json")])

    assert result.exit_code == 1
    assert isinstance(result.exception, NotImplementedError)
    assert "Simulating" in result.output
