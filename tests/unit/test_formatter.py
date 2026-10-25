"""Tests for the human-readable console output formatter."""

from __future__ import annotations

import io
import json
import sys
from typing import Any

import pytest

from windtunnel_shear.cli import formatter
from windtunnel_shear.cli.formatter import (
    format_jitter_diffs,
    format_request_json,
    format_request_line,
    format_request_verbose,
    format_response_json,
    format_response_line,
    format_response_verbose,
    log_exchange,
)
from windtunnel_shear.core.models import InterceptedRequest, InterceptedResponse


class _FakeStderr(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _request(**raw: Any) -> InterceptedRequest:
    body: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "one two three four five six seven eight"}],
    }
    body.update(raw)
    return InterceptedRequest(raw=body)


def _response(content: str = "Paris", **kwargs: Any) -> InterceptedResponse:
    raw: dict[str, Any] = {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"total_tokens": 33},
    }
    return InterceptedResponse(raw=raw, **kwargs)


def _tool_call_response() -> InterceptedResponse:
    tool_calls = [
        {"function": {"name": "get_weather"}},
        {"function": {"name": "get_time"}},
        {},
    ]
    raw = {"choices": [{"message": {"content": None, "tool_calls": tool_calls}}]}
    return InterceptedResponse(raw=raw)


# ---------------------------------------------------------------------------
# Color support
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("tty", "expected"), [(True, True), (False, False)])
def test_color_follows_whether_stderr_is_a_terminal(
    monkeypatch: pytest.MonkeyPatch, tty: bool, expected: bool
) -> None:
    monkeypatch.setattr(sys, "stderr", _FakeStderr(tty))

    assert formatter._supports_color() is expected


def test_no_color_when_stderr_has_no_isatty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stderr", object())

    assert formatter._supports_color() is False


@pytest.mark.parametrize(
    ("status", "color"),
    [
        (200, formatter._GREEN),
        (299, formatter._GREEN),
        (400, formatter._YELLOW),
        (499, formatter._YELLOW),
        (500, formatter._RED),
        (304, formatter._RESET),
    ],
)
def test_status_color(status: int, color: str) -> None:
    assert formatter._status_color(status) == color


# ---------------------------------------------------------------------------
# Request lines
# ---------------------------------------------------------------------------


def test_request_line_summarizes_model_messages_and_tokens() -> None:
    # 8 words x 1.3 + 4 framing tokens per message = ~14
    line = format_request_line(_request(), color=False)

    assert line == "→ gpt-4o | 1 messages | ~14 tokens"


def test_request_line_counts_tools_and_handles_missing_model() -> None:
    req = InterceptedRequest(
        raw={"messages": [{"role": "assistant", "content": ""}], "tools": [{}, {}]}
    )

    line = format_request_line(req, color=False)

    assert line == "→ unknown | 1 messages | ~5 tokens | 2 tools"


def test_request_line_with_color_uses_ansi_arrow() -> None:
    line = format_request_line(_request(), color=True)

    assert line.startswith(f"{formatter._CYAN}{formatter._BOLD}→{formatter._RESET} ")
    assert "gpt-4o" in line


# ---------------------------------------------------------------------------
# Response lines
# ---------------------------------------------------------------------------


def test_response_line_shows_status_tokens_latency_and_preview() -> None:
    line = format_response_line(_response(latency_ms=1234.0), color=False)

    assert line == '← 200 | 33 tokens | 1.2s | "Paris"'


def test_response_line_truncates_long_content() -> None:
    line = format_response_line(_response("x" * 100), color=False)

    assert f'"{"x" * 57}..."' in line


def test_response_line_keeps_content_of_exactly_60_chars() -> None:
    line = format_response_line(_response("x" * 60), color=False)

    assert f'"{"x" * 60}"' in line


def test_response_line_lists_tool_call_names() -> None:
    line = format_response_line(_tool_call_response(), color=False)

    assert line == "← 200 | tool_calls: get_weather, get_time, ?"


def test_response_line_marks_shear_injected_responses() -> None:
    resp = InterceptedResponse(raw={}, status_code=429, reason="shear:fault:rate_limit:0.3")

    line = format_response_line(resp, color=False)

    assert line == "← 429 | [shear: shear:fault:rate_limit:0.3]"


def test_response_line_with_color_uses_status_color() -> None:
    line = format_response_line(InterceptedResponse(raw={}, status_code=503), color=True)

    assert line.startswith(f"{formatter._RED}{formatter._BOLD}←{formatter._RESET} ")


# ---------------------------------------------------------------------------
# Verbose and JSON formats
# ---------------------------------------------------------------------------


def test_verbose_formats_are_full_pretty_json() -> None:
    req, resp = _request(), _response()

    assert json.loads(format_request_verbose(req)) == req.raw
    assert json.loads(format_response_verbose(resp)) == resp.raw
    assert "\n  " in format_request_verbose(req)


def test_request_json_summary() -> None:
    req = _request(stream=True, tools=[{}])

    assert json.loads(format_request_json(req)) == {
        "direction": "request",
        "model": "gpt-4o",
        "messages": 1,
        "stream": True,
        "tools": 1,
    }


def test_response_json_summary_includes_shear_reason_only_when_injected() -> None:
    real = json.loads(format_response_json(_response(latency_ms=50.0)))
    injected = json.loads(
        format_response_json(InterceptedResponse(raw={}, status_code=504, reason="shear:timeout"))
    )

    assert real == {
        "direction": "response",
        "status_code": 200,
        "latency_ms": 50.0,
        "usage": {"total_tokens": 33},
        "finish_reason": "stop",
    }
    assert injected["shear_reason"] == "shear:timeout"


def test_jitter_diffs() -> None:
    req = _request()
    assert format_jitter_diffs(req) == ""

    req.jitter_log.extend(["noise | a -> b", "dilute | c -> d"])

    assert (
        format_jitter_diffs(req, color=False)
        == "  jitter: noise | a -> b\n  jitter: dilute | c -> d"
    )
    assert format_jitter_diffs(req, color=True).startswith(
        f"  {formatter._YELLOW}jitter:{formatter._RESET} noise"
    )


# ---------------------------------------------------------------------------
# log_exchange
# ---------------------------------------------------------------------------


def _jittered_request() -> InterceptedRequest:
    req = _request()
    req.jitter_log.append("noise | before -> after")
    return req


def test_log_exchange_default_writes_two_summary_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_exchange(_jittered_request(), _response())

    lines = capsys.readouterr().err.splitlines()
    assert lines == ["→ gpt-4o | 1 messages | ~14 tokens", '← 200 | 33 tokens | "Paris"']


def test_log_exchange_verbose_adds_jitter_diffs(capsys: pytest.CaptureFixture[str]) -> None:
    log_exchange(_jittered_request(), _response(), verbose=True)

    lines = capsys.readouterr().err.splitlines()
    assert lines[1] == "  jitter: noise | before -> after"
    assert len(lines) == 3


def test_log_exchange_verbose_without_jitter(capsys: pytest.CaptureFixture[str]) -> None:
    log_exchange(_request(), _response(), verbose=True)

    assert len(capsys.readouterr().err.splitlines()) == 2


def test_log_exchange_debug_adds_full_payloads(capsys: pytest.CaptureFixture[str]) -> None:
    log_exchange(_jittered_request(), _response(), debug=True)

    err = capsys.readouterr().err
    assert "  jitter: noise | before -> after" in err
    assert '"model": "gpt-4o"' in err
    assert '"total_tokens": 33' in err


def test_log_exchange_debug_without_jitter(capsys: pytest.CaptureFixture[str]) -> None:
    log_exchange(_request(), _response(), debug=True)

    assert "jitter:" not in capsys.readouterr().err


def test_log_exchange_json_mode_writes_json_lines_without_color(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(formatter, "_supports_color", lambda: True)

    log_exchange(_request(), _response(), json_mode=True)

    err = capsys.readouterr().err
    request_line, response_line = (json.loads(line) for line in err.splitlines())
    assert request_line["direction"] == "request"
    assert response_line["direction"] == "response"
    assert "\033[" not in err


def test_log_exchange_uses_color_on_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(formatter, "_supports_color", lambda: True)

    log_exchange(_request(), _response())

    assert formatter._GREEN in capsys.readouterr().err


def test_log_exchange_quiet_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    log_exchange(_request(), _response(), quiet=True)

    assert capsys.readouterr().err == ""
