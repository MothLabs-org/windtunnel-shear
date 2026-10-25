"""Tests for token counting."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from windtunnel_shear.tokens.counter import TokenCounter


class _FakeEncoder:
    def encode(self, text: str) -> list[str]:
        return text.split()


def _install_fake_tiktoken(
    monkeypatch: pytest.MonkeyPatch, known_models: tuple[str, ...] = ("gpt-4o",)
) -> list[str]:
    """Install a fake ``tiktoken`` module and return the models it was asked for."""
    requested: list[str] = []

    def encoding_for_model(model: str) -> Any:
        requested.append(model)
        if model not in known_models:
            raise KeyError(model)
        return _FakeEncoder()

    fake = types.ModuleType("tiktoken")
    fake.encoding_for_model = encoding_for_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    return requested


def test_count_returns_zero_without_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # makes `import tiktoken` fail

    assert TokenCounter("gpt-4o").count("three little words") == 0


def test_count_uses_tiktoken_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = _install_fake_tiktoken(monkeypatch, known_models=("gpt-4o", "gpt-4o-mini"))

    assert TokenCounter("gpt-4o-mini").count("three little words") == 3
    assert requested == ["gpt-4o-mini"]


def test_count_defaults_to_gpt_4o_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = _install_fake_tiktoken(monkeypatch)

    assert TokenCounter().count("a b") == 2
    assert requested == ["gpt-4o"]


def test_count_returns_zero_for_model_unknown_to_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tiktoken(monkeypatch)

    assert TokenCounter("my-local-llama").count("a b c") == 0


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}, 12),
        ({"prompt_tokens": 5}, 0),
        ({}, 0),
    ],
)
def test_from_usage_reads_total_tokens(usage: dict[str, int], expected: int) -> None:
    assert TokenCounter.from_usage(usage) == expected
