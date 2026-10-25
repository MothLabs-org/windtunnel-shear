"""Tests for jitter injection."""

from __future__ import annotations

import pytest

from windtunnel_shear.core.models import InterceptedRequest
from windtunnel_shear.jitters.contradict import apply_contradict
from windtunnel_shear.jitters.dilute import apply_dilute
from windtunnel_shear.jitters.engine import JitterEngine, JitterSpec, parse_jitter_flag
from windtunnel_shear.jitters.noise import apply_noise
from windtunnel_shear.jitters.rephrase import apply_rephrase


def _make_req(
    user_msg: str = "Hello world",
    system_msg: str | None = None,
) -> InterceptedRequest:
    messages: list[dict[str, str]] = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    return InterceptedRequest(
        raw={"model": "gpt-4o", "messages": messages},
    )


class TestParseJitterFlag:
    def test_noise_with_param(self) -> None:
        spec = parse_jitter_flag("noise:0.1")
        assert spec.jitter_type == "noise"
        assert spec.params == "0.1"

    def test_contradict_no_param(self) -> None:
        spec = parse_jitter_flag("contradict")
        assert spec.jitter_type == "contradict"
        assert spec.params == ""

    def test_dilute_with_count(self) -> None:
        spec = parse_jitter_flag("dilute:5")
        assert spec.jitter_type == "dilute"
        assert spec.params == "5"

    def test_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown jitter type"):
            parse_jitter_flag("bogus:0.5")

    def test_tool_jitter_reserved(self) -> None:
        # Should not raise — just warn
        spec = parse_jitter_flag("tool-corrupt:fn:field")
        assert spec.jitter_type == "tool-corrupt"


class TestNoise:
    def test_noise_modifies_content(self) -> None:
        req = _make_req("Hello world, this is a test message")
        result = apply_noise(req, 1.0, seed=42)
        assert result.messages[-1]["content"] != "Hello world, this is a test message"

    def test_noise_zero_ratio(self) -> None:
        req = _make_req("Hello world")
        result = apply_noise(req, 0.0)
        assert result.messages[-1]["content"] == "Hello world"

    def test_noise_only_user_messages(self) -> None:
        req = _make_req("Hello", system_msg="You are helpful")
        original_system = req.messages[0]["content"]
        apply_noise(req, 1.0, seed=42)
        assert req.messages[0]["content"] == original_system

    def test_noise_deterministic_with_seed(self) -> None:
        req1 = _make_req("Hello world test")
        req2 = _make_req("Hello world test")
        apply_noise(req1, 0.5, seed=123)
        apply_noise(req2, 0.5, seed=123)
        assert req1.messages[-1]["content"] == req2.messages[-1]["content"]


class TestContradict:
    def test_adds_contradiction(self) -> None:
        req = _make_req("What is 2+2?", system_msg="Be helpful")
        original_len = len(req.messages)
        apply_contradict(req)
        assert len(req.messages) == original_len + 1

    def test_contradiction_is_user_message(self) -> None:
        req = _make_req("What is 2+2?")
        apply_contradict(req)
        # The inserted message should be a user message
        contradiction = req.messages[0]
        assert contradiction["role"] == "user"
        assert "ignore" in contradiction["content"].lower()


class TestDilute:
    def test_adds_filler(self) -> None:
        req = _make_req("What is 2+2?")
        original_len = len(req.messages)
        apply_dilute(req, 3)
        # 3 pairs = 6 messages added
        assert len(req.messages) == original_len + 6

    def test_zero_count(self) -> None:
        req = _make_req("Hello")
        original_len = len(req.messages)
        apply_dilute(req, 0)
        assert len(req.messages) == original_len

    def test_filler_preserves_last_user(self) -> None:
        req = _make_req("Important question")
        apply_dilute(req, 2)
        assert req.messages[-1]["content"] == "Important question"


class TestRephrase:
    def test_substitutes_words(self) -> None:
        req = _make_req("Hello", system_msg="You must always be helpful")
        apply_rephrase(req)
        system = req.messages[0]["content"]
        assert "must" not in system
        assert "should" in system
        assert "typically" in system

    def test_no_system_msg(self) -> None:
        req = _make_req("Hello")
        apply_rephrase(req)  # Should not raise
        assert req.messages[-1]["content"] == "Hello"

    def test_case_preservation(self) -> None:
        req = _make_req("Hello", system_msg="Never do that")
        apply_rephrase(req)
        # "Never" (capitalized) should become "Rarely" (capitalized)
        assert "Rarely" in req.messages[0]["content"]


class TestNoisePerWord:
    """Verify noise operates per-word, not per-character."""

    def test_one_char_per_word(self) -> None:
        """At ratio=1.0, every word gets exactly one char corrupted."""
        req = _make_req("Hello world test")
        apply_noise(req, 1.0, seed=42)
        result = req.messages[-1]["content"]
        original_words = ["Hello", "world", "test"]
        result_words = result.split()
        assert len(result_words) == len(original_words)
        for orig, res in zip(original_words, result_words, strict=True):
            # Exactly one character should differ
            diffs = sum(1 for a, b in zip(orig, res, strict=True) if a != b)
            assert diffs == 1, f"{orig!r} -> {res!r}: expected 1 diff, got {diffs}"

    def test_moderate_ratio_readable(self) -> None:
        """At ratio=0.3, most words should remain unchanged."""
        req = _make_req("What is the capital of France")
        apply_noise(req, 0.3, seed=99)
        result = req.messages[-1]["content"]
        original_words = ["What", "is", "the", "capital", "of", "France"]
        result_words = result.split()
        unchanged = sum(1 for a, b in zip(original_words, result_words, strict=True) if a == b)
        # With 6 words at 0.3 ratio, expect ~4 unchanged (probabilistic but seeded)
        assert unchanged >= 2, f"Too many words corrupted: {result}"


class TestJitterEngine:
    def test_apply_noise(self) -> None:
        engine = JitterEngine([JitterSpec("noise", "1.0")])
        req = _make_req("Hello world test message here")
        result = engine.apply(req)
        assert result.messages[-1]["content"] != "Hello world test message here"

    def test_apply_contradict(self) -> None:
        engine = JitterEngine([JitterSpec("contradict")])
        req = _make_req("Hello")
        original_len = len(req.messages)
        engine.apply(req)
        assert len(req.messages) > original_len

    def test_apply_dilute(self) -> None:
        engine = JitterEngine([JitterSpec("dilute", "2")])
        req = _make_req("Hello")
        original_len = len(req.messages)
        engine.apply(req)
        assert len(req.messages) == original_len + 4

    def test_apply_rephrase(self) -> None:
        engine = JitterEngine([JitterSpec("rephrase")])
        req = _make_req("Hello", system_msg="You must be helpful")
        engine.apply(req)
        assert "should" in req.messages[0]["content"]

    def test_tool_jitter_skipped(self) -> None:
        engine = JitterEngine([JitterSpec("tool-corrupt", "fn:field")])
        req = _make_req("Hello")
        result = engine.apply(req)
        assert result.messages[-1]["content"] == "Hello"

    def test_jitter_log_populated(self) -> None:
        """Engine should populate jitter_log with before/after diffs."""
        engine = JitterEngine([JitterSpec("noise", "1.0")])
        req = _make_req("Hello world test message here")
        result = engine.apply(req)
        assert len(result.jitter_log) == 1
        assert "noise:1.0" in result.jitter_log[0]
        assert "->" in result.jitter_log[0]

    def test_jitter_log_empty_when_no_change(self) -> None:
        """No log entry if jitter didn't change anything."""
        engine = JitterEngine([JitterSpec("noise", "0.0")])
        req = _make_req("Hello")
        result = engine.apply(req)
        assert len(result.jitter_log) == 0

    def test_multiple_jitters_chain(self) -> None:
        engine = JitterEngine(
            [
                JitterSpec("rephrase"),
                JitterSpec("dilute", "1"),
            ]
        )
        req = _make_req("Hello", system_msg="You must be helpful")
        result = engine.apply(req)
        assert "should" in result.messages[0]["content"]
        assert len(result.messages) > 2
