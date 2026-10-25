"""Tests for fault injection."""

from __future__ import annotations

import pytest

from windtunnel_shear.core.models import InterceptedRequest
from windtunnel_shear.faults.engine import FaultEngine, FaultSpec, parse_fault_flag
from windtunnel_shear.faults.errors import make_error_response
from windtunnel_shear.faults.latency import parse_duration


@pytest.fixture()
def req() -> InterceptedRequest:
    return InterceptedRequest(
        raw={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )


class TestParseFaultFlag:
    def test_rate_limit(self) -> None:
        spec = parse_fault_flag("rate-limit:0.3")
        assert spec.fault_type == "rate-limit"
        assert spec.params == "0.3"

    def test_latency_ms(self) -> None:
        spec = parse_fault_flag("latency:500ms")
        assert spec.fault_type == "latency"
        assert spec.params == "500ms"

    def test_error_with_probability(self) -> None:
        spec = parse_fault_flag("error:503:0.1")
        assert spec.fault_type == "error"
        assert spec.params == "503:0.1"

    def test_timeout(self) -> None:
        spec = parse_fault_flag("timeout:10s")
        assert spec.fault_type == "timeout"
        assert spec.params == "10s"

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid fault flag"):
            parse_fault_flag("invalid")

    def test_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown fault type"):
            parse_fault_flag("bogus:0.5")


class TestParseDuration:
    def test_milliseconds(self) -> None:
        assert parse_duration("500ms") == 500.0

    def test_seconds(self) -> None:
        assert parse_duration("1.5s") == 1500.0

    def test_bare_number(self) -> None:
        assert parse_duration("200") == 200.0

    def test_whitespace(self) -> None:
        assert parse_duration("  300ms  ") == 300.0


class TestMakeErrorResponse:
    def test_429(self) -> None:
        resp = make_error_response(429, "shear:fault:rate_limit:0.3")
        assert resp.status_code == 429
        assert resp.reason == "shear:fault:rate_limit:0.3"
        assert resp.is_shear_injected
        assert "Rate limit" in resp.raw["error"]["message"]

    def test_503(self) -> None:
        resp = make_error_response(503, "shear:fault:error:503")
        assert resp.status_code == 503


class TestFaultEngine:
    async def test_rate_limit_always_fires(self, req: InterceptedRequest) -> None:
        engine = FaultEngine([FaultSpec("rate-limit", "1.0")])
        result = await engine.maybe_inject(req)
        assert result is not None
        assert result.status_code == 429

    async def test_rate_limit_never_fires(self, req: InterceptedRequest) -> None:
        engine = FaultEngine([FaultSpec("rate-limit", "0.0")])
        result = await engine.maybe_inject(req)
        assert result is None

    async def test_error_always_fires(self, req: InterceptedRequest) -> None:
        engine = FaultEngine([FaultSpec("error", "503")])
        result = await engine.maybe_inject(req)
        assert result is not None
        assert result.status_code == 503

    async def test_error_with_probability(self, req: InterceptedRequest) -> None:
        engine = FaultEngine([FaultSpec("error", "500:1.0")])
        result = await engine.maybe_inject(req)
        assert result is not None
        assert result.status_code == 500

    async def test_latency_returns_none(self, req: InterceptedRequest) -> None:
        """Latency adds delay but doesn't block the request."""
        engine = FaultEngine([FaultSpec("latency", "1ms")])
        result = await engine.maybe_inject(req)
        assert result is None

    async def test_timeout_returns_504(self, req: InterceptedRequest) -> None:
        engine = FaultEngine([FaultSpec("timeout", "1ms")])
        result = await engine.maybe_inject(req)
        assert result is not None
        assert result.status_code == 504

    async def test_empty_engine(self, req: InterceptedRequest) -> None:
        engine = FaultEngine([])
        result = await engine.maybe_inject(req)
        assert result is None

    async def test_first_fault_wins(self, req: InterceptedRequest) -> None:
        """First fault that fires short-circuits."""
        engine = FaultEngine(
            [
                FaultSpec("rate-limit", "1.0"),
                FaultSpec("error", "503"),
            ]
        )
        result = await engine.maybe_inject(req)
        assert result is not None
        assert result.status_code == 429  # rate-limit fires first
