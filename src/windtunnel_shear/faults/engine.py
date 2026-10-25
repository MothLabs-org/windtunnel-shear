"""Fault engine: parses --fault CLI flags and applies fault injection.

Infrastructure faults test the APPLICATION's resilience. They operate
on the response side, returning synthetic error responses instead of
forwarding to upstream.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from windtunnel_shear.core.models import InterceptedRequest, InterceptedResponse
from windtunnel_shear.faults.errors import make_error_response
from windtunnel_shear.faults.latency import inject_latency, parse_duration

_VALID_FAULT_TYPES = frozenset({"rate-limit", "latency", "error", "timeout"})


@dataclass
class FaultSpec:
    """Parsed fault specification from a CLI flag.

    Args:
        fault_type: One of "rate-limit", "latency", "error", "timeout".
        params: Parameter string (e.g. "0.3", "500ms", "503:0.1", "10s").
    """

    fault_type: str
    params: str


def parse_fault_flag(flag: str) -> FaultSpec:
    """Parse a --fault flag value into a FaultSpec.

    Args:
        flag: The flag value, e.g. "rate-limit:0.3" or "latency:500ms".

    Returns:
        A parsed FaultSpec.

    Raises:
        ValueError: If the flag format is invalid.
    """
    parts = flag.split(":", 1)
    if len(parts) < 2:
        msg = f"Invalid fault flag format: '{flag}'. Expected TYPE:PARAMS."
        raise ValueError(msg)
    fault_type = parts[0]
    if fault_type not in _VALID_FAULT_TYPES:
        msg = (
            f"Unknown fault type: '{fault_type}'. "
            f"Valid types: {', '.join(sorted(_VALID_FAULT_TYPES))}"
        )
        raise ValueError(msg)
    return FaultSpec(fault_type=fault_type, params=parts[1])


class FaultEngine:
    """Applies infrastructure faults to intercepted responses.

    Faults operate on the response side — they simulate infrastructure
    failures that test the application's resilience.
    """

    def __init__(self, specs: list[FaultSpec] | None = None) -> None:
        self._specs = specs or []

    async def maybe_inject(
        self,
        request: InterceptedRequest,
    ) -> InterceptedResponse | None:
        """Possibly inject a fault response instead of forwarding to upstream.

        Args:
            request: The incoming request.

        Returns:
            A synthetic error response if a fault fires, or None to proceed.
        """
        for spec in self._specs:
            result = await self._apply_spec(spec)
            if result is not None:
                return result
        return None

    async def _apply_spec(self, spec: FaultSpec) -> InterceptedResponse | None:
        """Apply a single fault spec."""
        if spec.fault_type == "rate-limit":
            return self._apply_rate_limit(spec)
        if spec.fault_type == "latency":
            await self._apply_latency(spec)
            return None
        if spec.fault_type == "error":
            return self._apply_error(spec)
        if spec.fault_type == "timeout":
            return await self._apply_timeout(spec)
        return None

    def _apply_rate_limit(self, spec: FaultSpec) -> InterceptedResponse | None:
        """Return 429 with the given probability."""
        probability = float(spec.params)
        if random.random() < probability:
            return make_error_response(
                429,
                f"shear:fault:rate_limit:{spec.params}",
            )
        return None

    async def _apply_latency(self, spec: FaultSpec) -> None:
        """Add latency to the response."""
        duration_ms = parse_duration(spec.params)
        await inject_latency(duration_ms)

    def _apply_error(self, spec: FaultSpec) -> InterceptedResponse | None:
        """Return an error with the given status code and probability."""
        parts = spec.params.split(":")
        status_code = int(parts[0])
        probability = float(parts[1]) if len(parts) > 1 else 1.0
        if random.random() < probability:
            return make_error_response(
                status_code,
                f"shear:fault:error:{spec.params}",
            )
        return None

    async def _apply_timeout(self, spec: FaultSpec) -> InterceptedResponse:
        """Hang for the specified duration then return 504."""
        duration_ms = parse_duration(spec.params)
        await inject_latency(duration_ms)
        return make_error_response(
            504,
            f"shear:fault:timeout:{spec.params}",
        )
