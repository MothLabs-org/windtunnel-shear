"""Library mode wrapper tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from windtunnel_shear.core.hooks import HookRegistry
from windtunnel_shear.core.models import InterceptedRequest, InterceptedResponse
from windtunnel_shear.transport.library import wrap

# ---------------------------------------------------------------------------
# Fake OpenAI-like client for testing (no dependency on openai package)
# ---------------------------------------------------------------------------


@dataclass
class _FakeMessage:
    role: str = "assistant"
    content: str = "Hello from fake"
    tool_calls: list[Any] | None = None


@dataclass
class _FakeChoice:
    index: int = 0
    message: _FakeMessage = field(default_factory=_FakeMessage)
    finish_reason: str = "stop"


@dataclass
class _FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15


@dataclass
class _FakeResponse:
    id: str = "chatcmpl-fake"
    choices: list[_FakeChoice] = field(default_factory=lambda: [_FakeChoice()])
    usage: _FakeUsage = field(default_factory=_FakeUsage)
    model: str = "gpt-4o"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "choices": [
                {
                    "index": c.index,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                    },
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "model": self.model,
        }


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}
        self.response: _FakeResponse = _FakeResponse()
        self.call_count: int = 0

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.last_kwargs = kwargs
        self.call_count += 1
        return self.response


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()
        self.api_key = "sk-fake"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWrap:
    def test_passthrough_no_hooks(self) -> None:
        """Without hooks, wrap() should pass calls through transparently."""
        client = _FakeClient()
        wrapped = wrap(client, quiet=True)

        result = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result.choices[0].message.content == "Hello from fake"
        assert client.chat.completions.call_count == 1

    def test_kwargs_forwarded(self) -> None:
        """Request kwargs should be forwarded to the real client."""
        client = _FakeClient()
        wrapped = wrap(client, quiet=True)

        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.5,
        )

        assert client.chat.completions.last_kwargs["model"] == "gpt-4o"
        assert client.chat.completions.last_kwargs["temperature"] == 0.5

    def test_before_request_hook_modifies_kwargs(self) -> None:
        """A before_request hook can modify the request params."""
        client = _FakeClient()
        registry = HookRegistry()

        def set_temperature(req: InterceptedRequest) -> InterceptedRequest:
            req.raw["temperature"] = 0.0
            return req

        registry.register(set_temperature, "before_request", name="set_temp")
        wrapped = wrap(client, hooks=registry, quiet=True)

        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            temperature=1.0,
        )

        # Hook should have changed temperature to 0.0
        assert client.chat.completions.last_kwargs["temperature"] == 0.0

    def test_after_response_hook_modifies_content(self) -> None:
        """An after_response hook can modify the response content."""
        client = _FakeClient()
        registry = HookRegistry()

        def censor(_req: InterceptedRequest, resp: InterceptedResponse) -> InterceptedResponse:
            resp.content = "[CENSORED]"
            return resp

        registry.register(censor, "after_response", name="censor")
        wrapped = wrap(client, hooks=registry, quiet=True)

        result = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        # Hook should have modified the content on the original response object
        assert result.choices[0].message.content == "[CENSORED]"

    def test_on_error_hook_handles_exception(self) -> None:
        """An on_error hook can provide a fallback response."""
        client = _FakeClient()
        # Make the real create() raise
        client.chat.completions.response = None  # type: ignore[assignment]

        def fail_create(**kwargs: Any) -> None:
            msg = "API error"
            raise RuntimeError(msg)

        client.chat.completions.create = fail_create  # type: ignore[assignment]

        registry = HookRegistry()

        def handle_error(_req: InterceptedRequest, _err: Exception) -> InterceptedResponse:
            return InterceptedResponse(
                raw={"choices": [{"message": {"content": "fallback"}, "finish_reason": "stop"}]},
                status_code=200,
                reason="error_handler",
            )

        registry.register(handle_error, "on_error", name="handler")
        wrapped = wrap(client, hooks=registry, quiet=True)

        result = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        # Should get the fallback dict from the error handler
        assert result["choices"][0]["message"]["content"] == "fallback"

    def test_on_error_no_handler_reraises(self) -> None:
        """Without an error handler, exceptions propagate."""
        client = _FakeClient()

        def fail_create(**kwargs: Any) -> None:
            msg = "API error"
            raise RuntimeError(msg)

        client.chat.completions.create = fail_create  # type: ignore[assignment]
        wrapped = wrap(client, quiet=True)

        with pytest.raises(RuntimeError, match="API error"):
            wrapped.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_proxy_passes_through_other_attrs(self) -> None:
        """Non-intercepted attributes should pass through to the real client."""
        client = _FakeClient()
        wrapped = wrap(client, quiet=True)
        assert wrapped.api_key == "sk-fake"

    def test_repr(self) -> None:
        """Wrapped client should have a recognizable repr."""
        client = _FakeClient()
        wrapped = wrap(client, quiet=True)
        assert "Shear" in repr(wrapped)

    def test_multiple_calls(self) -> None:
        """Multiple calls should all flow through hooks."""
        client = _FakeClient()
        registry = HookRegistry()
        call_count = {"n": 0}

        def counter(req: InterceptedRequest) -> InterceptedRequest:
            call_count["n"] += 1
            return req

        registry.register(counter, "before_request", name="counter")
        wrapped = wrap(client, hooks=registry, quiet=True)

        wrapped.chat.completions.create(model="gpt-4o", messages=[])
        wrapped.chat.completions.create(model="gpt-4o", messages=[])
        wrapped.chat.completions.create(model="gpt-4o", messages=[])

        assert call_count["n"] == 3
        assert client.chat.completions.call_count == 3
