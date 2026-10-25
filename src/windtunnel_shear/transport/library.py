"""Library mode wrapper — the wrap() function.

Provides one-line integration for wrapping an existing LLM client
(e.g. OpenAI) so that all calls flow through Shear's hook pipeline.

Example::

    from windtunnel_shear import wrap, Hook
    from openai import OpenAI

    client = wrap(OpenAI())
    # All client.chat.completions.create() calls now flow through hooks.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from windtunnel_shear.cli.formatter import log_exchange
from windtunnel_shear.core.hooks import HookRegistry
from windtunnel_shear.core.models import InterceptedRequest, InterceptedResponse


class _Proxy:
    """Transparent attribute proxy that delegates to an underlying object."""

    __slots__ = ("_target",)

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)

    def __repr__(self) -> str:
        return f"<Shear:{object.__getattribute__(self, '_target')!r}>"


class _CompletionsProxy(_Proxy):
    """Proxy for ``client.chat.completions`` that intercepts ``create()``."""

    __slots__ = ("_json_output", "_quiet", "_registry", "_verbose")

    def __init__(
        self,
        target: Any,
        registry: HookRegistry,
        *,
        verbose: bool = False,
        json_output: bool = False,
        quiet: bool = False,
    ) -> None:
        super().__init__(target)
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_verbose", verbose)
        object.__setattr__(self, "_json_output", json_output)
        object.__setattr__(self, "_quiet", quiet)

    def create(self, **kwargs: Any) -> Any:
        """Intercept a sync chat.completions.create() call."""
        registry: HookRegistry = object.__getattribute__(self, "_registry")
        verbose: bool = object.__getattribute__(self, "_verbose")
        json_output: bool = object.__getattribute__(self, "_json_output")
        quiet: bool = object.__getattribute__(self, "_quiet")
        target = object.__getattribute__(self, "_target")

        # Build InterceptedRequest from kwargs
        intercepted_req = InterceptedRequest(raw=dict(kwargs), timestamp=time.monotonic())

        # Run before_request hooks (sync wrapper around async pipeline)
        intercepted_req = _run_sync(registry.run_before_request(intercepted_req))

        # Call the real create() with possibly-modified params
        start_time = time.monotonic()
        try:
            response = target.create(**intercepted_req.raw)
        except Exception as exc:
            error_resp = _run_sync(registry.run_on_error(intercepted_req, exc))
            if error_resp is not None:
                _log(
                    intercepted_req,
                    error_resp,
                    verbose=verbose,
                    json_output=json_output,
                    quiet=quiet,
                )
                return _response_to_object(
                    error_resp,
                    response_class=_guess_response_class(target),
                )
            raise

        latency_ms = (time.monotonic() - start_time) * 1000

        # Build InterceptedResponse from the SDK response
        resp_dict = _response_to_dict(response)
        intercepted_resp = InterceptedResponse(
            raw=resp_dict,
            status_code=200,
            timestamp=time.monotonic(),
            latency_ms=latency_ms,
        )

        # Run after_response hooks
        intercepted_resp = _run_sync(registry.run_after_response(intercepted_req, intercepted_resp))

        _log(
            intercepted_req,
            intercepted_resp,
            verbose=verbose,
            json_output=json_output,
            quiet=quiet,
        )

        # If hooks modified the response, patch the original object
        return _patch_response(response, intercepted_resp)


class _ChatProxy(_Proxy):
    """Proxy for ``client.chat`` that returns a _CompletionsProxy for ``completions``."""

    __slots__ = ("_json_output", "_quiet", "_registry", "_verbose")

    def __init__(
        self,
        target: Any,
        registry: HookRegistry,
        *,
        verbose: bool = False,
        json_output: bool = False,
        quiet: bool = False,
    ) -> None:
        super().__init__(target)
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_verbose", verbose)
        object.__setattr__(self, "_json_output", json_output)
        object.__setattr__(self, "_quiet", quiet)

    @property
    def completions(self) -> _CompletionsProxy:
        target = object.__getattribute__(self, "_target")
        registry = object.__getattribute__(self, "_registry")
        verbose = object.__getattribute__(self, "_verbose")
        json_output = object.__getattribute__(self, "_json_output")
        quiet = object.__getattribute__(self, "_quiet")
        return _CompletionsProxy(
            target.completions,
            registry,
            verbose=verbose,
            json_output=json_output,
            quiet=quiet,
        )


class _ClientProxy(_Proxy):
    """Top-level proxy for an LLM client (e.g. openai.OpenAI)."""

    __slots__ = ("_json_output", "_quiet", "_registry", "_verbose")

    def __init__(
        self,
        target: Any,
        registry: HookRegistry,
        *,
        verbose: bool = False,
        json_output: bool = False,
        quiet: bool = False,
    ) -> None:
        super().__init__(target)
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_verbose", verbose)
        object.__setattr__(self, "_json_output", json_output)
        object.__setattr__(self, "_quiet", quiet)

    @property
    def chat(self) -> _ChatProxy:
        target = object.__getattribute__(self, "_target")
        registry = object.__getattribute__(self, "_registry")
        verbose = object.__getattribute__(self, "_verbose")
        json_output = object.__getattribute__(self, "_json_output")
        quiet = object.__getattribute__(self, "_quiet")
        return _ChatProxy(
            target.chat,
            registry,
            verbose=verbose,
            json_output=json_output,
            quiet=quiet,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already in an async context — create a new thread to avoid deadlock
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _response_to_dict(response: Any) -> dict[str, Any]:
    """Convert an SDK response object to a plain dict."""
    if isinstance(response, dict):
        return response
    # OpenAI SDK objects have .model_dump()
    if hasattr(response, "model_dump"):
        return response.model_dump()  # type: ignore[no-any-return]
    # Fallback: try __dict__
    if hasattr(response, "__dict__"):
        return dict(response.__dict__)
    return {"raw": str(response)}


def _patch_response(original: Any, intercepted: InterceptedResponse) -> Any:
    """Apply modifications from hooks back to the SDK response object."""
    # If the response content was modified, update the original object
    if hasattr(original, "choices") and intercepted.choices:
        for i, choice in enumerate(intercepted.choices):
            if i < len(original.choices):
                msg = choice.get("message", {})
                if msg and hasattr(original.choices[i].message, "content"):
                    original.choices[i].message.content = msg.get("content")
    return original


def _guess_response_class(completions_obj: Any) -> type | None:
    """Try to determine the response class for synthetic responses."""
    return None


def _response_to_object(resp: InterceptedResponse, response_class: type | None = None) -> Any:
    """Convert an InterceptedResponse back to an SDK-like object if possible."""
    # Return the raw dict — callers should handle this
    return resp.raw


def _log(
    req: InterceptedRequest,
    resp: InterceptedResponse,
    *,
    verbose: bool,
    json_output: bool,
    quiet: bool,
) -> None:
    """Log the exchange via the formatter."""
    log_exchange(req, resp, verbose=verbose, json_mode=json_output, quiet=quiet)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wrap(
    client: Any,
    *,
    hooks: HookRegistry | None = None,
    verbose: bool = False,
    json_output: bool = False,
    quiet: bool = False,
) -> Any:
    """Wrap an LLM client so its calls flow through Shear.

    Example::

        from windtunnel_shear import wrap
        from openai import OpenAI

        client = wrap(OpenAI())

    Args:
        client: An LLM client instance (e.g. ``openai.OpenAI``).
        hooks: Optional hook registry. Uses a new empty registry if not provided.
        verbose: Show full request/response payloads in console output.
        json_output: Machine-readable JSON output.
        quiet: Suppress all console output.

    Returns:
        A wrapped client that intercepts all API calls through the hook pipeline.
    """
    registry = hooks if hooks is not None else HookRegistry()
    return _ClientProxy(
        client,
        registry,
        verbose=verbose,
        json_output=json_output,
        quiet=quiet,
    )
