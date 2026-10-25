"""``shear proxy`` command — start the HTTP proxy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import typer


def proxy_cmd(
    port: int = typer.Option(9800, help="Port to listen on."),
    upstream: str | None = typer.Option(None, help="Upstream LLM API URL."),
    hooks: Path | None = typer.Option(None, help="Python file with hook definitions."),
    fault: list[str] | None = typer.Option(
        None,
        help="Fault injection spec (e.g. rate-limit:0.3).",
    ),
    jitter: list[str] | None = typer.Option(
        None,
        help="Jitter spec (e.g. noise:0.1).",
    ),
    timeout: str = typer.Option("120s", help="Upstream request timeout (e.g. 30s, 5000ms)."),
    verbose: bool = typer.Option(False, help="Show jitter diffs and enriched output."),
    debug: bool = typer.Option(False, help="Show full JSON request/response payloads."),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Machine-readable JSON output.",
    ),
    quiet: bool = typer.Option(False, help="Suppress output."),
) -> None:
    """Start the Shear proxy server.

    Intercepts LLM API traffic, applies hooks/faults/jitters, and forwards
    to the upstream API. Auto-detects upstream from the Authorization header
    if --upstream is not provided.
    """
    import uvicorn

    from windtunnel_shear.core.hooks import HookRegistry
    from windtunnel_shear.core.models import InterceptedRequest
    from windtunnel_shear.faults.latency import parse_duration
    from windtunnel_shear.transport.proxy import ProxyConfig, create_app

    timeout_ms = parse_duration(timeout)
    timeout_s = timeout_ms / 1000.0

    registry = HookRegistry()

    # Load user hook file if provided
    if hooks is not None:
        if not hooks.exists():
            typer.echo(f"Error: Hook file not found: {hooks}", err=True)
            raise typer.Exit(1)
        _load_hooks_file(hooks)

    # Wire jitters as before_request hooks
    if jitter:
        from windtunnel_shear.jitters.engine import JitterEngine, parse_jitter_flag

        specs = [parse_jitter_flag(j) for j in jitter]
        engine = JitterEngine(specs)

        def jitter_hook(req: InterceptedRequest) -> InterceptedRequest:
            return engine.apply(req)

        registry.register(
            jitter_hook,
            "before_request",
            name="jitter",
            stage="mutate",
        )

    # Wire faults as before_request hooks (they short-circuit by returning
    # a synthetic response, handled specially in the proxy handler)
    if fault:
        from windtunnel_shear.faults.engine import FaultEngine, parse_fault_flag

        fault_specs = [parse_fault_flag(f) for f in fault]
        fault_engine = FaultEngine(fault_specs)

    config = ProxyConfig(
        upstream=upstream or "",
        verbose=verbose,
        debug=debug,
        json_output=json_output,
        quiet=quiet,
        timeout_s=timeout_s,
        hook_registry=registry,
        fault_engine=fault_engine if fault else None,
    )

    app = create_app(config)

    if not quiet:
        _print_banner(port, upstream, hooks, fault, jitter, timeout_s)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def _load_hooks_file(path: Path) -> None:
    """Load a Python file containing hook definitions."""
    spec = importlib.util.spec_from_file_location("user_hooks", path)
    if spec is None or spec.loader is None:
        typer.echo(f"Error: Could not load hook file: {path}", err=True)
        raise typer.Exit(1)
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_hooks"] = module
    spec.loader.exec_module(module)


def _print_banner(
    port: int,
    upstream: str | None,
    hooks: Path | None,
    faults: list[str] | None,
    jitters: list[str] | None,
    timeout_s: float = 120.0,
) -> None:
    """Print the startup banner to stderr."""
    typer.echo(f"\n  Shear proxy listening on http://0.0.0.0:{port}", err=True)
    if upstream:
        typer.echo(f"  Upstream: {upstream}", err=True)
    else:
        typer.echo(
            "  Upstream: auto-detect from Authorization header",
            err=True,
        )
    if hooks:
        typer.echo(f"  Hooks: {hooks}", err=True)
    if faults:
        for f in faults:
            typer.echo(f"  Fault: {f}", err=True)
    if jitters:
        for j in jitters:
            typer.echo(f"  Jitter: {j}", err=True)
    if timeout_s != 120.0:
        typer.echo(f"  Timeout: {timeout_s:.0f}s", err=True)
    typer.echo("", err=True)
