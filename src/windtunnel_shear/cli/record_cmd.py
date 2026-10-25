"""``shear record`` command — record LLM traffic to a session file."""

from __future__ import annotations

import json
from pathlib import Path

import typer


def record_cmd(
    output: Path = typer.Option(..., "-o", "--output", help="Output session file path."),
    port: int = typer.Option(9800, help="Port to listen on."),
    upstream: str | None = typer.Option(None, help="Upstream LLM API URL."),
    verbose: bool = typer.Option(False, help="Show full request/response payloads."),
    quiet: bool = typer.Option(False, help="Suppress output."),
) -> None:
    """Record LLM API traffic to a session file.

    Starts a proxy that records all request/response pairs to the specified
    output file in Shear's self-documenting JSON format.
    """
    import uvicorn

    from windtunnel_shear.core.hooks import HookRegistry
    from windtunnel_shear.core.models import InterceptedRequest, InterceptedResponse
    from windtunnel_shear.core.session import SessionRecorder
    from windtunnel_shear.transport.proxy import ProxyConfig, create_app

    recorder = SessionRecorder()
    registry = HookRegistry()

    # Register an after_response hook that records every exchange
    def record_hook(
        req: InterceptedRequest,
        resp: InterceptedResponse,
    ) -> InterceptedResponse:
        recorder.record_turn(req, resp)
        return resp

    registry.register(record_hook, "after_response", name="recorder", stage="observe")

    config = ProxyConfig(
        upstream=upstream or "",
        verbose=verbose,
        quiet=quiet,
        hook_registry=registry,
    )

    app = create_app(config)

    if not quiet:
        typer.echo(f"\n  Shear recording to {output}", err=True)
        typer.echo(f"  Listening on http://0.0.0.0:{port}", err=True)
        if upstream:
            typer.echo(f"  Upstream: {upstream}", err=True)
        else:
            typer.echo(
                "  Upstream: auto-detect from Authorization header",
                err=True,
            )
        typer.echo("  Press Ctrl+C to stop and save.\n", err=True)

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        session = recorder.finalize()
        output.write_text(
            json.dumps(session.to_json(), indent=2, ensure_ascii=False),
        )
        if not quiet:
            typer.echo(
                f"\n  Saved {session.total_turns} turns to {output}",
                err=True,
            )
