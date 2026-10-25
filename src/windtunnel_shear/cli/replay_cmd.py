"""``shear replay`` command — replay traffic from a session file."""

from __future__ import annotations

import json
from pathlib import Path

import typer


def replay_cmd(
    input_file: Path = typer.Option(
        ...,
        "-i",
        "--input",
        help="Input session file path.",
    ),
    port: int = typer.Option(9800, help="Port to listen on."),
    match_mode: str = typer.Option(
        "sequential",
        help="Matching strategy: sequential or exact.",
    ),
    on_miss: str = typer.Option(
        "error",
        help="Cache-miss behavior: error, passthrough, or fallback.",
    ),
) -> None:
    """Replay recorded responses from a session file.

    Starts a server that returns recorded responses without making real
    API calls. Useful for offline testing and CI.
    """
    import uvicorn

    from windtunnel_shear.core.models import Session
    from windtunnel_shear.transport.replay import ReplayServer

    if not input_file.exists():
        typer.echo(f"Error: Session file not found: {input_file}", err=True)
        raise typer.Exit(1)

    data = json.loads(input_file.read_text())
    session = Session.from_json(data)

    typer.echo(f"\n  Shear replay from {input_file}", err=True)
    typer.echo(
        f"  {session.total_turns} turns, mode={match_mode}, on_miss={on_miss}",
        err=True,
    )
    typer.echo(f"  Listening on http://0.0.0.0:{port}\n", err=True)

    server = ReplayServer(
        session,
        match_mode=match_mode,
        on_miss=on_miss,
    )
    app = server.create_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
