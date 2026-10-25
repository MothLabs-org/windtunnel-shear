"""Replay server — serves responses from recorded session files."""

from __future__ import annotations

import logging
import time
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from windtunnel_shear.core.matching import Matcher, create_matcher
from windtunnel_shear.core.models import (
    InterceptedRequest,
    ReplayMissError,
    Session,
)

logger = logging.getLogger(__name__)


class ReplayServer:
    """Serves recorded responses from a session file.

    Can operate in sequential or exact matching mode, with configurable
    cache-miss behavior.
    """

    def __init__(
        self,
        session: Session,
        match_mode: str = "sequential",
        on_miss: str = "error",
    ) -> None:
        self._session = session
        self._match_mode = match_mode
        self._on_miss = on_miss
        self._matcher: Matcher = create_matcher(session, match_mode)

    def create_app(self) -> Starlette:
        """Create the ASGI app for the replay server."""
        return Starlette(
            routes=[
                Route(
                    "/{path:path}",
                    self._handle,
                    methods=["POST", "GET", "PUT", "DELETE"],
                ),
            ],
        )

    async def _handle(self, request: Request) -> Response:
        """Handle an incoming replay request."""
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Invalid JSON", "source": "shear"}},
                status_code=400,
            )

        intercepted = InterceptedRequest(
            raw=body,
            timestamp=time.monotonic(),
        )

        try:
            response = self._matcher.match(intercepted)
            return JSONResponse(response.raw, status_code=response.status_code)
        except ReplayMissError as exc:
            if self._on_miss == "error":
                return JSONResponse(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "replay_miss",
                            "source": "shear",
                            "request_hash": exc.request_hash,
                        }
                    },
                    status_code=502,
                )
            if self._on_miss == "fallback":
                return JSONResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "[shear:replay:fallback]",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                    status_code=200,
                )
            # passthrough — not implemented in v0, return error
            return JSONResponse(
                {
                    "error": {
                        "message": "Passthrough on-miss not yet supported.",
                        "source": "shear",
                    }
                },
                status_code=501,
            )

    async def start(self, port: int = 9800) -> None:
        """Start the replay server.

        Args:
            port: Port to listen on.
        """
        import uvicorn

        app = self.create_app()
        await uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning"),
        ).serve()
