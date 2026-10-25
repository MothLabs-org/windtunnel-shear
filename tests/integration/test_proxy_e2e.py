"""End-to-end proxy round-trip tests.

The proxy app runs in-process through ``httpx.ASGITransport`` and the
upstream LLM API is an ``httpx.MockTransport``, so there are no network
calls and coverage measures the proxy code.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from windtunnel_shear.transport import proxy
from windtunnel_shear.transport.proxy import ProxyConfig, create_app


@pytest.fixture(autouse=True)
def _restore_proxy_state() -> Iterator[None]:
    """Put the proxy's module-level config and client back after each test."""
    saved_config = proxy._config
    saved_client = proxy._http_client
    yield
    proxy._config = saved_config
    proxy._http_client = saved_client


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


def test_create_app_builds_on_installed_starlette() -> None:
    """Regression: Starlette 1.0 removed ``on_shutdown``, which crashed ``shear proxy``."""
    app = create_app(ProxyConfig(quiet=True))

    assert len(app.routes) == 1


async def test_shutdown_closes_shared_http_client() -> None:
    app = create_app(ProxyConfig(quiet=True))

    async with app.router.lifespan_context(app):
        client = proxy._get_client()
        assert not client.is_closed

    assert client.is_closed
    assert proxy._http_client is None


async def test_shutdown_without_client_is_a_no_op() -> None:
    proxy._http_client = None
    app = create_app(ProxyConfig(quiet=True))

    async with app.router.lifespan_context(app):
        pass

    assert proxy._http_client is None
