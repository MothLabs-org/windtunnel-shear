"""Shared fixtures for integration tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from windtunnel_shear.transport import proxy
from windtunnel_shear.transport.proxy import ProxyConfig


@pytest.fixture(autouse=True)
def _restore_proxy_state() -> Iterator[None]:
    """Start each test with fresh proxy module state, and put the old state back after."""
    saved_config = proxy._config
    saved_client = proxy._http_client
    proxy._config = ProxyConfig()
    proxy._http_client = None
    yield
    proxy._config = saved_config
    proxy._http_client = saved_client
