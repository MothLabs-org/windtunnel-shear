# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `shear proxy` and `shear record` no longer crash at startup with `TypeError: ... unexpected keyword argument 'on_shutdown'` when Starlette 1.0 or newer is installed. The proxy now closes its upstream HTTP client through Starlette's `lifespan` handler, which works on every supported Starlette version.

## [0.1.0.post1] - 2026-10-23

### Changed

- Project links (homepage, repository, issues, changelog) now point to the renamed `MothLabs-org` GitHub organization, formerly `ButterflyLabs-org`, and the README credits Moth Labs (formerly Butterfly Labs). No code changes.

## [0.1.0] - 2026-03-05

### Added

- **HTTP proxy** — ASGI proxy (Starlette + uvicorn) with auto-detect upstream from API key format
- **Human-readable console output** — colored, compact request/response summaries with `--verbose` and `--debug` modes
- **Hook pipeline** — decorator API (`@Hook.before_request`, `@Hook.after_response`, `@Hook.on_error`) with stage ordering (mutate → fault → default → observe)
- **Library mode** — `wrap(client)` for one-line interception of OpenAI SDK clients
- **Provider auto-detection** — detects OpenAI (`sk-...`) and Azure endpoints from Authorization header
- **Infrastructure faults** — `--fault rate-limit:0.3`, `--fault latency:500ms`, `--fault error:503:0.1`, `--fault timeout:10s`
- **Prompt jitters** — `--jitter noise:0.1` (per-word typo injection), `--jitter contradict`, `--jitter dilute:5`, `--jitter rephrase`
- **Session recording** — `shear record -o session.json` with episode grouping and turn classification
- **Session replay** — `shear replay -i session.json` with sequential and exact matching modes
- **SSE streaming** — reassemble upstream SSE chunks into complete response for hook processing, then re-stream to client
- **Jitter diff logging** — `--verbose` shows before/after diffs when jitters modify requests
- **Configurable timeout** — `--timeout 30s` for upstream request timeout
- **Custom exception hierarchy** — ShearError, HookError, ReplayMissError, UpstreamError, FaultInjectedError with full context

[Unreleased]: https://github.com/MothLabs-org/windtunnel-shear/compare/v0.1.0.post1...HEAD
[0.1.0.post1]: https://github.com/MothLabs-org/windtunnel-shear/releases/tag/v0.1.0.post1
[0.1.0]: https://github.com/MothLabs-org/windtunnel-shear/releases/tag/v0.1.0
