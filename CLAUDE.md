# windtunnel-shear (Shear)
> Wind shear for LLM APIs — intercept, mutate, record, replay, and stress-test any LLM conversation.
## Project Identity
- **Package name:** `windtunnel-shear`
- **CLI command:** `shear`
- **Import:** `from windtunnel_shear import ...`
- **License:** MIT
- **Python:** 3.10+
- **Organization:** Moth Labs
- **Part of:** The Wind Tunnel family (first building block)
---
## What Shear Is
Shear is a Python library and CLI that sits between any application and any LLM API. It can observe, modify, delay, block, replay, or simulate LLM API traffic. It understands LLM-specific semantics: message roles, tool calls, streaming, conversation structure, and token accounting.
Think of it as: **mitmproxy, but purpose-built for the LLM conversation protocol.**
## What Shear Is NOT
- Not a gateway (no routing, failover, or load balancing — that's LiteLLM)
- Not an observability platform (no dashboards or analytics — that's Langfuse)
- Not a guardrail system (no content filtering or policy enforcement)
- Not a testing framework (that is Wind Tunnel, which uses Shear as a building block)
---
## Design Principles (MUST follow these in every decision)
### P1. Zero-Config Start
The first experience requires NO configuration files, NO YAML, NO understanding of hooks.
```bash
pip install windtunnel-shear
shear proxy
```
That's it. Shear starts on port 9800, auto-detects the upstream from the first request's Authorization header, and logs every request/response in human-readable format. No setup wizard, no API keys to register, no accounts.
### P2. Progressive Complexity
Each step adds exactly ONE concept. A developer can stop at any level and get full value:
- **Level 0:** `shear proxy` — see your traffic (zero config)
- **Level 1:** `shear record -o session.json` — save your traffic (one flag)
- **Level 2:** `shear replay -i session.json` — replay without API calls (one flag)
- **Level 3:** `shear proxy --fault rate-limit:0.3` — test your app's resilience (one flag)
- **Level 4:** `shear proxy --jitter noise:0.1` — test your LLM's robustness (one flag)
- **Level 5:** `shear proxy --hooks my_hooks.py` — custom logic (one file)
No level depends on understanding a later level.
### P3. Inline Faults and Jitters Without Code
80% of testing scenarios use CLI flags, not Python:
```bash
# Infrastructure faults (test the application)
shear proxy --fault rate-limit:0.3
shear proxy --fault latency:500ms
shear proxy --fault error:503:0.1
shear proxy --fault timeout:10s
# Prompt jitters (test the LLM)
shear proxy --jitter noise:0.1
shear proxy --jitter contradict
shear proxy --jitter dilute:5
shear proxy --jitter rephrase
# Combine freely
shear proxy --fault latency:200ms --jitter noise:0.1
```
### P4. Auto-Detect Upstream
Never require `--upstream` for common cases. Read the Authorization header:
- `sk-...` → api.openai.com
- Azure deployment URLs → extracted from request
- `--upstream` exists for explicit control but is never required
### P5. Human-Readable Console Output
Default output shows what developers care about:
```
→ gpt-4o | 3 messages | 847 tokens
← 200 | 234 tokens | 1.2s | "The capital of France is..."
```
Not raw JSON. Not HTTP headers. Colored, compact, scannable.
- `--verbose` shows full payloads
- `--json` gives machine-readable output for piping
- `--quiet` suppresses output
### P6. One-Line Library Integration
```python
from windtunnel_shear import wrap
from openai import OpenAI
client = wrap(OpenAI())  # that's it
```
### P7. Self-Documenting Session Files
Session JSON is human-readable. Metadata at top, clean request/response pairs. A developer can read it in any editor without docs.
### P8. Helpful Error Messages
- **Replay miss:** Shows expected vs received with a diff
- **Hook exception:** Shows hook name, exception, source file + line
- **Upstream error:** Labels whether Shear injected it or the real API returned it
- **Connection failure:** Suggests fixes (wrong URL? invalid key? port in use?)
---
## Architecture Overview
```
┌─────────────────────────────────────────────────┐
│                   Application                     │
│            (OpenAI SDK, LangChain, etc.)          │
└────────────────────┬────────────────────────────┘
                     │  OPENAI_BASE_URL=localhost:9800
                     ▼
┌─────────────────────────────────────────────────┐
│                     SHEAR                         │
│                                                   │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Jitters │→ │   Hook   │→ │   Transport    │  │
│  │ Engine  │  │ Pipeline │  │ (Proxy/Library) │  │
│  └─────────┘  └──────────┘  └────────┬───────┘  │
│                                       │          │
│  ┌─────────┐  ┌──────────┐           │          │
│  │  Fault  │← │  Session │←──────────┘          │
│  │ Engine  │  │ Recorder │                       │
│  └─────────┘  └──────────┘                       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Upstream LLM API                     │
│         (OpenAI, Azure, via LiteLLM, etc.)       │
└─────────────────────────────────────────────────┘
```
**Center of gravity:** The session format and hook pipeline are the core primitives. The proxy is one transport — library mode and replay mode feed the same pipeline. Wind Tunnel uses Shear primarily in library mode.
---
## Repository Structure
```
windtunnel-shear/
├── CLAUDE.md                    # This file — project spec for Claude Code
├── LICENSE                      # MIT
├── README.md                    # User-facing docs (install, quickstart, examples)
├── CONTRIBUTING.md              # Contribution guide
├── CHANGELOG.md                 # Keep-a-changelog format
├── pyproject.toml               # Project metadata, dependencies, build config
├── Makefile                     # Common dev commands (lint, test, format, etc.)
│
├── src/
│   └── windtunnel_shear/
│       ├── __init__.py          # Public API: wrap(), Interceptor, Hook
│       ├── py.typed             # PEP 561 marker for type checking
│       │
│       ├── core/                # Core abstractions (framework-independent)
│       │   ├── __init__.py
│       │   ├── models.py        # InterceptedRequest, InterceptedResponse, Session, Episode, HookResult
│       │   ├── hooks.py         # Hook registry, decorator API, pipeline executor
│       │   ├── session.py       # Session recorder, episode grouping logic
│       │   └── matching.py      # Replay matching strategies (sequential, exact)
│       │
│       ├── faults/              # Infrastructure fault injection
│       │   ├── __init__.py
│       │   ├── engine.py        # Fault engine: parses --fault flags, applies faults
│       │   ├── errors.py        # Error injection (429, 500, 503, timeout)
│       │   ├── latency.py       # Latency injection
│       │   └── streaming.py     # Partial stream interruption
│       │
│       ├── jitters/             # Conversation perturbation
│       │   ├── __init__.py
│       │   ├── engine.py        # Jitter engine: parses --jitter flags, applies jitters
│       │   ├── noise.py         # Typo/Unicode noise injection
│       │   ├── contradict.py    # Instruction contradiction
│       │   ├── dilute.py        # Context dilution
│       │   ├── rephrase.py      # Prompt rephrasing
│       │   └── tool.py          # Tool jitter stubs (v0.1 — reserved namespace, returns helpful message)
│       │
│       ├── transport/           # How traffic reaches the hook pipeline
│       │   ├── __init__.py
│       │   ├── proxy.py         # HTTP proxy server (ASGI/uvicorn)
│       │   ├── library.py       # Library mode wrapper (wrap() function)
│       │   └── replay.py        # Replay server (serves from session files)
│       │
│       ├── streaming/           # SSE handling
│       │   ├── __init__.py
│       │   ├── reassembly.py    # Collect SSE chunks → complete response
│       │   └── restream.py      # Complete response → SSE chunks for client
│       │
│       ├── providers/           # Provider-specific logic
│       │   ├── __init__.py
│       │   ├── detect.py        # Auto-detect upstream from auth header
│       │   └── openai.py        # OpenAI-compatible request/response parsing
│       │
│       ├── tokens/              # Token counting
│       │   ├── __init__.py
│       │   └── counter.py       # tiktoken + provider-reported usage
│       │
│       └── cli/                 # CLI entry points
│           ├── __init__.py
│           ├── main.py          # Click/Typer app, top-level command routing
│           ├── proxy_cmd.py     # `shear proxy` command
│           ├── record_cmd.py    # `shear record` command
│           ├── replay_cmd.py    # `shear replay` command
│           ├── simulate_cmd.py  # `shear simulate` command
│           └── inspect_cmd.py   # `shear inspect` command
│
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── unit/
│   │   ├── test_models.py       # Data model tests
│   │   ├── test_hooks.py        # Hook pipeline tests
│   │   ├── test_session.py      # Session recording + episode grouping
│   │   ├── test_matching.py     # Replay matching strategies
│   │   ├── test_faults.py       # Fault injection
│   │   ├── test_jitters.py      # Jitter injection
│   │   ├── test_streaming.py    # SSE reassembly/restream
│   │   ├── test_detect.py       # Provider auto-detection
│   │   └── test_tokens.py       # Token counting
│   │
│   ├── integration/
│   │   ├── test_proxy_e2e.py    # Full proxy round-trip tests
│   │   ├── test_library_mode.py # Library wrapper tests
│   │   ├── test_record_replay.py # Record → replay cycle
│   │   └── test_cli.py          # CLI command tests
│   │
│   └── fixtures/
│       ├── sessions/            # Recorded session files for replay tests
│       ├── responses/           # Synthetic response fixtures
│       └── hooks/               # Example hook files for testing
│
├── examples/
│   ├── basic_proxy.py           # Simplest usage
│   ├── record_replay.py         # Record and replay a conversation
│   ├── custom_hooks.py          # Hook examples
│   ├── fault_injection.py       # Infrastructure fault testing
│   ├── jitter_testing.py        # Prompt jitter testing
│   └── agent_testing.py         # Testing an agentic tool-calling loop
│
└── docs/
    ├── quickstart.md            # 5-minute getting started
    ├── concepts.md              # Faults vs jitters, episodes, matching
    ├── hooks.md                 # Hook API reference
    ├── cli.md                   # CLI reference
    ├── session-format.md        # Session JSON schema documentation
    └── windtunnel.md            # How Shear connects to Wind Tunnel
```
---
## Data Model (src/windtunnel_shear/core/models.py)
Use Python dataclasses with `__slots__` for performance. All models are immutable after creation (frozen dataclasses) except where mutation is explicitly needed in hooks.
### Design: Raw Passthrough + Typed Accessors
Every request/response stores the complete raw body. Typed accessors provide convenience for common fields. Unknown fields pass through untouched.
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
@dataclass
class InterceptedRequest:
"""Raw LLM API request with typed accessors."""
    raw: dict[str, Any]                    # Complete raw request body — never drop fields
    timestamp: float                        # Monotonic timestamp
    request_hash: str = ""                  # Computed after normalization, used for replay matching
# Typed accessors (read from raw, write back to raw)
@property
def messages(self) -> list[dict[str, Any]]:
return self.raw.get("messages", [])
@messages.setter
def messages(self, value: list[dict[str, Any]]) -> None:
self.raw["messages"] = value
@property
def model(self) -> str:
return self.raw.get("model", "")
@property
def tools(self) -> list[dict[str, Any]]:
return self.raw.get("tools", [])
@property
def stream(self) -> bool:
return self.raw.get("stream", False)
@property
def temperature(self) -> Optional[float]:
return self.raw.get("temperature")
# Agent-aware accessors
@property
def is_tool_result(self) -> bool:
"""True if this request contains tool result messages."""
return any(m.get("role") == "tool" for m in self.messages)
@property
def pending_tool_calls(self) -> list[dict[str, Any]]:
"""Tool calls from the most recent assistant message."""
for msg in reversed(self.messages):
if msg.get("role") == "assistant" and msg.get("tool_calls"):
return msg["tool_calls"]
return []
@property
def tool_call_count(self) -> int:
"""Total tool calls observed in message history."""
        count = 0
for msg in self.messages:
if msg.get("role") == "assistant" and msg.get("tool_calls"):
                count += len(msg["tool_calls"])
return count
@dataclass
class InterceptedResponse:
"""Raw LLM API response with typed accessors."""
    raw: dict[str, Any]                    # Complete raw response body
    status_code: int = 200
    timestamp: float = 0.0
    latency_ms: float = 0.0
    reason: str = ""                        # Shear-injected reason string (empty if from real API)
@property
def choices(self) -> list[dict[str, Any]]:
return self.raw.get("choices", [])
@property
def content(self) -> str:
"""First choice message content."""
if self.choices and self.choices[0].get("message"):
return self.choices[0]["message"].get("content", "")
return ""
@content.setter
def content(self, value: str) -> None:
if self.choices and self.choices[0].get("message"):
self.choices[0]["message"]["content"] = value
@property
def usage(self) -> dict[str, int]:
return self.raw.get("usage", {})
@property
def finish_reason(self) -> str:
if self.choices:
return self.choices[0].get("finish_reason", "")
return ""
@property
def tool_calls(self) -> list[dict[str, Any]]:
if self.choices and self.choices[0].get("message"):
return self.choices[0]["message"].get("tool_calls", [])
return []
@property
def has_tool_calls(self) -> bool:
return len(self.tool_calls) > 0
@property
def is_shear_injected(self) -> bool:
"""True if this response was injected by Shear (fault/simulation), not from real API."""
return self.reason != ""
class TurnType(str, Enum):
    USER_MESSAGE = "user_message"
    TOOL_CALL_REQUEST = "tool_call_request"
    TOOL_RESULT_SUBMISSION = "tool_result_submission"
    FINAL_RESPONSE = "final_response"
@dataclass
class Turn:
"""Single request/response pair with type annotation."""
type: TurnType
    request: InterceptedRequest
    response: InterceptedResponse
@dataclass
class Episode:
"""Group of related API calls forming one agentic task execution."""
    episode_id: str
    turns: list[Turn] = field(default_factory=list)
@property
def tool_call_count(self) -> int:
return sum(t.request.tool_call_count for t in self.turns)
@property
def total_tokens(self) -> int:
return sum(t.response.usage.get("total_tokens", 0) for t in self.turns)
@dataclass
class Session:
"""Complete recorded session with episode structure."""
    session_id: str
    model: str = ""
    started_at: str = ""
    episodes: list[Episode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
@property
def total_turns(self) -> int:
return sum(len(ep.turns) for ep in self.episodes)
@property
def total_tokens(self) -> int:
return sum(ep.total_tokens for ep in self.episodes)
def to_json(self) -> dict:
"""Serialize to human-readable JSON."""
# Implementation: clean, self-documenting format
        ...
@classmethod
def from_json(cls, data: dict) -> Session:
"""Deserialize from JSON."""
        ...
class HookAction(str, Enum):
    ALLOW = "allow"          # Pass through unchanged
    MODIFY = "modify"        # Return altered request/response
    BLOCK = "block"          # Return error with reason string
    SIMULATE = "simulate"    # Return synthetic response
@dataclass
class HookResult:
    action: HookAction
    request: Optional[InterceptedRequest] = None
    response: Optional[InterceptedResponse] = None
    reason: str = ""
```
---
## Hook System (src/windtunnel_shear/core/hooks.py)
Hooks are Python functions registered via decorator. Multiple hooks chain in registration order. Any hook can short-circuit.
```python
from windtunnel_shear import Hook
@Hook.before_request(name="inject_policy", category="jitter")
def inject_policy(req: InterceptedRequest) -> InterceptedRequest:
    req.messages.insert(0, {"role": "system", "content": "New policy"})
return req
@Hook.after_response(name="check_compliance")
def check_compliance(req: InterceptedRequest, resp: InterceptedResponse) -> InterceptedResponse:
if "forbidden" in resp.content:
        resp.content = "[BLOCKED]"
return resp
@Hook.on_error(name="retry_on_429")
def retry_on_429(req: InterceptedRequest, err: Exception) -> Optional[InterceptedResponse]:
if hasattr(err, 'status_code') and err.status_code == 429:
return None  # returning None means "propagate error"
return None
```
### Hook metadata fields:
- `name` (str): Human-readable name for traces/errors
- `category` (str, optional): Group hooks logically ("jitter", "fault", "logging")
- `stage` (str, optional): Ordering group — "mutate" runs before "fault" runs before "observe"
### Implementation notes:
- Hook registry is a singleton per Interceptor instance
- Hooks are stored as ordered list, sorted by stage then registration order
- Each hook receives immutable copies (for before_request) or mutable references (for after_response)
- Hook exceptions are caught, logged with full context (hook name, source file, line), and optionally propagated
---
## Replay Matching (src/windtunnel_shear/core/matching.py)
### v0 Matching Modes
**Sequential (default):** Return recorded responses in order regardless of request content. Fails fast if more requests than recorded.
**Exact:** Hash normalized request body and match against recorded hashes. Ignores non-deterministic fields.
### Request Normalization (before hashing)
Strip these non-deterministic fields:
- Request IDs / client-generated metadata
- Timestamps
- `stream` parameter (preserved in session but not used for matching)
- Provider-specific headers
Keep these fields:
- `messages` array (exact content)
- `model`
- `tools` / `functions`
- `temperature`, `max_tokens`, `top_p`
### Cache-Miss Behavior (`--on-miss`)
- `error` (default): Return HTTP 502 with JSON identifying the unmatched request (hash, expected hash, content)
- `passthrough`: Forward to real upstream API
- `fallback`: Return configurable static response
---
## Fault Engine (src/windtunnel_shear/faults/)
Infrastructure faults test the APPLICATION's resilience. They operate on the response side.
### CLI flag format: `--fault TYPE:PARAMS`
| Flag | Effect |
|------|--------|
| `--fault rate-limit:0.3` | 30% of requests return 429 |
| `--fault latency:500ms` | Add 500ms to every response |
| `--fault error:503:0.1` | 10% return 503 |
| `--fault timeout:10s` | Hang for 10s then drop |
Every fault includes a `reason` field in the response: `"shear:fault:rate_limit:0.3"`.
---
## Jitter Engine (src/windtunnel_shear/jitters/)
Prompt jitters test the LLM's robustness. They operate on the request side (before_request).
### CLI flag format: `--jitter TYPE:PARAMS`
| Flag | Effect |
|------|--------|
| `--jitter noise:0.1` | 10% of words in user messages get a typo (1 char each) |
| `--jitter contradict` | Inject contradictory user message opposing system prompt |
| `--jitter dilute:5` | Pad with 5 irrelevant turns |
| `--jitter rephrase` | Reword system prompt policies (seeded) |
Every jitter records original + mutated messages in session metadata with reason `"shear:jitter:noise:0.1"`.
### Reserved tool-* namespace (v0.1)
These flags are parsed but return a helpful "tool jitters ship in v0.1" message:
- `--jitter tool-corrupt:TOOL:FIELD`
- `--jitter tool-drop:TOOL:FIELD`
- `--jitter tool-fail:TOOL:PROBABILITY`
- `--jitter tool-delay:TOOL:DURATION`
---
## Streaming (src/windtunnel_shear/streaming/)
v0 handles streaming via **reassembly**: collect all SSE chunks into a complete response, run hooks, then re-stream to client.
**DO NOT implement chunk-level mutation (on_stream_chunk) in v0.** This is explicitly deferred due to edge-case complexity around partial JSON fragments and tool call data spanning chunks.
### Reassembly flow:
1. Intercept SSE stream from upstream
2. Collect chunks, accumulating delta content
3. Build complete InterceptedResponse
4. Run after_response hooks on complete response
5. Re-serialize to SSE chunks
6. Stream to client
### Passthrough mode:
When no hooks need the complete response, stream chunks directly without reassembly (zero latency overhead).
---
## CLI (src/windtunnel_shear/cli/)
Use **Typer** (or Click) for CLI framework. Commands:
```bash
# Basic
shear proxy                                         # zero-config
shear proxy --port 9800 --upstream https://api.openai.com/v1
shear proxy --hooks my_hooks.py
shear proxy --verbose                               # full payloads
shear proxy --json                                  # machine-readable
# Record & Replay
shear record -o session.json
shear replay -i session.json
shear replay -i session.json --match-mode exact --on-miss passthrough
shear inspect session.json
# Inline Faults
shear proxy --fault rate-limit:0.3
shear proxy --fault latency:500ms
shear proxy --fault error:503:0.1
shear proxy --fault timeout:10s
# Inline Jitters
shear proxy --jitter noise:0.1
shear proxy --jitter contradict
shear proxy --jitter dilute:5
shear proxy --jitter rephrase
# Combined
shear proxy --fault latency:200ms --jitter noise:0.1
# Simulation
shear simulate --responses fixtures.json
```
---
## Software Engineering Principles
### Code Quality
- **Type hints everywhere.** All function signatures, all return types. Use `from __future__ import annotations` in every file.
- **PEP 561 compliant.** Ship `py.typed` marker so downstream type checkers work.
- **Strict linting.** Use `ruff` for linting + formatting (replaces black + isort + flake8). Configure in pyproject.toml.
- **No `Any` leakage.** Use `Any` only in the raw passthrough dict. All Shear-authored interfaces use concrete types.
- **Docstrings on every public function.** Google-style docstrings. Include Args, Returns, Raises, and Examples.
### Testing
- **pytest** as the test framework. No unittest.
- **Minimum 90% coverage** on core/, faults/, jitters/, transport/. Measured via `pytest-cov`.
- **Unit tests are fast.** No network calls, no real LLM APIs. Use recorded sessions and mock responses.
- **Integration tests use a local proxy.** Spin up Shear's proxy in a subprocess, send requests, verify behavior.
- **Every bug fix gets a regression test.** No exceptions.
- **Test the error paths.** Hook exceptions, replay misses, malformed requests, stream interruptions. These matter more than happy paths.
### Dependencies
- **Minimal production dependencies.** Core should depend only on:
- `httpx` (async HTTP client for upstream calls)
- `uvicorn` + `starlette` (ASGI server for proxy mode)
- `typer` (CLI framework)
- `tiktoken` (token counting — optional, graceful fallback)
- `orjson` (fast JSON — optional, fallback to stdlib json)
- **No heavy frameworks.** No FastAPI (too opinionated), no aiohttp (httpx is better), no requests (blocking).
- **Dev dependencies separate.** pytest, ruff, mypy, pytest-cov, pre-commit in dev group only.
### Error Handling
- **Custom exception hierarchy:**
  ```python
class ShearError(Exception): ...
class HookError(ShearError): ...          # Hook raised an exception
class ReplayMissError(ShearError): ...    # No matching recorded response
class UpstreamError(ShearError): ...      # Real API returned error
class FaultInjectedError(ShearError): ... # Shear intentionally returned error
  ```
- **Never swallow exceptions silently.** Log with full context, then decide whether to propagate or handle.
- **Distinguish Shear errors from upstream errors.** Every error response includes `"source": "shear"` or `"source": "upstream"` so developers never chase phantom bugs.
### Async
- **Async-first internally.** The proxy, HTTP client, and hook pipeline are all async.
- **Sync wrapper for library mode.** `wrap(client)` works in sync code via `asyncio.run()` or `nest_asyncio`.
- **No blocking calls on the event loop.** All I/O is async. File writes (session recording) use `aiofiles` or offload to thread pool.
---
## Open Source Best Practices
### README.md Structure
1. **One-line description** + badge row (PyPI version, Python versions, license, CI status, coverage)
2. **GIF demo** — record a conversation, inject a jitter, show the result. 15 seconds max.
3. **Install:** `pip install windtunnel-shear`
4. **Quickstart:** 3 examples (basic proxy, record/replay, fault injection). Copy-pasteable.
5. **Why Shear?** One paragraph. Not a gateway, not observability — the "what if" tool.
6. **Feature table** — what's in v0 vs what's coming
7. **CLI reference** — all commands with examples
8. **Hook API** — decorator examples
9. **Links:** docs, contributing, changelog, Wind Tunnel project
### CONTRIBUTING.md
- How to set up dev environment (`make install`)
- How to run tests (`make test`)
- How to lint (`make lint`)
- PR process: fork, branch, test, PR
- Code style: ruff config, type hints required, docstrings required
- Issue labels: `bug`, `feature`, `good-first-issue`, `help-wanted`
### CHANGELOG.md
- Follow [Keep a Changelog](https://keepachangelog.com/) format
- Sections: Added, Changed, Deprecated, Removed, Fixed, Security
- Every PR that changes user-facing behavior updates the changelog
### CI/CD (GitHub Actions)
```yaml
# .github/workflows/ci.yml
# Triggers: push to main, all PRs
# Jobs:
#   - lint (ruff check, ruff format --check, mypy)
#   - test (pytest across Python 3.10, 3.11, 3.12, 3.13)
#   - coverage (fail if below 90%)
#   - build (build wheel, verify install)
```
### Release Process
- Semantic versioning: MAJOR.MINOR.PATCH
- Tag-based releases: push a git tag → CI builds and publishes to PyPI
- Every release has a GitHub Release with changelog excerpt
---
## Key Technical Decisions
1. **ASGI over WSGI** — async-first, handles streaming natively
2. **httpx over requests** — async, HTTP/2, better streaming support
3. **Starlette over FastAPI** — lighter, less opinionated, we don't need OpenAPI docs for a proxy
4. **Typer over argparse** — better DX, auto-generated help, type-safe
5. **ruff over black+isort+flake8** — single tool, faster, actively maintained
6. **dataclasses over Pydantic** — lighter for internal models, no validation overhead on hot path
7. **orjson optional** — fast JSON for high-throughput proxy, stdlib fallback for minimal installs
8. **tiktoken optional** — not everyone needs token counting, graceful degradation
---
## Implementation Order
### Week 1: Core + Proxy
1. Set up repo structure, pyproject.toml, Makefile, CI
2. Implement `InterceptedRequest`, `InterceptedResponse`, `HookResult` models
3. Implement hook registry and pipeline executor
4. Implement proxy server (Starlette ASGI app + uvicorn)
5. Implement provider auto-detection
6. Implement human-readable console output
7. Basic CLI: `shear proxy`
8. Tests for models, hooks, proxy round-trip
### Week 2: Library Mode + Streaming
1. Implement `wrap()` function for library mode
2. Implement SSE reassembly and re-streaming
3. Implement streaming passthrough mode (when no hooks need complete response)
4. Agent-aware typed accessors (is_tool_result, pending_tool_calls, etc.)
5. Tests for library mode, streaming, agent accessors
### Week 3: Record + Session Format
1. Implement session recorder with episode grouping
2. Implement `Session.to_json()` / `Session.from_json()`
3. Implement request normalization and hashing
4. Implement `shear record` CLI command
5. Implement `shear inspect` CLI command
6. Tests for session recording, episode grouping, serialization
### Week 4: Replay + Simulate
1. Implement sequential matching strategy
2. Implement exact matching strategy
3. Implement cache-miss behavior (error/passthrough/fallback)
4. Implement replay server
5. Implement `shear replay` CLI command
6. Implement response simulation with fixture files
7. Implement `shear simulate` CLI command
8. Tests for matching, replay, simulation, cache-miss behavior
### Week 5: Faults + Jitters
1. Implement fault engine and `--fault` flag parser
2. Implement error injection, latency injection, partial stream interruption
3. Implement jitter engine and `--jitter` flag parser
4. Implement noise, contradict, dilute, rephrase jitters
5. Implement reserved tool-* namespace (parsed, returns helpful message)
6. Implement reason strings in traces and error responses
7. Tests for all fault types and jitter types
### Week 6: Polish + Ship
1. Token counting (tiktoken + provider-reported)
2. End-to-end integration tests
3. README with GIF demo
4. CONTRIBUTING.md, CHANGELOG.md
5. Documentation (quickstart, concepts, hook API, CLI reference, session format)
6. Examples directory
7. CI/CD pipeline
8. First PyPI publish
---
## Development Progress

### Completed
- **Week 0:** Repo scaffolding, pyproject.toml, Makefile, CI, .gitignore, LICENSE, README, CONTRIBUTING, CHANGELOG
- **Week 1:** Core models, hook registry + pipeline, proxy server (ASGI), provider auto-detection, console formatter, CLI (`shear proxy`), library mode `wrap()`
- **Week 2:** Fault engine (rate-limit, latency, error, timeout), jitter engine (noise, contradict, dilute, rephrase), session recording with episode grouping, replay with sequential/exact matching, CLI wiring (`shear record`, `shear replay`, `--fault`, `--jitter`)
- **Streaming:** SSE reassembly + re-streaming. Proxy detects `stream: true`, collects chunks via httpx streaming, reassembles into complete response for hooks, then re-streams SSE to client.
- **Tests:** 148 passing (unit + integration), ruff clean, mypy clean across 38 source files

### Bug Fixes Applied
- **Double /v1 in proxy URL:** When `--upstream http://host/v1` and request path is `/v1/chat/completions`, the path was duplicated (`/v1/v1/...`). Fixed by stripping overlapping path prefix in `transport/proxy.py`.
- **Episode splitting:** `test_new_episode_after_final` failed because episode split checked `turn_type == USER_MESSAGE` but both turns classified as `FINAL_RESPONSE`. Fixed by checking `!= TOOL_RESULT_SUBMISSION` instead.

### Revisions Applied (from live testing on DGX Spark)

#### P0 — Noise jitter changed to per-word (DONE)
- Changed from per-character to per-word corruption. Each word has `ratio` probability of getting 1 char corrupted.
- Files changed: `jitters/noise.py`, `tests/unit/test_jitters.py`

#### P1 — Jitter diff logging (DONE)
- Added `jitter_log` field to InterceptedRequest. JitterEngine captures before/after diffs. `--verbose` shows diffs.
- Files changed: `core/models.py`, `jitters/engine.py`, `cli/formatter.py`

#### P2 — `--timeout` flag (DONE)
- Added `--timeout` CLI option (e.g. `--timeout 30s`), configurable httpx timeout.
- Files changed: `cli/proxy_cmd.py`, `transport/proxy.py`

#### P3 — Token estimate fixed (DONE)
- Changed from chars/4 to words x 1.3 + 4 per message framing overhead.
- Files changed: `cli/formatter.py`

#### P4 — Verbose/debug split (DONE)
- `--verbose`: compact + jitter diffs. `--debug`: full JSON payloads.
- Files changed: `cli/proxy_cmd.py`, `cli/formatter.py`, `transport/proxy.py`

### Not Yet Implemented (stubs remaining)
- `tokens/counter.py` — tiktoken integration
- `cli/simulate_cmd.py` — `shear simulate` command
- `cli/inspect_cmd.py` — `shear inspect` command
- `jitters/tool.py` — Tool-level jitters (reserved for v0.1)
- Replay `--on-miss passthrough` mode

---
## What Success Looks Like
- **Week 2:** `pip install windtunnel-shear && shear proxy` works. Developer sees colored request/response summaries.
- **Week 4:** Developer records a conversation, replays it offline, gets identical behavior.
- **Week 6:** Developer runs `shear proxy --jitter contradict` against GPT-4 and sees whether it holds system prompt boundaries. This is the GIF for the README.
The GIF-worthy demo: **"How robust is your LLM, really?"**
