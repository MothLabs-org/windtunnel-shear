# windtunnel-shear

> Wind shear for LLM APIs — intercept, mutate, record, replay, and stress-test any LLM conversation.

[![PyPI version](https://img.shields.io/pypi/v/windtunnel-shear.svg)](https://pypi.org/project/windtunnel-shear/)
[![Python versions](https://img.shields.io/pypi/pyversions/windtunnel-shear.svg)](https://pypi.org/project/windtunnel-shear/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/MothLabs-org/windtunnel-shear/actions/workflows/ci.yml/badge.svg)](https://github.com/MothLabs-org/windtunnel-shear/actions)

## Install

```bash
pip install windtunnel-shear
```

## Quickstart

### 1. See your LLM traffic (zero config)

```bash
shear proxy
# Point your app at localhost:9800 and watch requests flow
```

```
→ gpt-4o | 3 messages | 847 tokens
← 200 | 234 tokens | 1.2s | "The capital of France is..."
```

### 2. Record and replay

```bash
# Record a session
shear record -o session.json

# Replay without API calls
shear replay -i session.json
```

### 3. Inject faults and jitters

```bash
# Test your app's resilience
shear proxy --fault rate-limit:0.3

# Test your LLM's robustness
shear proxy --jitter noise:0.1

# Combine freely
shear proxy --fault latency:200ms --jitter contradict
```

### 4. Library mode

```python
from windtunnel_shear import wrap
from openai import OpenAI

client = wrap(OpenAI())  # that's it
```

## Why Shear?

Shear is **not** a gateway (no routing or load balancing), **not** an observability platform (no dashboards), and **not** a guardrail system (no content filtering). It's the "what if" tool — what if 30% of requests get rate-limited? What if the user prompt has typos? What if the system prompt gets contradicted? Shear answers these questions without changing your application code.

## Features

| Feature | Status |
|---------|--------|
| HTTP proxy with auto-detect upstream | ✅ v0.1 |
| Human-readable console output | ✅ v0.1 |
| Record & replay sessions | ✅ v0.1 |
| Library mode (`wrap()`) | ✅ v0.1 |
| Infrastructure faults (rate-limit, latency, error, timeout) | ✅ v0.1 |
| Prompt jitters (noise, contradict, dilute, rephrase) | ✅ v0.1 |
| Hook pipeline (before_request, after_response, on_error) | ✅ v0.1 |
| SSE streaming support (reassemble + re-stream) | ✅ v0.1 |
| Token counting (tiktoken) | coming |
| Session inspector (`shear inspect`) | coming |
| Response simulation (`shear simulate`) | coming |
| Tool call jitters | coming |
| Chunk-level stream mutation | coming |

## CLI Reference

```bash
shear proxy                              # Zero-config proxy on port 9800
shear proxy --port 8080                  # Custom port
shear proxy --upstream https://api.openai.com/v1  # Explicit upstream
shear proxy --hooks my_hooks.py          # Custom hook file
shear proxy --verbose                    # Full payloads
shear proxy --json                       # Machine-readable output
shear proxy --fault rate-limit:0.3       # 30% rate limiting
shear proxy --fault latency:500ms        # Add 500ms latency
shear proxy --fault error:503:0.1        # 10% 503 errors
shear proxy --fault timeout:10s          # Timeout after 10s
shear proxy --jitter noise:0.1           # 10% word-level typos
shear proxy --jitter contradict          # Contradict system prompt
shear proxy --jitter dilute:5            # Pad with 5 irrelevant turns
shear proxy --jitter rephrase            # Reword system prompt
shear record -o session.json             # Record traffic
shear replay -i session.json             # Replay from file
shear proxy --timeout 30s                # Upstream timeout
shear proxy --verbose                    # Compact + jitter diffs
shear proxy --debug                      # Full JSON payloads
```

## Hook API

```python
from windtunnel_shear import Hook
from windtunnel_shear.core.models import InterceptedRequest, InterceptedResponse

@Hook.before_request(name="add_system_prompt")
def add_system_prompt(req: InterceptedRequest) -> InterceptedRequest:
    req.messages.insert(0, {"role": "system", "content": "Be concise."})
    return req

@Hook.after_response(name="log_usage")
def log_usage(req: InterceptedRequest, resp: InterceptedResponse) -> InterceptedResponse:
    print(f"Tokens used: {resp.usage}")
    return resp
```

## Part of Wind Tunnel

Shear is the first building block of the [Wind Tunnel](https://github.com/MothLabs-org) family of LLM testing tools by [Butterfly Labs](https://www.mothlabs.org).

## License

[MIT](LICENSE)
