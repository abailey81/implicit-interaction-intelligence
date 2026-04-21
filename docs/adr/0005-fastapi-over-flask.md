# ADR-0005 — FastAPI over Flask

- **Status**: Accepted
- **Date**: 2026-01-22
- **Deciders**: Tamer Atesyakar
- **Technical area**: server

## Context and problem statement { #context }

The I³ server surfaces both a REST API and a real-time WebSocket channel.
The REST layer is request/response, JSON-in, JSON-out. The WebSocket
layer streams keystroke events and returns pipeline results. Both sit in
front of the same `async` pipeline (`i3/pipeline/engine.py`), which is
natively coroutine-based.

The framework needs to:

- Speak ASGI (so `websockets` and the async pipeline compose cleanly).
- Validate inbound payloads against typed models, enforced before handler
  entry.
- Emit OpenAPI for tooling and Swagger-UI at `/docs`.
- Have a small, legible middleware story for rate limiting and CORS.

## Decision drivers { #drivers }

- Native `async` handlers and WebSocket support.
- Pydantic-based validation at the edge.
- OpenAPI schema out of the box.
- Mature ecosystem, actively maintained.

## Considered options { #options }

1. **FastAPI** (Starlette underneath).
2. **Flask** + **Flask-SocketIO** + **webargs** + hand-rolled OpenAPI.
3. **aiohttp** or **Sanic**.

## Decision outcome { #outcome }

> **Chosen option**: Option 1 — FastAPI. It is the shortest path to
> `async` REST + WebSocket + Pydantic validation + OpenAPI + a maintained
> ecosystem. Flask would require stitching three or four packages for
> parity, and aiohttp lacks the Pydantic-native validation flow.

### Consequences — positive { #pos }

- Path regex validation (`Path(..., pattern=…)`) is a first-class
  feature — we use it to anchor `user_id` to `^[a-zA-Z0-9_-]{1,64}$`
  pre-handler (see `server/routes.py`).
- Swagger UI at `/docs` for free during development.
- Starlette's WebSocket primitives play natively with our `async`
  pipeline.
- Active security patch cadence (closed the 2024-47874 multipart DoS).

### Consequences — negative { #neg }

- Pydantic v2 migration pinned us to FastAPI ≥ 0.115; older extensions
  had to be re-written or dropped. *Mitigation*: see
  [ADR-0010 — Pydantic v2 config](0010-pydantic-v2-config.md).
- FastAPI's dependency-injection system is magical; new contributors
  need a 5-minute primer. *Mitigation*: `CONTRIBUTING.md` points at the
  canonical docs.
- WebSocket is supplied by Starlette/ASGI, not by FastAPI itself, so
  some security controls must be re-implemented in the handler (see the
  Origin allow-list in `server/websocket.py`). *Mitigation*: we document
  this explicitly.

## Pros and cons of the alternatives { #alternatives }

### Option 2 — Flask + extensions { #opt-2 }

- Yes Broadest third-party ecosystem.
- No Sync-first; async support is a retrofit.
- No Extension matrix is fragile and historically drifts with Flask
  majors.
- No No native OpenAPI — bolt-ons tend to lag framework updates.

### Option 3 — aiohttp / Sanic { #opt-3 }

- Yes Async-native, lower overhead per request.
- No No first-class Pydantic integration; validation must be hand-written.
- No Smaller ecosystem for things like CORS/rate-limiting middleware.
- No Less production-audit-friendly OpenAPI story.

## References { #refs }

- [REST API](../api/rest.md)
- [WebSocket API](../api/websocket.md)
- [SECURITY.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md)
