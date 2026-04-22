"""Translate provider-neutral ``CompletionRequest`` to provider-specific shapes.

Each provider family expects a subtly different message format:

- **Anthropic**: ``{system: str, messages: [{role, content}]}`` with
  the system prompt in a *separate* top-level field.
- **OpenAI / Azure / LiteLLM / OpenRouter / PanGu**: flat
  ``messages`` list with a leading ``{"role": "system", ...}`` turn.
- **Google Gemini**: ``contents: [{role, parts: [{text}]}]`` with
  ``role`` restricted to ``"user"`` / ``"model"``; system prompt is
  injected as a leading user turn.
- **Cohere**: ``preamble`` + ``chat_history`` + ``message``.
- **Bedrock**: body format depends on the underlying model family
  (``anthropic.*``, ``amazon.titan-*``, ``meta.llama3-*``, ``mistral.*``).

Most adapters inline their translation in their own module for
locality.  This file exposes the ones that are more involved -- the
Bedrock per-family dispatch, and a set of thin helpers that tests can
exercise independently.

Tool / function-call normalisation
----------------------------------
I3 does not currently emit tool-use requests.  For providers that
*require* a ``tools: []`` field to be present, the helpers below
return an empty list; for providers that interpret its absence as
"no tools", the field is simply omitted.
"""

from __future__ import annotations

import logging
from typing import Any

from i3.cloud.providers.base import ChatMessage, CompletionRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def anthropic_payload(
    request: CompletionRequest, model: str, default_max: int
) -> dict[str, Any]:
    """Build an Anthropic Messages API payload from ``request``."""
    system = request.system or ""
    if not system:
        for m in request.messages:
            if m.role == "system":
                system = m.content
                break
    messages = [
        {"role": m.role, "content": m.content}
        for m in request.messages
        if m.role != "system"
    ]
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_tokens or default_max,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if request.stop:
        payload["stop_sequences"] = request.stop
    payload["temperature"] = request.temperature
    return payload


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


def openai_messages(request: CompletionRequest) -> list[dict[str, str]]:
    """Return the flat OpenAI-style messages list for ``request``."""
    msgs: list[dict[str, str]] = []
    if request.system:
        msgs.append({"role": "system", "content": request.system})
    for m in request.messages:
        msgs.append({"role": m.role, "content": m.content})
    return msgs


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


def google_contents(
    request: CompletionRequest,
) -> list[dict[str, Any]]:
    """Build Gemini's ``contents`` list from ``request``."""
    contents: list[dict[str, Any]] = []
    if request.system:
        contents.append(
            {"role": "user", "parts": [{"text": request.system}]}
        )
        contents.append(
            {"role": "model", "parts": [{"text": "Understood."}]}
        )
    for m in request.messages:
        role = "model" if m.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m.content}]})
    return contents


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------


def cohere_parts(
    request: CompletionRequest,
) -> tuple[str, list[dict[str, str]], str]:
    """Return ``(preamble, chat_history, message)`` for Cohere chat."""
    preamble = request.system or ""
    non_system: list[ChatMessage] = [
        m for m in request.messages if m.role != "system"
    ]
    if not non_system:
        return preamble, [], ""
    last = non_system[-1]
    if last.role == "user":
        current = last.content
        prior = non_system[:-1]
    else:
        current = ""
        prior = non_system
    role_map = {"user": "USER", "assistant": "CHATBOT"}
    history = [
        {"role": role_map.get(m.role, "USER"), "message": m.content}
        for m in prior
    ]
    return preamble, history, current


# ---------------------------------------------------------------------------
# Bedrock -- per-family dispatch
# ---------------------------------------------------------------------------


def _bedrock_anthropic_body(
    request: CompletionRequest, default_max: int
) -> dict[str, Any]:
    system = request.system or ""
    if not system:
        for m in request.messages:
            if m.role == "system":
                system = m.content
                break
    messages = [
        {"role": m.role, "content": m.content}
        for m in request.messages
        if m.role != "system"
    ]
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": request.max_tokens or default_max,
        "messages": messages,
        "temperature": request.temperature,
    }
    if system:
        body["system"] = system
    if request.stop:
        body["stop_sequences"] = request.stop
    return body


def _bedrock_titan_body(
    request: CompletionRequest, default_max: int
) -> dict[str, Any]:
    # Titan uses a single ``inputText`` prompt.  Collapse the
    # conversation into a simple transcript.
    lines: list[str] = []
    if request.system:
        lines.append(f"System: {request.system}")
    for m in request.messages:
        if m.role == "user":
            lines.append(f"User: {m.content}")
        elif m.role == "assistant":
            lines.append(f"Assistant: {m.content}")
    lines.append("Assistant:")
    return {
        "inputText": "\n".join(lines),
        "textGenerationConfig": {
            "maxTokenCount": request.max_tokens or default_max,
            "temperature": request.temperature,
            "stopSequences": request.stop or [],
        },
    }


def _bedrock_llama_body(
    request: CompletionRequest, default_max: int
) -> dict[str, Any]:
    # Llama 3 / 3.3 on Bedrock uses the Meta-chat prompt format.
    parts: list[str] = ["<|begin_of_text|>"]
    if request.system:
        parts.append(
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{request.system}<|eot_id|>"
        )
    for m in request.messages:
        if m.role == "system":
            # system already emitted above; skip if duplicated.
            continue
        parts.append(
            f"<|start_header_id|>{m.role}<|end_header_id|>\n"
            f"{m.content}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return {
        "prompt": "".join(parts),
        "max_gen_len": request.max_tokens or default_max,
        "temperature": request.temperature,
    }


def _bedrock_mistral_body(
    request: CompletionRequest, default_max: int
) -> dict[str, Any]:
    # Mistral on Bedrock expects an INST-wrapped single prompt.
    lines: list[str] = []
    if request.system:
        lines.append(f"<<SYS>>\n{request.system}\n<</SYS>>\n")
    for m in request.messages:
        if m.role == "user":
            lines.append(f"[INST] {m.content} [/INST]")
        elif m.role == "assistant":
            lines.append(m.content)
    body: dict[str, Any] = {
        "prompt": "\n".join(lines),
        "max_tokens": request.max_tokens or default_max,
        "temperature": request.temperature,
    }
    if request.stop:
        body["stop"] = request.stop
    return body


def bedrock_body(
    model: str, request: CompletionRequest, default_max: int
) -> dict[str, Any]:
    """Build the Bedrock request body for ``model``.

    Dispatches on the leading ``<provider>.`` prefix of the model id.
    """
    prefix = model.split(".", 1)[0].lower()
    if prefix == "anthropic":
        return _bedrock_anthropic_body(request, default_max)
    if prefix == "amazon":
        return _bedrock_titan_body(request, default_max)
    if prefix == "meta":
        return _bedrock_llama_body(request, default_max)
    if prefix == "mistral":
        return _bedrock_mistral_body(request, default_max)
    # Fallback: anthropic-style works for Claude-on-Bedrock and many
    # newer models (Cohere-on-Bedrock accepts similar shapes).
    logger.debug(
        "Unknown Bedrock model prefix %r; falling back to anthropic body",
        prefix,
    )
    return _bedrock_anthropic_body(request, default_max)


def parse_bedrock_response(
    model: str, data: dict[str, Any]
) -> tuple[str, int, int, str]:
    """Extract ``(text, prompt_tokens, completion_tokens, finish)``.

    Dispatch on the model prefix mirrors :func:`bedrock_body`.
    """
    prefix = model.split(".", 1)[0].lower()
    if prefix == "anthropic":
        blocks = data.get("content") or []
        text = "".join(
            b.get("text", "")
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        usage = data.get("usage", {}) or {}
        return (
            text,
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            str(data.get("stop_reason", "stop") or "stop"),
        )
    if prefix == "amazon":
        results = data.get("results") or []
        first = results[0] if results else {}
        return (
            str(first.get("outputText", "") or ""),
            int(data.get("inputTextTokenCount", 0) or 0),
            int(first.get("tokenCount", 0) or 0),
            str(first.get("completionReason", "FINISH") or "FINISH").lower(),
        )
    if prefix == "meta":
        return (
            str(data.get("generation", "") or ""),
            int(data.get("prompt_token_count", 0) or 0),
            int(data.get("generation_token_count", 0) or 0),
            str(data.get("stop_reason", "stop") or "stop"),
        )
    if prefix == "mistral":
        outputs = data.get("outputs") or []
        first = outputs[0] if outputs else {}
        return (
            str(first.get("text", "") or ""),
            0,  # Mistral-on-Bedrock does not return token counts.
            0,
            str(first.get("stop_reason", "stop") or "stop"),
        )
    # Unknown: try anthropic layout.
    return ("", 0, 0, "unknown")


__all__ = [
    "anthropic_payload",
    "bedrock_body",
    "cohere_parts",
    "google_contents",
    "openai_messages",
    "parse_bedrock_response",
]
