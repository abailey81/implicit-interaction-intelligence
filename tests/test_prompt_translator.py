"""Tests for :mod:`i3.cloud.prompt_translator`."""

from __future__ import annotations

import pytest

from i3.cloud.prompt_translator import (
    anthropic_payload,
    bedrock_body,
    cohere_parts,
    google_contents,
    openai_messages,
    parse_bedrock_response,
)
from i3.cloud.providers.base import ChatMessage, CompletionRequest


def _req(
    *, system: str | None = "You are helpful.", msgs: list[ChatMessage] | None = None
) -> CompletionRequest:
    return CompletionRequest(
        system=system,
        messages=msgs
        or [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi!"),
            ChatMessage(role="user", content="How are you?"),
        ],
        max_tokens=64,
        temperature=0.7,
    )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_payload_splits_system() -> None:
    payload = anthropic_payload(_req(), model="claude-sonnet-4-5", default_max=512)
    assert payload["system"] == "You are helpful."
    assert all(m["role"] != "system" for m in payload["messages"])
    assert payload["model"] == "claude-sonnet-4-5"
    assert payload["max_tokens"] == 64


def test_anthropic_payload_extracts_system_from_messages() -> None:
    req = _req(
        system=None,
        msgs=[
            ChatMessage(role="system", content="Be formal."),
            ChatMessage(role="user", content="Hi"),
        ],
    )
    payload = anthropic_payload(req, model="x", default_max=512)
    assert payload["system"] == "Be formal."
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]


def test_anthropic_payload_handles_no_system() -> None:
    req = CompletionRequest(
        system=None,
        messages=[ChatMessage(role="user", content="yo")],
        max_tokens=0,
        temperature=0.0,
    )
    payload = anthropic_payload(req, model="x", default_max=100)
    assert "system" not in payload
    assert payload["max_tokens"] == 100  # falls back to default


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


def test_openai_messages_prepends_system() -> None:
    msgs = openai_messages(_req())
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert len(msgs) == 4


def test_openai_messages_omits_system_when_absent() -> None:
    req = CompletionRequest(
        system=None,
        messages=[ChatMessage(role="user", content="hi")],
    )
    msgs = openai_messages(req)
    assert msgs == [{"role": "user", "content": "hi"}]


def test_openai_messages_roundtrip_roles() -> None:
    msgs = openai_messages(_req())
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


def test_google_contents_injects_system_as_user_turn() -> None:
    contents = google_contents(_req())
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "You are helpful."
    assert contents[1]["role"] == "model"  # synthetic ack


def test_google_contents_maps_assistant_to_model() -> None:
    contents = google_contents(_req())
    # After the system injection (2 turns), the real messages follow.
    real = contents[2:]
    assert real[0]["role"] == "user"
    assert real[1]["role"] == "model"
    assert real[2]["role"] == "user"


def test_google_contents_handles_no_system() -> None:
    req = CompletionRequest(
        system=None,
        messages=[ChatMessage(role="user", content="hi")],
    )
    contents = google_contents(req)
    assert len(contents) == 1
    assert contents[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------


def test_cohere_parts_splits_preamble_history_message() -> None:
    preamble, history, message = cohere_parts(_req())
    assert preamble == "You are helpful."
    # First user + assistant turns go into history, last user is message.
    assert len(history) == 2
    assert history[0]["role"] == "USER"
    assert history[1]["role"] == "CHATBOT"
    assert message == "How are you?"


def test_cohere_parts_empty_messages() -> None:
    req = CompletionRequest(system="s", messages=[])
    preamble, history, message = cohere_parts(req)
    assert preamble == "s"
    assert history == []
    assert message == ""


def test_cohere_parts_trailing_assistant() -> None:
    req = CompletionRequest(
        system=None,
        messages=[
            ChatMessage(role="user", content="a"),
            ChatMessage(role="assistant", content="b"),
        ],
    )
    _, history, message = cohere_parts(req)
    # No trailing user; current message is empty, both turns enter history.
    assert message == ""
    assert len(history) == 2


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------


def test_bedrock_body_anthropic_family() -> None:
    body = bedrock_body("anthropic.claude-sonnet-4-5", _req(), default_max=512)
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["system"] == "You are helpful."
    assert body["max_tokens"] == 64
    assert all(m["role"] != "system" for m in body["messages"])


def test_bedrock_body_amazon_family_uses_input_text() -> None:
    body = bedrock_body("amazon.titan-text-premier-v1:0", _req(), 512)
    assert "inputText" in body
    assert "User:" in body["inputText"]
    assert body["textGenerationConfig"]["maxTokenCount"] == 64


def test_bedrock_body_meta_family_uses_prompt() -> None:
    body = bedrock_body(
        "meta.llama3-3-70b-instruct-v1:0", _req(), 512
    )
    assert "<|begin_of_text|>" in body["prompt"]
    assert "<|start_header_id|>system<|end_header_id|>" in body["prompt"]
    assert body["max_gen_len"] == 64


def test_bedrock_body_mistral_family_uses_inst_markers() -> None:
    body = bedrock_body("mistral.mistral-large-2407-v1:0", _req(), 512)
    assert "[INST]" in body["prompt"]
    assert "<<SYS>>" in body["prompt"]


def test_bedrock_body_unknown_prefix_falls_back_to_anthropic() -> None:
    body = bedrock_body("unknownfamily.model-1", _req(), 512)
    assert body["anthropic_version"] == "bedrock-2023-05-31"


def test_parse_bedrock_response_anthropic_shape() -> None:
    data = {
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 10, "output_tokens": 3},
        "stop_reason": "end_turn",
    }
    text, p, c, finish = parse_bedrock_response("anthropic.claude", data)
    assert text == "hello"
    assert p == 10
    assert c == 3
    assert finish == "end_turn"


def test_parse_bedrock_response_amazon_shape() -> None:
    data = {
        "inputTextTokenCount": 12,
        "results": [{"outputText": "hi", "tokenCount": 2, "completionReason": "FINISH"}],
    }
    text, p, c, finish = parse_bedrock_response("amazon.titan", data)
    assert text == "hi"
    assert p == 12
    assert c == 2
    assert finish == "finish"


def test_parse_bedrock_response_meta_shape() -> None:
    data = {
        "generation": "llama says hi",
        "prompt_token_count": 5,
        "generation_token_count": 4,
        "stop_reason": "stop",
    }
    text, p, c, finish = parse_bedrock_response("meta.llama3", data)
    assert text == "llama says hi"
    assert p == 5
    assert c == 4


def test_parse_bedrock_response_mistral_shape() -> None:
    data = {"outputs": [{"text": "bonjour", "stop_reason": "stop"}]}
    text, p, c, finish = parse_bedrock_response("mistral.foo", data)
    assert text == "bonjour"
    assert p == 0  # Mistral-on-Bedrock omits token counts.
