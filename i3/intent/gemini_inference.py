"""Gemini 2.5 Flash inference for HMI command-intent.

Iter 51 (2026-04-27).  The closed-weight cloud counterpart to
:class:`i3.intent.qwen_inference.QwenIntentParser`.

Loads the tuned-model name from
``checkpoints/intent_gemini/tuning_result.json`` (written by
``training/train_intent_gemini.py``) and calls the AI Studio API.
Falls back to a polite "not configured" :class:`IntentResult` when
the API key is absent or the tuning hasn't run.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from i3.intent.qwen_inference import _strip_to_json
from i3.intent.types import (
    ACTION_SLOTS,
    SUPPORTED_ACTIONS,
    IntentResult,
)

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TUNING_RESULT = REPO_ROOT / "checkpoints" / "intent_gemini" / "tuning_result.json"


def _build_schema_block() -> str:
    """Render the canonical action+slot schema for the prompt."""
    lines = []
    for act in SUPPORTED_ACTIONS:
        slots = sorted(ACTION_SLOTS.get(act, set()))
        lines.append(
            f"  {act!r}: params={slots}" if slots else f"  {act!r}: params=[]"
        )
    return "\n".join(lines)


_PROMPT_TEMPLATE = (
    "You are an HMI command parser for a vehicle assistant. "
    "Convert the user's voice utterance into a JSON object "
    "{{\"action\": <one of the supported actions>, \"params\": "
    "{{...}}}}. Reply with ONLY the JSON, no prose.\n\n"
    "Supported actions (use these names EXACTLY; pick 'unsupported' "
    "if none apply):\n"
    + _build_schema_block() + "\n\n"
    "Slot rules:\n"
    "  * 'duration_seconds' is an integer count of seconds (e.g. 300 "
    "for five minutes).\n"
    "  * 'time' is a clock time string (e.g. '07:00', '6pm').\n"
    "  * 'video' is a boolean.\n"
    "  * Use only the slot names listed for the chosen action.\n\n"
    "Utterance: {utterance}\nJSON:"
)


# Per-action mapping of Gemini-returned slot names → canonical slot
# names defined in :mod:`i3.intent.types`.  Keys are case-insensitive.
# Whatever Gemini emits for a slot we don't recognise gets dropped (so
# the ``unexpected slot(s)`` rejection path still fires for genuine
# hallucinations rather than minor naming drift).
_SLOT_ALIASES: dict[str, dict[str, str]] = {
    "set_timer": {
        "duration": "duration_seconds",
        "duration_seconds": "duration_seconds",
        "duration_secs": "duration_seconds",
        "duration_minutes": "duration_seconds",
        "duration_min": "duration_seconds",
        "seconds": "duration_seconds",
        "minutes": "duration_seconds",
        "time": "duration_seconds",
        "length": "duration_seconds",
        "for": "duration_seconds",
    },
    "set_alarm": {
        "time": "time",
        "alarm_time": "time",
        "at": "time",
        "when": "time",
    },
    "send_message": {
        "recipient": "recipient",
        "to": "recipient",
        "name": "recipient",
        "contact": "recipient",
        "message": "message",
        "text": "message",
        "body": "message",
        "content": "message",
    },
    "play_music": {
        "genre": "genre",
        "type": "genre",
        "style": "genre",
        "artist": "artist",
        "singer": "artist",
        "band": "artist",
        "musician": "artist",
    },
    "navigate": {
        "location": "location",
        "destination": "location",
        "address": "location",
        "place": "location",
        "to": "location",
        "where": "location",
    },
    "weather_query": {
        "location": "location",
        "city": "location",
        "place": "location",
        "where": "location",
        "for": "location",
    },
    "call": {
        "recipient": "recipient",
        "name": "recipient",
        "contact": "recipient",
        "to": "recipient",
        "video": "video",
        "video_call": "video",
        "is_video": "video",
    },
    "set_reminder": {
        "task": "task",
        "what": "task",
        "activity": "task",
        "description": "task",
        "reminder": "task",
        "time": "time",
        "at": "time",
        "when": "when",
    },
    "control_device": {
        "device": "device",
        "device_name": "device",
        "appliance": "device",
        "state": "state",
        "status": "state",
        "on_off": "state",
        "value": "value",
        "level": "value",
        "amount": "value",
    },
    "cancel": {},
    "unsupported": {},
}


# Words that reliably indicate seconds/minutes/hours when Gemini gives
# us a free-form duration string (``"5 minutes"``, ``"1 hour"``, …).
_DURATION_UNITS: dict[str, int] = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
    "day": 86400, "days": 86400, "d": 86400,
}


def _coerce_duration_seconds(value: object) -> int | None:
    """Best-effort conversion of a Gemini ``duration`` value to seconds.

    Handles ``300``, ``"300"``, ``"5 minutes"``, ``"1 hour 30 mins"``,
    etc.  Returns ``None`` when nothing parseable is found, so the
    caller can drop the slot and the schema check will catch it.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is a subclass of int — refuse it explicitly.
        return None
    if isinstance(value, (int, float)):
        v = int(value)
        return v if v > 0 else None
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if not s:
        return None
    # Pure-number string ("300") → assume seconds.
    try:
        return int(s) if int(s) > 0 else None
    except ValueError:
        pass
    # Tokenise alternating number/unit pairs.
    tokens = re.findall(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?", s)
    if not tokens:
        return None
    total = 0.0
    for num, unit in tokens:
        try:
            n = float(num)
        except ValueError:
            continue
        u = (unit or "").lower().strip()
        # Default to minutes when no unit attached — the most common
        # natural-language phrasing ("set a 5 timer" -> 5 minutes).
        mult = _DURATION_UNITS.get(u, 60 if not u else 0)
        total += n * mult
    return int(total) if total > 0 else None


def _normalize_slots(action: str, params: dict) -> dict:
    """Map Gemini's slot names + values onto our canonical schema."""
    aliases = _SLOT_ALIASES.get(action, {})
    out: dict = {}
    for key, val in params.items():
        if not isinstance(key, str):
            continue
        canonical = aliases.get(key.lower(), key)
        # If we already have a value for this canonical slot, keep the
        # first one (deterministic).
        if canonical in out:
            continue
        out[canonical] = val

    # Per-slot value coercions (just the high-impact ones for now).
    if action == "set_timer" and "duration_seconds" in out:
        coerced = _coerce_duration_seconds(out["duration_seconds"])
        if coerced is None:
            del out["duration_seconds"]
        else:
            out["duration_seconds"] = coerced
    if action == "call" and "video" in out:
        v = out["video"]
        if isinstance(v, str):
            out["video"] = v.strip().lower() in {"true", "yes", "1", "video"}
        elif not isinstance(v, bool):
            out["video"] = bool(v)
    return out


class GeminiIntentParser:
    """Calls a tuned Gemini 2.5 Flash via the direct AI Studio API."""

    def __init__(
        self,
        api_key: str | None = None,
        tuned_model_name: str | None = None,
        fallback_model: str = "gemini-2.5-flash",
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.tuned_model_name = tuned_model_name
        self.fallback_model = fallback_model
        self._client = None
        self._model_handle = None
        self._configured = False

    def _ensure_configured(self) -> bool:
        if self._configured:
            return self._model_handle is not None
        self._configured = True
        if not self.api_key:
            logger.info(
                "GeminiIntentParser: GEMINI_API_KEY not set — "
                "intent route will skip the cloud comparison",
            )
            return False

        # Resolve the model name.
        resolved = self.tuned_model_name
        if not resolved and TUNING_RESULT.exists():
            try:
                with TUNING_RESULT.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                resolved = data.get("tuned_model_name")
            except Exception:
                pass
        if not resolved:
            resolved = self.fallback_model
            logger.info(
                "GeminiIntentParser: no tuned model id available — "
                "using base %s as a comparison reference", resolved,
            )

        # Prefer the new ``google-genai`` SDK (the supported route for
        # 2026); fall back to the deprecated ``google-generativeai``
        # only when the new one isn't installed.  Either path produces
        # a uniform ``self._model_handle`` interface that
        # :meth:`parse` calls into.
        try:
            from google import genai as new_genai
            self._client = new_genai.Client(api_key=self.api_key)
            self._sdk = "new"
            self._resolved_model = resolved
            self._model_handle = self._client  # placeholder; we use
                                                # client.models.generate_content
            logger.info(
                "GeminiIntentParser ready (model=%s, sdk=google-genai)",
                resolved,
            )
            return True
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("GeminiIntentParser (new SDK) init failed: %s", exc)
            # fall through to old SDK

        try:
            import google.generativeai as old_genai
        except ImportError:
            logger.info(
                "GeminiIntentParser: neither google-genai nor "
                "google-generativeai installed — skipping",
            )
            return False
        try:
            old_genai.configure(api_key=self.api_key)
            self._model_handle = old_genai.GenerativeModel(model_name=resolved)
            self._client = old_genai
            self._sdk = "old"
            self._resolved_model = resolved
            logger.info(
                "GeminiIntentParser ready (model=%s, sdk=google-generativeai-deprecated)",
                resolved,
            )
            return True
        except Exception as exc:
            logger.warning("GeminiIntentParser (old SDK) init failed: %s", exc)
            return False

    def parse(self, utterance: str) -> IntentResult:
        utterance = (utterance or "").strip()
        result = IntentResult(
            raw_input=utterance, raw_output="",
            parsed=None,
            valid_json=False, valid_action=False, valid_slots=False,
            backend="gemini-aistudio",
        )
        if not utterance:
            result.error = "empty utterance"
            return result
        if not self._ensure_configured():
            result.error = "gemini not configured"
            return result
        prompt = _PROMPT_TEMPLATE.format(utterance=utterance)
        t0 = time.time()
        try:
            if getattr(self, "_sdk", "old") == "new":
                # google-genai (the supported SDK).  Different call
                # surface than the deprecated package.
                from google.genai import types as gtypes
                # Gemini 2.5 Flash defaults to "thinking-mode" which
                # consumes the first N tokens internally.  For a tight
                # JSON-only intent task we (a) raise the budget so the
                # final JSON has room, and (b) disable thinking
                # explicitly.  The ``thinking_config`` field is
                # backwards-compatible with older Gemini versions
                # (they ignore it).
                cfg_kwargs: dict = dict(
                    temperature=0.0,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                )
                try:
                    cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(
                        thinking_budget=0,
                    )
                except Exception:
                    # Older google-genai didn't ship ThinkingConfig.
                    pass
                response = self._client.models.generate_content(
                    model=self._resolved_model,
                    contents=prompt,
                    config=gtypes.GenerateContentConfig(**cfg_kwargs),
                )
                # ``response.text`` is a property that raises when the
                # candidate has no text part (safety filter, MAX_TOKENS
                # before content, etc.).  Catch it explicitly so we
                # surface a meaningful ``error`` instead of a stray
                # ValueError.
                full = ""
                try:
                    full = (response.text or "").strip()
                except Exception:
                    if response.candidates:
                        cand = response.candidates[0]
                        finish = getattr(cand, "finish_reason", None)
                        result.error = (
                            f"gemini empty response (finish_reason={finish})"
                        )
                    else:
                        result.error = "gemini empty response"
                    return result
            else:
                # Deprecated google-generativeai path (legacy fallback).
                response = self._model_handle.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 64,
                        "response_mime_type": "application/json",
                    },
                )
                full = (response.text or "").strip()
        except Exception as exc:
            result.error = (
                f"gemini call failed: {type(exc).__name__}: {exc}"
            )
            return result
        result.raw_output = full
        result.latency_ms = (time.time() - t0) * 1000
        json_str = _strip_to_json(full) or full
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            result.error = f"json parse failed: {exc}"
            return result
        if not isinstance(parsed, dict):
            result.error = "JSON not an object"
            return result
        result.parsed = parsed
        result.valid_json = True
        action = parsed.get("action")
        if isinstance(action, str) and action in SUPPORTED_ACTIONS:
            result.action = action
            result.valid_action = True
        else:
            result.error = f"unknown action: {action!r}"
            return result
        params = parsed.get("params", {})
        if not isinstance(params, dict):
            params = {}
        # Iter 51 Phase 5: Gemini happily returns natural-language slots
        # (e.g. ``{"duration": "5 minutes"}``) that don't match our
        # canonical schema (``duration_seconds: 300``).  Normalise them
        # here so a Gemini-backed parse can still satisfy
        # ``valid_slots`` and short-circuit the cascade.
        params = _normalize_slots(action, params)
        result.params = params
        allowed = ACTION_SLOTS.get(action, set())
        extra = set(params.keys()) - allowed
        if extra:
            result.error = (
                f"unexpected slot(s) for {action}: {sorted(extra)}"
            )
            return result
        result.valid_slots = True
        return result
