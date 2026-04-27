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


_PROMPT_TEMPLATE = (
    "You are an HMI command parser. Convert the user's voice "
    "utterance into a JSON object {{\"action\": ..., \"params\": "
    "{{...}}}}. Reply with ONLY the JSON, no prose.\n\n"
    "Utterance: {utterance}\nJSON:"
)


class GeminiIntentParser:
    """Calls a tuned Gemini 2.5 Flash via the direct AI Studio API."""

    def __init__(
        self,
        api_key: str | None = None,
        tuned_model_name: str | None = None,
        fallback_model: str = "models/gemini-2.5-flash-001",
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
        try:
            import google.generativeai as genai
        except ImportError:
            logger.info(
                "GeminiIntentParser: google-generativeai not installed — "
                "skipping",
            )
            return False
        genai.configure(api_key=self.api_key)
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
        try:
            self._model_handle = genai.GenerativeModel(model_name=resolved)
            self._client = genai
            logger.info("GeminiIntentParser ready (model=%s)", resolved)
            return True
        except Exception as exc:
            logger.warning("GeminiIntentParser init failed: %s", exc)
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
            result.error = f"gemini call failed: {exc}"
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
