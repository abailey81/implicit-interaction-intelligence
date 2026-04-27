"""Qwen3.5-2B + LoRA inference for HMI command-intent.

Iter 51 (2026-04-27).  Loads the LoRA adapter trained by
``training/train_intent_lora.py`` on top of the base Qwen3.5-2B model
and exposes a synchronous ``parse(utterance) -> IntentResult`` API.

The base model is loaded lazily on first ``parse`` call so importing
the module is cheap.  Subsequent calls reuse the cached
``(tokenizer, model)`` pair.

Privacy: inference runs entirely on the host GPU/CPU.  No network
calls.  The structured output is validated against
:data:`i3.intent.types.SUPPORTED_ACTIONS` and per-action
:data:`i3.intent.types.ACTION_SLOTS`.

Why not always-loaded:
    * Qwen3.5-2B + LoRA at bf16 is ~5 GB.  Loading on every server
      startup slows cold-boot to ~20 s and reserves VRAM that would
      otherwise be available for the from-scratch SLM.
    * The intent path is opt-in via ``/api/intent`` — most chat turns
      don't need it.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from i3.intent.types import (
    ACTION_SLOTS,
    SUPPORTED_ACTIONS,
    IntentResult,
)

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ADAPTER = REPO_ROOT / "checkpoints" / "intent_lora" / "qwen3.5-2b"
BEST_ADAPTER = REPO_ROOT / "checkpoints" / "intent_lora" / "qwen3.5-2b_best"


_PROMPT_TEMPLATE = (
    "You are an HMI command parser. Convert the user's voice "
    "utterance into a JSON object {{\"action\": ..., \"params\": "
    "{{...}}}}. Reply with ONLY the JSON, no prose.\n\n"
    "Utterance: {utterance}\nJSON:"
)


def _strip_to_json(text: str) -> str | None:
    """Extract the first balanced ``{...}`` JSON object from *text*.

    The model is trained to emit ONLY the JSON, but some generations
    leak prose (especially with the reasoning-distilled bases).  This
    extracts the first balanced top-level brace pair.
    """
    if not text:
        return None
    text = text.strip()
    # Fast-path: already starts with `{`.
    if text.startswith("{"):
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[: i + 1]
        return None
    # Find the first `{` and parse from there.
    idx = text.find("{")
    if idx < 0:
        return None
    return _strip_to_json(text[idx:])


class QwenIntentParser:
    """LoRA-fine-tuned Qwen3.5-2B intent parser (lazy-loaded)."""

    def __init__(
        self,
        adapter_dir: Path | str | None = None,
        base_model: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 64,
    ) -> None:
        # Prefer the best-checkpoint adapter when present.
        if adapter_dir is None:
            if BEST_ADAPTER.exists():
                adapter_dir = BEST_ADAPTER
            else:
                adapter_dir = DEFAULT_ADAPTER
        self.adapter_dir = Path(adapter_dir)
        self.base_model_id = base_model
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        """Lazy-load model + adapter.  Returns False on any failure
        (the engine should fall through to a fallback path)."""
        if self._loaded:
            return self._model is not None
        with self._lock:
            if self._loaded:
                return self._model is not None
            self._loaded = True
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                from peft import PeftModel
            except ImportError as exc:
                logger.warning(
                    "QwenIntentParser unavailable — missing dep: %s", exc,
                )
                return False
            if not self.adapter_dir.exists():
                logger.warning(
                    "QwenIntentParser: adapter dir not found at %s — "
                    "skipping load (run training/train_intent_lora.py first)",
                    self.adapter_dir,
                )
                return False
            # Read base model id from adapter_config.json.
            cfg_path = self.adapter_dir / "adapter_config.json"
            base_id = self.base_model_id
            if cfg_path.exists() and base_id is None:
                try:
                    with cfg_path.open("r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    base_id = cfg.get("base_model_name_or_path")
                except Exception:  # pragma: no cover
                    pass
            if not base_id:
                base_id = "Qwen/Qwen2.5-1.5B-Instruct"  # safe fallback
            try:
                logger.info(
                    "QwenIntentParser: loading base=%s + adapter=%s",
                    base_id, self.adapter_dir,
                )
                tok = AutoTokenizer.from_pretrained(
                    self.adapter_dir, trust_remote_code=True,
                )
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                device_arg = self._device or (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                base = AutoModelForCausalLM.from_pretrained(
                    base_id,
                    torch_dtype=torch.bfloat16
                    if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    device_map={"": device_arg},
                )
                model = PeftModel.from_pretrained(
                    base, str(self.adapter_dir),
                )
                model.eval()
                self._tokenizer = tok
                self._model = model
                logger.info("QwenIntentParser ready (device=%s)", device_arg)
                return True
            except Exception as exc:
                logger.exception(
                    "QwenIntentParser load failed: %s", exc,
                )
                self._model = None
                self._tokenizer = None
                return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def parse(self, utterance: str) -> IntentResult:
        """Parse *utterance* into structured intent.

        Always returns an :class:`IntentResult` — never raises.  When
        the adapter isn't loaded (training never ran, missing libs),
        returns a result with ``backend="qwen-lora"``,
        ``valid_json=False`` and an ``error`` string explaining the
        skip reason.
        """
        utterance = (utterance or "").strip()
        result = IntentResult(
            raw_input=utterance,
            raw_output="",
            parsed=None,
            valid_json=False,
            valid_action=False,
            valid_slots=False,
            backend="qwen-lora",
        )
        if not utterance:
            result.error = "empty utterance"
            return result
        if not self._ensure_loaded():
            result.error = "adapter not loaded"
            return result
        import torch
        prompt = _PROMPT_TEMPLATE.format(utterance=utterance)
        t0 = time.time()
        try:
            inputs = self._tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False,
            ).to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            full = self._tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
        except Exception as exc:
            result.error = f"generation failed: {exc}"
            return result
        result.raw_output = full
        result.latency_ms = (time.time() - t0) * 1000
        # Parse JSON.
        json_str = _strip_to_json(full)
        if not json_str:
            result.error = "no JSON object found"
            return result
        try:
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                result.error = "JSON not an object"
                return result
        except json.JSONDecodeError as exc:
            result.error = f"json parse failed: {exc}"
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
        # Slot validation.
        params = parsed.get("params", {})
        if not isinstance(params, dict):
            params = {}
            result.error = "params not a dict"
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
