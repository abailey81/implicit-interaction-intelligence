"""HMI command-intent parsing via fine-tuned small LLM.

Iter 51 (2026-04-27).  Two interchangeable backends:
    * :class:`QwenIntentParser` — open-weight Qwen3.5-2B + LoRA
      adapter, runs entirely on-device.
    * :class:`GeminiIntentParser` — closed-weight Gemini 2.5 Flash
      via Vertex AI supervised tuning, hosted on Google Cloud.

Both expose :meth:`parse(utterance: str) -> IntentResult`.
"""
from i3.intent.types import IntentResult
from i3.intent.qwen_inference import QwenIntentParser

__all__ = ["IntentResult", "QwenIntentParser"]
