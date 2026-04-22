"""CLI demo for the HMAF agentic runtime.

Boots an :class:`~i3.huawei.agentic_core_runtime.HMAFAgentRuntime`,
feeds it five canned HMAF intents, prints the structured responses,
and exits.  Demonstrates the end-to-end agentic flow -- from intent to
plan to execution to telemetry -- without requiring any external HMAF
runtime, network access, or trained checkpoint.

Usage::

    python scripts/run_hmaf_runtime_demo.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Allow running the script directly from a fresh checkout.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from i3.huawei.agentic_core_runtime import (  # noqa: E402  (sys.path mutation)
    HMAFAgentRuntime,
    HMAFIntent,
    HMAFResponse,
)


logger = logging.getLogger("run_hmaf_runtime_demo")


_CANNED_INTENTS: list[HMAFIntent] = [
    HMAFIntent(
        name="get_user_adaptation",
        parameters={"user_id": "demo_user"},
        source_device="phone",
    ),
    HMAFIntent(
        name="summarise_session",
        parameters={"turns": 12, "avg_engagement": 0.82},
        source_device="phone",
    ),
    HMAFIntent(
        name="translate",
        parameters={"target_language": "fr", "length_in": 42},
        source_device="ai_glasses",
    ),
    HMAFIntent(
        name="route_recommendation",
        parameters={
            "prefer_cloud": False,
            "sensitive_topic": True,
        },
        source_device="watch",
    ),
    HMAFIntent(
        name="explain_adaptation",
        parameters={"dimensions_requested": 4},
        source_device="phone",
    ),
]


def _response_to_printable(resp: HMAFResponse) -> dict[str, Any]:
    """Render a response as a pretty-printable dict.

    Args:
        resp: The :class:`HMAFResponse` to render.

    Returns:
        A dict ready for :func:`json.dumps`.
    """
    data: dict[str, Any] = resp.model_dump()
    data["latency_ms"] = round(float(data["latency_ms"]), 3)
    return data


async def _run() -> int:
    """Run the demo.

    Returns:
        Shell exit code (0 = success).
    """
    runtime = HMAFAgentRuntime()
    await runtime.start()
    try:
        for idx, intent in enumerate(_CANNED_INTENTS, start=1):
            response = await runtime.plan_and_execute(intent)
            printable = _response_to_printable(response)
            print(f"--- Intent {idx}: {intent.name} ---")
            print(json.dumps(printable, indent=2))
            print()
    finally:
        await runtime.stop()
    return 0


def main() -> int:
    """CLI entry point.

    Returns:
        The shell exit code from the async run.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
