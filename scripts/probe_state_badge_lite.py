"""Lightweight probe for the Live State Badge + Accessibility Mode features.

Drives the state classifier and accessibility controller directly,
plus the reasoning-trace builder, without booting the full pipeline.
The integration with the engine has separate unit-test coverage in
``tests/test_state_badge_and_accessibility.py``; this probe walks
the same 13-turn scenario the brief asked for and prints the
state-badge label + accessibility flags + reasoning-trace sentences
for each turn so the operator can read off the closed-loop behaviour.

Run::

    python scripts/probe_state_badge_lite.py

Total runtime is ~50 ms — no torch, no SLM, no retrieval index.
"""

from __future__ import annotations

import json
import sys

from i3.affect.accessibility_mode import AccessibilityController
from i3.affect.state_classifier import classify_user_state
from i3.explain.reasoning_trace import build_reasoning_trace


# ---------------------------------------------------------------------------
# Per-turn driver
# ---------------------------------------------------------------------------


def _run_turn(
    *,
    turn_no: int,
    phase: str,
    controller: AccessibilityController,
    user_id: str,
    session_id: str,
    composition_ms: float,
    edit_count: int,
    iki_mean: float,
    iki_std: float,
    cognitive_load: float,
    accessibility_axis: float,
    formality: float,
    engagement: float,
    deviation: float,
    msg_count: int,
    baseline_established: bool,
) -> dict:
    """Run the classifier + controller for one synthetic turn."""
    adapt_dict = {
        "cognitive_load": cognitive_load,
        "accessibility": accessibility_axis,
        "formality": formality,
        "verbosity": 0.5,
        "emotional_tone": 0.0,
    }
    label = classify_user_state(
        adaptation=adapt_dict,
        composition_time_ms=composition_ms,
        edit_count=edit_count,
        iki_mean=iki_mean,
        iki_std=iki_std,
        engagement_score=engagement,
        deviation_from_baseline=deviation,
        baseline_established=baseline_established,
        messages_in_session=msg_count,
    )
    access = controller.observe(
        user_id, session_id,
        edit_count=float(edit_count),
        iki_mean=iki_mean,
        iki_std=iki_std,
        cognitive_load=cognitive_load,
        accessibility_axis=accessibility_axis,
    )

    label_dict = label.to_dict()
    access_dict = access.to_dict()

    # Build reasoning trace + extract the two sentences the brief asks for.
    trace = build_reasoning_trace(
        keystroke_metrics={
            "composition_time_ms": composition_ms,
            "edit_count": edit_count,
            "pause_before_send_ms": 200,
            "keystroke_timings": [iki_mean] * 12 + [iki_mean + iki_std] * 4,
        },
        adaptation=adapt_dict,
        adaptation_changes=[],
        engagement_score=engagement,
        deviation_from_baseline=deviation,
        messages_in_session=msg_count,
        baseline_established=baseline_established,
        routing_confidence={"local_slm": 1.0, "cloud_llm": 0.0},
        response_path="retrieval",
        retrieval_score=1.0,
        user_message_preview="(synthetic probe)",
        response_preview="(probe)",
        user_state_label=label_dict,
        accessibility=access_dict,
    )

    paragraphs = trace["narrative_paragraphs"]
    para2 = paragraphs[1] if len(paragraphs) >= 2 else ""
    para3 = paragraphs[2] if len(paragraphs) >= 3 else ""

    state_sentence = ""
    idx = para2.find("The state classifier")
    if idx >= 0:
        end = para2.find(".", idx + 60)
        end = end + 1 if end >= 0 else len(para2)
        # Capture both the classifier sentence and the signals one.
        next_dot = para2.find(".", end)
        next_dot = next_dot + 1 if next_dot >= 0 else end
        state_sentence = para2[idx:next_dot].strip()

    access_sentence = ""
    idx = para3.find("Accessibility mode is active")
    if idx >= 0:
        end = para3.find(".", idx)
        access_sentence = para3[idx : end + 1].strip() if end >= 0 else para3[idx:]

    return {
        "turn": turn_no,
        "phase": phase,
        "label": label_dict,
        "accessibility": access_dict,
        "para2_state_sentence": state_sentence,
        "para3_access_sentence": access_sentence,
    }


def _fmt_turn(row: dict) -> str:
    lbl = row["label"]
    acc = row["accessibility"]
    sigs = ", ".join(lbl.get("contributing_signals") or [])
    flags = []
    if acc.get("activated_this_turn"):
        flags.append("RISING")
    if acc.get("deactivated_this_turn"):
        flags.append("FALLING")
    flag_str = (" [" + " ".join(flags) + "]") if flags else ""
    return (
        f"T{row['turn']:>2} ({row['phase']}): "
        f"badge='{lbl['state']} {lbl['confidence']:.2f}' "
        f"signals=[{sigs}] | "
        f"access.active={acc['active']}{flag_str}"
    )


# ---------------------------------------------------------------------------
# Probe scenario
# ---------------------------------------------------------------------------


def main() -> int:
    user_id = "probe_user"
    session_id = "probe_session"
    controller = AccessibilityController()

    scenarios: list[tuple[str, str, dict]] = []

    # Phase 1: calm typing, baseline still warming up (turn 1) then warm.
    for t in range(1, 4):
        baseline_est = t > 4  # always False here — baseline=5 turns
        scenarios.append((
            f"turn_{t}",
            "calm",
            dict(
                turn_no=t,
                phase="calm-1",
                composition_ms=1200,
                edit_count=0,
                iki_mean=100.0,
                iki_std=10.0,
                cognitive_load=0.3,
                accessibility_axis=0.0,
                formality=0.5,
                engagement=0.6,
                deviation=0.0,
                msg_count=t,
                baseline_established=baseline_est,
            ),
        ))

    # Phase 2: rushed (turns 4-6).
    for t in range(4, 7):
        scenarios.append((
            f"turn_{t}",
            "rushed",
            dict(
                turn_no=t,
                phase="rushed-1",
                composition_ms=3500,
                edit_count=4,
                iki_mean=180.0,
                iki_std=40.0,
                cognitive_load=0.7,
                accessibility_axis=0.4,
                formality=0.5,
                engagement=0.5,
                deviation=0.3,
                msg_count=t,
                baseline_established=True,
            ),
        ))

    # Phase 3: keep rushed (turns 7-9).
    for t in range(7, 10):
        scenarios.append((
            f"turn_{t}",
            "rushed-sustained",
            dict(
                turn_no=t,
                phase="rushed-2",
                composition_ms=3500,
                edit_count=5,
                iki_mean=185.0,
                iki_std=45.0,
                cognitive_load=0.75,
                accessibility_axis=0.55,
                formality=0.5,
                engagement=0.45,
                deviation=0.35,
                msg_count=t,
                baseline_established=True,
            ),
        ))

    # Phase 4: calm recovery (turns 10-13).
    for t in range(10, 14):
        scenarios.append((
            f"turn_{t}",
            "recovery",
            dict(
                turn_no=t,
                phase="calm-recovery",
                composition_ms=1200,
                edit_count=0,
                iki_mean=100.0,
                iki_std=10.0,
                cognitive_load=0.3,
                accessibility_axis=0.0,
                formality=0.5,
                engagement=0.6,
                deviation=0.05,
                msg_count=t,
                baseline_established=True,
            ),
        ))

    print("=" * 78)
    print("Live State Badge + Accessibility Mode probe (lite, no pipeline)")
    print("=" * 78)

    transcript: list[dict] = []
    for name, _phase, kwargs in scenarios:
        row = _run_turn(controller=controller, user_id=user_id,
                        session_id=session_id, **kwargs)
        transcript.append(row)
        print(_fmt_turn(row))
        if row["para2_state_sentence"]:
            print(f"     para2: {row['para2_state_sentence']}")
        if row["para3_access_sentence"]:
            print(f"     para3: {row['para3_access_sentence']}")

    # ------------------------------------------------------------------
    # Manual toggle endpoint smoke-test (engine-side, no HTTP).
    # ------------------------------------------------------------------
    print("\n--- Manual force=True ---")
    state = controller.force(user_id, "manual_session", force=True)
    print(f"  active={state.active} font_scale={state.font_scale}"
          f" tts_rate_multiplier={state.tts_rate_multiplier}"
          f" rising={state.activated_this_turn}")
    print("--- Manual force=None (clear, re-enable auto) ---")
    state = controller.force(user_id, "manual_session", force=None)
    print(f"  active={state.active} reason={state.reason!r}")

    # ------------------------------------------------------------------
    # Summary checks against the brief's expectations.
    # ------------------------------------------------------------------
    print("\n--- Brief acceptance checks ---")
    summary = {
        "turn1_warming_up": transcript[0]["label"]["state"] == "warming up",
        "turn5_or_6_stressed": (
            transcript[4]["label"]["state"] == "stressed"
            or transcript[5]["label"]["state"] == "stressed"
        ),
        "any_rising_in_phase3": any(
            t["accessibility"].get("activated_this_turn")
            for t in transcript[3:9]  # turns 4-9 (rushed)
        ),
        "any_falling_in_phase4": any(
            t["accessibility"].get("deactivated_this_turn")
            for t in transcript[9:]  # turns 10-13 (recovery)
        ),
    }
    for name, ok in summary.items():
        print(f"  [{'OK' if ok else 'MISS'}] {name}")

    print("\n--- Transcript JSON ---")
    print(json.dumps(transcript, indent=2, default=str))

    return 0 if all(summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
