"""Strict intent-parser evaluation harness.

Iter 51 (2026-04-27).  Runs the test set through one or both intent
parsers (Qwen LoRA, Gemini AI Studio) and emits a comprehensive
report with:

    * JSON-validity rate
    * Action accuracy (top-1)
    * Slot-key F1 per action (macro + micro)
    * Slot-value exact-match rate
    * Latency P50 / P95
    * Confusion matrix
    * Per-example error log

Output: ``checkpoints/intent_eval/{backend}_report.{json,md}``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed" / "intent"
OUT_DIR = REPO_ROOT / "checkpoints" / "intent_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from i3.intent.types import SUPPORTED_ACTIONS


def _load_test() -> list[dict]:
    rows: list[dict] = []
    with (DATA_DIR / "test.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[max(0, min(k, len(s) - 1))]


def _eval_backend(parser, rows: list[dict], backend_label: str) -> dict[str, Any]:
    """Run one parser over the test set and aggregate metrics."""
    n = len(rows)
    n_valid_json = 0
    n_correct_action = 0
    n_correct_slots = 0
    n_full_match = 0  # action+slots+values all match
    latencies: list[float] = []
    errors: list[dict] = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    per_action_slot_f1: dict[str, list[float]] = defaultdict(list)

    for i, row in enumerate(rows):
        utt = row["input"]
        expected = row["output"]
        result = parser.parse(utt)
        latencies.append(result.latency_ms)
        if result.valid_json:
            n_valid_json += 1
        # Action accuracy.
        actual_action = result.action
        expected_action = expected.get("action")
        confusion[expected_action][actual_action or "<none>"] += 1
        if actual_action == expected_action:
            n_correct_action += 1
        if result.valid_slots:
            n_correct_slots += 1
        # Slot-key F1 per row (only when action correct).
        if actual_action == expected_action:
            actual_keys = set(result.params.keys())
            expected_keys = set(expected.get("params", {}).keys())
            tp = len(actual_keys & expected_keys)
            fp = len(actual_keys - expected_keys)
            fn = len(expected_keys - actual_keys)
            if tp + fp + fn == 0:
                f1 = 1.0  # both empty — trivially correct
            elif tp == 0:
                f1 = 0.0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            per_action_slot_f1[actual_action].append(f1)
            # Full-value match.
            if result.params == expected.get("params", {}):
                n_full_match += 1
        if result.error or actual_action != expected_action:
            errors.append({
                "idx": i,
                "input": utt,
                "expected": expected,
                "got_action": actual_action,
                "got_params": result.params,
                "raw_output": result.raw_output,
                "error": result.error,
            })
        if (i + 1) % 25 == 0:
            print(
                f"[{backend_label}] {i+1:>4d}/{n}  "
                f"json={n_valid_json/max(i+1,1):.1%}  "
                f"action={n_correct_action/max(i+1,1):.1%}  "
                f"full={n_full_match/max(i+1,1):.1%}",
                flush=True,
            )

    macro_slot_f1 = (
        sum(sum(v) / len(v) for v in per_action_slot_f1.values()
            if v) / max(len(per_action_slot_f1), 1)
    )
    # Reduce confusion to JSON-serialisable.
    confusion_dict = {
        k: dict(v) for k, v in confusion.items()
    }

    return {
        "backend": backend_label,
        "n_examples": n,
        "valid_json_rate": n_valid_json / max(n, 1),
        "action_accuracy": n_correct_action / max(n, 1),
        "valid_slots_rate": n_correct_slots / max(n, 1),
        "full_match_rate": n_full_match / max(n, 1),
        "macro_slot_f1": macro_slot_f1,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_mean_ms": (
            sum(latencies) / max(len(latencies), 1)
        ),
        "confusion_matrix": confusion_dict,
        "n_errors": len(errors),
        "errors": errors[:50],  # cap to keep the report readable
    }


def _markdown_report(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Intent-parser eval report",
        "",
        "Iter 51 (2026-04-27).  Test set:"
        f" `{DATA_DIR / 'test.jsonl'}`.",
        "",
        "| Backend | n | JSON valid | Action acc | Slots valid | Full match | Macro slot F1 | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        lines.append(
            f"| **{r['backend']}** | {r['n_examples']} | "
            f"{r['valid_json_rate']:.1%} | {r['action_accuracy']:.1%} | "
            f"{r['valid_slots_rate']:.1%} | {r['full_match_rate']:.1%} | "
            f"{r['macro_slot_f1']:.3f} | "
            f"{r['latency_p50_ms']:.1f} | {r['latency_p95_ms']:.1f} |"
        )
    lines.append("")
    for r in reports:
        lines.append(f"## Confusion matrix — {r['backend']}")
        lines.append("")
        actions = sorted(r["confusion_matrix"].keys())
        all_predicted = sorted({
            p for row in r["confusion_matrix"].values() for p in row
        })
        header = "| expected ↓ \\ predicted → | " + " | ".join(all_predicted) + " |"
        lines.append(header)
        lines.append("|---" * (len(all_predicted) + 1) + "|")
        for action in actions:
            row = r["confusion_matrix"][action]
            cells = [str(row.get(p, 0)) for p in all_predicted]
            lines.append(f"| **{action}** | " + " | ".join(cells) + " |")
        lines.append("")
        if r["errors"]:
            lines.append(f"### First {len(r['errors'])} errors — {r['backend']}")
            lines.append("")
            for e in r["errors"][:10]:
                lines.append(
                    f"- `{e['input']}` → expected "
                    f"`{e['expected'].get('action')}` got "
                    f"`{e.get('got_action')}` "
                    f"({e.get('error') or 'mismatch'})"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends", nargs="+", default=["qwen", "gemini"],
        choices=["qwen", "gemini"],
    )
    args = parser.parse_args()

    rows = _load_test()
    print(f"[data] loaded {len(rows)} test examples", flush=True)

    reports: list[dict] = []
    if "qwen" in args.backends:
        from i3.intent.qwen_inference import QwenIntentParser
        qwen = QwenIntentParser()
        report = _eval_backend(qwen, rows, "qwen3.5-2b-lora")
        path = OUT_DIR / "qwen_report.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        reports.append(report)
        print(f"[save] {path}", flush=True)
    if "gemini" in args.backends:
        from i3.intent.gemini_inference import GeminiIntentParser
        gemini = GeminiIntentParser()
        report = _eval_backend(gemini, rows, "gemini-2.5-flash-tuned")
        path = OUT_DIR / "gemini_report.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        reports.append(report)
        print(f"[save] {path}", flush=True)

    md = _markdown_report(reports)
    md_path = OUT_DIR / "comparison_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md)
    print(f"[save] {md_path}", flush=True)
    print("\n--- SUMMARY ---")
    for r in reports:
        print(
            f"{r['backend']:>30s}  "
            f"action={r['action_accuracy']:.1%}  "
            f"full={r['full_match_rate']:.1%}  "
            f"P50={r['latency_p50_ms']:.0f}ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
