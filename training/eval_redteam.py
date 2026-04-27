"""Iter 59 — Run the redteam ATTACK_CORPUS through the safety classifier.

Companion to ``training/eval_intent.py`` and ``training/eval_slm_v2.py``.
Reads the curated attack corpus (`i3/redteam/attack_corpus.py`),
sends each payload through `i3.safety.classifier.classify_text`, and
emits per-category aggregates plus a per-attack outcome log.

Outputs:
* ``reports/redteam_eval.json``
* ``reports/redteam_eval.md``

Usage::

    poetry run python training/eval_redteam.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("I3_QUIET", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Redteam batch eval")
    parser.add_argument("--out-prefix", type=str,
                        default=str(REPO_ROOT / "reports" / "redteam_eval"))
    args = parser.parse_args()

    from i3.redteam.attack_corpus import ATTACK_CORPUS
    from i3.safety.classifier import get_global_classifier

    clf = get_global_classifier()
    n = len(ATTACK_CORPUS)
    print(f"[redteam] running {n} attacks through safety classifier")

    results: list[dict] = []
    cat_counts: dict[str, dict[str, int]] = {}
    sev_counts: dict[str, dict[str, int]] = {}

    for atk in ATTACK_CORPUS:
        a = atk.model_dump()
        category = a["category"]
        severity = a["severity"]
        try:
            v = clf.classify(a["payload"])
            verdict = v.verdict if hasattr(v, "verdict") else str(v)
            confidence = float(getattr(v, "confidence", 0.0))
        except Exception as exc:
            verdict = f"error:{type(exc).__name__}"
            confidence = 0.0
        results.append({
            "id": a["id"],
            "category": category,
            "severity": severity,
            "expected": a["expected_outcome"],
            "verdict": verdict,
            "confidence": round(confidence, 3),
        })
        cat_counts.setdefault(category, {"total": 0, "flagged": 0})
        cat_counts[category]["total"] += 1
        if verdict in ("refuse", "review"):
            cat_counts[category]["flagged"] += 1
        sev_counts.setdefault(severity, {"total": 0, "flagged": 0})
        sev_counts[severity]["total"] += 1
        if verdict in ("refuse", "review"):
            sev_counts[severity]["flagged"] += 1

    flagged = sum(1 for r in results if r["verdict"] in ("refuse", "review"))
    expected_blocked = sum(1 for r in results
                           if r["expected"] in ("blocked", "refused"))
    correctly_blocked = sum(
        1 for r in results
        if r["expected"] in ("blocked", "refused")
        and r["verdict"] in ("refuse", "review")
    )

    out = {
        "n_attacks": n,
        "n_flagged_total": flagged,
        "n_expected_blocked": expected_blocked,
        "n_correctly_blocked": correctly_blocked,
        "block_recall": round(
            correctly_blocked / max(1, expected_blocked), 3,
        ),
        "block_rate_overall": round(flagged / max(1, n), 3),
        "by_category": cat_counts,
        "by_severity": sev_counts,
        "per_attack": results,
        "wall_time_s": round(time.time() - 0, 1),  # not really useful here
    }

    out_json = Path(args.out_prefix + ".json")
    out_md = Path(args.out_prefix + ".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[ok] wrote {out_json}")

    md = ["# Redteam evaluation — iter 59", ""]
    md.append(f"Total attacks: **{n}**")
    md.append(f"Total flagged (review or refuse): **{flagged}**")
    md.append(f"Block recall (flagged / expected-blocked): **{out['block_recall']}**")
    md.append("")
    md.append("## By category")
    md.append("")
    md.append("| category | total | flagged | rate |")
    md.append("|---|---|---|---|")
    for cat, c in sorted(cat_counts.items()):
        rate = c["flagged"] / max(1, c["total"])
        md.append(f"| {cat} | {c['total']} | {c['flagged']} | {rate:.0%} |")
    md.append("")
    md.append("## By severity")
    md.append("")
    md.append("| severity | total | flagged | rate |")
    md.append("|---|---|---|---|")
    for sev, c in sorted(sev_counts.items()):
        rate = c["flagged"] / max(1, c["total"])
        md.append(f"| {sev} | {c['total']} | {c['flagged']} | {rate:.0%} |")
    md.append("")
    md.append("## Per-attack outcome (first 25)")
    md.append("")
    md.append("| id | category | severity | expected | verdict | conf |")
    md.append("|---|---|---|---|---|---|")
    for r in results[:25]:
        md.append(
            f"| {r['id']} | {r['category']} | {r['severity']} "
            f"| {r['expected']} | {r['verdict']} | {r['confidence']} |"
        )
    md.append("")
    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[ok] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
