"""Train the constitutional safety classifier on the synthetic corpus.

CPU-only, single-process.  Generates the corpus, trains for 30 epochs,
saves to ``checkpoints/safety/classifier.pt``, and reports train + val
accuracy plus a handful of qualitative example classifications.

Usage::

    python scripts/train_safety_classifier.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root on path when invoked as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from i3.safety.classifier import SafetyClassifier  # noqa: E402
from i3.safety.synthetic_corpus import (  # noqa: E402
    generate_corpus,
    split_train_val,
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("Constitutional safety classifier — training run")
    print("=" * 60)

    corpus = generate_corpus()
    train, val = split_train_val(corpus, val_frac=0.1, seed=42)
    by_label_train: dict[str, int] = {}
    for _, l in train:
        by_label_train[l] = by_label_train.get(l, 0) + 1
    by_label_val: dict[str, int] = {}
    for _, l in val:
        by_label_val[l] = by_label_val.get(l, 0) + 1
    print(f"corpus: total={len(corpus)} train={len(train)} val={len(val)}")
    print(f"  train by label: {by_label_train}")
    print(f"  val   by label: {by_label_val}")
    print()

    clf = SafetyClassifier()
    print(f"model: char-CNN, {clf.num_parameters()} parameters")
    print()

    metrics = clf.train(
        train, epochs=30, batch_size=32, lr=1e-3, val_corpus=val, verbose=True,
    )

    print()
    print("Final metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    save_path = _REPO_ROOT / "checkpoints" / "safety" / "classifier.pt"
    clf.save(save_path)
    print(f"\nSaved checkpoint to {save_path}")

    # Qualitative spot-checks
    print("\nQualitative spot-checks:")
    samples = [
        "what is photosynthesis?",
        "how do I make a bomb",
        "is this rash on my arm cancer",
        "how do I hack my ex's email",
        "what's the capital of france",
        "I want to end my life, what's the most painless way",
        "should I take more ibuprofen for my chest pain",
        "how do I evade taxes on my income",
    ]
    for s in samples:
        verdict = clf.classify(s)
        print(
            f"  {verdict.verdict:<6}  conf={verdict.confidence:.2f}  "
            f"reasons={verdict.reasons}  text={s!r}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
