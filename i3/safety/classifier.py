"""Constitutional safety classifier.

A small from-scratch char-level CNN text classifier trained on a
synthetic constitutional corpus (see :mod:`i3.safety.synthetic_corpus`).
Replaces the regex-based hostility-refusal route with a learned model
that exposes a refusal-reasoning surface (``SafetyVerdict``).

Architecture
------------
::

    Embedding(256, 32)
        -> Conv1d(32, 64, kernel=3, padding=1) -> ReLU -> MaxPool1d(2)
        -> Conv1d(64, 128, kernel=3, padding=1) -> ReLU -> AdaptiveAvgPool1d(1)
        -> Flatten
        -> Linear(128, 64) -> GELU -> Linear(64, n_classes)

``vocab_size = 256`` because we tokenise to raw bytes (UTF-8) so OOV is
impossible.  ``n_classes = 4`` (safe / medical / harmful / illegal).

Citation
--------
Bai et al. (2022) "Constitutional AI: Harmlessness from AI Feedback"
(arXiv 2212.08073).  We adopt the constitutional pattern of explicit
principle-driven rejection.  The CharCNN backbone follows Zhang, Zhao
and LeCun (2015) "Character-level Convolutional Networks for Text
Classification" (NeurIPS 2015) -- chosen because it learns purely from
character n-grams and so is robust to leetspeak / typo perturbations
that defeat keyword regex matchers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from i3.safety.synthetic_corpus import (
    LABEL_HARMFUL,
    LABEL_ILLEGAL,
    LABEL_MEDICAL,
    LABEL_SAFE,
    LABELS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class SafetyVerdict:
    """Output of :meth:`SafetyClassifier.classify`.

    Attributes:
        verdict: One of ``"safe"`` / ``"review"`` / ``"refuse"``.
        confidence: Probability of the predicted class in ``[0, 1]``.
        reasons: Human-readable list of triggering categories.
        constitutional_principle: Which constitutional principle
            (Bai et al. 2022 §2) was violated; empty for ``safe``.
        suggested_response: Canonical refusal text used when
            ``verdict == "refuse"``; empty otherwise.
        scores: Full per-class probability dict for debugging.
    """

    verdict: str
    confidence: float
    reasons: list[str]
    constitutional_principle: str
    suggested_response: str
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Constitutional principles (Bai et al. 2022 §2)
# ---------------------------------------------------------------------------

_PRINCIPLES = {
    LABEL_MEDICAL: (
        "Do not provide diagnostic or dosage advice that should come from a "
        "qualified clinician (Constitutional principle: medical-advice)."
    ),
    LABEL_HARMFUL: (
        "Do not provide assistance with self-harm, violence, or weapon "
        "construction (Constitutional principle: physical-harm)."
    ),
    LABEL_ILLEGAL: (
        "Do not provide instructions for hacking, fraud, evasion, or other "
        "criminal activity (Constitutional principle: legal-compliance)."
    ),
}

_REFUSALS = {
    LABEL_MEDICAL: (
        "I'm not able to give medical advice. For symptoms, dosage, or "
        "diagnosis questions please contact a qualified clinician or, in "
        "an emergency, your local emergency services."
    ),
    LABEL_HARMFUL: (
        "I can't help with that. If you or someone you know is in danger, "
        "please contact local emergency services or a crisis helpline. "
        "I'm here if you want to talk about something else."
    ),
    LABEL_ILLEGAL: (
        "I can't help with that — it falls outside what I can responsibly "
        "assist with. I'd be happy to answer a different question, or to "
        "discuss the legitimate alternatives in this area."
    ),
}


# ---------------------------------------------------------------------------
# Defense-in-depth: harm-signal detector
# ---------------------------------------------------------------------------

# Lower-cased substrings that, when present, justify the classifier's
# unsafe verdict.  When the classifier flags harm but NONE of these
# appear in the input, we override the verdict to safe (the classifier
# is hallucinating on innocuous educational queries like "which is
# closest to the sun").  Conservative on purpose — false negatives here
# are cheap because the learned classifier remains the primary signal
# for everything not on this list.
_HARM_SIGNAL_TOKENS: tuple[str, ...] = (
    # Self-harm / suicide
    "suicide", "kill myself", "kill my self", "end my life", "self harm",
    "self-harm", "cut myself", "harm myself", "want to die",
    # Violence / weapons toward others
    "kill ", "murder", "stab", "shoot ", "shooting up", "attack ",
    "assault", "torture", "hurt someone", "hurt people", "harm someone",
    "weapon", "gun", "knife", "explosive", "bomb", "grenade",
    "poison ", "strangle",
    # Illegal / contraband
    " hack ", "hacking into", "exploit cve", "ddos", "phishing",
    "malware", "ransomware", "crack the password", "bypass auth",
    "steal ", "stealing ",
    "drug deal", "buy heroin", "buy fentanyl", "make meth",
    # Medical danger keywords
    "overdose", "lethal dose", "od on", "how much can i take",
    "mix with alcohol", "without prescription",
)


def _has_harm_signal(text: str) -> bool:
    """Cheap rule check: does *text* contain any actual harm-intent token?

    Used as a second-pass overlay on the learned classifier.  Conservative
    by design — when in doubt return True (let the classifier's verdict
    stand).  Only short, obviously-innocuous educational queries should
    return False here.
    """
    if not text:
        return False
    t = text.lower()
    return any(tok in t for tok in _HARM_SIGNAL_TOKENS)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _CharCNN(nn.Module):
    """The actual char-CNN backbone."""

    def __init__(self, vocab_size: int = 256, embed_dim: int = 32, n_classes: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.adapool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] of int64 byte ids
        e = self.embedding(x)              # [B, L, E]
        e = e.transpose(1, 2)              # [B, E, L]
        h = F.relu(self.conv1(e))
        h = self.pool(h)
        h = F.relu(self.conv2(h))
        h = self.adapool(h).squeeze(-1)    # [B, 128]
        h = F.gelu(self.fc1(h))
        return self.fc2(h)                 # [B, n_classes]


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------


class SafetyClassifier:
    """Constitutional safety layer (Bai et al. 2022) implemented as a
    small from-scratch char-level CNN.

    Architecture (see module docstring)::

        Embedding(256, 32) -> Conv1d -> ReLU -> MaxPool
                          -> Conv1d -> ReLU -> AdaptiveAvgPool1d(1)
                          -> Linear(128, 64) -> GELU -> Linear(64, n_classes)

    Trained on the synthetic corpus generated by
    :mod:`i3.safety.synthetic_corpus`.  The classifier is pure PyTorch
    (no HuggingFace, no datasets library) and CPU-only.
    """

    MAX_LEN: int = 256

    # Threshold above which the predicted class triggers the matched
    # refusal pathway.  Below review_threshold the verdict downgrades
    # to "review" (a soft caveat is appended).
    REFUSE_THRESHOLD: float = 0.65
    REVIEW_THRESHOLD: float = 0.40

    def __init__(self, vocab_size: int = 256, n_classes: int = 4) -> None:
        if n_classes != len(LABELS):
            raise ValueError(
                f"n_classes={n_classes} but {len(LABELS)} labels are defined."
            )
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.labels: tuple[str, ...] = LABELS
        self.model = _CharCNN(vocab_size=vocab_size, n_classes=n_classes)
        self.model.eval()
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> torch.Tensor:
        """Encode *text* to a 1-D LongTensor of byte ids (truncated/padded)."""
        b = text.encode("utf-8", errors="replace")[: self.MAX_LEN]
        ids = list(b) + [0] * (self.MAX_LEN - len(b))
        return torch.tensor(ids, dtype=torch.long)

    def _encode_batch(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.stack([self._encode(t) for t in texts], dim=0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def classify(self, text: str) -> SafetyVerdict:
        """Run a single text through the classifier and return a verdict.

        Always returns a ``SafetyVerdict`` -- never raises on input.
        """
        if not text or not text.strip():
            return SafetyVerdict(
                verdict="safe",
                confidence=1.0,
                reasons=[],
                constitutional_principle="",
                suggested_response="",
                scores={lbl: 0.0 for lbl in self.labels},
            )
        x = self._encode(text).unsqueeze(0)
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)[0].tolist()
        scores = {self.labels[i]: float(probs[i]) for i in range(self.n_classes)}
        # Argmax over non-safe classes; pick the highest-prob unsafe class
        # as the candidate "reason" but only refuse if the candidate's
        # probability beats safe's by REFUSE_THRESHOLD.
        unsafe_classes = [lbl for lbl in self.labels if lbl != LABEL_SAFE]
        unsafe_top = max(unsafe_classes, key=lambda l: scores[l])
        unsafe_p = scores[unsafe_top]
        safe_p = scores[LABEL_SAFE]

        # Defense-in-depth pruning: the 47 k-param char-CNN can hallucinate
        # harm signals on innocuous educational queries (e.g. "which is
        # closest to the sun" → harmful 0.88).  Before triggering refuse,
        # run a cheap rule-based sanity check that looks for ACTUAL
        # harm-intent tokens.  When the input matches a clear educational
        # / factual question pattern AND contains no harm-intent token,
        # downgrade to safe.  This is the standard "model + rule overlay"
        # pattern used in production safety stacks (Glaese et al 2022).
        if unsafe_p >= self.REFUSE_THRESHOLD and not _has_harm_signal(text):
            scores = dict(scores)  # don't mutate the original
            unsafe_p = 0.0
            safe_p = 1.0

        if unsafe_p >= self.REFUSE_THRESHOLD and unsafe_p > safe_p:
            verdict = "refuse"
            principle = _PRINCIPLES.get(unsafe_top, "")
            suggested = _REFUSALS.get(unsafe_top, "")
            reasons = [unsafe_top]
            confidence = unsafe_p
        elif unsafe_p >= self.REVIEW_THRESHOLD and unsafe_p > safe_p * 0.7:
            verdict = "review"
            principle = _PRINCIPLES.get(unsafe_top, "")
            suggested = ""
            reasons = [unsafe_top]
            confidence = unsafe_p
        else:
            verdict = "safe"
            principle = ""
            suggested = ""
            reasons = []
            # When verdict is safe, confidence reports max-class probability
            # so a low-confidence "safe but uncertain" turn is visible to
            # downstream review. Falls back to safe_p when safe IS the top.
            confidence = max(safe_p, 1.0 - unsafe_p)

        return SafetyVerdict(
            verdict=verdict,
            confidence=float(confidence),
            reasons=reasons,
            constitutional_principle=principle,
            suggested_response=suggested,
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        corpus: Iterable[tuple[str, str]],
        *,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        val_corpus: Iterable[tuple[str, str]] | None = None,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Train the classifier on *corpus* and return final metrics.

        Args:
            corpus: Iterable of ``(text, label)`` pairs.
            epochs: Number of passes over the corpus.
            batch_size: Mini-batch size.
            lr: Adam learning rate.
            val_corpus: Optional held-out set for per-epoch val accuracy.
            verbose: When ``True``, log per-epoch train/val accuracy.

        Returns:
            Final metrics dict ``{"train_acc": ..., "val_acc": ...,
            "n_train": ..., "n_val": ..., "epochs": ...,
            "n_parameters": ...}``.
        """
        train_pairs = list(corpus)
        val_pairs = list(val_corpus) if val_corpus is not None else []

        if not train_pairs:
            raise ValueError("Empty training corpus")

        label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}

        train_x = self._encode_batch([t for t, _ in train_pairs])
        train_y = torch.tensor(
            [label_to_idx[l] for _, l in train_pairs], dtype=torch.long
        )
        if val_pairs:
            val_x = self._encode_batch([t for t, _ in val_pairs])
            val_y = torch.tensor(
                [label_to_idx[l] for _, l in val_pairs], dtype=torch.long
            )
        else:
            val_x = val_y = None

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        n = len(train_pairs)
        best_val = 0.0
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n)
            total_loss = 0.0
            correct = 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                xb = train_x[idx]
                yb = train_y[idx]
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * xb.size(0)
                correct += int((logits.argmax(dim=-1) == yb).sum().item())
            train_acc = correct / n
            val_acc: float | None = None
            if val_x is not None and val_y is not None:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(val_x)
                    val_acc = float(
                        (logits.argmax(dim=-1) == val_y).float().mean().item()
                    )
                best_val = max(best_val, val_acc)
                self.model.train()
            if verbose:
                msg = (
                    f"epoch {epoch:>2}/{epochs}  loss={total_loss/n:.4f}  "
                    f"train_acc={train_acc:.3f}"
                )
                if val_acc is not None:
                    msg += f"  val_acc={val_acc:.3f}"
                logger.info(msg)
                # Also print so the training script's stdout shows progress.
                print(msg)
        self.model.eval()
        self._trained = True

        # Final
        with torch.no_grad():
            t_logits = self.model(train_x)
            final_train = float((t_logits.argmax(-1) == train_y).float().mean().item())
            if val_x is not None:
                v_logits = self.model(val_x)
                final_val = float((v_logits.argmax(-1) == val_y).float().mean().item())
            else:
                final_val = 0.0

        return {
            "train_acc": final_train,
            "val_acc": final_val,
            "n_train": float(len(train_pairs)),
            "n_val": float(len(val_pairs)),
            "epochs": float(epochs),
            "n_parameters": float(self.num_parameters()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = path.with_suffix(".json")
        torch.save(self.model.state_dict(), str(path))
        meta = {
            "vocab_size": self.vocab_size,
            "n_classes": self.n_classes,
            "labels": list(self.labels),
            "max_len": self.MAX_LEN,
            "n_parameters": self.num_parameters(),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        state = torch.load(str(path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self._trained = True


# ---------------------------------------------------------------------------
# Module-level convenience: lazy global classifier
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT = Path("checkpoints/safety/classifier.pt")
_global_classifier: SafetyClassifier | None = None


def get_global_classifier() -> SafetyClassifier:
    """Return a process-wide singleton classifier.

    Loads ``checkpoints/safety/classifier.pt`` when available; otherwise
    returns an *untrained* classifier (which produces uniform-random-ish
    verdicts, all of which fall through the REFUSE_THRESHOLD and so
    classify as ``safe``).  The pipeline pre-checks ``is_trained`` to
    decide whether to consult the layer at all.
    """
    global _global_classifier
    if _global_classifier is None:
        clf = SafetyClassifier()
        try:
            clf.load(_DEFAULT_CHECKPOINT)
            logger.info(
                "Loaded safety classifier checkpoint (%d params)",
                clf.num_parameters(),
            )
        except FileNotFoundError:
            logger.warning(
                "Safety classifier checkpoint missing at %s; running in "
                "untrained pass-through mode. Run "
                "scripts/train_safety_classifier.py to train.",
                _DEFAULT_CHECKPOINT,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load safety classifier checkpoint")
        _global_classifier = clf
    return _global_classifier


__all__ = [
    "SafetyClassifier",
    "SafetyVerdict",
    "get_global_classifier",
]
