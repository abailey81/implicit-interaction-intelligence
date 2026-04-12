"""Query complexity estimation for the Intelligent Router.

Provides a heuristic-based estimator that scores user queries from 0
(trivial greeting) to 1 (complex analytical request) without requiring
any trained ML model.  The estimator combines multiple orthogonal
signals -- length, question structure, technical vocabulary, syntactic
depth, and conditional language -- into a single scalar score.
"""

from __future__ import annotations

import re

import numpy as np


# ---------------------------------------------------------------------------
# Technical / complex vocabulary
# ---------------------------------------------------------------------------

_TECHNICAL_WORDS: frozenset[str] = frozenset(
    {
        # Analytical / reasoning
        "analyze", "analyse", "evaluate", "synthesize", "synthesise",
        "compare", "contrast", "differentiate", "correlate", "extrapolate",
        "interpolate", "deduce", "infer", "hypothesize", "hypothesise",
        "derive", "formulate", "critique", "assess", "benchmark",
        # Technical domains
        "algorithm", "architecture", "implementation", "framework",
        "infrastructure", "protocol", "specification", "abstraction",
        "encapsulation", "polymorphism", "inheritance", "recursion",
        "concurrency", "parallelism", "asynchronous", "synchronous",
        "latency", "throughput", "bandwidth", "optimization", "optimisation",
        "regression", "classification", "clustering", "dimensionality",
        "eigenvalue", "gradient", "heuristic", "stochastic", "deterministic",
        "probabilistic", "bayesian", "posterior", "likelihood", "prior",
        # Scientific
        "hypothesis", "methodology", "empirical", "theoretical", "quantitative",
        "qualitative", "statistical", "variance", "deviation", "coefficient",
        "parameter", "variable", "dependent", "independent", "controlled",
        # Complex request patterns
        "implications", "consequences", "tradeoffs", "trade-offs",
        "constraints", "requirements", "prerequisites", "dependencies",
        "scalability", "maintainability", "extensibility", "interoperability",
        # Multi-step reasoning
        "furthermore", "moreover", "nevertheless", "consequently",
        "subsequently", "alternatively", "notwithstanding", "whereas",
        "whereby", "thereof", "therein", "hitherto", "aforementioned",
        # Domain-specific (HMI / AI)
        "embedding", "transformer", "attention", "convolution", "encoder",
        "decoder", "tokenizer", "tokeniser", "backpropagation", "feedforward",
        "activation", "regularization", "regularisation", "dropout",
        "batch-norm", "layer-norm", "fine-tuning", "pre-training",
        "reinforcement", "supervised", "unsupervised", "semi-supervised",
        "distillation", "quantization", "quantisation", "pruning",
    }
)

# Words/phrases that signal question complexity
_COMPLEX_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "why", "how", "explain", "compare", "analyze", "analyse",
        "evaluate", "discuss", "elaborate", "describe", "justify",
        "distinguish", "illustrate", "summarize", "summarise",
        "what if", "suppose", "consider", "contrast",
    }
)

# Conditional / hypothetical markers
_CONDITIONAL_WORDS: frozenset[str] = frozenset(
    {
        "if", "would", "could", "should", "might", "suppose",
        "assuming", "hypothetically", "imagine", "consider",
        "provided", "unless", "whereas", "although", "despite",
        "notwithstanding", "regardless", "irrespective",
    }
)

# Sentence-ending punctuation pattern
_SENTENCE_END_RE = re.compile(r"[.!?]+")

# Multi-part question pattern (multiple question marks or question words)
_QUESTION_MARK_RE = re.compile(r"\?")

# SEC: Truncate over-long inputs before analysis to bound CPU usage.
# 32 KB is well above any plausible user query and prevents pathological
# multi-megabyte payloads from causing slow text scans / regex evaluation.
_MAX_TEXT_LEN = 32_768


class QueryComplexityEstimator:
    """Estimates query complexity using a weighted combination of heuristic
    signals.

    No ML model is required.  The estimator is deterministic and
    instantaneous, making it suitable for real-time routing decisions.

    Signals:
        1. **Length**: Longer queries tend to be more complex.
        2. **Question complexity**: Presence and count of analytical question
           words (why, how, explain, compare, ...).
        3. **Technical vocabulary**: Fraction of words that are domain-
           specific or academically complex.
        4. **Sentence count**: Multiple sentences suggest multi-part
           requests.
        5. **Conditional language**: Hypothetical and conditional markers
           indicate higher-order reasoning.
        6. **Multi-part questions**: Multiple question marks or
           conjunctions joining questions.

    The final score is a weighted average of the six signals, clipped
    to [0, 1].
    """

    def __init__(
        self,
        weights: tuple[float, ...] | None = None,
    ) -> None:
        """Initialise the estimator.

        Args:
            weights: Optional 6-element tuple of signal weights.  If None,
                uses the default weights ``(0.20, 0.25, 0.20, 0.10, 0.15, 0.10)``.
        """
        if weights is None:
            self.weights = np.array(
                [0.20, 0.25, 0.20, 0.10, 0.15, 0.10], dtype=np.float64
            )
        else:
            if len(weights) != 6:
                raise ValueError(f"Expected 6 weights, got {len(weights)}")
            self.weights = np.array(weights, dtype=np.float64)
            total = self.weights.sum()
            if total > 0:
                self.weights /= total  # Normalise to sum to 1

    def estimate(self, text: str) -> float:
        """Estimate query complexity on a 0-1 scale.

        Args:
            text: The raw user query text.

        Returns:
            A float in [0, 1] where 0 = trivial and 1 = highly complex.
        """
        if not text or not text.strip():
            return 0.0

        # SEC: Bound input length to prevent slow text processing on
        # adversarial multi-megabyte payloads.
        if len(text) > _MAX_TEXT_LEN:
            text = text[:_MAX_TEXT_LEN]

        text_lower = text.lower().strip()
        words = text_lower.split()
        word_count = len(words)

        if word_count == 0:
            return 0.0

        signals = np.zeros(6, dtype=np.float64)

        # Signal 1: Length (saturates around 50 words)
        signals[0] = min(1.0, word_count / 50.0)

        # Signal 2: Question complexity
        # Count complex question words (including 2-word phrases)
        question_word_count = 0
        word_set = set(words)
        for qw in _COMPLEX_QUESTION_WORDS:
            if " " in qw:
                # Multi-word phrase: check in original text
                if qw in text_lower:
                    question_word_count += 1
            else:
                if qw in word_set:
                    question_word_count += 1
        signals[1] = min(1.0, question_word_count / 3.0)

        # Signal 3: Technical vocabulary density
        tech_count = sum(1 for w in words if w.strip(".,;:!?()[]{}\"'") in _TECHNICAL_WORDS)
        tech_density = tech_count / word_count
        # Scale: 10%+ technical words = max complexity
        signals[2] = min(1.0, tech_density / 0.10)

        # Signal 4: Sentence count (more sentences = more complex request)
        sentences = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
        # Also count question marks as sentence boundaries
        n_questions = len(_QUESTION_MARK_RE.findall(text))
        sentence_count = max(len(sentences), 1) + n_questions
        # Saturates around 5 sentences
        signals[3] = min(1.0, (sentence_count - 1) / 4.0)

        # Signal 5: Conditional / hypothetical language
        conditional_count = sum(1 for w in words if w in _CONDITIONAL_WORDS)
        signals[4] = min(1.0, conditional_count / 3.0)

        # Signal 6: Multi-part question indicator
        # Multiple question marks or list-like structure
        multi_indicators = 0
        if n_questions > 1:
            multi_indicators += min(n_questions - 1, 3)
        # Numbered lists: "1.", "2.", etc.
        numbered_items = len(re.findall(r"\b\d+\.\s", text))
        if numbered_items > 1:
            multi_indicators += min(numbered_items - 1, 3)
        # Bullet points
        bullet_items = text.count("\n-") + text.count("\n*") + text.count("\n+")
        if bullet_items > 0:
            multi_indicators += min(bullet_items, 3)
        signals[5] = min(1.0, multi_indicators / 3.0)

        # Weighted average
        score = float(np.dot(self.weights, signals))
        return float(np.clip(score, 0.0, 1.0))

    def estimate_detailed(self, text: str) -> dict[str, float]:
        """Return the complexity score along with individual signal values.

        Useful for debugging and explainability.

        Args:
            text: The raw user query text.

        Returns:
            A dict with keys ``"score"``, ``"length"``, ``"question_complexity"``,
            ``"technical_vocab"``, ``"sentence_count"``, ``"conditional_language"``,
            and ``"multi_part"``.
        """
        if not text or not text.strip():
            return {
                "score": 0.0,
                "length": 0.0,
                "question_complexity": 0.0,
                "technical_vocab": 0.0,
                "sentence_count": 0.0,
                "conditional_language": 0.0,
                "multi_part": 0.0,
            }

        # SEC: Bound input length (mirror of estimate())
        if len(text) > _MAX_TEXT_LEN:
            text = text[:_MAX_TEXT_LEN]

        text_lower = text.lower().strip()
        words = text_lower.split()
        word_count = len(words)

        if word_count == 0:
            return {
                "score": 0.0,
                "length": 0.0,
                "question_complexity": 0.0,
                "technical_vocab": 0.0,
                "sentence_count": 0.0,
                "conditional_language": 0.0,
                "multi_part": 0.0,
            }

        # Reuse estimate logic -- compute signals
        score = self.estimate(text)

        # Recompute individual signals for the breakdown
        sig_length = min(1.0, word_count / 50.0)

        question_word_count = 0
        word_set = set(words)
        for qw in _COMPLEX_QUESTION_WORDS:
            if " " in qw:
                if qw in text_lower:
                    question_word_count += 1
            else:
                if qw in word_set:
                    question_word_count += 1
        sig_question = min(1.0, question_word_count / 3.0)

        tech_count = sum(1 for w in words if w.strip(".,;:!?()[]{}\"'") in _TECHNICAL_WORDS)
        sig_tech = min(1.0, (tech_count / word_count) / 0.10)

        sentences = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
        n_questions = len(_QUESTION_MARK_RE.findall(text))
        sentence_count = max(len(sentences), 1) + n_questions
        sig_sentence = min(1.0, (sentence_count - 1) / 4.0)

        conditional_count = sum(1 for w in words if w in _CONDITIONAL_WORDS)
        sig_conditional = min(1.0, conditional_count / 3.0)

        multi_indicators = 0
        if n_questions > 1:
            multi_indicators += min(n_questions - 1, 3)
        numbered_items = len(re.findall(r"\b\d+\.\s", text))
        if numbered_items > 1:
            multi_indicators += min(numbered_items - 1, 3)
        bullet_items = text.count("\n-") + text.count("\n*") + text.count("\n+")
        if bullet_items > 0:
            multi_indicators += min(bullet_items, 3)
        sig_multi = min(1.0, multi_indicators / 3.0)

        return {
            "score": score,
            "length": sig_length,
            "question_complexity": sig_question,
            "technical_vocab": sig_tech,
            "sentence_count": sig_sentence,
            "conditional_language": sig_conditional,
            "multi_part": sig_multi,
        }
