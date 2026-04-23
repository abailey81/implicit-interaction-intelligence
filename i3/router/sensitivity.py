"""Topic sensitivity detection for the Intelligent Router.

Detects sensitive topics in user queries that should be routed to the
local SLM for privacy reasons rather than sent to a cloud LLM.

Sensitivity categories include:
    - Medical / health information
    - Financial / monetary data
    - Relationship / personal life
    - Mental health and crisis
    - Credentials and secrets
    - Abuse, violence, and safety
    - Legal matters
    - Employment and workplace issues

Each category has a base sensitivity weight reflecting how critical
privacy is for that topic.  The detector returns the maximum sensitivity
score across all matched categories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# SEC: Cap input length before regex evaluation. The patterns are
# audited to be free of catastrophic backtracking, but capping length
# is a defense-in-depth measure against pathological inputs.
_MAX_TEXT_LEN = 32_768


@dataclass(frozen=True)
class SensitivityPattern:
    """A compiled regex pattern paired with its sensitivity weight.

    Attributes:
        pattern: Compiled regex for matching sensitive content.
        weight: Base sensitivity score in [0, 1] if this pattern matches.
        category: Human-readable category name for logging.
    """

    pattern: re.Pattern[str]
    weight: float
    category: str


# ---------------------------------------------------------------------------
# Sensitivity patterns ordered by category
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS: tuple[SensitivityPattern, ...] = (
    # --- Mental health and crisis (highest priority) ---
    SensitivityPattern(
        pattern=re.compile(
            r"\b(suicid[ae]l?|self[- ]harm|kill\s+my\s*self|end\s+it\s+all"
            r"|want\s+to\s+die|panic\s+attack|ptsd|trauma|bipolar"
            r"|schizophren|eating\s+disorder|anorex|bulimi"
            r"|self[- ]medic|overdos|cutting\s+my\s*self)\b",
            re.IGNORECASE,
        ),
        weight=0.95,
        category="mental_health_crisis",
    ),
    SensitivityPattern(
        pattern=re.compile(
            r"\b(mental\s+health|anxiety|anxious|depressed|depression"
            r"|therap(y|ist)|counsell?(or|ing)|psychiatr"
            r"|insomnia|panic|phobia|ocd|adhd|burnout)\b",
            re.IGNORECASE,
        ),
        weight=0.85,
        category="mental_health_general",
    ),
    # --- Credentials and secrets ---
    SensitivityPattern(
        pattern=re.compile(
            r"\b(password|passphrase|secret\s+key|api[- ]?key|token"
            r"|credential|private\s+key|ssn|social[- ]security"
            r"|pin\s+number|cvv|credit\s+card\s+number"
            r"|bank\s+account\s+number|routing\s+number)\b",
            re.IGNORECASE,
        ),
        weight=0.95,
        category="credentials",
    ),
    SensitivityPattern(
        pattern=re.compile(
            r"\b(confidential|classified|nda|non[- ]disclosure"
            r"|proprietary|trade\s+secret|restricted|sensitive\s+data"
            r"|encrypt(ed|ion)?|private\s+information)\b",
            re.IGNORECASE,
        ),
        weight=0.85,
        category="confidential_info",
    ),
    # --- Abuse and safety ---
    SensitivityPattern(
        pattern=re.compile(
            r"\b(abuse|abus(ed|ing|ive)|domestic\s+violence"
            r"|sexual\s+assault|rape|molest|harass(ed|ment|ing)"
            r"|stalk(ed|er|ing)|batter(ed|ing)|victim"
            r"|restraining\s+order|protective\s+order)\b",
            re.IGNORECASE,
        ),
        weight=0.90,
        category="abuse_safety",
    ),
    # --- Medical / health ---
    SensitivityPattern(
        pattern=re.compile(
            r"\b(diagnos(is|ed|e)|symptom|medical\s+record"
            r"|prescription|medication|dosage|side\s+effect"
            r"|surgery|biopsy|cancer|tumor|tumour"
            r"|chronic|terminal|prognosis|mri|ct\s+scan)\b",
            re.IGNORECASE,
        ),
        weight=0.85,
        category="medical_records",
    ),
    SensitivityPattern(
        pattern=re.compile(
            r"\b(health|medical|doctor|physician|hospital|clinic"
            r"|sick|illness|disease|condition|pain|ache"
            r"|treatment|therapy|rehab|nurse|specialist"
            r"|blood\s+test|lab\s+result|allerg(y|ic|ies))\b",
            re.IGNORECASE,
        ),
        weight=0.75,
        category="health_general",
    ),
    # --- Financial ---
    SensitivityPattern(
        pattern=re.compile(
            r"\b(salary|income|wage|pay\s*check|compensation"
            r"|net\s+worth|debt|owe|bankrupt|foreclosur"
            r"|tax\s+return|irs|audit|tax\s+evasion"
            r"|insider\s+trad|money\s+launder)\b",
            re.IGNORECASE,
        ),
        weight=0.80,
        category="financial_sensitive",
    ),
    SensitivityPattern(
        pattern=re.compile(
            r"\b(money|financial|bank|loan|mortgage|credit"
            r"|invest(ment|ing)?|savings?|budget|expense"
            r"|insurance|retirement|pension|401k|ira)\b",
            re.IGNORECASE,
        ),
        weight=0.65,
        category="financial_general",
    ),
    # --- Relationships / personal ---
    SensitivityPattern(
        pattern=re.compile(
            r"\b(divorce|custody|infidel|cheat(ed|ing)\s+on"
            r"|affair|separat(ed|ion)|breakup|break\s+up"
            r"|pregnant|abortion|miscarriage|fertility"
            r"|sexual\s+orientation|coming\s+out|gender\s+identity)\b",
            re.IGNORECASE,
        ),
        weight=0.75,
        category="relationship_sensitive",
    ),
    SensitivityPattern(
        pattern=re.compile(
            r"\b(relationship|dating|marriage|partner|spouse"
            r"|boyfriend|girlfriend|family\s+problem"
            r"|parent(s|ing)|child(ren)?|sibling)\b",
            re.IGNORECASE,
        ),
        weight=0.55,
        category="relationship_general",
    ),
    # --- Legal ---
    SensitivityPattern(
        pattern=re.compile(
            r"\b(lawyer|attorney|lawsuit|legal\s+advice"
            r"|court\s+order|subpoena|arrest(ed)?"
            r"|criminal\s+record|felony|misdemeanor"
            r"|probation|parole|bail|plea\s+deal)\b",
            re.IGNORECASE,
        ),
        weight=0.75,
        category="legal",
    ),
    # --- Employment / workplace ---
    SensitivityPattern(
        pattern=re.compile(
            r"\b(fired|terminated|laid\s+off|wrongful\s+termination"
            r"|workplace\s+discrimination|sexual\s+harassment"
            r"|whistleblow|hr\s+complaint|hostile\s+work"
            r"|performance\s+review|disciplinary)\b",
            re.IGNORECASE,
        ),
        weight=0.70,
        category="employment",
    ),
)


class TopicSensitivityDetector:
    """Detects sensitive topics in user queries and returns a sensitivity
    score indicating how important it is to keep the query local (private).

    The detector scans the query text against a battery of regex patterns
    across multiple sensitivity categories.  The returned score is the
    maximum sensitivity weight among all matched patterns, reflecting a
    conservative (privacy-first) approach.

    Example::

        detector = TopicSensitivityDetector()
        score = detector.detect("I'm feeling really anxious about my diagnosis")
        # score ~ 0.85
    """

    def __init__(
        self,
        patterns: tuple[SensitivityPattern, ...] | None = None,
        min_score: float = 0.0,
    ) -> None:
        """Initialise the detector.

        Args:
            patterns: Optional override for the built-in sensitivity
                patterns.  If None, uses the default battery.
            min_score: Floor for the returned sensitivity score.  Useful
                if you want a non-zero baseline sensitivity for all queries.
        """
        self.patterns: tuple[SensitivityPattern, ...] = (
            patterns if patterns is not None else _SENSITIVE_PATTERNS
        )
        self.min_score: float = min_score

    def detect(self, text: str) -> float:
        """Return a sensitivity score in [min_score, 1.0].

        Higher scores indicate more sensitive content that should be
        routed to the local SLM for privacy.

        Args:
            text: The raw user query text.

        Returns:
            A float in [min_score, 1.0].  Returns ``min_score`` if no
            sensitive patterns are detected.
        """
        if not text or not text.strip():
            return self.min_score

        # SEC: Truncate over-long inputs to bound regex CPU cost.
        if len(text) > _MAX_TEXT_LEN:
            text = text[:_MAX_TEXT_LEN]

        max_sensitivity = self.min_score
        for sp in self.patterns:
            if sp.pattern.search(text):
                max_sensitivity = max(max_sensitivity, sp.weight)
                # Early exit: can't exceed 1.0
                if max_sensitivity >= 1.0:
                    return 1.0

        return max_sensitivity

    def detect_detailed(
        self, text: str
    ) -> dict[str, float | list[str] | dict[str, float]]:
        """Return sensitivity score with matched category details.

        Useful for debugging, logging, and explainability.

        Args:
            text: The raw user query text.

        Returns:
            A dict with keys:
                ``"score"`` (float): The overall sensitivity score.
                ``"matched_categories"`` (list[str]): Names of matched
                    sensitivity categories, sorted by weight descending.
                ``"category_scores"`` (dict[str, float]): Mapping from
                    matched category name to its weight.
        """
        if not text or not text.strip():
            return {
                "score": self.min_score,
                "matched_categories": [],
                "category_scores": {},
            }

        # SEC: Mirror estimate() length cap.
        if len(text) > _MAX_TEXT_LEN:
            text = text[:_MAX_TEXT_LEN]

        matched: dict[str, float] = {}
        for sp in self.patterns:
            if sp.pattern.search(text):
                # Keep the highest weight if multiple patterns share a category
                if sp.category not in matched or sp.weight > matched[sp.category]:
                    matched[sp.category] = sp.weight

        if not matched:
            return {
                "score": self.min_score,
                "matched_categories": [],
                "category_scores": {},
            }

        sorted_categories = sorted(matched.keys(), key=lambda c: matched[c], reverse=True)
        max_score = max(matched.values())

        return {
            "score": max(max_score, self.min_score),
            "matched_categories": sorted_categories,
            "category_scores": matched,
        }
