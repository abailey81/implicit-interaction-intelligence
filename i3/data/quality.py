"""Quality filtering for ingested records.

A :class:`QualityFilter` is a composition of named rules; each rule is
a pure callable that returns ``True`` iff the record is *acceptable*.
A rejected record is logged with the rule name, allowing the
:class:`QualityReport` to surface rejection-rate breakdowns.

Built-in rules:

- ``min_length`` / ``max_length`` — token count bounds.
- ``latin_ratio`` — fraction of ASCII letters must exceed a threshold
  (catches OCR garbage and non-primary-language records when the
  pipeline is configured for English-only corpora).
- ``unique_token_ratio`` — rejects repetitive junk like "lol lol lol …".
- ``no_embedded_urls`` — optional.  Rejects records that look like
  pure URL dumps.
- ``no_email_density`` — rejects records where emails are the
  majority of tokens (spam fingerprint).
- ``no_control_density`` — rejects records with abnormally high
  proportion of non-printable characters (OCR / binary leakage).
- ``profanity_budget`` — a small in-file profanity list; rejects when
  the ratio exceeds a threshold.  Deliberately conservative — false
  positives reject less data than false negatives would poison the
  model.

Every rule is independently testable and carries a short description
so the data card can cite exactly which rules rejected how many
records.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

from i3.data.lineage import RecordSchema

#: Tiny profanity list — English, mild + severe.  Deliberately kept
#: small to minimise false positives on research content.  Users who
#: need stricter filtering should provide a custom :class:`QualityRule`.
_PROFANITY: frozenset[str] = frozenset({
    "damn", "hell", "crap", "shit", "fuck", "fucking", "bitch",
    "bastard", "asshole", "dick", "piss",
})

_URL_RE: re.Pattern[str] = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE: re.Pattern[str] = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
)
_TOKEN_RE: re.Pattern[str] = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class QualityRule:
    """A named predicate over :class:`RecordSchema`.

    Attributes:
        name: Stable identifier — used as the key in
            :attr:`QualityReport.rejected_by_rule`.
        description: One-line human-readable explanation.
        check: Callable returning ``True`` iff the record is acceptable.
    """

    name: str
    description: str
    check: Callable[[RecordSchema], bool]

    def __call__(self, record: RecordSchema) -> bool:
        return self.check(record)


@dataclass(slots=True)
class QualityReport:
    """Aggregate rejection-rate statistics for a run.

    Attributes:
        total_seen: Number of records the filter saw.
        accepted: Number of records that passed every rule.
        rejected_by_rule: Mapping ``rule_name -> rejection count``.
        length_histogram: Bucketed token-length histogram over
            accepted records (10, 25, 50, 100, 250, 500, 1000+).
    """

    total_seen: int = 0
    accepted: int = 0
    rejected_by_rule: Counter[str] = field(default_factory=Counter)
    length_histogram: Counter[str] = field(default_factory=Counter)

    @property
    def rejection_rate(self) -> float:
        if self.total_seen == 0:
            return 0.0
        return 1.0 - (self.accepted / self.total_seen)

    def as_dict(self) -> dict:
        return {
            "total_seen": self.total_seen,
            "accepted": self.accepted,
            "rejection_rate": round(self.rejection_rate, 4),
            "rejected_by_rule": dict(self.rejected_by_rule),
            "length_histogram": dict(self.length_histogram),
        }


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


def rule_min_length(min_tokens: int = 2) -> QualityRule:
    """Reject records shorter than ``min_tokens`` alphanumeric tokens."""
    def _check(r: RecordSchema) -> bool:
        return len(_TOKEN_RE.findall(r.text)) >= min_tokens
    return QualityRule(
        name="min_length",
        description=f"token count >= {min_tokens}",
        check=_check,
    )


def rule_max_length(max_tokens: int = 2048) -> QualityRule:
    """Reject records longer than ``max_tokens``."""
    def _check(r: RecordSchema) -> bool:
        return len(_TOKEN_RE.findall(r.text)) <= max_tokens
    return QualityRule(
        name="max_length",
        description=f"token count <= {max_tokens}",
        check=_check,
    )


def rule_latin_ratio(min_ratio: float = 0.7) -> QualityRule:
    """Reject when the ASCII-letter ratio falls below ``min_ratio``.

    This is a cheap proxy for "predominantly English content"; the
    pipeline does not ship a real langid model so this rule gates on
    the character distribution instead.
    """
    def _check(r: RecordSchema) -> bool:
        if not r.text:
            return False
        letters = sum(1 for ch in r.text if ch.isalpha())
        if letters == 0:
            return False
        ascii_letters = sum(
            1 for ch in r.text if "a" <= ch.lower() <= "z"
        )
        return (ascii_letters / letters) >= min_ratio
    return QualityRule(
        name="latin_ratio",
        description=f"ASCII-letter ratio >= {min_ratio:.2f}",
        check=_check,
    )


def rule_unique_token_ratio(min_ratio: float = 0.25) -> QualityRule:
    """Reject repetitive junk (``"lol lol lol lol"``)."""
    def _check(r: RecordSchema) -> bool:
        tokens = _TOKEN_RE.findall(r.text.lower())
        if len(tokens) < 4:
            return True  # skip for short records
        return (len(set(tokens)) / len(tokens)) >= min_ratio
    return QualityRule(
        name="unique_token_ratio",
        description=f"unique/total token ratio >= {min_ratio:.2f}",
        check=_check,
    )


def rule_no_url_dump(max_url_density: float = 0.3) -> QualityRule:
    """Reject records where URLs are the majority of tokens."""
    def _check(r: RecordSchema) -> bool:
        tokens = _TOKEN_RE.findall(r.text)
        if not tokens:
            return False
        urls = len(_URL_RE.findall(r.text))
        return (urls / max(1, len(tokens))) < max_url_density
    return QualityRule(
        name="no_url_dump",
        description=f"URL density < {max_url_density:.2f}",
        check=_check,
    )


def rule_no_email_dump(max_email_density: float = 0.2) -> QualityRule:
    def _check(r: RecordSchema) -> bool:
        tokens = _TOKEN_RE.findall(r.text)
        if not tokens:
            return False
        emails = len(_EMAIL_RE.findall(r.text))
        return (emails / max(1, len(tokens))) < max_email_density
    return QualityRule(
        name="no_email_dump",
        description=f"email density < {max_email_density:.2f}",
        check=_check,
    )


def rule_no_control_density(max_density: float = 0.05) -> QualityRule:
    """Reject records where >5 % of chars are non-printable junk."""
    def _check(r: RecordSchema) -> bool:
        if not r.text:
            return False
        import unicodedata
        bad = sum(
            1 for ch in r.text
            if ch not in "\n\t" and unicodedata.category(ch) in {"Cc", "Cn"}
        )
        return (bad / len(r.text)) <= max_density
    return QualityRule(
        name="no_control_density",
        description=f"non-printable-char density <= {max_density:.2f}",
        check=_check,
    )


def rule_profanity_budget(max_ratio: float = 0.1) -> QualityRule:
    """Reject records whose profanity ratio exceeds ``max_ratio``."""
    def _check(r: RecordSchema) -> bool:
        tokens = _TOKEN_RE.findall(r.text.lower())
        if not tokens:
            return True
        bad = sum(1 for t in tokens if t in _PROFANITY)
        return (bad / len(tokens)) <= max_ratio
    return QualityRule(
        name="profanity_budget",
        description=f"profanity ratio <= {max_ratio:.2f}",
        check=_check,
    )


def default_rules() -> list[QualityRule]:
    """Return the canonical English-dialogue rule set."""
    return [
        rule_min_length(min_tokens=2),
        rule_max_length(max_tokens=2048),
        rule_latin_ratio(min_ratio=0.7),
        rule_unique_token_ratio(min_ratio=0.25),
        rule_no_url_dump(max_url_density=0.3),
        rule_no_email_dump(max_email_density=0.2),
        rule_no_control_density(max_density=0.05),
        rule_profanity_budget(max_ratio=0.1),
    ]


# ---------------------------------------------------------------------------
# Filter orchestrator
# ---------------------------------------------------------------------------


class QualityFilter:
    """Apply a sequence of :class:`QualityRule` to every record.

    The filter is fail-fast per record: the first rule that rejects
    decides the rejection reason.  ``report`` is updated in-place.
    """

    def __init__(self, rules: Iterable[QualityRule] | None = None) -> None:
        self.rules: list[QualityRule] = list(rules or default_rules())
        self.report: QualityReport = QualityReport()

    def accept(self, record: RecordSchema) -> bool:
        """Return ``True`` iff ``record`` passes every rule."""
        self.report.total_seen += 1
        for rule in self.rules:
            if not rule(record):
                self.report.rejected_by_rule[rule.name] += 1
                return False
        self.report.accepted += 1
        n_tokens = len(_TOKEN_RE.findall(record.text))
        self.report.length_histogram[_bucket_len(n_tokens)] += 1
        return True

    def filter(self, records: Iterable[RecordSchema]) -> list[RecordSchema]:
        return [r for r in records if self.accept(r)]


def _bucket_len(n: int) -> str:
    for cap in (10, 25, 50, 100, 250, 500, 1000):
        if n <= cap:
            return f"<={cap}"
    return ">1000"


__all__ = [
    "QualityFilter",
    "QualityReport",
    "QualityRule",
    "default_rules",
    "rule_latin_ratio",
    "rule_max_length",
    "rule_min_length",
    "rule_no_control_density",
    "rule_no_email_dump",
    "rule_no_url_dump",
    "rule_profanity_budget",
    "rule_unique_token_ratio",
]
