"""Per-session privacy budget + PII counter for the cloud LLM route.

Tracks every byte that crosses the network boundary towards the cloud
LLM, every PII redaction performed by :class:`PrivacySanitizer` before
that crossing happens, and per-category counts (email / phone / IP /
SSN / credit-card / address / DOB / URL).  When the per-session budget
is exhausted the engine falls back to the edge SLM with a clear
``routing_decision.reason``.

Design rationale
~~~~~~~~~~~~~~~~
The cloud route is *opt-in*.  Even when the user has flipped the
consent toggle on, the system enforces hard upper bounds on:

* total cloud calls per session  (default 50)
* total bytes shipped to the cloud per session  (default 1 MB)

These bounds are NOT user-tunable from the UI.  They are policy
invariants of the demo: a misbehaving caller, an unbounded
conversation, or an overly-eager bandit cannot blow past them.

The companion :data:`bytes_redacted_total` counter tells a *positive*
story: how many bytes WOULD have been shipped if we hadn't sanitised
them.  This is the visible "value-add of the sanitiser" number the
Privacy tab surfaces.

Privacy
~~~~~~~
This module never persists raw text and never makes a network call.
It accepts already-sanitised strings + scalar metrics and returns
scalar counters.  All state lives in memory and is bounded.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# Default policy bounds.  Operators can override via env vars but the
# UI cannot.
_DEFAULT_MAX_CLOUD_CALLS_PER_SESSION = 50
_DEFAULT_MAX_BYTES_PER_SESSION = 1_000_000  # 1 MB

# Maximum number of (user_id, session_id) pairs to track concurrently.
# Bounded LRU to prevent unbounded growth on a long-running server.
_MAX_TRACKED_SESSIONS = 1_000


@dataclass
class PrivacyBudgetSnapshot:
    """JSON-safe snapshot of a per-(user, session) privacy budget.

    Attributes:
        cloud_calls_total: Number of cloud LLM calls billed against the
            budget so far.
        cloud_calls_max: Hard ceiling on cloud calls for the session.
        pii_redactions_total: Number of individual PII redactions
            performed across all calls (e.g. 3 emails + 1 phone = 4).
        bytes_transmitted_total: Cumulative bytes that left the host
            *after* sanitisation (the actual wire cost).
        bytes_transmitted_max: Hard ceiling on bytes transmitted.
        bytes_redacted_total: Cumulative bytes that WOULD have left the
            host had the sanitiser not been in the path.  This number
            equals the bytes of the original sanitised tokens — the
            visible "value-add of the sanitiser".
        sensitive_categories: Per-category redaction counts.  Keys
            include ``email``, ``phone``, ``url``, ``ip_address``,
            ``credit_card``, ``ssn``, ``address``, ``dob``.  Always
            present (zeroed when no hits) so the UI can render bars
            without conditionals.
        last_call_ts: Unix epoch seconds of the most recent cloud call,
            or ``None`` if no call has been made.
        budget_remaining_calls: ``cloud_calls_max - cloud_calls_total``.
        budget_remaining_bytes: ``bytes_transmitted_max -
            bytes_transmitted_total``, never below ``0``.
        consent_enabled: ``True`` if the user has explicitly opted in
            for cloud route on this session.  Mirrors the consent
            toggle state for the UI's convenience.
    """

    cloud_calls_total: int = 0
    cloud_calls_max: int = _DEFAULT_MAX_CLOUD_CALLS_PER_SESSION
    pii_redactions_total: int = 0
    bytes_transmitted_total: int = 0
    bytes_transmitted_max: int = _DEFAULT_MAX_BYTES_PER_SESSION
    bytes_redacted_total: int = 0
    sensitive_categories: dict[str, int] = field(default_factory=dict)
    last_call_ts: float | None = None
    budget_remaining_calls: int = _DEFAULT_MAX_CLOUD_CALLS_PER_SESSION
    budget_remaining_bytes: int = _DEFAULT_MAX_BYTES_PER_SESSION
    consent_enabled: bool = False

    def to_dict(self) -> dict:
        """Serialise to a plain JSON-safe dict."""
        return asdict(self)


# Always-present category keys so UI bars never disappear.
_DEFAULT_CATEGORIES: tuple[str, ...] = (
    "email", "phone", "url", "ip_address",
    "credit_card", "ssn", "address", "dob",
)


def _zero_categories() -> dict[str, int]:
    return {k: 0 for k in _DEFAULT_CATEGORIES}


@dataclass
class _Bucket:
    """Mutable per-(user, session) state held in memory."""

    cloud_calls_total: int = 0
    pii_redactions_total: int = 0
    bytes_transmitted_total: int = 0
    bytes_redacted_total: int = 0
    sensitive_categories: dict[str, int] = field(default_factory=_zero_categories)
    last_call_ts: float | None = None
    consent_enabled: bool = False


class PrivacyBudget:
    """Per-(user, session) cloud privacy budget tracker.

    Public API:

    * :meth:`set_consent` — flip the per-user cloud-consent flag
      (default OFF).  Returns the new state.
    * :meth:`consent` — query the per-user cloud-consent flag.
    * :meth:`can_call` — check whether a cloud call is currently
      allowed for ``(user_id, session_id)``.  Returns ``(allowed,
      reason)``.
    * :meth:`record_call` — bill a successful cloud call against the
      budget.  Updates byte / redaction counters atomically.
    * :meth:`snapshot` — JSON-safe :class:`PrivacyBudgetSnapshot` for
      the UI.
    * :meth:`reset_session` — drop the per-(user, session) bucket
      (called from the engine on ``end_session``).

    Concurrency: every public method takes ``self._lock`` so concurrent
    handlers cannot race the counters.

    Privacy: this class NEVER stores raw text.  It accepts
    already-sanitised strings and scalar metrics and returns scalar
    counters.
    """

    def __init__(
        self,
        *,
        max_cloud_calls_per_session: int = _DEFAULT_MAX_CLOUD_CALLS_PER_SESSION,
        max_bytes_per_session: int = _DEFAULT_MAX_BYTES_PER_SESSION,
        max_tracked_sessions: int = _MAX_TRACKED_SESSIONS,
    ) -> None:
        self.max_cloud_calls_per_session: int = int(max_cloud_calls_per_session)
        self.max_bytes_per_session: int = int(max_bytes_per_session)
        self.max_tracked_sessions: int = int(max_tracked_sessions)
        # Per-(user, session) buckets, LRU-bounded.
        self._buckets: OrderedDict[tuple[str, str], _Bucket] = OrderedDict()
        # Per-user consent flag.  Independent of session lifetime so
        # toggling the consent before a new session is created still
        # has effect.
        self._consent: dict[str, bool] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Consent
    # ------------------------------------------------------------------

    def set_consent(self, user_id: str, enabled: bool) -> bool:
        """Set the cloud-route consent flag for *user_id*.

        Returns the new state (mirrors the input).  Default for any
        unknown ``user_id`` is ``False`` — opt-in by design.
        """
        with self._lock:
            self._consent[str(user_id)] = bool(enabled)
            # Reflect the new state on every existing bucket for this
            # user so the UI snapshot is consistent without an extra
            # round-trip.
            for (uid, _sid), bucket in self._buckets.items():
                if uid == user_id:
                    bucket.consent_enabled = bool(enabled)
            logger.info(
                "Cloud-route consent for user_id=%s set to %s",
                user_id,
                bool(enabled),
            )
            return bool(enabled)

    def consent(self, user_id: str) -> bool:
        """Return the cloud-route consent flag for *user_id*.

        Default is ``False`` (opt-in).
        """
        with self._lock:
            return bool(self._consent.get(str(user_id), False))

    # ------------------------------------------------------------------
    # Budget gating
    # ------------------------------------------------------------------

    def can_call(self, user_id: str, session_id: str) -> tuple[bool, str]:
        """Return whether a cloud call is currently permitted.

        Returns:
            A two-tuple ``(allowed, reason)``.  ``reason`` is a short
            string suitable for a routing-decision message, e.g.
            ``"cloud budget exhausted"``.
        """
        with self._lock:
            if not self._consent.get(str(user_id), False):
                return False, "cloud consent off"
            bucket = self._get_or_create_bucket(user_id, session_id)
            if bucket.cloud_calls_total >= self.max_cloud_calls_per_session:
                return False, "cloud call budget exhausted"
            if bucket.bytes_transmitted_total >= self.max_bytes_per_session:
                return False, "cloud byte budget exhausted"
            return True, "within budget"

    # ------------------------------------------------------------------
    # Accounting
    # ------------------------------------------------------------------

    def record_call(
        self,
        user_id: str,
        session_id: str,
        *,
        sanitised_prompt: str,
        response_text: str,
        pii_redactions: int,
        pii_categories: dict[str, int] | None = None,
        bytes_in: int | None = None,
        bytes_out: int | None = None,
        bytes_redacted: int = 0,
    ) -> PrivacyBudgetSnapshot:
        """Bill a single cloud LLM call against the budget.

        Args:
            user_id: Owning user identifier.
            session_id: Owning session identifier.
            sanitised_prompt: The prompt text *after* sanitisation.
                Only used to compute byte size if ``bytes_in`` is
                ``None``.  Never persisted.
            response_text: The cloud reply text.  Only used to compute
                byte size if ``bytes_out`` is ``None``.  Never
                persisted.
            pii_redactions: Total number of PII redactions performed
                across the prompt + history before transmission.
            pii_categories: Optional per-category counts (matches
                :data:`PrivacySanitizer.PII_PATTERNS` keys).
            bytes_in: Bytes shipped to the cloud (after sanitisation).
                If ``None``, computed from ``sanitised_prompt``.
            bytes_out: Bytes received from the cloud.  If ``None``,
                computed from ``response_text``.
            bytes_redacted: Bytes that WOULD have been shipped if the
                sanitiser hadn't redacted them.  This is the
                "value-add of the sanitiser" number.

        Returns:
            A :class:`PrivacyBudgetSnapshot` with all counters updated.
        """
        if bytes_in is None:
            bytes_in = len((sanitised_prompt or "").encode("utf-8"))
        if bytes_out is None:
            bytes_out = len((response_text or "").encode("utf-8"))

        with self._lock:
            bucket = self._get_or_create_bucket(user_id, session_id)
            bucket.cloud_calls_total += 1
            bucket.pii_redactions_total += int(max(0, pii_redactions))
            bucket.bytes_transmitted_total += int(max(0, bytes_in)) + int(
                max(0, bytes_out)
            )
            bucket.bytes_redacted_total += int(max(0, bytes_redacted))
            bucket.last_call_ts = time.time()
            if pii_categories:
                for k, v in pii_categories.items():
                    if not isinstance(k, str):
                        continue
                    try:
                        bucket.sensitive_categories[k] = (
                            bucket.sensitive_categories.get(k, 0) + int(v)
                        )
                    except (TypeError, ValueError):
                        continue
            return self._snapshot_locked(user_id, session_id, bucket)

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot(
        self, user_id: str, session_id: str
    ) -> PrivacyBudgetSnapshot:
        """Return a JSON-safe snapshot of the current budget.

        Always returns a populated snapshot, even when no cloud calls
        have been made (counters are zero, categories are
        zero-filled).
        """
        with self._lock:
            bucket = self._get_or_create_bucket(user_id, session_id)
            return self._snapshot_locked(user_id, session_id, bucket)

    def reset_session(self, user_id: str, session_id: str) -> None:
        """Drop the per-(user, session) bucket.

        Called from the engine on ``end_session``.  Idempotent — a
        second call after a missing bucket is a no-op.
        """
        with self._lock:
            self._buckets.pop((str(user_id), str(session_id)), None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_create_bucket(
        self, user_id: str, session_id: str
    ) -> _Bucket:
        """Look up (or create) the bucket for ``(user_id, session_id)``.

        LRU-touches the entry so frequent users stay in memory; evicts
        the oldest entry when the cap is hit.  MUST be called under
        ``self._lock``.
        """
        key = (str(user_id), str(session_id))
        if key in self._buckets:
            self._buckets.move_to_end(key)
            bucket = self._buckets[key]
        else:
            bucket = _Bucket()
            bucket.consent_enabled = bool(self._consent.get(str(user_id), False))
            self._buckets[key] = bucket
            # Bound the dict.
            while len(self._buckets) > self.max_tracked_sessions:
                self._buckets.popitem(last=False)
        return bucket

    def _snapshot_locked(
        self,
        user_id: str,
        session_id: str,
        bucket: _Bucket,
    ) -> PrivacyBudgetSnapshot:
        """Build a :class:`PrivacyBudgetSnapshot` from *bucket*.

        MUST be called under ``self._lock``.
        """
        # Make sure every default category is present so UI bars
        # render consistently.
        cats = dict(_zero_categories())
        cats.update(bucket.sensitive_categories or {})
        consent = bool(self._consent.get(str(user_id), False))
        return PrivacyBudgetSnapshot(
            cloud_calls_total=bucket.cloud_calls_total,
            cloud_calls_max=self.max_cloud_calls_per_session,
            pii_redactions_total=bucket.pii_redactions_total,
            bytes_transmitted_total=bucket.bytes_transmitted_total,
            bytes_transmitted_max=self.max_bytes_per_session,
            bytes_redacted_total=bucket.bytes_redacted_total,
            sensitive_categories=cats,
            last_call_ts=bucket.last_call_ts,
            budget_remaining_calls=max(
                0,
                self.max_cloud_calls_per_session - bucket.cloud_calls_total,
            ),
            budget_remaining_bytes=max(
                0,
                self.max_bytes_per_session - bucket.bytes_transmitted_total,
            ),
            consent_enabled=consent,
        )
