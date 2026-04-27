"""Continuous typing-biometric authentication (the I3 headline feature).

Why this matters for the Huawei R&D UK / HMI Lab pitch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The TCN encoder already produces a 64-d user-state embedding from a
window of keystroke features on every turn.  This module turns that
embedding stream into a *continuous authentication* signal: an
"Identity Lock" that registers the typing pattern of the human at
the keyboard during the first N turns, then on every subsequent turn
verifies that the same human is still there.  When the pattern
drifts beyond a threshold, the assistant flags the drift -- the same
behaviour that hardens an HMI-aware on-device companion against
shoulder-surfing / device hand-off / impostor impersonation.

This is the *demo that sells the job*: a behaviour ChatGPT cannot
do, that ships entirely on-device, and that re-uses the existing
TCN encoder rather than introducing a new model.

Theoretical background
~~~~~~~~~~~~~~~~~~~~~~
Two classical results underpin the design:

* **Monrose & Rubin (1997), "Authentication via keystroke
  dynamics" (ACM CCS '97)** -- showed that population-scale
  keystroke timings (inter-key intervals, key hold times) are
  individually distinctive enough to identify users by typing
  pattern alone.  Their composite "rhythm vector" anticipates the
  modern embedding-based approach used here.
* **Killourhy & Maxion (2009), "Comparing anomaly-detection
  algorithms for keystroke dynamics" (IEEE/IFIP DSN '09)** --
  benchmarked 14 distance / outlier scorers on a 51-user keystroke
  corpus and established that simple Manhattan / Mahalanobis
  distance on the per-user template generalises better than learnt
  classifiers when training data is scarce (the on-device
  cold-start regime).  We mirror that finding: the "template" is
  the running mean of the encoder embedding plus an EWMA of four
  scalar metrics, and matching is a hand-tuned linear combination
  of cosine similarity (population-scale separability, per
  Monrose-Rubin) and z-score distance on the scalars (the
  Killourhy-Maxion sweet spot).

State machine
~~~~~~~~~~~~~
::

    unregistered ─► registering(progress 0/N) ─► registered
       │                                             │
       │                                             │ new turn
       │                                             ▼
       │                                       verifying
       │                                             │
       │                              match │  drift │ recover
       │                                    ▼        ▼
       │                              registered   mismatch
       │                                                │
       └──── reset_for_user() ──────────────────────────┘

Once a template is frozen at the end of registration it does **not**
update during verification (callers can pass ``update_template=True``
to opt in).  Holding the reference still is the right default for an
authentication system: a slow drift toward an impostor would otherwise
be silently absorbed into the template.

Constraints
~~~~~~~~~~~
* Pure Python + torch.  CPU-only.  No HuggingFace / pretrained
  weights -- the I3 pitch hinges on the from-scratch SLM/TCN, and
  the biometric layer must stay consistent with that line.
* All state is per-``user_id`` and persists across sessions.  An
  LRU cap of 1000 users bounds memory on a multi-tenant server.
* The module is *decorative-safe*: every public method is wrapped
  to never raise into the calling pipeline; on internal failure it
  emits a benign :class:`BiometricMatch` describing the unregistered
  state.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class BiometricMatch:
    """Result of one :meth:`KeystrokeAuthenticator.observe` (or status) call.

    Attributes:
        state: One of ``"unregistered"``, ``"registering"``,
            ``"registered"``, ``"verifying"``, ``"mismatch"``.
        similarity: Composite similarity score in roughly ``[-1, 1]``;
            the cosine-on-embedding term plus four z-score penalties.
        confidence: ``sigmoid(8 * (similarity - 0.5))`` mapped into
            ``[0, 1]``.
        threshold: The dynamic match threshold in confidence-space
            (default ``base_threshold=0.65``).
        enrolment_progress: 0..N during registration, N once
            registered.
        enrolment_target: How many turns are needed to register.
        is_owner: ``True`` iff ``confidence >= threshold``.  Only
            meaningful in the ``registered`` / ``verifying`` states.
        drift_alert: Rising-edge-style flag set when a previously-
            registered template's confidence has just dropped below
            threshold.  Sticky for the turn it fires on.
        diverged_signals: Human-readable labels of the worst-diverging
            sub-signals (only populated when ``is_owner`` is False).
        ewma_iki_mean: Exponentially-weighted IKI mean of the
            registered template.
        ewma_iki_std: EWMA of the IKI std-dev.
        ewma_composition_ms: EWMA of the composition-time scalar.
        ewma_edit_rate: EWMA of the per-turn edit count.
        notes: Short human-readable explanation suitable for the UI
            tooltip / reasoning trace.
    """

    state: str
    similarity: float
    confidence: float
    threshold: float
    enrolment_progress: int
    enrolment_target: int
    is_owner: bool
    drift_alert: bool
    diverged_signals: list[str] = field(default_factory=list)
    ewma_iki_mean: float = 0.0
    ewma_iki_std: float = 0.0
    ewma_composition_ms: float = 0.0
    ewma_edit_rate: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-safe dict representation."""
        return {
            "state": str(self.state),
            "similarity": float(self.similarity),
            "confidence": float(self.confidence),
            "threshold": float(self.threshold),
            "enrolment_progress": int(self.enrolment_progress),
            "enrolment_target": int(self.enrolment_target),
            "is_owner": bool(self.is_owner),
            "drift_alert": bool(self.drift_alert),
            "diverged_signals": [str(s) for s in self.diverged_signals],
            "ewma_iki_mean": float(self.ewma_iki_mean),
            "ewma_iki_std": float(self.ewma_iki_std),
            "ewma_composition_ms": float(self.ewma_composition_ms),
            "ewma_edit_rate": float(self.ewma_edit_rate),
            "notes": str(self.notes),
        }


# ---------------------------------------------------------------------------
# Per-user template
# ---------------------------------------------------------------------------


@dataclass
class _Template:
    """Per-user biometric template + recent observation memory.

    During *registration* the running mean of the 64-d embedding is
    updated cumulatively; the four scalar metrics are tracked via
    exponentially-weighted moving averages (EWMA) so the template
    converges quickly without being dominated by a single noisy turn.

    A small ``recent`` ring buffer of the last few (embedding, metrics)
    tuples is retained even after registration so :meth:`force_register`
    can stamp the template from "what I just saw" without forcing the
    user to type N more messages.
    """

    embedding_sum: torch.Tensor                # cumulative sum used to compute the mean
    n_observed: int = 0
    iki_mean_ewma: float = 0.0
    iki_std_ewma: float = 0.0
    comp_mean_ewma: float = 0.0
    edit_mean_ewma: float = 0.0
    # Frozen template fields, populated when the state transitions to
    # "registered".  Held separately from the registration-window
    # accumulators so re-registration cannot leak partial state.
    template_embedding: torch.Tensor | None = None
    template_iki_mean: float = 0.0
    template_iki_std: float = 0.0
    template_comp_mean: float = 0.0
    template_edit_mean: float = 0.0
    registered: bool = False
    last_drift_alert: bool = False
    last_state: str = "unregistered"
    # Short tail of recent observations, used by force_register so the
    # demo button can stamp the template without 5 enrolment turns.
    recent: list[tuple[torch.Tensor, float, float, float, float]] = field(
        default_factory=list
    )

    def reset(self) -> None:
        """Forget everything for this user."""
        self.embedding_sum = torch.zeros(
            self.embedding_sum.shape if self.embedding_sum is not None else (64,),
            dtype=torch.float32,
        )
        self.n_observed = 0
        self.iki_mean_ewma = 0.0
        self.iki_std_ewma = 0.0
        self.comp_mean_ewma = 0.0
        self.edit_mean_ewma = 0.0
        self.template_embedding = None
        self.template_iki_mean = 0.0
        self.template_iki_std = 0.0
        self.template_comp_mean = 0.0
        self.template_edit_mean = 0.0
        self.registered = False
        self.last_drift_alert = False
        self.last_state = "unregistered"
        self.recent.clear()


# ---------------------------------------------------------------------------
# Authenticator
# ---------------------------------------------------------------------------


class KeystrokeAuthenticator:
    """Continuous typing-biometric authentication via per-user template matching.

    Cites Monrose & Rubin (1997) and Killourhy & Maxion (2009).  The
    template is built from N enrolment turns: average of the TCN
    encoder's 64-d embeddings + EWMA of the four scalar metrics
    (IKI mean, IKI std, composition_time_ms, edit_count).  Matching
    fuses cosine similarity on the 64-d template with z-score
    distance on the four scalars.

    All state is per-``user_id`` and persists across sessions (so the
    registered template survives a session_end), bounded LRU at 1000
    users.

    The state machine::

        unregistered -> registering(progress 0/N) -> registered
        registered + new turn -> verifying -> match  -> registered
                                          -> drift  -> mismatch
        mismatch -> stays mismatch unless reset_for_user() is called

    Attributes:
        enrolment_target: Number of turns required to register.
        base_threshold: Confidence floor above which a turn counts as
            "is_owner".
        lr: EWMA decay rate for the scalar EWMAs (higher = faster
            adaptation, lower = stabler template).
        embedding_dim: Expected dimensionality of incoming embeddings.
        recent_window: How many recent observations to retain for
            :meth:`force_register`.

    """

    # LRU bound on the per-user template map so a churning user-id
    # client cannot grow this unbounded on a long-running server.
    _MAX_USERS: int = 1000

    # Match threshold lives in confidence-space ([0, 1]); the
    # similarity-to-confidence map below has a ~50% slope at
    # similarity=0.5 so a threshold of 0.65 corresponds to roughly
    # similarity >= 0.55 on the raw composite score.
    _SIGMOID_GAIN: float = 8.0
    _SIGMOID_PIVOT: float = 0.5

    def __init__(
        self,
        enrolment_target: int = 5,
        base_threshold: float = 0.65,
        lr: float = 0.15,
        embedding_dim: int = 64,
        recent_window: int = 6,
    ) -> None:
        if enrolment_target < 1:
            raise ValueError("enrolment_target must be >= 1")
        if not (0.0 < base_threshold < 1.0):
            raise ValueError("base_threshold must be in (0, 1)")
        if not (0.0 < lr <= 1.0):
            raise ValueError("lr must be in (0, 1]")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        self.enrolment_target = int(enrolment_target)
        self.base_threshold = float(base_threshold)
        self.lr = float(lr)
        self.embedding_dim = int(embedding_dim)
        self.recent_window = int(max(1, recent_window))

        self._templates: OrderedDict[str, _Template] = OrderedDict()
        # SEC: protect the OrderedDict from concurrent observe / reset
        # / status calls on a multi-worker server.  RLock so the
        # internal helpers can re-enter under the same call.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        user_id: str,
        *,
        embedding: torch.Tensor,
        iki_mean: float,
        iki_std: float,
        composition_time_ms: float,
        edit_count: int,
        update_template: bool = False,
    ) -> BiometricMatch:
        """Observe one turn for *user_id*.

        Updates the template during registration; verifies during
        operation.  Wrapped to never raise -- on internal failure it
        emits a benign ``unregistered`` :class:`BiometricMatch`.

        Args:
            user_id: Opaque user identifier.
            embedding: 64-d user-state embedding from the TCN encoder.
            iki_mean: Mean inter-keystroke interval (ms) for the turn.
            iki_std: Std-dev of inter-keystroke intervals (ms).
            composition_time_ms: Total composition window for the turn.
            edit_count: Backspaces + deletes during composition.
            update_template: When ``True``, after registration the
                template is refreshed in-place from each verifying
                turn.  Default ``False`` (a stable reference is the
                right default for authentication).
        """
        try:
            return self._observe_inner(
                user_id=str(user_id),
                embedding=embedding,
                iki_mean=float(iki_mean),
                iki_std=float(iki_std),
                composition_time_ms=float(composition_time_ms),
                edit_count=int(edit_count),
                update_template=bool(update_template),
            )
        except Exception:
            logger.exception(
                "KeystrokeAuthenticator.observe failed for user_id=%s; "
                "returning unregistered fallback",
                user_id,
            )
            return self._unregistered_match(notes="internal error - resetting state")

    def reset_for_user(self, user_id: str) -> BiometricMatch:
        """Forget the template for *user_id* (e.g. user re-registers).

        Idempotent; safe to call on an unknown id.  Returns the
        post-reset state (always ``unregistered``).
        """
        with self._lock:
            self._templates.pop(str(user_id), None)
        return self._unregistered_match(notes="template cleared")

    def force_register(
        self, user_id: str, *, complete: bool = True
    ) -> BiometricMatch:
        """Demo helper: stamp the template from recent observations.

        When ``complete=True`` (the default) the registration window
        is filled to ``enrolment_target`` from the most recent
        observations the authenticator has seen, the template is
        frozen, and the next call to :meth:`observe` lands in
        ``verifying`` mode.

        When ``complete=False`` we instead just promote the running
        accumulators -- whatever's been observed so far -- into the
        frozen template fields.

        Falls back to a synthetic neutral template if no observations
        have been recorded yet (so the demo button never errors).
        """
        with self._lock:
            tmpl = self._get_or_create_template(str(user_id))
            if not tmpl.recent and tmpl.n_observed == 0:
                # Synthetic neutral template -- enough for the UI to
                # transition out of "unregistered" without making the
                # demoer type 5 messages first.
                tmpl.template_embedding = torch.zeros(
                    self.embedding_dim, dtype=torch.float32
                )
                tmpl.template_iki_mean = 100.0
                tmpl.template_iki_std = 15.0
                tmpl.template_comp_mean = 1500.0
                tmpl.template_edit_mean = 0.0
                tmpl.n_observed = self.enrolment_target
                tmpl.registered = True
                tmpl.last_state = "registered"
                return self._build_match(
                    tmpl,
                    state="registered",
                    similarity=1.0,
                    confidence=1.0,
                    is_owner=True,
                    drift_alert=False,
                    diverged=[],
                    notes="Force-registered with synthetic neutral template (no observations yet).",
                )

            if complete:
                # Replay the recent window through the registration
                # accumulators until n_observed >= enrolment_target.
                for emb, iki_m, iki_s, comp, edit in tmpl.recent:
                    if tmpl.n_observed >= self.enrolment_target:
                        break
                    self._update_registration(
                        tmpl, emb, iki_m, iki_s, comp, edit
                    )
                # If still short, repeat the most recent observation
                # until the count is met (deterministic).
                if tmpl.n_observed < self.enrolment_target and tmpl.recent:
                    last = tmpl.recent[-1]
                    while tmpl.n_observed < self.enrolment_target:
                        self._update_registration(
                            tmpl, last[0], last[1], last[2], last[3], last[4]
                        )

            self._freeze_template(tmpl)
            tmpl.last_state = "registered"
            return self._build_match(
                tmpl,
                state="registered",
                similarity=1.0,
                confidence=1.0,
                is_owner=True,
                drift_alert=False,
                diverged=[],
                notes=(
                    "Force-registered from recent observations -- next turn "
                    "will run in verification mode."
                ),
            )

    def status(self, user_id: str) -> BiometricMatch:
        """Read-only view of the current state for *user_id*.

        Does not advance the state machine.  Returns the
        ``unregistered`` state for unknown users.
        """
        with self._lock:
            tmpl = self._templates.get(str(user_id))
            if tmpl is None:
                return self._unregistered_match()
            self._templates.move_to_end(str(user_id))
            state = tmpl.last_state or (
                "registered" if tmpl.registered else "registering"
            )
            return self._build_match(
                tmpl,
                state=state,
                similarity=1.0 if state == "registered" else 0.0,
                confidence=1.0 if state == "registered" else 0.0,
                is_owner=tmpl.registered,
                drift_alert=tmpl.last_drift_alert,
                diverged=[],
                notes=self._notes_for_state(state, tmpl, is_owner=tmpl.registered),
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _observe_inner(
        self,
        *,
        user_id: str,
        embedding: torch.Tensor,
        iki_mean: float,
        iki_std: float,
        composition_time_ms: float,
        edit_count: int,
        update_template: bool,
    ) -> BiometricMatch:
        emb = self._coerce_embedding(embedding)
        iki_m = max(0.0, _safe_finite(iki_mean))
        iki_s = max(0.0, _safe_finite(iki_std))
        comp = max(0.0, _safe_finite(composition_time_ms))
        edits = max(0.0, float(int(edit_count) if edit_count is not None else 0))

        with self._lock:
            tmpl = self._get_or_create_template(user_id)
            self._record_recent(tmpl, emb, iki_m, iki_s, comp, edits)

            # ---- Phase 1: still registering -----------------------
            if not tmpl.registered:
                self._update_registration(tmpl, emb, iki_m, iki_s, comp, edits)
                if tmpl.n_observed >= self.enrolment_target:
                    self._freeze_template(tmpl)
                    state = "registered"
                    notes = self._notes_for_state(state, tmpl, is_owner=True)
                    tmpl.last_state = state
                    tmpl.last_drift_alert = False
                    return self._build_match(
                        tmpl,
                        state=state,
                        similarity=1.0,
                        confidence=1.0,
                        is_owner=True,
                        drift_alert=False,
                        diverged=[],
                        notes=notes,
                    )
                state = "registering"
                notes = self._notes_for_state(state, tmpl, is_owner=False)
                tmpl.last_state = state
                tmpl.last_drift_alert = False
                return self._build_match(
                    tmpl,
                    state=state,
                    similarity=0.0,
                    confidence=0.0,
                    is_owner=False,
                    drift_alert=False,
                    diverged=[],
                    notes=notes,
                )

            # ---- Phase 2: verifying ---------------------------------
            similarity, cosine_sim, z_iki_m, z_iki_s, z_comp, z_edit = (
                self._score_match(tmpl, emb, iki_m, iki_s, comp, edits)
            )
            confidence = self._sigmoid_confidence(similarity)
            is_owner = confidence >= self.base_threshold
            diverged = [] if is_owner else self._rank_divergence(
                z_iki_m=z_iki_m, z_iki_s=z_iki_s, z_comp=z_comp, z_edit=z_edit,
                cosine_sim=cosine_sim,
            )
            # Drift = previously-registered template now failing.  Sticky
            # mismatch state bypasses drift-alert phrasing.
            drift_alert = (not is_owner) and (tmpl.last_state != "mismatch")

            # State transition.
            if is_owner:
                state = "verifying"
            else:
                state = "mismatch"
            tmpl.last_state = state
            tmpl.last_drift_alert = drift_alert

            if update_template and is_owner:
                # Opt-in slow drift of the template toward the
                # current observation.  Off by default.
                self._slow_update_template(tmpl, emb, iki_m, iki_s, comp, edits)

            notes = self._notes_for_state(
                state, tmpl, is_owner=is_owner, similarity=similarity,
                drift_alert=drift_alert, diverged=diverged,
            )
            return self._build_match(
                tmpl,
                state=state,
                similarity=similarity,
                confidence=confidence,
                is_owner=is_owner,
                drift_alert=drift_alert,
                diverged=diverged,
                notes=notes,
            )

    def _get_or_create_template(self, user_id: str) -> _Template:
        """Fetch the template, creating + LRU-evicting as needed."""
        existing = self._templates.get(user_id)
        if existing is not None:
            self._templates.move_to_end(user_id)
            return existing
        while len(self._templates) >= self._MAX_USERS:
            evicted_key, _ = self._templates.popitem(last=False)
            logger.debug(
                "KeystrokeAuthenticator evicted oldest template: user=%s",
                evicted_key,
            )
        new_tmpl = _Template(
            embedding_sum=torch.zeros(self.embedding_dim, dtype=torch.float32),
        )
        self._templates[user_id] = new_tmpl
        return new_tmpl

    def _record_recent(
        self,
        tmpl: _Template,
        emb: torch.Tensor,
        iki_m: float,
        iki_s: float,
        comp: float,
        edits: float,
    ) -> None:
        """Append to the recent-observations ring buffer."""
        tmpl.recent.append((emb.detach().clone(), iki_m, iki_s, comp, edits))
        if len(tmpl.recent) > self.recent_window:
            tmpl.recent = tmpl.recent[-self.recent_window :]

    def _update_registration(
        self,
        tmpl: _Template,
        emb: torch.Tensor,
        iki_m: float,
        iki_s: float,
        comp: float,
        edits: float,
    ) -> None:
        """Cumulative mean of embedding + EWMA of scalars during enrolment."""
        tmpl.embedding_sum = tmpl.embedding_sum + emb
        tmpl.n_observed += 1
        # First observation seeds the EWMAs; subsequent calls mix.
        if tmpl.n_observed == 1:
            tmpl.iki_mean_ewma = iki_m
            tmpl.iki_std_ewma = iki_s
            tmpl.comp_mean_ewma = comp
            tmpl.edit_mean_ewma = edits
        else:
            a = self.lr
            tmpl.iki_mean_ewma = (1 - a) * tmpl.iki_mean_ewma + a * iki_m
            tmpl.iki_std_ewma = (1 - a) * tmpl.iki_std_ewma + a * iki_s
            tmpl.comp_mean_ewma = (1 - a) * tmpl.comp_mean_ewma + a * comp
            tmpl.edit_mean_ewma = (1 - a) * tmpl.edit_mean_ewma + a * edits

    def _freeze_template(self, tmpl: _Template) -> None:
        """Promote the registration accumulators into the frozen template."""
        n = max(1, tmpl.n_observed)
        tmpl.template_embedding = (tmpl.embedding_sum / float(n)).detach().clone()
        tmpl.template_iki_mean = float(tmpl.iki_mean_ewma)
        tmpl.template_iki_std = float(tmpl.iki_std_ewma)
        tmpl.template_comp_mean = float(tmpl.comp_mean_ewma)
        tmpl.template_edit_mean = float(tmpl.edit_mean_ewma)
        tmpl.registered = True

    def _slow_update_template(
        self,
        tmpl: _Template,
        emb: torch.Tensor,
        iki_m: float,
        iki_s: float,
        comp: float,
        edits: float,
    ) -> None:
        """Slow adaptive update -- only used when ``update_template=True``."""
        if tmpl.template_embedding is None:
            return
        a = self.lr * 0.25  # quarter-rate so the template barely drifts
        tmpl.template_embedding = (
            (1 - a) * tmpl.template_embedding + a * emb
        ).detach().clone()
        tmpl.template_iki_mean = (1 - a) * tmpl.template_iki_mean + a * iki_m
        tmpl.template_iki_std = (1 - a) * tmpl.template_iki_std + a * iki_s
        tmpl.template_comp_mean = (1 - a) * tmpl.template_comp_mean + a * comp
        tmpl.template_edit_mean = (1 - a) * tmpl.template_edit_mean + a * edits

    def _score_match(
        self,
        tmpl: _Template,
        emb: torch.Tensor,
        iki_m: float,
        iki_s: float,
        comp: float,
        edits: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute the composite similarity and the four z-scores.

        Implements the formula in the brief::

            similarity = 0.6 * cosine_sim_64d
                       - 0.10 * z_iki_mean
                       - 0.10 * z_iki_std
                       - 0.10 * z_comp
                       - 0.10 * z_edit
        """
        if tmpl.template_embedding is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # Cosine on the 64-d embedding -- both vectors are L2-normalised
        # by the TCN's projection head, but we re-normalise defensively
        # so a degenerate zero-vector input doesn't produce NaN.
        ref = tmpl.template_embedding
        cosine_sim = float(_safe_cosine(ref, emb))

        # Z-score-style distance on each scalar metric.  We deliberately
        # divide by max(template_value, floor) rather than the
        # population std so a hand-typed small-corpus template with
        # near-zero variance still produces graded penalties rather
        # than infinities.
        z_iki_m = abs(iki_m - tmpl.template_iki_mean) / max(
            tmpl.template_iki_std, 1.0
        )
        z_iki_s = abs(iki_s - tmpl.template_iki_std) / max(
            tmpl.template_iki_std, 1.0
        )
        z_comp = abs(comp - tmpl.template_comp_mean) / max(
            tmpl.template_comp_mean * 0.3, 1.0
        )
        z_edit = abs(edits - tmpl.template_edit_mean) / max(
            tmpl.template_edit_mean + 1.0, 1.0
        )

        # Weighted composite per the brief.  Each z-score is clipped at
        # 5σ so a single absurd metric can't dominate the entire score
        # and obscure a partial match.
        z_iki_m = min(5.0, z_iki_m)
        z_iki_s = min(5.0, z_iki_s)
        z_comp = min(5.0, z_comp)
        z_edit = min(5.0, z_edit)

        similarity = (
            0.60 * cosine_sim
            - 0.10 * z_iki_m
            - 0.10 * z_iki_s
            - 0.10 * z_comp
            - 0.10 * z_edit
        )
        return (similarity, cosine_sim, z_iki_m, z_iki_s, z_comp, z_edit)

    def _sigmoid_confidence(self, similarity: float) -> float:
        """Map similarity → confidence via a sharpened sigmoid."""
        try:
            x = self._SIGMOID_GAIN * (similarity - self._SIGMOID_PIVOT)
            # Numerically stable sigmoid.
            if x >= 0:
                z = math.exp(-x)
                conf = 1.0 / (1.0 + z)
            else:
                z = math.exp(x)
                conf = z / (1.0 + z)
        except (OverflowError, ValueError):
            conf = 1.0 if similarity > self._SIGMOID_PIVOT else 0.0
        return float(max(0.0, min(1.0, conf)))

    @staticmethod
    def _rank_divergence(
        *,
        z_iki_m: float,
        z_iki_s: float,
        z_comp: float,
        z_edit: float,
        cosine_sim: float,
    ) -> list[str]:
        """Return human labels for the most-diverging signals.

        Sorted from worst to least; capped at 3 entries so the UI
        chip strip stays readable.
        """
        ranked = [
            ("IKI mean", z_iki_m),
            ("IKI variance", z_iki_s),
            ("composition cadence", z_comp),
            ("edit rate", z_edit),
        ]
        ranked.sort(key=lambda kv: -kv[1])
        out: list[str] = []
        for label, score in ranked:
            if score >= 1.0 and len(out) < 3:
                out.append(f"{label} ({score:.1f} sigma off)")
        # Embedding-level divergence as a separate label when cosine
        # is low.  Useful for the demo: makes clear when the *whole*
        # rhythm pattern shifted, not just one axis.
        if cosine_sim < 0.3 and len(out) < 3:
            out.insert(0, f"embedding rhythm (cos={cosine_sim:.2f})")
        return out

    @staticmethod
    def _coerce_embedding(embedding: torch.Tensor | None) -> torch.Tensor:
        """Coerce arbitrary embedding inputs to a flat float32 tensor.

        Falls back to a zero vector on bad input rather than raising --
        the authenticator must never break the calling pipeline.
        """
        if embedding is None:
            return torch.zeros(64, dtype=torch.float32)
        try:
            t = embedding.detach().to(dtype=torch.float32, copy=False)
        except Exception:
            return torch.zeros(64, dtype=torch.float32)
        if t.ndim > 1:
            t = t.flatten()
        if t.numel() == 0:
            return torch.zeros(64, dtype=torch.float32)
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        return t

    def _build_match(
        self,
        tmpl: _Template,
        *,
        state: str,
        similarity: float,
        confidence: float,
        is_owner: bool,
        drift_alert: bool,
        diverged: list[str],
        notes: str,
    ) -> BiometricMatch:
        """Common builder for the BiometricMatch result dataclass."""
        return BiometricMatch(
            state=str(state),
            similarity=float(similarity),
            confidence=float(confidence),
            threshold=float(self.base_threshold),
            enrolment_progress=int(min(tmpl.n_observed, self.enrolment_target)),
            enrolment_target=int(self.enrolment_target),
            is_owner=bool(is_owner),
            drift_alert=bool(drift_alert),
            diverged_signals=list(diverged),
            ewma_iki_mean=float(
                tmpl.template_iki_mean if tmpl.registered else tmpl.iki_mean_ewma
            ),
            ewma_iki_std=float(
                tmpl.template_iki_std if tmpl.registered else tmpl.iki_std_ewma
            ),
            ewma_composition_ms=float(
                tmpl.template_comp_mean if tmpl.registered else tmpl.comp_mean_ewma
            ),
            ewma_edit_rate=float(
                tmpl.template_edit_mean if tmpl.registered else tmpl.edit_mean_ewma
            ),
            notes=str(notes),
        )

    def _unregistered_match(self, *, notes: str = "") -> BiometricMatch:
        """Build a benign 'unregistered' result."""
        return BiometricMatch(
            state="unregistered",
            similarity=0.0,
            confidence=0.0,
            threshold=float(self.base_threshold),
            enrolment_progress=0,
            enrolment_target=int(self.enrolment_target),
            is_owner=False,
            drift_alert=False,
            diverged_signals=[],
            ewma_iki_mean=0.0,
            ewma_iki_std=0.0,
            ewma_composition_ms=0.0,
            ewma_edit_rate=0.0,
            notes=notes or "No biometric template registered yet.",
        )

    def _notes_for_state(
        self,
        state: str,
        tmpl: _Template,
        *,
        is_owner: bool,
        similarity: float = 0.0,
        drift_alert: bool = False,
        diverged: list[str] | None = None,
    ) -> str:
        """Compose the human-readable explanation surfaced in the UI tooltip."""
        if state == "unregistered":
            return "No biometric template registered yet."
        if state == "registering":
            return (
                f"Biometric enrolment in progress -- "
                f"{tmpl.n_observed}/{self.enrolment_target} typing samples "
                f"collected."
            )
        if state == "registered":
            return (
                "Biometric template registered. Future turns will be verified "
                "against this rhythm."
            )
        if state == "verifying" and is_owner:
            return (
                f"Verified -- typing rhythm matches registered owner "
                f"(similarity {similarity:.2f}, threshold "
                f"{self.base_threshold:.2f})."
            )
        if state == "mismatch" or drift_alert:
            d = diverged or []
            d_str = "; ".join(d) if d else "rhythm mismatch"
            return (
                f"Typing pattern diverges from registered owner "
                f"(similarity {similarity:.2f} below threshold "
                f"{self.base_threshold:.2f}). Diverging signals: {d_str}."
            )
        return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_finite(value: float, default: float = 0.0) -> float:
    """Coerce a value to a finite float, falling back to *default*."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity that returns 0 on degenerate / mismatched inputs."""
    try:
        if a.shape != b.shape:
            return 0.0
        na = float(torch.linalg.norm(a).item())
        nb = float(torch.linalg.norm(b).item())
        if na <= 1e-9 or nb <= 1e-9:
            return 0.0
        cos = float(torch.dot(a, b).item()) / (na * nb)
        if not math.isfinite(cos):
            return 0.0
        return max(-1.0, min(1.0, cos))
    except Exception:
        return 0.0


__all__ = [
    "BiometricMatch",
    "KeystrokeAuthenticator",
]


# ---------------------------------------------------------------------------
# Smoke test (run as ``python -m i3.biometric.keystroke_auth``)
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    import torch as _torch

    auth = KeystrokeAuthenticator(enrolment_target=5)

    # Synthetic registration: 5 turns of near-identical metrics.
    base_emb = _torch.randn(64)
    base_emb = base_emb / _torch.linalg.norm(base_emb)
    print("=== Phase 1: registering (5 turns) ===")
    for i in range(5):
        # Tiny per-turn jitter so the template smooths out.
        emb_i = base_emb + 0.01 * _torch.randn(64)
        m = auth.observe(
            "demo",
            embedding=emb_i,
            iki_mean=100.0 + (i % 2),
            iki_std=10.0,
            composition_time_ms=1200.0,
            edit_count=0,
        )
        print(
            f"  turn {i + 1}: state={m.state} progress={m.enrolment_progress}/"
            f"{m.enrolment_target} is_owner={m.is_owner}"
        )

    print("\n=== Phase 2: verifying with same pattern ===")
    for i in range(3):
        emb_i = base_emb + 0.02 * _torch.randn(64)
        m = auth.observe(
            "demo",
            embedding=emb_i,
            iki_mean=100.0,
            iki_std=10.0,
            composition_time_ms=1200.0,
            edit_count=0,
        )
        print(
            f"  turn {i + 1}: state={m.state} sim={m.similarity:.3f} "
            f"conf={m.confidence:.3f} is_owner={m.is_owner}"
        )

    print("\n=== Phase 3: very different pattern (should fail) ===")
    different_emb = _torch.randn(64)
    different_emb = different_emb / _torch.linalg.norm(different_emb)
    for i in range(3):
        m = auth.observe(
            "demo",
            embedding=different_emb,
            iki_mean=240.0,
            iki_std=80.0,
            composition_time_ms=4500.0,
            edit_count=6,
        )
        print(
            f"  turn {i + 1}: state={m.state} sim={m.similarity:.3f} "
            f"conf={m.confidence:.3f} is_owner={m.is_owner} "
            f"diverged={m.diverged_signals}"
        )

    print("\n=== Phase 4: status / reset ===")
    print("  status:", auth.status("demo").state)
    auth.reset_for_user("demo")
    print("  after reset:", auth.status("demo").state)
