"""Pipeline Orchestration Engine for Implicit Interaction Intelligence (I3).

This module contains the central :class:`Pipeline` class that connects every
subsystem into a single end-to-end message-processing flow:

    InteractionMonitor --> PrivacySanitizer --> TCN Encoder --> UserModel
        --> AdaptationController --> Router --> Response Generation
        --> DiaryLogger --> PipelineOutput

Design principles
~~~~~~~~~~~~~~~~~
1. **Privacy by architecture** -- raw user text is used transiently for
   feature extraction and response generation but is **never persisted**.
   Only embeddings, scalar metrics, and topic keywords reach the diary.
2. **Graceful degradation** -- if any component is unavailable (e.g. no
   cloud API key, no trained SLM checkpoint), the pipeline falls back to
   the next-best option rather than crashing.
3. **Async throughout** -- all I/O-bound operations (database writes,
   cloud API calls) are awaited, keeping the event loop responsive.
4. **Per-user isolation** -- each user gets their own
   :class:`~src.user_model.model.UserModel` instance, ensuring that
   baseline tracking and session state are fully independent.

Lifecycle
~~~~~~~~~
1. Construct: ``pipeline = Pipeline(config)``
2. Initialise: ``await pipeline.initialize()``
3. Start session: ``session_id = await pipeline.start_session(user_id)``
4. Process messages: ``output = await pipeline.process_message(input)``
5. End session: ``summary = await pipeline.end_session(user_id, session_id)``
6. Shutdown: ``await pipeline.shutdown()``
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from i3.config import Config
from i3.pipeline.types import (
    EngagementEstimator,
    PipelineInput,
    PipelineOutput,
)

logger = logging.getLogger(__name__)


def _looks_coherent(text: str) -> bool:
    """Heuristic: does *text* look like a real reply, not word salad?

    The tiny autoregressive SLM emits confused strings on novel
    prompts — duplicate tokens, orphan punctuation, split contractions
    like ``you 're``, or aimless repetitions of the same word.  Those
    should fall through to the OOD branch rather than be shown to the
    user.  Accept only when *all* of these hold:

    * 4–60 words long,
    * at least one alphabetic word of length ≥ 3,
    * no more than one orphan single-letter non-``I`` token,
    * not almost entirely filler words,
    * no word is repeated 3+ times consecutively (catches
      "your your your your"),
    * the most common word doesn't dominate (catches
      "transformer transformer transformer ..."),
    * at least 60% of lowercase content words are unique (catches
      sentence-level loops).
    """
    if not text:
        return False
    words = text.strip().split()
    if not (4 <= len(words) <= 80):
        return False
    if not any(len(w) >= 3 and w.isalpha() for w in words):
        return False
    orphan_count = sum(
        1 for w in words if len(w) == 1 and w.lower() != "i" and w.isalpha()
    )
    if orphan_count > 1:
        return False
    filler = {"a", "an", "the", "and", "to", "of", "in", "is", "it", "so"}
    lower = [w.lower().strip(".,!?;:'\"") for w in words]
    if sum(1 for w in lower if w in filler) >= len(lower) * 0.7:
        return False

    # Adjacent-repetition: "your your your your" or "the the the".
    streak = 1
    for a, b in zip(lower, lower[1:]):
        if a == b and a:
            streak += 1
            if streak >= 3:
                return False
        else:
            streak = 1

    # Single-word dominance: any one token making up more than 28% of
    # the reply is a strong word-salad signal.
    if lower:
        from collections import Counter

        most = Counter(lower).most_common(1)[0]
        if most[1] / len(lower) > 0.28 and len(most[0]) > 1:
            return False

    # Unique-content ratio: too few unique non-stopwords means the
    # generator is looping on a small set of tokens.
    content = [w for w in lower if w and w not in filler and len(w) > 1]
    if content and len(set(content)) / len(content) < 0.6:
        return False

    # Malformed contractions — the SLM at low training maturity emits
    # bogus tokens like "the's", "what'm", "i's", "small's", "can's"
    # by appending an apostrophe-suffix to a word that can't host one.
    # Strategy: allow the standard English contraction inventory verbatim
    # plus possessive 's on regular nouns, reject everything else.
    known_contractions = {
        "i'm", "i've", "i'd", "i'll",
        "you're", "you've", "you'd", "you'll",
        "he's", "he'd", "he'll", "she's", "she'd", "she'll", "it's",
        "we're", "we've", "we'd", "we'll",
        "they're", "they've", "they'd", "they'll",
        "isn't", "aren't", "wasn't", "weren't",
        "hasn't", "haven't", "hadn't",
        "don't", "doesn't", "didn't", "won't", "shan't",
        "can't", "couldn't", "shouldn't", "wouldn't",
        "mustn't", "needn't", "ain't",
        "that's", "who's", "what's", "where's", "when's", "how's", "why's",
        "let's", "there's", "here's", "y'all", "o'clock",
    }
    bad_contractions = 0
    for raw in words:
        if "'" not in raw:
            continue
        norm = raw.lower().strip(".,!?;:\"")
        if norm in known_contractions:
            continue
        parts = norm.split("'")
        if len(parts) != 2:
            bad_contractions += 1
            continue
        head, tail = parts
        if not head or not tail:
            bad_contractions += 1
            continue
        if not (head.isalpha() and tail.isalpha()):
            bad_contractions += 1
            continue
        # Possessive 's on a normal noun is fine ("demo's", "shenzhen's"),
        # but determiners and bare pronouns can never take a possessive 's.
        if tail == "s":
            if head in {"the", "a", "an", "this", "these", "those"}:
                bad_contractions += 1
                continue
            if head in {"i", "you", "we", "they"}:
                bad_contractions += 1
                continue
            continue  # accept generic possessive
        # Any non-'s, non-known contraction shape ("what'm", "small're",
        # "can'd") is malformed.
        bad_contractions += 1
    if bad_contractions >= 1:
        return False

    return True


class Pipeline:
    """Main orchestration pipeline connecting all I3 components.

    Flow per message:

    1. **InteractionMonitor**: extract 32-dim feature vector from
       keystroke metrics, message text, and session history.
    2. **PrivacySanitizer**: strip PII from message text before cloud
       transmission.
    3. **TCN Encoder**: encode the feature-vector window into a 64-dim
       ``UserStateEmbedding``.
    4. **UserModel**: update three-timescale representation and compute
       deviations from the user's personal baseline.
    5. **AdaptationController**: compute 8-dim ``AdaptationVector`` from
       the current feature vector and deviation metrics.
    6. **Router**: decide between local SLM and cloud LLM using the
       contextual Thompson sampling bandit.
    7. **Response Generation**: generate the response via the chosen
       route, conditioned on the adaptation vector and user state.
    8. **DiaryLogger**: log the exchange to SQLite (no raw text stored).
    9. **Return** :class:`PipelineOutput` with state updates for the
       frontend dashboard.

    Parameters:
        config: Fully validated :class:`~src.config.Config` instance
            providing settings for every subsystem.

    Example::

        from i3.config import load_config
        from i3.pipeline import Pipeline, PipelineInput

        config = load_config("configs/default.yaml")
        pipeline = Pipeline(config)
        await pipeline.initialize()

        session_id = await pipeline.start_session("user_42")
        output = await pipeline.process_message(PipelineInput(
            user_id="user_42",
            session_id=session_id,
            message_text="Hello, how are you?",
            timestamp=time.time(),
            composition_time_ms=3200.0,
            edit_count=1,
            pause_before_send_ms=450.0,
        ))
        print(output.response_text)
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # ---- Import all component classes --------------------------------
        # Imports are deferred to __init__ rather than module level so that
        # the pipeline module can be imported without triggering heavy
        # transitive imports (torch, numpy, etc.) at parse time.
        from i3.adaptation.controller import AdaptationController
        from i3.affect.accessibility_mode import AccessibilityController
        from i3.affect.shift_detector import AffectShift, AffectShiftDetector
        from i3.affect.state_classifier import classify_user_state  # noqa: F401
        from i3.cloud.client import CloudLLMClient
        from i3.cloud.postprocess import ResponsePostProcessor
        from i3.cloud.prompt_builder import PromptBuilder
        from i3.diary.store import DiaryStore
        from i3.encoder.inference import EncoderInference
        from i3.interaction.monitor import InteractionMonitor
        from i3.privacy.encryption import ModelEncryptor
        from i3.privacy.sanitizer import PrivacySanitizer
        from i3.router.bandit import ContextualThompsonBandit
        from i3.router.complexity import QueryComplexityEstimator
        from i3.router.sensitivity import TopicSensitivityDetector

        # ---- Interaction monitoring --------------------------------------
        self.monitor = InteractionMonitor(
            feature_window=config.interaction.feature_window,
            baseline_warmup=config.user_model.baseline_warmup,
        )

        # ---- TCN encoder (lazy -- needs checkpoint path) -----------------
        self._encoder: EncoderInference | None = None
        self._encoder_config = config.encoder

        # ---- Voice-prosody multimodal fusion (FLAGSHIP feature #2) -------
        # Browser-side WebAudio extracts 8 prosodic scalars (pace, pitch
        # mean/var, RMS energy mean/var, voiced ratio, pause density,
        # spectral centroid) — never raw audio — and ships them on the
        # ``message`` frame.  The encoder maps them to a 32-d prosody
        # embedding, fused with the 64-d keystroke embedding into a 96-d
        # multimodal user-state.  Identity-init means the keystroke half
        # passes through unchanged when joint training hasn't run yet,
        # so this is a strict superset of the keystroke-only baseline.
        # See :mod:`i3.multimodal.prosody` for the privacy contract +
        # citations (Schuller 2009, Eyben et al. 2010 openSMILE).
        from i3.multimodal.prosody import (
            GazeEncoder,
            MultimodalFusion,
            ProsodyEncoder,
        )
        self._prosody_encoder: ProsodyEncoder = ProsodyEncoder()
        self._prosody_encoder.eval()
        # Vision-gaze flagship feature: an 8-d gaze feature vector
        # (4 class probs + presence + blink + stability + confidence)
        # is fed through GazeEncoder → 32-d gaze embedding and fused
        # alongside keystroke + prosody into a 128-d multimodal vector.
        # See :mod:`i3.multimodal.gaze_classifier` for the fine-tuned
        # MobileNetV3-small backbone story.
        self._gaze_encoder: GazeEncoder = GazeEncoder()
        self._gaze_encoder.eval()
        self._multimodal_fusion: MultimodalFusion = MultimodalFusion(
            key_dim=config.encoder.embedding_dim,
            prosody_dim=32,
            gaze_dim=32,
            out_dim=config.encoder.embedding_dim + 32 + 32,
        )
        self._multimodal_fusion.eval()

        # Per-user fine-tuned gaze classifier head.  Each user gets
        # their own :class:`~i3.multimodal.GazeClassifier` instance,
        # mapped by SHA-256(biometric template) like the personalisation
        # adapters.  The frozen MobileNetV3-small backbone is shared
        # across all instances (loaded once, lazy).  Capped at 1000
        # cached classifiers to bound memory; per-head ≤300 KB on disk.
        from collections import OrderedDict as _OD2  # local alias
        self._gaze_classifiers: _OD2[str, Any] = _OD2()
        self._max_gaze_classifiers: int = 1000
        # Path on disk where calibrated heads persist.
        self._gaze_ckpt_dir = "checkpoints/gaze"

        # ---- Per-user models (LRU-capped) --------------------------------
        # PERF (H-4, 2026-04-23 audit): bound the per-user map so a
        # long-running multi-tenant server (or a client rotating
        # user ids) cannot grow this unbounded to OOM.  ``OrderedDict``
        # is indexed with ``move_to_end`` on access in
        # ``_get_or_create_user_model`` to keep eviction LRU-ordered.
        from collections import OrderedDict  # local import to avoid top-level churn
        self._max_users: int = int(
            os.environ.get("I3_MAX_TRACKED_USERS", "10000")
        )
        self.user_models: OrderedDict[str, Any] = OrderedDict()

        # ---- Per-session conversation history (LRU-capped) ---------------
        # Sliding-window history of the last N user/assistant turn pairs
        # keyed by ``"{user_id}::{session_id}"``. Each value is a list of
        # ``(user_text, assistant_text)`` tuples ordered oldest first and
        # capped at :data:`_history_max_turns` pairs. The retriever and
        # SLM both consume this so prompts like "what about that?" or
        # "explain the first one" can be resolved against prior context.
        #
        # The dict is bounded LRU-style (cap = 1000 sessions) so a long-
        # running server cannot grow this unbounded if clients churn
        # session ids without ever calling ``end_session``.
        # Phase B.5 (2026-04-25): bumped from 4 → 8 default turns.  The
        # hierarchical memory adds topic-stack + user-fact tiers on top
        # so the engine effectively sees 8 verbatim pairs PLUS a 50-turn
        # decayed topic stack PLUS user-stated facts.  Override with
        # I3_HISTORY_TURNS env var.
        self._history_max_turns: int = max(
            1, int(os.environ.get("I3_HISTORY_TURNS", "8"))
        )
        self._max_sessions_tracked: int = int(
            os.environ.get("I3_MAX_TRACKED_SESSIONS", "1000")
        )
        self._session_histories: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
        # Last-turn history-length recorded per session for the WS layer
        # to surface in the reasoning trace ("Working from N prior
        # turns of context"). Keyed by the same ``user_id::session_id``.
        self._last_history_turns_used: dict[str, int] = {}

        # ---- Per-(user, session) entity tracker + co-reference resolver ---
        # Closes the multi-turn understanding gap: a follow-up like
        # "where are they located?" gets rewritten to
        # "where is huawei located?" before it reaches retrieval, so
        # the entity-knowledge tool route in the retriever can answer
        # deterministically instead of falling back to a high-cosine
        # but semantically-wrong dialogue match.  Wrapped in try/except
        # at every call site so a tracker failure can never block a
        # turn — single-turn behaviour is the worst-case degradation.
        # ---- Hierarchical session memory (Phase B.5, 2026-04-25) ----------
        # Topic stack + user-stated facts + recurring thread summary.
        # Bounded internally; graceful no-op when the import fails.
        try:
            from i3.dialogue.memory import HierarchicalMemory
            self._hierarchical_memory: HierarchicalMemory | None = (
                HierarchicalMemory()
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to construct HierarchicalMemory; long-context "
                "memory will be disabled."
            )
            self._hierarchical_memory = None

        from i3.dialogue.coref import EntityTracker
        self._entity_tracker: EntityTracker = EntityTracker(
            max_entities_per_session=16, max_sessions=1000,
        )
        # Stash the most recent ResolutionResult per session so the WS
        # layer can pull it onto the response/response_done frame
        # without having to re-run the resolver.
        self._last_coref: dict[str, Any] = {}

        # ---- Adaptation --------------------------------------------------
        self.adaptation = AdaptationController(config.adaptation)

        # ---- Router (contextual Thompson sampling bandit) ----------------
        self.router = ContextualThompsonBandit(
            n_arms=len(config.router.arms),
            context_dim=config.router.context_dim,
            exploration_bonus=config.router.exploration_bonus,
        )
        self.complexity_estimator = QueryComplexityEstimator()
        self.sensitivity_detector = TopicSensitivityDetector()

        # Cloud-vs-edge prompt-complexity estimator (CPU-only).  Used
        # by the LinUCB bandit's feature vector to decide whether the
        # current prompt is hard enough to justify a cloud round-trip.
        # Distinct from ``complexity_estimator`` above which feeds the
        # general routing context — this one specifically scores
        # "how hard is this prompt for the on-device SLM?".  See
        # :mod:`i3.router.complexity_estimator` for the design.
        from i3.router.complexity_estimator import PromptComplexityEstimator
        self._prompt_complexity_estimator = PromptComplexityEstimator(
            tokenizer=None,  # tokenizer is wired up after SLM init
        )

        # ---- Response generation -----------------------------------------
        self._slm_generator: Any | None = None  # Lazy init
        self._slm_retriever: Any | None = None  # Lazy init, paired with SLM
        # Version tag — populated by ``_load_slm_v2`` / ``load_slm`` so
        # the /api/stack endpoint and the audit script can tell the
        # current SLM generation apart from the legacy 4.5 M v1 model.
        self._slm_version: str | None = None
        self.cloud_client = CloudLLMClient(config)
        self.prompt_builder = PromptBuilder()
        self.postprocessor = ResponsePostProcessor()

        # ---- Self-critique loop (Phase 7 HMI pitch piece) ----------------
        # Rule-based critic that scores SLM drafts against on-topic /
        # well-formed / non-repetitive / safe / adaptation-match
        # rubrics.  When the score is below threshold the SLM is
        # re-decoded once with tighter sampling and the better of the
        # two attempts is kept.  See ``i3.critique.critic`` for the
        # rubric details and the constitutional-AI / self-refine
        # references behind the design.  Pure Python, deterministic,
        # <5 ms per call — never invoked outside the SLM path.
        from i3.critique.critic import SelfCritic
        self._self_critic: SelfCritic = SelfCritic()
        # Stash for the WS layer: populated by ``_generate_response_inner``
        # on every SLM turn, cleared (or left stale) on retrieval / tool /
        # OOD turns.  ``_process_message_inner`` only forwards it to
        # ``PipelineOutput.critique`` when ``response_path == "slm"``,
        # so leakage from a previous SLM turn cannot mislabel a later
        # retrieval turn.
        self._last_critique: dict = {}

        # ---- Privacy -----------------------------------------------------
        self.sanitizer = PrivacySanitizer(config.privacy.strip_pii)

        # ---- Cloud-route privacy budget ---------------------------------
        # Enforces hard upper bounds on per-session cloud LLM calls and
        # bytes transmitted, plus the per-user opt-in consent flag.
        # Defaults: 50 calls, 1 MB.  Operators can override via env
        # vars; the UI cannot lift the ceiling.  See
        # :mod:`i3.privacy.budget` for the full contract.
        from i3.privacy.budget import PrivacyBudget
        self.privacy_budget: PrivacyBudget = PrivacyBudget(
            max_cloud_calls_per_session=int(
                os.environ.get("I3_CLOUD_MAX_CALLS_PER_SESSION", "50")
            ),
            max_bytes_per_session=int(
                os.environ.get("I3_CLOUD_MAX_BYTES_PER_SESSION", "1000000")
            ),
        )
        # Most-recent routing decision, populated on every turn by
        # ``_make_routing_decision``.  Forwarded onto
        # :attr:`PipelineOutput.routing_decision` and stashed on a
        # per-pipeline ring buffer (last 50) so
        # ``GET /api/routing/decision/recent`` can replay them for the
        # Routing-tab scatter plot.
        from collections import deque as _deque
        self._last_routing_decision: dict | None = None
        self._recent_routing_decisions: _deque[dict] = _deque(maxlen=50)
        # Most-recent privacy-budget snapshot, populated on every turn
        # so the WS layer can ship it without a separate REST call.
        self._last_privacy_budget_snapshot: dict | None = None

        # ---- Retrieval routing floor (Phase B.1, 2026-04-25) -------------
        # The cosine threshold above which we *prefer retrieval over SLM*
        # on substantive queries (≥4 content words).  The 0.92 path is
        # always-commit, the 0.85 path is short-query-commit, and this
        # floor (default 0.75) is the substantive-query commit threshold.
        # Mutable at runtime via :meth:`set_retrieval_floor`; the
        # playground tab uses that to demonstrate routing sensitivity.
        try:
            self._retrieval_floor: float = float(
                os.environ.get("I3_RETRIEVAL_FLOOR", "0.75")
            )
        except Exception:
            self._retrieval_floor = 0.75

        # SEC: Instantiate the embedding encryptor whenever the config
        # enables encryption at rest. When no key is available, the
        # encryptor warns and falls back to an ephemeral key; the pipeline
        # still routes embeddings through the versioned envelope format
        # so operators can retrofit a real key later without a migration.
        self._encryptor: ModelEncryptor | None = None
        if getattr(config.privacy, "encrypt_embeddings", False):
            try:
                self._encryptor = ModelEncryptor(
                    key_env_var=getattr(
                        config.privacy, "encryption_key_env", "I3_ENCRYPTION_KEY"
                    )
                )
                self._encryptor.initialize()
                logger.info(
                    "Embedding encryption ENABLED; stores will write Fernet envelopes."
                )
            except Exception as exc:
                logger.error(
                    "Failed to initialise ModelEncryptor (%s); embeddings will "
                    "be written plaintext. Set I3_ENCRYPTION_KEY to enable "
                    "encryption at rest.",
                    type(exc).__name__,
                )
                self._encryptor = None

        # ---- Diary -------------------------------------------------------
        # SEC: DiaryStore receives the same encryptor so per-exchange embeddings
        # are written via the versioned envelope.
        self.diary_store = DiaryStore(
            config.diary.db_path, encryptor=self._encryptor
        )
        self._diary_logger: Any | None = None  # After store init

        # ---- Engagement tracking (per-user) ------------------------------
        self._last_response_time: dict[str, float] = {}
        self._last_response_length: dict[str, int] = {}
        # Iter 48: lightweight session-stated user name.  When the user
        # says "my name is X" / "call me X" / "I'm X" we record X here
        # keyed by ``(user_id, session_id)`` so subsequent "what's my
        # name" probes can return it.  Volatile (in-memory only) and
        # cleared by ``end_session``.
        self._stated_user_name: dict[tuple[str, str], str] = {}
        # Iter 49: multi-fact session memory — generalises the iter-48
        # name slot to a per-(user, session) dict of {slot: value}.
        # Slots populated by ``_maybe_handle_fact_statement``: name
        # (mirrored), favourite_color, favourite_food, favourite_music,
        # occupation, location, hobby, age, pet.  Recall via
        # "what's my favourite color", "where do I live", etc.
        self._stated_facts: dict[tuple[str, str], dict[str, str]] = {}
        # Iter 55: rolling per-cascade-arm latency tracker.
        # Keys: "slm" / "qwen_intent" / "gemini_cloud" / "retrieval" /
        # "tool" / "other".  Values: collections.deque(maxlen=200) of
        # latency_ms floats from each turn that fired the arm.  Used by
        # GET /api/cascade/stats to expose live p50/p95 to the dashboard.
        from collections import deque as _deque
        self._cascade_arm_latencies: dict[str, Any] = {
            "slm": _deque(maxlen=200),
            "qwen_intent": _deque(maxlen=200),
            "gemini_cloud": _deque(maxlen=200),
            "retrieval": _deque(maxlen=200),
            "tool": _deque(maxlen=200),
            "other": _deque(maxlen=200),
        }
        self._previous_engagement: dict[str, float] = {}
        self._previous_route: dict[str, int] = {}
        # SEC: the contextual bandit's Laplace approximation requires the
        # *same* routing-context vector used at arm-selection time to be
        # passed back when the engagement reward arrives. Storing only
        # the integer arm index (``_previous_route``) and then fabricating
        # a zero context at update time degenerates the posterior — the
        # bandit learns reward-vs-zero-vector instead of
        # reward-vs-actual-routing-context. We persist the vector here
        # so ``compute_engagement()`` can replay it verbatim.
        self._previous_routing_context: dict[str, np.ndarray] = {}
        self.engagement_estimator = EngagementEstimator()

        # ---- Affect-shift detector (HMI showpiece) -----------------------
        # Tracks per-session rolling user-state embeddings + keystroke
        # metrics and emits an :class:`AffectShift` per turn.  When a
        # shift fires, ``_process_message_inner`` appends the
        # detector's polite check-in to the visible response.  See
        # :mod:`i3.affect.shift_detector` for the full contract.
        self._shift_detector: AffectShiftDetector = AffectShiftDetector()
        # The most recent AffectShift produced for any user; the WS
        # layer reads this back when building the reasoning trace.
        # Keyed on ``"{user_id}::{session_id}"`` so concurrent users
        # don't cross-pollute.
        self._last_affect_shift: dict[str, AffectShift | None] = {}

        # ---- Discrete user-state classifier (Live State Badge) -----------
        # Pure-Python, deterministic, no torch.  Classifies the live
        # 8-d adaptation + raw keystroke metrics into one of six
        # discrete states (calm / focused / stressed / tired /
        # distracted / warming up) for the nav badge.  The result is
        # stashed alongside the affect shift so the WS layer can
        # surface it on every state_update *and* every response.
        self._last_user_state_label: dict[str, dict | None] = {}

        # ---- Accessibility-mode auto-switch ------------------------------
        # Sticky per-session state machine: when sustained motor
        # difficulty / dyslexia signals are observed we engage
        # accessibility mode (font scale up, vocab simplified, TTS
        # rate slowed).  Deactivates only after a 4-turn recovery
        # window so a single calmer turn doesn't undo it.
        self._accessibility_controller: AccessibilityController = AccessibilityController()
        self._last_accessibility_state: dict[str, dict | None] = {}

        # ---- Continuous typing-biometric auth (Identity Lock) ------------
        # The HEADLINE I3 feature: per-user keystroke template + match
        # decision on every turn.  See i3.biometric.keystroke_auth for
        # the state machine + Monrose-Rubin / Killourhy-Maxion design
        # rationale.  State is per-user_id (not per-session), so a
        # registered user's template survives session_end.
        from i3.biometric.keystroke_auth import (
            BiometricMatch,
            KeystrokeAuthenticator,
        )
        self._keystroke_auth: KeystrokeAuthenticator = KeystrokeAuthenticator()
        self._last_biometric: dict[str, dict | None] = {}
        # Track previous biometric state per user_id so the WS layer
        # can emit rising-edge biometric_event frames (registered,
        # drift_alert, mismatch).  Keyed on user_id.
        self._last_biometric_state: dict[str, str] = {}

        # ---- Per-biometric LoRA personalisation (FLAGSHIP novelty) ------
        # Per-user low-rank adapter layered onto the base
        # AdaptationVector, keyed by SHA-256 hash of the typing-
        # biometric template embedding.  Trained online from the A/B
        # preference picker; never federated, never leaves the device.
        # See i3.personalisation.lora_adapter for the full design,
        # citations (Hu et al. 2021 LoRA, Houlsby et al. 2019).
        from pathlib import Path as _Path

        from i3.personalisation import PersonalisationManager
        self._personalisation: PersonalisationManager = PersonalisationManager(
            d_state=64,
            d_adapt=8,
            rank=4,
            alpha=8.0,
            lr=1.0,
            storage_dir=_Path("checkpoints") / "personalisation",
        )
        # Most recent personalisation result per user_id — surfaced on
        # PipelineOutput.personalisation and read by the WS layer.
        self._last_personalisation: dict[str, dict] = {}
        # Most recent 64-d TCN user_state embedding per user_id — the
        # personalisation manager needs this to project the residual
        # via W_a @ user_state.  Refreshed on every turn.
        self._last_user_state_embedding: dict[str, torch.Tensor] = {}
        # Per-user_id cache of the picked / rejected adaptation profiles
        # last offered to the user via the A/B selector.  The
        # preference-record route uses this to look up the right
        # 8-d profiles when applying a contrastive update.  Bounded
        # at the same _MAX_USERS as the keystroke auth.
        self._preference_offer_cache: OrderedDict[str, dict] = OrderedDict()

        # ---- Per-(user_id, session_id) cognitive-profile aggregator -----
        # Bounded LRU map of running stats consumed by the Profile tab.
        # Each entry holds histories + counts so /api/profile/{user}/
        # {session} can return a snapshot dict without hitting the
        # diary store.  See _profile_update / _profile_snapshot below.
        self._profile_aggregator: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._profile_max_sessions: int = 1000

        # ---- Per-turn pipeline trace collector (Flow dashboard) ---------
        # Records per-stage timings + arrow flows for every turn so the
        # front-end "Flow" tab can animate every component pulsing in
        # real time.  Bounded ring buffer of 200 traces fronts the
        # ``GET /api/flow/recent`` route.  The collector is pure-Python,
        # never blocks, and only ships ≤ 4 KB of summary per turn.
        from i3.observability.pipeline_trace import PipelineTraceCollector
        self._trace_collector: PipelineTraceCollector = PipelineTraceCollector(
            max_traces_in_memory=int(
                os.environ.get("I3_FLOW_TRACE_BUFFER", "200")
            )
        )

        # ---- Constitutional safety classifier (char-CNN, ~48k params) ----
        # Replaces the regex-based hostility filter with a learned
        # 4-class char-level CNN trained on a synthetic constitutional
        # corpus.  Verdict drives an early return on `refuse` and a
        # post-hoc caveat on `review`. See i3/safety/classifier.py for
        # the architecture + Bai et al. 2022 reference.
        try:
            from i3.safety.classifier import get_global_classifier
            self._safety_classifier = get_global_classifier()
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Safety classifier failed to load; pipeline will run "
                "without the constitutional safety layer."
            )
            self._safety_classifier = None
        # Most recent SafetyVerdict per (user, session) so the WS layer
        # can attach the chip without re-running classification.
        self._last_safety: dict[str, dict | None] = {}

        # ---- Playground overrides (Deliverable 3) ------------------------
        # Per-session counter for the "max 100 playground requests per
        # session" cap.  Bounded LRU-style alongside ``_session_histories``.
        self._playground_call_counts: dict[str, int] = {}
        self._max_playground_calls_per_session: int = 100

        # ---- Pipeline state ----------------------------------------------
        self._initialized = False

        # SEC: Track fire-and-forget tasks in a set so they are not garbage-
        # collected mid-flight (Python GC can drop unreferenced tasks, which
        # silently cancels them and loses their exceptions). See PEP 3156 and
        # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # SEC: asyncio.Lock guarding mutation of self.user_models. Without it,
        # two concurrent process_message() calls for a *new* user could each
        # construct a UserModel and the second would clobber the first, losing
        # any state mutations that happened on the first instance.
        self._user_models_lock = asyncio.Lock()

    @property
    def encoder(self):
        """Public alias for the lazily-loaded TCN encoder ``nn.Module``.

        ``server/routes_explain.py`` and ``server/routes_whatif.py`` look
        up ``pipeline.encoder`` and expect a raw :class:`torch.nn.Module`
        (so ``MCDropoutAdaptationEstimator`` can flip its dropout layers
        into train mode).  ``self._encoder`` is the
        :class:`EncoderInference` wrapper, so we drill one level deeper
        and return its ``.model`` — the actual TCN.
        """
        if self._encoder is None:
            return None
        return getattr(self._encoder, "model", None)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Perform async initialisation (database setup, optional model loading).

        Must be awaited before :meth:`process_message` is called.  This
        method is idempotent -- calling it more than once is safe.
        """
        if self._initialized:
            logger.debug("Pipeline.initialize() called but already initialised.")
            return

        # 1. Diary store (creates SQLite tables)
        await self.diary_store.initialize()

        # 2. Diary logger (depends on store being ready)
        try:
            from i3.diary.logger import DiaryLogger
            self._diary_logger = DiaryLogger(self.diary_store, self.config)
            logger.info("DiaryLogger initialised.")
        except ImportError:
            logger.warning(
                "DiaryLogger not available (src.diary.logger not found). "
                "Exchange logging will be skipped."
            )

        # 3. Try to load the TCN encoder if a checkpoint exists
        self._try_load_encoder()

        # 4. Try to load the trained SLM if a checkpoint exists.  Without
        #    this the pipeline always falls back to the stock three-line
        #    template responder ("That's interesting! Tell me more." etc.)
        #    even though ``train-slm`` has produced a real model.
        self._try_load_slm()

        self._initialized = True
        logger.info("Pipeline initialised successfully.")

    @staticmethod
    def _classify_cascade_arm(response_path: str, route_chosen: str) -> str:
        """Bucket *response_path* / *route_chosen* into a cascade-arm label.

        Iter 55: maps the engine's per-turn ``response_path`` and
        ``route_chosen`` strings into the cascade-arm name used by
        ``/api/cascade/stats`` and the dashboard's cascade ribbon.

            slm           — local SLM generation (the every-turn arm)
            qwen_intent   — Qwen LoRA intent parser (command turns)
            gemini_cloud  — Gemini AI Studio cloud arm
            retrieval     — curated retrieval (no model gen)
            tool          — any tool short-circuit (recap/fact/safety/intent)
            other         — anything not classified above
        """
        rp = (response_path or "").lower()
        rc = (route_chosen or "").lower()
        if rp == "tool:intent":
            return "qwen_intent"
        if "gemini" in rc or rc == "cloud_llm":
            return "gemini_cloud"
        if rp.startswith("tool:"):
            return "tool"
        if rp.startswith("retrieval"):
            return "retrieval"
        if rp == "slm" or rc == "local_slm":
            return "slm"
        return "other"

    def cascade_arm_stats(self) -> dict[str, Any]:
        """Expose live per-arm latency stats for ``/api/cascade/stats``.

        Iter 55.  Returns a dict::

            {
              "<arm>": {
                "n": int,           # number of samples in window
                "p50_ms": float,
                "p95_ms": float,
                "mean_ms": float,
                "max_ms": float,
              },
              ...
              "_window_size": 200,
            }

        Empty arms still appear (with all zeros) so the dashboard can
        render a stable layout.
        """
        out: dict[str, Any] = {"_window_size": 200}
        if not hasattr(self, "_cascade_arm_latencies"):
            return out
        for arm, dq in self._cascade_arm_latencies.items():
            if not dq:
                out[arm] = {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0,
                            "mean_ms": 0.0, "max_ms": 0.0}
                continue
            xs = sorted(dq)
            n = len(xs)
            def _pct(p: float) -> float:
                k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
                return float(xs[k])
            out[arm] = {
                "n": n,
                "p50_ms": round(_pct(50), 2),
                "p95_ms": round(_pct(95), 2),
                "mean_ms": round(sum(xs) / n, 2),
                "max_ms": round(xs[-1], 2),
            }
        return out

    async def get_profiling_report(self) -> dict[str, Any]:
        """Return the edge-feasibility profiling report.

        Iter 51 (2026-04-27): wires the existing
        :class:`i3.profiling.EdgeProfiler` measurements into a recruiter-
        readable dashboard summary.  The values reflect the v2 stack:
        204 M-param custom transformer + 50 k-param TCN encoder, INT8
        quantised, measured on a 5 W TDP wearable equivalent extrapolated
        from the host RTX 4050 P50.  The numbers are static here (loaded
        from `data/profiling/edge_profile.json` when the file exists,
        else baked-in defaults) because re-running the full profiler on
        every dashboard hit would cost 5-10 s of forward passes.

        Returns a dict that downstream filters through
        ``server.routes._PROFILING_ALLOWED_FIELDS`` so the response is
        guaranteed to be PII/path-free.
        """
        # Prefer a cached profile dropped by `scripts/profile_edge.py` if
        # the user re-ran profiling recently.  Falls back to baked-in
        # defaults from the v2 technical report (Section 6.4).
        try:
            import json
            from pathlib import Path
            cache_path = Path("data/profiling/edge_profile.json")
            if cache_path.exists():
                with cache_path.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, dict) and cached.get("components"):
                    return cached
        except Exception:  # pragma: no cover — defensive cache read
            pass
        # Defaults match the technical report (docs/paper/I3_research_paper.md
        # §6.4) for the v2 stack on Kirin 9000-class hardware (κ=1.5
        # INT8 efficiency factor).  Encoder values are measured;
        # SLM values are extrapolated from RTX 4050 host P50 latency.
        return {
            "components": [
                {
                    "name": "PII sanitiser",
                    "params_m": 0.0,
                    "fp32_mb": 0.0,
                    "int8_mb": 0.0,
                    "p50_ms": 1.2,
                },
                {
                    "name": "Keystroke + linguistic features",
                    "params_m": 0.0,
                    "fp32_mb": 0.0,
                    "int8_mb": 0.0,
                    "p50_ms": 0.8,
                },
                {
                    "name": "TCN encoder",
                    "params_m": 0.05,
                    "fp32_mb": 0.20,
                    "int8_mb": 0.06,
                    "p50_ms": 2.4,
                },
                {
                    "name": "User-state projection (96-d fusion)",
                    "params_m": 0.025,
                    "fp32_mb": 0.10,
                    "int8_mb": 0.03,
                    "p50_ms": 0.5,
                },
                {
                    "name": "Adaptation controller (8-axis)",
                    "params_m": 0.0,
                    "fp32_mb": 0.0,
                    "int8_mb": 0.0,
                    "p50_ms": 0.4,
                },
                {
                    "name": "LinUCB router (Thompson sample)",
                    "params_m": 0.0,
                    "fp32_mb": 0.0,
                    "int8_mb": 0.0,
                    "p50_ms": 0.6,
                },
                {
                    "name": "Custom SLM (AdaptiveTransformerV2 INT8)",
                    "params_m": 204.4,
                    "fp32_mb": 818.0,
                    "int8_mb": 205.0,
                    "p50_ms": 48.0,
                },
                {
                    "name": "Postprocess + safety overlay",
                    "params_m": 0.05,
                    "fp32_mb": 0.10,
                    "int8_mb": 0.04,
                    "p50_ms": 1.8,
                },
                {
                    "name": "Diary write (background)",
                    "params_m": 0.0,
                    "fp32_mb": 0.0,
                    "int8_mb": 0.0,
                    "p50_ms": 0.0,  # off-thread, doesn't count toward critical-path
                },
                # Iter 51 — second arm of the cascade.  Only fires on
                # command-shaped utterances (gated by the cheap regex);
                # otherwise costs nothing.  Numbers are LoRA-on-base
                # measurements: 17.4 M trainable params on top of a
                # 1.74 B frozen base; we report only the LoRA-relevant
                # delta.  Latency reflects bf16 + 8-bit AdamW weights.
                {
                    "name": "Qwen3-1.7B + LoRA intent parser (cascade arm B)",
                    "params_m": 17.4,
                    "fp32_mb": 70.0,
                    "int8_mb": 17.5,
                    "p50_ms": 78.0,  # bf16 generate, 64-token max
                },
                # Iter 51 — third arm: Gemini cloud fallback.  Only
                # fires when LinUCB picks the cloud arm AND the user
                # has consented.  No on-device parameters; the entry
                # documents the network round-trip cost.
                {
                    "name": "Gemini 2.5 Flash (cascade arm C, network)",
                    "params_m": 0.0,
                    "fp32_mb": 0.0,
                    "int8_mb": 0.0,
                    "p50_ms": 220.0,  # AI Studio P50 from London
                },
            ],
            "total_latency_ms": 55.7,
            "memory_mb": 205.13,
            "fits_budget": True,
            "budget_ms": 100.0,
            "device_class": "Kirin 9000-class (8 GB DRAM, NPU; κ=1.5 INT8)",
            # Iter 51 — explicit per-arm budget breakdown so the
            # dashboard can show "the SLM arm fits in 56 ms; cascade
            # arms only fire on demand and do not blow the budget".
            "cascade_arms": {
                "A_slm": {"latency_ms": 55.7, "memory_mb": 205.13,
                          "fires": "every chat turn"},
                "B_qwen_intent": {"latency_ms": 78.0, "memory_mb": 17.5,
                                  "fires": "command-shaped turns only (~5–10 % of utterances)"},
                "C_gemini_cloud": {"latency_ms": 220.0, "memory_mb": 0.0,
                                   "fires": "explicit cloud opt-in only; round-trip"},
            },
        }

    async def shutdown(self) -> None:
        """Release resources held by the pipeline.

        Closes the cloud HTTP client, the diary store connection, and
        drains any in-flight background tasks (diary writes, etc.).
        Safe to call multiple times.
        """
        # SEC: Drain background tasks before closing dependencies, otherwise
        # an in-flight diary write could race against diary_store.close().
        # We give them a brief grace period, then cancel any stragglers.
        if self._background_tasks:
            pending = list(self._background_tasks)
            logger.info(
                "Awaiting %d in-flight background task(s) before shutdown.",
                len(pending),
            )
            try:
                done, still_pending = await asyncio.wait(
                    pending, timeout=5.0
                )
                for task in still_pending:
                    task.cancel()
                if still_pending:
                    # Allow cancellations to propagate; suppress CancelledError.
                    await asyncio.gather(*still_pending, return_exceptions=True)
            except Exception:
                logger.exception(
                    "Error while draining background tasks during shutdown."
                )

        # SEC: Close cloud HTTP client (httpx connection pool).
        try:
            await self.cloud_client.close()
        except Exception:
            logger.exception("Error closing cloud_client during shutdown.")

        # SEC: Close DiaryStore (SQLite/aiosqlite connection). Previously this
        # was leaked on shutdown.
        try:
            close_diary = getattr(self.diary_store, "close", None)
            if close_diary is not None:
                result = close_diary()
                if asyncio.iscoroutine(result):
                    await result
        except Exception:
            logger.exception("Error closing diary_store during shutdown.")

        self._initialized = False
        logger.info("Pipeline shut down.")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def start_session(
        self, user_id: str, session_id: str | None = None,
    ) -> str:
        """Start a new interaction session for the given user.

        Creates the per-user :class:`~src.user_model.model.UserModel` if
        it does not already exist, initialises a new session in the user
        model, records the session in the diary store, and resets the
        adaptation controller.

        Args:
            user_id: Unique user identifier.
            session_id: Optional caller-supplied UUID — when the
                WebSocket layer has already generated one for routing
                messages it must pass it here so the diary `sessions`
                row matches the id used in subsequent
                ``log_exchange`` calls (otherwise FOREIGN KEY fails
                silently).  If ``None``, a fresh UUID4 is allocated.

        Returns:
            The session identifier (echoed back when the caller supplied
            one, or freshly generated otherwise).
        """
        self._ensure_initialized()

        if not session_id:
            session_id = str(uuid.uuid4())
        user_model = self._get_or_create_user_model(user_id)
        user_model.start_session()

        await self.diary_store.create_session(session_id, user_id)
        self.adaptation.reset()

        # Clear per-user route tracking for the new session
        self._previous_route.pop(user_id, None)
        self._previous_routing_context.pop(user_id, None)

        # Iter 50: cross-session fact recall — load any stored facts
        # for this user_id into the new session's fact dict so
        # "what's my name" works on session 2 after the user said
        # their name in session 1.  Wrapped so a DB failure never
        # blocks session creation.
        try:
            stored_facts = await self.diary_store.get_user_facts(user_id)
            if stored_facts:
                self._stated_facts[(user_id, session_id)] = dict(stored_facts)
                # Mirror name into the iter-48 single-name slot so
                # the iter-48 recall regex still works.
                if "name" in stored_facts:
                    self._stated_user_name[(user_id, session_id)] = (
                        stored_facts["name"]
                    )
                logger.debug(
                    "Loaded %d cross-session facts for user_id=%s",
                    len(stored_facts), user_id,
                )
        except Exception:  # pragma: no cover — never block session start
            logger.debug(
                "Failed to load cross-session facts for user_id=%s",
                user_id, exc_info=True,
            )

        logger.info(
            "Session started: session_id=%s, user_id=%s", session_id, user_id
        )
        return session_id

    async def end_session(
        self, user_id: str, session_id: str
    ) -> dict[str, Any] | None:
        """End an active session and generate a summary.

        Updates the user model's long-term profile, optionally generates
        a diary summary via the cloud LLM, and finalises the session
        record in the diary store.

        Args:
            user_id: The user whose session is ending.
            session_id: The session to close.

        Returns:
            A summary dict with session metrics and a natural-language
            summary, or ``None`` if no active session was found.
        """
        self._ensure_initialized()

        user_model = self.user_models.get(user_id)
        if user_model is None or user_model.current_session is None:
            logger.warning(
                "end_session called but no active session for user_id=%s",
                user_id,
            )
            return None

        # End the session in the user model (updates long-term EMA)
        session_summary = user_model.end_session()

        # Generate a privacy-safe diary summary
        summary_text = self._build_fallback_summary(session_summary)
        try:
            if self.cloud_client.is_available:
                # PERF (M-7, 2026-04-23 audit): wrap the cloud summary
                # call in a hard deadline so a slow upstream cannot turn
                # the session-end round-trip into a ~45 s tail (the
                # retry + backoff budget inside CloudLLMClient).
                summary_budget = float(
                    getattr(self.cloud_client, "timeout", 10.0)
                ) * 1.2
                summary_text = await asyncio.wait_for(
                    self.cloud_client.generate_session_summary(
                        session_summary
                    ),
                    timeout=summary_budget,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Cloud session-summary timed out after %.1fs for session %s; "
                "using fallback.",
                summary_budget if 'summary_budget' in locals() else -1.0,
                session_id,
            )
        except Exception:
            logger.exception(
                "Failed to generate cloud summary for session %s; "
                "using fallback.",
                session_id,
            )

        # Finalise the session record in the diary store
        try:
            await self.diary_store.end_session(
                session_id=session_id,
                summary=summary_text,
                dominant_emotion=session_summary.get("dominant_emotion", "neutral"),
                topics=session_summary.get("topics", []),
                mean_engagement=session_summary.get("mean_engagement", 0.0),
                mean_cognitive_load=0.5,
                mean_accessibility=0.0,
                relationship_strength=session_summary.get(
                    "relationship_strength", 0.0
                ),
            )
        except Exception:
            logger.exception(
                "Failed to finalise session %s in diary store.", session_id
            )

        # Clean up per-user engagement tracking
        self._last_response_time.pop(user_id, None)
        self._last_response_length.pop(user_id, None)
        self._previous_route.pop(user_id, None)
        self._previous_routing_context.pop(user_id, None)

        # Drop the per-session conversation-history buffer so an
        # ended session can't keep its prior turns in memory.
        self._drop_session_history(user_id, session_id)

        # Drop the per-session entity tracker stack so a reused
        # session id can't bleed entities into a fresh session.
        try:
            self._entity_tracker.end_session(user_id, session_id)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "EntityTracker.end_session failed for user_id=%s "
                "session_id=%s", user_id, session_id,
            )
        # Drop the per-session hierarchical memory (B.5) so a reused
        # session id can't keep prior topic stack / user facts.
        try:
            if self._hierarchical_memory is not None:
                self._hierarchical_memory.end_session(user_id, session_id)
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "HierarchicalMemory.end_session failed (non-blocking)",
                exc_info=True,
            )
        self._last_coref.pop(self._history_key(user_id, session_id), None)

        # Drop the affect-shift detector's rolling window for this
        # session so it can't bleed into a subsequent session that
        # happens to reuse the same id.
        try:
            self._shift_detector.end_session(user_id, session_id)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "AffectShiftDetector.end_session failed for user_id=%s "
                "session_id=%s", user_id, session_id,
            )
        self._last_affect_shift.pop(f"{user_id}::{session_id}", None)

        # Drop the state-classifier and accessibility-controller
        # bookkeeping for this session so a reused session id can
        # never bleed into a fresh session.
        try:
            self._accessibility_controller.end_session(user_id, session_id)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "AccessibilityController.end_session failed for user_id=%s "
                "session_id=%s", user_id, session_id,
            )
        self._last_user_state_label.pop(f"{user_id}::{session_id}", None)
        self._last_accessibility_state.pop(f"{user_id}::{session_id}", None)

        # Drop the per-(user, session) profile aggregator entry so a
        # reused session id can never bleed into a new session's
        # snapshot.  The biometric template is *user-scoped* (not
        # session-scoped) so it deliberately survives session_end.
        self._profile_aggregator.pop(f"{user_id}::{session_id}", None)

        logger.info("Session ended: session_id=%s, user_id=%s", session_id, user_id)
        return {"summary": summary_text, **session_summary}

    # ------------------------------------------------------------------
    # Core message processing
    # ------------------------------------------------------------------

    async def process_message(
        self,
        input: PipelineInput,
        on_token: Any = None,
    ) -> PipelineOutput:
        """Process a single user message through the full pipeline.

        This is the main entry point called for every user message.  It
        executes the nine-step pipeline described in the class docstring
        and returns a :class:`PipelineOutput` containing the response and
        all state updates for the frontend dashboard.

        SEC: This method NEVER returns ``None`` and only raises
        :class:`RuntimeError` (when the pipeline is uninitialised).
        Any other exception during processing is caught at the outer
        boundary and converted into an error-marked
        :class:`PipelineOutput` so the calling FastAPI handler can
        always serialise a response.

        Args:
            input: A :class:`PipelineInput` carrying the raw message,
                keystroke timing data, and session metadata.
            on_token: Optional callback ``f(delta: str) -> None``
                invoked once per generated token when the response path
                is the SLM generator.  Retrieval / tool / OOD paths do
                not stream (there are no intermediate tokens to emit);
                for those callers receive the single ``PipelineOutput``
                return value in the usual way.  The callback may be sync
                or async — both are awaited if necessary.

        Returns:
            A :class:`PipelineOutput` with the AI response, routing
            decision, adaptation vector, and user-state metrics.

        Raises:
            RuntimeError: If the pipeline has not been initialised.
        """
        self._ensure_initialized()
        start_time = time.perf_counter()

        try:
            return await self._process_message_inner(
                input, start_time, on_token=on_token
            )
        except asyncio.CancelledError:
            # SEC: Never swallow CancelledError - propagate so the event
            # loop can shut down cleanly.
            raise
        except Exception as exc:
            # SEC: Never return None on error. Build a degraded
            # PipelineOutput so callers always have a serialisable result.
            logger.exception(
                "process_message failed for user_id=%s session_id=%s",
                input.user_id,
                input.session_id,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return self._build_error_output(latency_ms, exc)

    async def _process_message_inner(
        self, input: PipelineInput, start_time: float,
        on_token: Any = None,
    ) -> PipelineOutput:
        """Inner pipeline implementation, wrapped by :meth:`process_message`."""
        # Per-turn flow-trace handle.  Carries through every stage so
        # the third flagship surface (Flow dashboard) can render real
        # measurements rather than synthetic timings.  Wrapped to never
        # raise: if the collector ever fails the rest of the pipeline
        # keeps working — pipeline_trace just lands as None.
        try:
            _trace = self._trace_collector.start_turn(
                input.user_id, input.session_id,
            )
        except Exception:  # pragma: no cover - decorative
            _trace = None
            logger.warning("trace.start_turn failed", exc_info=True)

        # ---- Step 1: Extract interaction features ------------------------
        with self._trace_stage(_trace, "interaction", "Interaction monitor") as _rec:
            features = await self.monitor.process_message(
                user_id=input.user_id,
                text=input.message_text,
                composition_time_ms=input.composition_time_ms,
                edit_count=input.edit_count,
                pause_before_send_ms=input.pause_before_send_ms,
            )
            self._trace_note(
                _trace, "interaction",
                _input={"text_len": len(input.message_text)},
                _output={"features_dim": 32},
                _notes="composition + edit + pause -> 32-dim feature vector",
            )
        logger.debug(
            "Step 1 complete: extracted 32-dim features for user_id=%s",
            input.user_id,
        )

        # ---- Step 2: Encode user state via TCN ---------------------------
        feature_window = await self.monitor.get_feature_window(input.user_id)
        # SEC: TCN encoder is sync + CPU-heavy (PyTorch forward pass).
        # Run it in the default thread-pool executor so it does not block
        # the event loop and starve other concurrent users.
        loop = asyncio.get_running_loop()
        with self._trace_stage(_trace, "encoder", "TCN encoder"):
            user_state_embedding = await loop.run_in_executor(
                None, self._encode_features, feature_window
            )
            embedding_2d = self._project_2d(user_state_embedding)
            self._trace_note(
                _trace, "encoder",
                _input={"window": "32x10"},
                _output={"embedding_dim": 64},
                _notes="dilated TCN -> 64-d state",
            )
        self._trace_arrow(
            _trace, "interaction", "encoder",
            payload_summary="32-d feature vector",
            size_bytes=128,
        )
        self._trace_arrow(
            _trace, "encoder", "adaptation",
            payload_summary="64-d embedding",
            size_bytes=256,
        )
        # Stash the 64-d user-state embedding so the preference-record
        # route can feed it to the personalisation manager when a
        # user A/B pick comes in (often without the user sending
        # another full message first).
        try:
            self._last_user_state_embedding[input.user_id] = (
                user_state_embedding.detach().clone()
            )
        except Exception:  # pragma: no cover - decorative
            pass
        logger.debug(
            "Step 2 complete: 64-dim embedding -> 2D projection (%.3f, %.3f)",
            embedding_2d[0],
            embedding_2d[1],
        )

        # ---- Step 2b: Multimodal voice-prosody fusion (optional) ----------
        # When the browser shipped a validated ``prosody_features`` dict on
        # the message frame, project it through the prosody encoder and
        # fuse with the 64-d keystroke embedding into a 96-d multimodal
        # user-state vector.  When the dict is absent / invalid, we fuse
        # with ``None`` so the output is still 96-d (zero-padded prosody
        # half) and the downstream consumers see a stable shape.
        # The 96-d vector is purely a *flagship-feature* artefact at the
        # moment — the rest of the pipeline (state classifier, biometric,
        # affect-shift, LoRA) keeps consuming the 64-d keystroke half so
        # legacy behaviour is preserved bit-for-bit when the mic is off.
        with self._trace_stage(_trace, "multimodal_fusion", "Multimodal fusion") as _rec:
            multimodal_dict = await loop.run_in_executor(
                None,
                self._encode_multimodal_features,
                user_state_embedding,
                input.prosody_features,
                input.gaze_features,
            )
            _prosody_active = bool(
                isinstance(multimodal_dict, dict)
                and multimodal_dict.get("prosody_active")
            )
            _gaze_active = bool(
                isinstance(multimodal_dict, dict)
                and multimodal_dict.get("gaze_active")
            )
            _trace_notes = []
            if _prosody_active:
                _trace_notes.append("prosody mic ON")
            if _gaze_active:
                _trace_notes.append("gaze camera ON")
            if not _trace_notes:
                _trace_notes.append("keystroke-only (mic off, camera off)")
            self._trace_note(
                _trace, "multimodal_fusion",
                _output={
                    "fused_dim": int(
                        (multimodal_dict or {}).get("fused_dim", 128)
                    ),
                    "prosody_active": _prosody_active,
                    "gaze_active": _gaze_active,
                },
                _notes=" + ".join(_trace_notes),
            )

        # ---- Step 2c: Build gaze output dict for the WS frame -----------
        # Snapshot the validated gaze payload onto a JSON-safe dict
        # that gets surfaced on PipelineOutput.gaze.  When the user
        # was looking away (presence=False) we annotate
        # ``gaze_aware_note`` with the gaze-conditioned response-timing
        # message; the reasoning trace + UI surface that as the HCI
        # demonstration that gaze can hold a response on a real device.
        gaze_output_dict: dict | None = None
        if _gaze_active and isinstance(input.gaze_features, dict):
            try:
                gaze_output_dict = {
                    "label": str(input.gaze_features.get("label", "")),
                    "confidence": float(
                        input.gaze_features.get("confidence", 0.0)
                    ),
                    "label_probs": dict(
                        input.gaze_features.get("label_probs") or {}
                    ),
                    "presence": bool(
                        input.gaze_features.get("presence", True)
                    ),
                    "blink_rate_norm": float(
                        input.gaze_features.get("blink_rate_norm", 0.0)
                    ),
                    "head_stability": float(
                        input.gaze_features.get("head_stability", 0.0)
                    ),
                    "captured_seconds": float(
                        input.gaze_features.get("captured_seconds", 0.0)
                    ),
                    "samples_count": int(
                        input.gaze_features.get("samples_count", 0)
                    ),
                    "gaze_aware_note": None,
                }
                if not gaze_output_dict["presence"]:
                    gaze_output_dict["gaze_aware_note"] = (
                        "I noticed you weren't looking at the screen. "
                        "Continuing anyway, but on a phone I'd hold the "
                        "response until you looked back."
                    )
            except (TypeError, ValueError):
                gaze_output_dict = None

        # ---- Step 3: Update user model -----------------------------------
        user_model = await self._aget_or_create_user_model(input.user_id)
        if user_model.current_session is None:
            # Auto-start session if the caller forgot
            user_model.start_session()
            logger.warning(
                "Auto-started session for user_id=%s (caller should "
                "call start_session explicitly).",
                input.user_id,
            )

        deviation = user_model.update_state(user_state_embedding, features)
        logger.debug(
            "Step 3 complete: deviation magnitude=%.4f, engagement=%.4f",
            deviation.magnitude,
            deviation.engagement_score,
        )

        # ---- Step 4: Compute adaptation vector --------------------------
        with self._trace_stage(_trace, "adaptation", "Adaptation controller"):
            adaptation = self.adaptation.compute(features, deviation)
            self._trace_note(
                _trace, "adaptation",
                _output={
                    "cog_load": float(adaptation.cognitive_load),
                    "emo_tone": float(adaptation.emotional_tone),
                    "access": float(adaptation.accessibility),
                    "axes": 8,
                },
                _notes=(
                    f"8-axis vector — cog={adaptation.cognitive_load:.2f} "
                    f"tone={adaptation.emotional_tone:.2f} "
                    f"access={adaptation.accessibility:.2f}"
                ),
            )
        logger.debug(
            "Step 4 complete: cognitive_load=%.2f, emotional_tone=%.2f, "
            "accessibility=%.2f",
            adaptation.cognitive_load,
            adaptation.emotional_tone,
            adaptation.accessibility,
        )

        # ---- Step 5: Sanitize text for cloud transmission ----------------
        sanitized = self.sanitizer.sanitize(input.message_text)
        if sanitized.pii_detected:
            logger.info(
                "PII detected and sanitised (%d replacements, types=%s)",
                sanitized.replacements_made,
                sanitized.pii_types,
            )

        # ---- Step 5a: Constitutional safety classification ---------------
        # Char-CNN trained on a synthetic constitutional corpus
        # (i3.safety.synthetic_corpus).  Verdict drives an early
        # return on `refuse` (the canonical refusal text replaces the
        # response, no retrieval / generation runs) and a soft caveat
        # on `review`.  Wrapped to never block — a classifier-init
        # failure means the layer goes silent, never raises.  Honours
        # ``playground_overrides["safety"] = False`` to allow the
        # Playground tab to bypass it for what-if exploration.
        safety_dict: dict | None = None
        safety_refuse_text: str | None = None
        safety_caveat: str | None = None
        playground_overrides = (
            input.playground_overrides
            if isinstance(getattr(input, "playground_overrides", None), dict)
            else None
        )
        _safety_enabled = True
        if playground_overrides is not None:
            _safety_enabled = bool(playground_overrides.get("safety", True))
        with self._trace_stage(_trace, "safety", "Constitutional safety", is_tool=True):
            if _safety_enabled and self._safety_classifier is not None:
                try:
                    verdict = self._safety_classifier.classify(
                        sanitized.sanitized_text
                    )
                    safety_dict = verdict.to_dict()
                    # Phase B (2026-04-25): the char-CNN classifier is
                    # over-eager on benign factoid follow-ups (e.g.
                    # "who proposed it?", "now back to apple — who
                    # founded it?").  Whitelist obvious entity/factoid
                    # query shapes so the safety route can't intercept
                    # them.  This is a *narrow* whitelist — anything
                    # with self-harm / weapon keywords still falls
                    # through to the classifier.  Order matters: we
                    # check the whitelist BEFORE acting on the verdict.
                    benign_pattern = self._is_benign_factoid_query(
                        sanitized.sanitized_text
                    )
                    if benign_pattern and verdict.verdict in ("refuse", "review"):
                        # Override only when there are no actual
                        # safety-trigger keywords (gun / kill / suicide
                        # / etc.).  The classifier can still fire on
                        # genuinely harmful text — we just refuse to
                        # let a benign factoid look harmful.
                        if not self._has_safety_trigger_word(
                            sanitized.sanitized_text
                        ):
                            logger.debug(
                                "Safety override: benign factoid pattern "
                                "matched %r — downgrading %s to safe.",
                                sanitized.sanitized_text[:60],
                                verdict.verdict,
                            )
                            safety_dict = dict(safety_dict)
                            safety_dict["verdict"] = "safe"
                            safety_dict["reasons"] = ["benign_factoid_override"]
                            safety_dict["constitutional_principle"] = ""
                            safety_dict["suggested_response"] = ""
                            verdict = type(verdict)(
                                verdict="safe",
                                confidence=verdict.confidence,
                                reasons=["benign_factoid_override"],
                                constitutional_principle="",
                                suggested_response="",
                                scores=verdict.scores,
                            )
                    if verdict.verdict == "refuse":
                        safety_refuse_text = verdict.suggested_response
                    elif verdict.verdict == "review":
                        safety_caveat = (
                            f"⚠ {verdict.constitutional_principle} "
                            "This answer is for general information only."
                        )
                    self._trace_note(
                        _trace, "safety",
                        _output={
                            "verdict": verdict.verdict,
                            "confidence": float(verdict.confidence),
                        },
                        _notes=(
                            f"char-CNN ({self._safety_classifier.num_parameters()} "
                            f"params) → {verdict.verdict}"
                            f"{' (' + verdict.reasons[0] + ')' if verdict.reasons else ''}"
                        ),
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "Safety classifier failed for user_id=%s; "
                        "continuing without safety verdict",
                        input.user_id,
                    )
                    safety_dict = None
        self._last_safety[
            self._history_key(input.user_id, input.session_id)
        ] = safety_dict

        # ---- Step 5b: Co-reference resolution ----------------------------
        # Rewrite pronoun-laden follow-ups ("where are they located?")
        # using the per-session entity recency stack so retrieval (and
        # in particular the entity-knowledge tool route) sees the
        # resolved form ("where is huawei located?") instead of the
        # short, ambiguous user text.  Bounded, pure-Python, never
        # raises — see :mod:`i3.dialogue.coref`.
        coref_resolution = None
        query_for_retrieval = sanitized.sanitized_text
        # Topic-carryover prefix for the embedding query (Fix 3 of the
        # 2026-04-25 corpus-quality overhaul).  When the user sends a
        # short follow-up like "most famous product" right after
        # "tell me about apple", coref doesn't fire (no pronoun) and
        # the cosine retrieval lands on whatever short-query mean-pool
        # noise looks like — typically Tokyo/Japan paragraphs.  We
        # prepend the most recent topic entity to the *embedding*
        # query (via ``query_for_embedding``) so the cosine search
        # has the right anchor, while leaving the keyword-overlap /
        # exact-match gate looking at the raw user text.
        topic_prefix_for_embedding: str | None = None
        # Iter 40: when a negation-pivot fires ("not Apple, Microsoft"),
        # the new subject canonical is captured here so the
        # post-response ``observe()`` call can promote it via
        # ``priority_canonical`` — making the next bare follow-up
        # resolve to the new subject rather than the negated one.
        # ``_neg_negated_surface`` is the surface phrase to scrub from
        # the user_text passed to observe, so the negated entity is
        # NOT re-anchored on this turn (otherwise both end up user-
        # anchored at the same turn → ambiguity → clarifier).
        _neg_anchored: str | None = None
        _neg_negated_surface: str | None = None
        try:
            current_turn = (
                user_model.current_session.message_count + 1
                if user_model.current_session is not None
                else 1
            )
            coref_resolution = self._entity_tracker.resolve(
                user_id=input.user_id,
                session_id=input.session_id,
                turn_idx=current_turn,
                user_text=sanitized.sanitized_text,
            )
            # Phase B (2026-04-25): "back to X / about X" override.
            # When the user explicitly names the topic at the start of
            # the message, override coref's most-recent-entity choice
            # so a topic switch can't be hijacked by a stale pronoun
            # binding (e.g. "now back to apple — who founded it?"
            # should resolve "it" → apple, not the just-discussed
            # linux).  We mutate the resolved_query directly.
            try:
                # Iter 40 (2026-04-26): negation-pivot detection — "not
                # apple, microsoft" / "actually not apple, microsoft" /
                # "not apple but microsoft" / "i mean microsoft, not
                # apple" — anchor on the *non-negated* surface so the
                # next bare follow-up ("their CEO") resolves to the new
                # subject.  Run BEFORE the leading-pivot regex so we
                # don't get fooled by the leading "actually" / "not".
                _neg_text = (sanitized.sanitized_text or "").strip()
                neg_match = re.match(
                    r"^(?:actually[,\s]+|well[,\s]+|sorry[,\s]+|wait[,\s]+|hmm[,\s]+)?"
                    r"not\s+([a-z][a-z\s]{1,30}?)"
                    r"\s*(?:[,;:]\s*|\s+but\s+|\s+rather\s+|\s+but\s+rather\s+|\s*[—–\-]\s*)"
                    r"([a-z][a-z\s]{1,30}?)\s*[?.!]?$",
                    _neg_text,
                    re.I,
                )
                if not neg_match:
                    neg_match = re.match(
                        r"^(?:actually[,\s]+)?(?:i\s+mean(?:t)?[,\s]+|i\s+meant[,\s]+)"
                        r"([a-z][a-z\s]{1,30}?)\s*[,;:]\s*not\s+"
                        r"([a-z][a-z\s]{1,30}?)\s*[?.!]?$",
                        _neg_text,
                        re.I,
                    )
                    if neg_match:
                        # "I mean Y, not X" — Y is in group(1)
                        _kept = neg_match.group(1).strip().lower()
                    else:
                        _kept = None
                else:
                    # "not X, Y" / "not X but Y" — Y is in group(2)
                    _kept = neg_match.group(2).strip().lower()
                if neg_match and _kept:
                    from i3.dialogue.coref import _ALIAS_TO_CANONICAL
                    _kept_normed = re.sub(
                        r"^(?:the\s+|a\s+|an\s+)", "", _kept,
                    )
                    forced_canonical = (
                        _ALIAS_TO_CANONICAL.get(_kept_normed)
                        or _ALIAS_TO_CANONICAL.get(_kept)
                    )
                    if forced_canonical:
                        # Anchor on the new entity: (1) synthesise
                        # "tell me about {canonical}" as the retrieval
                        # query so the response is about the new
                        # subject, (2) save the canonical in
                        # ``_neg_anchored`` so the post-response
                        # ``observe()`` call promotes it via
                        # ``priority_canonical`` — letting the next
                        # bare follow-up ("their CEO") resolve to the
                        # new subject rather than the negated one,
                        # (3) capture the NEGATED surface in
                        # ``_neg_negated_surface`` so observe() doesn't
                        # re-anchor it on this turn (passing both
                        # would create ambiguity → clarifier on the
                        # very next user message).
                        _neg_anchored = forced_canonical
                        # Determine which group is the negated surface
                        # for the two pattern variants:
                        # - "not X, Y" / "not X but Y" → group(1) is X
                        # - "I mean Y, not X"          → group(2) is X
                        try:
                            if "not " in _neg_text.lower().split(",", 1)[0]:
                                _neg_negated_surface = neg_match.group(1).strip()
                            else:
                                _neg_negated_surface = neg_match.group(2).strip()
                        except Exception:
                            _neg_negated_surface = None
                        query_for_retrieval = (
                            f"tell me about {forced_canonical}"
                        )
                        logger.debug(
                            "Negation pivot: %r -> anchor=%r, "
                            "negated=%r, retrieval=%r",
                            _neg_text, forced_canonical,
                            _neg_negated_surface,
                            query_for_retrieval,
                        )

                # Iter 21 (2026-04-26): broader leading-pivot vocabulary
                # so "back to apple", "going back to apple", "returning
                # to apple", "switch back to apple", "now back to
                # apple", "let's talk about apple", and the original
                # comma/dash separator forms all force a re-anchor.
                # The pattern is anchored to the START of the message
                # so a mid-sentence "about" doesn't trigger.
                # Iter 43: strip a leading discourse marker ("ok",
                # "alright", "so", "well", "hey", "anyway", "umm")
                # before testing the leading-pivot regex so phrases
                # like "OK back to gravity" / "anyway back to apple"
                # still re-anchor.
                # Iter 44: greedy strip — repeat until no marker
                # consumed.  Catches stacked markers like "wait, sorry,
                # scrap that — ..." or "well, OK, anyway ...".  Also
                # added retraction markers (nevermind, scrap that,
                # forget that, drop that, never mind).
                _lead_input = (sanitized.sanitized_text or "").strip()
                _DISCOURSE_RE = re.compile(
                    r"^(?:ok|okay|alright|all\s+right|so|well|hey|"
                    r"anyway|anyhow|um|umm|uh|hmm|listen|wait|sorry|"
                    r"actually|oh|you\s+know|i\s+mean|"
                    r"by\s+the\s+way|btw|"
                    r"(?:please\s+)?(?:never[\s-]?mind|nevermind)(?:\s+that)?|"
                    r"scrap\s+that|forget\s+that|drop\s+that|"
                    r"strike\s+that|"
                    r"on\s+second\s+thought|second\s+thought|"
                    # Iter 46: "no" / "nope" / "yeah no" — soft
                    # disagreement before a corrective pivot.
                    r"no|nope|yeah\s+no|nah)"
                    r"\s*[,\-:—.]*\s+",
                    re.I,
                )
                # Repeatedly strip leading discourse markers (max 4
                # iterations to avoid pathological inputs).  Iter 44:
                # don't strip if it would leave a single-word remainder
                # (e.g. "oh great" → "great" loses the curated "oh
                # great" cosine match).  We need at least 2 content
                # tokens after strip to be sure the strip is helpful.
                for _ in range(4):
                    new_input = _DISCOURSE_RE.sub("", _lead_input, count=1)
                    if new_input == _lead_input:
                        break
                    if len(new_input.split()) < 2:
                        break
                    _lead_input = new_input
                lead_match = re.match(
                    r"^(?:now\s+back\s+to|now\s+about|back\s+to|going\s+back\s+to|"
                    r"go\s+back\s+to|"
                    r"returning\s+to|switch(?:ing)?\s+back\s+to|"
                    # Iter 46: contradiction pivots — "I meant X" /
                    # "I really meant X".  These signal the user is
                    # correcting a prior topic mention and should
                    # re-anchor on X.
                    r"i\s+(?:really\s+)?meant|"
                    r"i\s+(?:was\s+)?actually\s+(?:asking\s+about|talking\s+about)|"
                    r"i'?m\s+(?:asking|talking)\s+about|"
                    # Iter 41: explicit topic-switch — "let's switch
                    # to X" / "let's discuss X" / "let's go to X" /
                    # "switch to X" / "now let's talk about X" / "now
                    # let's discuss X".
                    r"let'?s\s+(?:go\s+back\s+to|talk\s+about|switch\s+back\s+to|"
                    r"switch\s+to|move\s+on\s+to|move\s+to|jump\s+to|"
                    r"discuss|consider|look\s+at)|"
                    r"now\s+let'?s\s+(?:talk\s+about|discuss|consider|"
                    r"look\s+at|switch\s+to)|"
                    r"switch\s+to|jump\s+to|move\s+on\s+to|move\s+to|"
                    r"talk\s+about|about|"
                    # Iter 32 (2026-04-26): "and X" / "what about X" /
                    # "now X" / "or X" / "then X" / "next X" /
                    # "how about X" — when X is a known catalog
                    # entity, treat as a new topic anchor instead of
                    # letting the bare-rewriter pivot back to the
                    # current active topic.
                    r"and|or|then|next|how\s+about|what\s+about|now)"
                    r"\s+([a-z][a-z\s]{1,30}?)"
                    r"\s*(?:[—\-,:;.?!]|—|$)",
                    _lead_input,
                    re.I,
                )
                # Iter 44: when no lead_match but discourse was
                # stripped, hand the cleaned form to retrieval so
                # phrases like "by the way what time is it" cosine
                # against the curated "what time is it" entry instead
                # of being polluted by the discourse prefix.
                if not lead_match:
                    _raw_stripped = (sanitized.sanitized_text or "").strip()
                    if _lead_input and _lead_input != _raw_stripped:
                        query_for_retrieval = _lead_input
                if lead_match:
                    from i3.dialogue.coref import _ALIAS_TO_CANONICAL
                    surface = lead_match.group(1).strip().lower()
                    # Iter 31 (2026-04-26): normalise the surface form
                    # so "back to the AI question" / "back to the
                    # transformer topic" / "back to the apple thread"
                    # also resolve.  Strip leading "the " and trailing
                    # frame nouns.
                    _surface_normed = re.sub(
                        r"^(?:the\s+|that\s+|this\s+)", "", surface,
                    )
                    _surface_normed = re.sub(
                        r"\s+(?:question|topic|thread|subject|"
                        r"discussion|chat|conversation|thing|stuff)$",
                        "",
                        _surface_normed,
                    ).strip()
                    forced_canonical = (
                        _ALIAS_TO_CANONICAL.get(_surface_normed)
                        or _ALIAS_TO_CANONICAL.get(surface)
                    )
                    # Iter 46: definite-description surfaces in lead-
                    # pivot.  "go back to the company" / "I meant the
                    # company" — resolve "company"/"firm"/"brand" to
                    # the topmost ORG canonical.  Similarly
                    # "the fruit" → if topmost ORG is "apple", use
                    # "apple fruit" (alt-sense); for other orgs no
                    # alt-sense is defined so it falls through.
                    if forced_canonical is None and _surface_normed in {
                        "company", "firm", "brand", "organization",
                        "organisation", "corporation", "business",
                    }:
                        try:
                            snap = self._entity_tracker.snapshot(
                                input.user_id, input.session_id,
                            )
                            for f in snap:
                                if f.kind == "org":
                                    forced_canonical = f.canonical
                                    break
                        except Exception:
                            pass
                    if forced_canonical is None and _surface_normed in {
                        "fruit", "the fruit",
                    }:
                        try:
                            snap = self._entity_tracker.snapshot(
                                input.user_id, input.session_id,
                            )
                            for f in snap:
                                if f.canonical == "apple" and f.kind == "org":
                                    forced_canonical = "apple fruit"
                                    break
                        except Exception:
                            pass
                    # "back to the start" / "back to the beginning" /
                    # "back to where we started" → resolve to the
                    # FIRST user-anchored topic in the session.
                    if forced_canonical is None and _surface_normed in {
                        "start", "beginning", "first one",
                        "first thing", "first topic",
                        "where we started", "where we began",
                        "the original", "original",
                    }:
                        try:
                            snap = self._entity_tracker.snapshot(
                                input.user_id, input.session_id,
                            )
                            anchored = [
                                f for f in snap
                                if f.first_anchor_turn is not None
                                and f.kind in {"org", "topic"}
                            ]
                            if anchored:
                                # The FIRST anchored is the one with
                                # the smallest first_anchor_turn — the
                                # session's original topic, even if
                                # the user has re-anchored it since.
                                first = min(
                                    anchored,
                                    key=lambda f: f.first_anchor_turn,
                                )
                                forced_canonical = first.canonical
                        except Exception:
                            forced_canonical = None
                    if forced_canonical:
                        forced_text = re.sub(
                            r"\b(?:it|them|they)\b",
                            forced_canonical,
                            sanitized.sanitized_text,
                            flags=re.I,
                        )
                        # Iter 31: when the user message is JUST a
                        # leading-pivot ("back to apple", "back to
                        # the start") with no follow-up question,
                        # synthesise "tell me about {canonical}" so
                        # the downstream retrieval / decomposer has
                        # something concrete to answer.  Cheaper
                        # heuristic than the full regex strip: count
                        # words.  A pure pivot is at most ~6 words
                        # ("now let's go back to the apple thread")
                        # and contains no question word other than
                        # the pivot frame.  If so, synthesise.
                        _raw = (sanitized.sanitized_text or "").strip()
                        _word_count = len(_raw.split())
                        _has_question_tail = bool(re.search(
                            r"\b(?:who|what|when|where|why|how|which|whose|whom|is|are|was|were|do|does|did|can|could|should|would|will)\b",
                            _raw[len(lead_match.group(0)):],
                            re.I,
                        ))
                        if _word_count <= 7 and not _has_question_tail:
                            forced_text = f"tell me about {forced_canonical}"
                        # Build a coref-like result with the forced
                        # entity so the downstream code thinks it
                        # resolved.
                        from i3.dialogue.coref import (
                            EntityFrame, ResolutionResult,
                        )
                        # Synthesise a minimal frame.
                        forced_frame = EntityFrame(
                            text=surface,
                            canonical=forced_canonical,
                            kind="org",
                            last_turn_idx=current_turn,
                        )
                        coref_resolution = ResolutionResult(
                            original_query=sanitized.sanitized_text,
                            resolved_query=forced_text,
                            used_entity=forced_frame,
                            used_pronoun="(forced via 'back to' lead-in)",
                            confidence=1.0,
                            reasoning=(
                                "User explicitly switched topic to "
                                f"{forced_canonical} at the message start; "
                                "overriding pronoun resolution."
                            ),
                        )
            except Exception:
                logger.debug("'back to' override failed", exc_info=True)
            # Iter 36 (2026-04-26): single-word topic auto-anchor.  If
            # the user types a single word (or a short multi-word
            # phrase) that maps to a known catalog topic AND no other
            # leading-pivot fired, treat the whole message as
            # "tell me about {topic}".  Without this, terse one-word
            # queries like "time" / "love" / "happiness" cosine-match
            # poorly and fall to OOD.
            try:
                raw_lower = (sanitized.sanitized_text or "").strip().lower().rstrip("?!.")
                if (
                    raw_lower
                    and len(raw_lower) >= 3
                    and len(raw_lower.split()) <= 3
                    and not lead_match
                    and coref_resolution.used_entity is None
                ):
                    from i3.dialogue.coref import _ALIAS_TO_CANONICAL
                    canon_single = _ALIAS_TO_CANONICAL.get(raw_lower)
                    if canon_single:
                        # Build a coref-like result with the auto-anchored
                        # topic so downstream code thinks coref resolved.
                        from i3.dialogue.coref import (
                            EntityFrame, ResolutionResult,
                        )
                        forced_frame = EntityFrame(
                            text=raw_lower,
                            canonical=canon_single,
                            kind="topic",
                            last_turn_idx=current_turn,
                        )
                        coref_resolution = ResolutionResult(
                            original_query=sanitized.sanitized_text,
                            # Iter 36: rewrite to "what is X" rather
                            # than "tell me about X" because the
                            # curated overlay uses the "what is X"
                            # form, and the exact-match fast path
                            # in retrieval picks it up cleanly.
                            resolved_query=f"what is {canon_single}",
                            used_entity=forced_frame,
                            used_pronoun="(single-word topic auto-anchor)",
                            confidence=1.0,
                            reasoning=(
                                f"User typed bare topic word {raw_lower!r}; "
                                f"auto-rewriting to 'what is {canon_single}' "
                                "so retrieval hits the curated entry."
                            ),
                        )
            except Exception:
                logger.debug(
                    "single-word topic auto-anchor failed", exc_info=True,
                )
            if coref_resolution.used_entity is not None:
                query_for_retrieval = coref_resolution.resolved_query
                logger.debug(
                    "coref resolved %r -> %r (entity=%s, conf=%.2f)",
                    coref_resolution.original_query,
                    coref_resolution.resolved_query,
                    coref_resolution.used_entity.canonical,
                    coref_resolution.confidence,
                )
            else:
                # No pronoun resolution fired.  If the user message is
                # short (≤6 tokens) and we have a recent topic entity
                # on the recency stack, prepend it to the embedding
                # query.  Bounded by max_age_turns=5 so a stale topic
                # from way upthread can't haunt every short reply.
                raw_text = sanitized.sanitized_text or ""
                token_count = len(raw_text.split())
                if 0 < token_count <= 6:
                    # Iteration 14/15 (2026-04-26): for bare follow-ups
                    # we ALWAYS prefer the topmost ORG/TOPIC over an
                    # incidental PERSON the assistant mentioned in the
                    # prior answer.  This is what stops the
                    # "Steve Jobs hijacks Apple" / "neural network
                    # hijacks transformer" failure modes.  Generic
                    # pivots ("how", "why", "more", "explain that")
                    # AND entity-slot probes ("ceo", "founder",
                    # "products") all benefit.
                    recent = self._entity_tracker.get_recent_entity(
                        input.user_id,
                        input.session_id,
                        max_age_turns=3,
                        current_turn=current_turn,
                        prefer_kinds=("org", "topic", "place", "person"),
                    )
                    # Iter 44: meta-topic probes that are intrinsically
                    # self-contained — time/date/weather/location-of-
                    # user — should NEVER inherit the active topic via
                    # the embedding prefix.  Without this, "what time
                    # is it" after an "overfitting" thread embeds as
                    # "[overfitting] what time is it" → cosines into
                    # the overfitting curated entry.
                    _META_SELF_CONTAINED = re.compile(
                        r"^(?:"
                        # "what time is it" / "what date is it" / "what day is it"
                        r"what\s+(?:time|date|day|year|month)\s+is\s+(?:it|today)|"
                        # "what's the time" / "what is the time/date/day/...weather"
                        r"what(?:['']?s|\s+is)?\s+(?:the\s+)?"
                        r"(?:time|date|day|year|month|weather|temperature)(?:\s+(?:now|today))?|"
                        # "what's today's date"
                        r"what(?:['']?s|\s+is)?\s+today(?:['']?s\s+date)?|"
                        # "how's the weather"
                        r"how['']?s\s+the\s+weather|"
                        # "do you know the time"
                        r"do\s+you\s+(?:know|have)\s+the\s+(?:time|date)"
                        r")\s*\??\s*$",
                        re.I,
                    )
                    _meta_self_contained = bool(
                        _META_SELF_CONTAINED.match(
                            (query_for_retrieval or "").strip()
                        )
                    )
                    if recent is not None and recent.canonical and not _meta_self_contained:
                        topic_prefix_for_embedding = recent.canonical
                        logger.debug(
                            "Short-prompt topic carryover: prefixing "
                            "embedding query with %r (entity kind=%s, "
                            "last_turn=%d, current_turn=%d)",
                            recent.canonical,
                            recent.kind,
                            recent.last_turn_idx,
                            current_turn,
                        )
                        # Bare-noun-followup rewriter: when the user
                        # types a single bare slot-noun ("location",
                        # "CEO", "founder", "products") right after a
                        # topic prompt, rewrite the retrieval query
                        # into the canonical entity-tool shape so the
                        # tool route ("where is apple located?", "who
                        # is the ceo of apple?", ...) fires
                        # deterministically rather than landing on a
                        # generic clarifier or a Tokyo paragraph.
                        # We DO NOT mutate the raw user message — the
                        # SLM and post-processor still see what the
                        # user typed.
                        bare = raw_text.strip().rstrip("?!. ").lower()
                        canon = recent.canonical
                        # Country-attribute follow-ups (Phase 14,
                        # 2026-04-25).  When the recency-stack topic is
                        # a country, "language?" / "currency?" /
                        # "capital?" / "population?" / "flag?" /
                        # "government?" should bind to the country-fact
                        # tool, not to whatever generic retrieval row
                        # the embedding cosine landed on.
                        from i3.dialogue.coref import COUNTRY_CANONICALS
                        if canon in COUNTRY_CANONICALS:
                            country_bare_rewrites: dict[str, str] = {
                                "language": f"what is the language of {canon}",
                                "languages": f"what is the language of {canon}",
                                "currency": f"what is the currency of {canon}",
                                "money": f"what is the currency of {canon}",
                                "capital": f"what is the capital of {canon}",
                                "capital city": f"what is the capital of {canon}",
                                "population": f"what is the population of {canon}",
                                "people": f"what is the population of {canon}",
                                "flag": f"what is the flag of {canon}",
                                "government": f"what is the government of {canon}",
                            }
                            if bare in country_bare_rewrites:
                                query_for_retrieval = country_bare_rewrites[bare]
                                logger.debug(
                                    "Country-attribute rewriter: %r -> %r",
                                    raw_text, query_for_retrieval,
                                )
                        bare_rewrites: dict[str, str] = {
                            "location": f"where is {canon} located",
                            "where": f"where is {canon} located",
                            "headquarters": f"where is {canon} located",
                            "headquartered": f"where is {canon} located",
                            "ceo": f"who is the ceo of {canon}",
                            "the ceo": f"who is the ceo of {canon}",
                            # Iter 40: possessive-pronoun entity-slot
                            # probes — "their CEO", "their founder",
                            # "their location" — should bind to the
                            # active topic just like the bare-noun
                            # forms.  Without this, the SLM hallucinates
                            # generic "Which company did you have in
                            # mind?" replies even when the tracker has a
                            # clear topic.
                            "their ceo": f"who is the ceo of {canon}",
                            "its ceo": f"who is the ceo of {canon}",
                            "his ceo": f"who is the ceo of {canon}",
                            "her ceo": f"who is the ceo of {canon}",
                            "their founder": f"who founded {canon}",
                            "its founder": f"who founded {canon}",
                            "their founders": f"who founded {canon}",
                            "their location": f"where is {canon} located",
                            "its location": f"where is {canon} located",
                            "their headquarters": f"where is {canon} located",
                            "its headquarters": f"where is {canon} located",
                            "their hq": f"where is {canon} located",
                            "its hq": f"where is {canon} located",
                            "their products": f"what does {canon} sell",
                            "its products": f"what does {canon} sell",
                            "their main product": f"what does {canon} sell",
                            "their competitors": f"who are {canon} competitors",
                            "its competitors": f"who are {canon} competitors",
                            "founder": f"who founded {canon}",
                            "founders": f"who founded {canon}",
                            "founded": f"when was {canon} founded",
                            "products": f"what does {canon} sell",
                            "main product": f"what does {canon} sell",
                            "what they sell": f"what does {canon} sell",
                            "what they make": f"what does {canon} sell",
                            # Phase B (2026-04-25) — bare temporal /
                            # event follow-ups.  When the user's most
                            # recent topic is an org / event / concept,
                            # the right entity-tool query depends on
                            # that topic (e.g. "when?" after "tell me
                            # about apple" → "when was apple founded?").
                            "when": f"when was {canon} founded",
                            "who": f"who founded {canon}",
                            "who founded": f"who founded {canon}",
                            "who created": f"who founded {canon}",
                            "who started": f"who founded {canon}",
                            "who made": f"who founded {canon}",
                            "who built": f"who founded {canon}",
                            "who invented": f"who founded {canon}",
                            "who won": f"who won {canon}",
                            "competitors": f"who are {canon} competitors",
                            "competitor": f"who are {canon} competitors",
                            "owners": f"who owns {canon}",
                            "owner": f"who owns {canon}",
                            "fall": f"when did {canon} fall",
                            "ended": f"when did {canon} end",
                            "started": f"when was {canon} founded",
                            "start": f"when was {canon} founded",
                            # Iteration 14 (2026-04-26): topic-pivot
                            # rewrites — bare generic prompts after a
                            # topic should ride that topic forward.
                            # Without these, "how" / "why" / "more"
                            # after "what is gravity" hit the curated
                            # clarifier entry instead of expanding to
                            # "how does gravity work".
                            "how": f"how does {canon} work",
                            "why": f"why does {canon} matter",
                            "more": f"tell me more about {canon}",
                            "tell me more": f"tell me more about {canon}",
                            "explain": f"explain {canon}",
                            "explain that": f"explain {canon}",
                            "explain it": f"explain {canon}",
                            "elaborate": f"tell me more about {canon}",
                            "elaborate on that": f"tell me more about {canon}",
                            "details": f"explain {canon}",
                            "more details": f"tell me more about {canon}",
                            "go on": f"tell me more about {canon}",
                            "go deeper": f"explain {canon}",
                            "deeper": f"explain {canon}",
                            "and": f"tell me more about {canon}",
                            "expand": f"explain {canon}",
                            "continue": f"tell me more about {canon}",
                            "keep going": f"tell me more about {canon}",
                            "what else": f"tell me more about {canon}",
                            "what about it": f"tell me more about {canon}",
                            "more about it": f"tell me more about {canon}",
                            "more on that": f"tell me more about {canon}",
                            "say more": f"tell me more about {canon}",
                            # Iter 42 (2026-04-26): short ack words —
                            # when the active topic is still on the
                            # stack within max_age_turns=3, treat
                            # "yeah" / "right" / "uh huh" / "got it"
                            # as topic-pivots ("tell me more about X")
                            # rather than letting them fall to the
                            # generic curated ack entry which doesn't
                            # carry topic forward.  The bare-rewrite
                            # path is gated by `recent is not None` so
                            # this only fires when there's a topic to
                            # pivot back to.
                            "yeah": f"tell me more about {canon}",
                            "yes": f"tell me more about {canon}",
                            "right": f"tell me more about {canon}",
                            "uh huh": f"tell me more about {canon}",
                            "uh-huh": f"tell me more about {canon}",
                            "mhm": f"tell me more about {canon}",
                            "mm": f"tell me more about {canon}",
                            "got it": f"tell me more about {canon}",
                            "makes sense": f"tell me more about {canon}",
                            "i see": f"tell me more about {canon}",
                            "ah": f"tell me more about {canon}",
                            "aha": f"tell me more about {canon}",
                            "interesting": f"tell me more about {canon}",
                            "nice": f"tell me more about {canon}",
                            "cool": f"tell me more about {canon}",
                            "wow": f"tell me more about {canon}",
                        }
                        # Iter 41 (2026-04-26): same-surface alt-sense
                        # disambiguation.  Some catalog entities share
                        # a surface name with another concept (Apple
                        # the company vs apple the fruit, Java the
                        # language vs Java the island).  When the
                        # user explicitly asks about "the fruit" /
                        # "the island" / "the snake" while the active
                        # topic is the alternate-sense canonical, route
                        # to the curated alt-sense entry.
                        if canon == "apple":
                            bare_rewrites["fruit"] = "what is the apple fruit"
                            bare_rewrites["the fruit"] = "what is the apple fruit"
                            bare_rewrites["what about the fruit"] = "what is the apple fruit"
                            bare_rewrites["what about fruit"] = "what is the apple fruit"
                            bare_rewrites["the apple fruit"] = "what is the apple fruit"
                            bare_rewrites["apple the fruit"] = "what is the apple fruit"
                            # Stock price probe — bare and possessive
                            # forms pivot to the org's stock entry.
                            bare_rewrites["stock price"] = f"what is {canon}'s stock price"
                            bare_rewrites["the stock price"] = f"what is {canon}'s stock price"
                            bare_rewrites["their stock price"] = f"what is {canon}'s stock price"
                            bare_rewrites["its stock price"] = f"what is {canon}'s stock price"
                            bare_rewrites["share price"] = f"what is {canon}'s stock price"
                            bare_rewrites["their share price"] = f"what is {canon}'s stock price"
                        if canon == "microsoft":
                            bare_rewrites["stock price"] = f"what is {canon}'s stock price"
                            bare_rewrites["the stock price"] = f"what is {canon}'s stock price"
                            bare_rewrites["their stock price"] = f"what is {canon}'s stock price"
                            bare_rewrites["its stock price"] = f"what is {canon}'s stock price"
                            bare_rewrites["share price"] = f"what is {canon}'s stock price"
                            bare_rewrites["their share price"] = f"what is {canon}'s stock price"
                        if bare in bare_rewrites:
                            query_for_retrieval = bare_rewrites[bare]
                            logger.debug(
                                "Bare-noun rewriter: %r -> %r",
                                raw_text, query_for_retrieval,
                            )
                        else:
                            # Phase B (2026-04-25): phrase-shape rewrites.
                            # Common follow-up shapes that include a
                            # pronoun ("it", "they") which the formal
                            # coref didn't catch (because the entity is
                            # in a multi-word topic the resolver
                            # hadn't yet wired in).  We still leave
                            # the SLM's prompt looking at the raw
                            # user text — only the retrieval cosine /
                            # tool path sees the rewrite.
                            phrase_rewrites = (
                                (r"^when\s+did\s+(?:it|they)\s+(?:start|begin)\??$",
                                 f"when was {canon} founded"),
                                (r"^when\s+did\s+(?:it|they)\s+(?:end|finish)\??$",
                                 f"when did {canon} end"),
                                (r"^when\s+did\s+(?:it|they)\s+fall\??$",
                                 f"when did {canon} fall"),
                                (r"^who\s+won\??$", f"who won {canon}"),
                                (r"^who\s+(?:founded|created|started)\s+(?:it|them)\??$",
                                 f"who founded {canon}"),
                                (r"^who\s+(?:proposed|discovered|invented)\s+(?:it|them)\??$",
                                 f"who proposed {canon}"),
                                (r"^who\s+runs\s+(?:it|them)\??$",
                                 f"who is the ceo of {canon}"),
                                (r"^who\s+leads\s+(?:it|them)\??$",
                                 f"who is the ceo of {canon}"),
                                (r"^where\s+(?:is|are)\s+(?:it|they)\??$",
                                 f"where is {canon} located"),
                                # Iteration 16 (2026-04-26): register-
                                # pivot rewrites — "explain in plain
                                # english" / "in simpler terms" / "give
                                # me an analogy" should be topic-pivots,
                                # NOT taken as new topics.  Without
                                # these, "explain in plain english" got
                                # interpreted as "explain English" and
                                # returned the English-language entry.
                                (r"^explain\s+(?:in\s+)?(?:plain\s+english|simply|simple\s+terms|simpler\s+terms|plain\s+language)\??$",
                                 f"explain {canon} simply"),
                                (r"^(?:in\s+)?simpler\s+(?:terms|words)\??$",
                                 f"explain {canon} simply"),
                                (r"^(?:in\s+)?plain\s+(?:english|terms|language)\??$",
                                 f"explain {canon} simply"),
                                # Iter 47: include bare "give an
                                # analogy" / "give a metaphor" / "give
                                # me an analogy" / "an analogy".  The
                                # earlier pattern only matched the
                                # "give me ..." or bare "an analogy"
                                # forms — "give an analogy" (no "me")
                                # fell through and the topic-prefix
                                # alone determined retrieval.
                                (r"^(?:give\s+(?:me\s+)?|can\s+you\s+give\s+(?:me\s+)?|share\s+|do\s+you\s+have\s+)?an?\s+(?:analogy|metaphor)(?:\s+for\s+it|\s+for\s+that)?\??$",
                                 f"give an analogy for {canon}"),
                                (r"^(?:what['']?s\s+)?(?:the\s+)?(?:simplest|simple|short)\s+(?:version|explanation|answer)\??$",
                                 f"explain {canon} simply"),
                                (r"^(?:break|walk)\s+it\s+down\??$",
                                 f"explain {canon} step by step"),
                                (r"^(?:in\s+a\s+)?nutshell\??$",
                                 f"explain {canon} simply"),
                                # Iter 43 (2026-04-26): "the science
                                # behind it" / "what's the science
                                # behind it" — pivot to the active
                                # topic so the curated science-behind-X
                                # entries don't mis-fire on whatever
                                # "the science behind ..." entry has
                                # the highest free-text cosine.
                                (r"^(?:what['']?s\s+|tell\s+me\s+)?(?:the\s+)?science\s+behind\s+(?:it|that|this)\??$",
                                 f"what is the science behind {canon}"),
                                (r"^how\s+does\s+(?:it|that|this)\s+(?:actually\s+)?work\??$",
                                 f"how does {canon} work"),
                                (r"^why\s+does\s+(?:it|that|this)\s+matter\??$",
                                 f"why does {canon} matter"),
                                (r"^how\s+does\s+(?:it|that|this)\s+scale\??$",
                                 f"how does {canon} scale"),
                                (r"^how\s+(?:big|fast)\s+(?:is|does)\s+(?:it|that|this)(?:\s+get|run)?\??$",
                                 f"how does {canon} scale"),
                                # "how do we prevent it" / "how do you avoid it"
                                (r"^how\s+(?:do\s+we|do\s+you|to)\s+(?:prevent|avoid|stop|combat|fight|deal\s+with|fix|address|mitigate)\s+(?:it|that|this)\??$",
                                 f"how do we prevent {canon}"),
                                # "how do we get rid of it" / "how to get rid of it"
                                (r"^how\s+(?:do\s+we|do\s+you|to)\s+get\s+rid\s+of\s+(?:it|that|this)\??$",
                                 f"how do we prevent {canon}"),
                                # Math chain ("and 30 percent of 100")
                                # — strip leading "and " for math
                                # follow-ups, otherwise the OOD branch
                                # eats them.
                                (r"^(?:and|or|then)\s+(\d+(?:\.\d+)?)\s*(?:percent|%)\s+of\s+(\d+(?:\.\d+)?)\s*\??$",
                                 None),  # signal: re-run as raw math
                            )
                            for pat_str, rewrite in phrase_rewrites:
                                m = re.match(pat_str, bare, re.I)
                                if m:
                                    if rewrite is None:
                                        # Math-chain signal: peel the
                                        # "and"/"or"/"then" prefix and
                                        # let the math tool see the
                                        # raw expression.
                                        try:
                                            stripped = re.sub(
                                                r"^(?:and|or|then)\s+",
                                                "",
                                                bare,
                                                count=1,
                                                flags=re.I,
                                            )
                                            query_for_retrieval = stripped
                                            logger.debug(
                                                "Math-chain peel: %r -> %r",
                                                raw_text,
                                                query_for_retrieval,
                                            )
                                        except Exception:
                                            pass
                                        break
                                    query_for_retrieval = rewrite
                                    logger.debug(
                                        "Phrase-rewriter: %r -> %r",
                                        raw_text, query_for_retrieval,
                                    )
                                    break
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "EntityTracker.resolve failed for user_id=%s session_id=%s",
                input.user_id,
                input.session_id,
            )
            coref_resolution = None
        # Stash for the WS layer / reasoning trace.
        self._last_coref[
            self._history_key(input.user_id, input.session_id)
        ] = coref_resolution

        # Record entity-tracker activity as a virtual "stage" — the
        # tracker ran inline above; this records its outcome for the
        # Flow dashboard.
        try:
            _coref_used = bool(
                coref_resolution is not None
                and getattr(coref_resolution, "used_entity", None) is not None
            )
            self._trace_collector.record_skipped(
                _trace, "entity_tracker", "Entity tracker / coref",
                reason=(
                    f"resolved {coref_resolution.original_query!r} → "
                    f"{coref_resolution.resolved_query!r}"
                    if _coref_used
                    else "no pronoun on this turn"
                ),
            ) if _trace is not None else None
            # Mark as fired=True manually if coref actually ran a resolve
            if _trace is not None and len(_trace.stages) > 0:
                _last = _trace._stage_index.get("entity_tracker")
                if _last is not None:
                    _last.fired = True
                    if _coref_used:
                        _last.outputs["resolved"] = True
                        _last.outputs["entity"] = (
                            coref_resolution.used_entity.canonical
                            if coref_resolution.used_entity is not None
                            else None
                        )
        except Exception:  # pragma: no cover - decorative
            pass

        # ---- Step 6: Route decision -------------------------------------
        with self._trace_stage(_trace, "router", "Router (LinUCB)"):
            query_complexity = self.complexity_estimator.estimate(
                input.message_text
            )
            topic_sensitivity = self.sensitivity_detector.detect(
                input.message_text
            )

            route_chosen, routing_confidence, routing_ctx_vec = self._make_routing_decision(
                user_id=input.user_id,
                user_state_embedding=user_state_embedding,
                features=features,
                query_complexity=query_complexity,
                topic_sensitivity=topic_sensitivity,
                user_model=user_model,
                message_text=input.message_text,
                retrieval_top_score=None,
                session_id=input.session_id,
            )
            self._trace_note(
                _trace, "router",
                _input={
                    "complexity": float(query_complexity),
                    "sensitivity": float(topic_sensitivity),
                },
                _output={
                    "route": str(route_chosen),
                    "p_local": float(routing_confidence.get("local_slm", 0.0)),
                    "p_cloud": float(routing_confidence.get("cloud_llm", 0.0)),
                },
                _notes=(
                    f"contextual Thompson bandit -> {route_chosen}"
                ),
            )
        # If the cloud route is off (no credit, explicitly disabled, or
        # circuit-broken after a prior 4xx) rewrite the decision to the
        # local SLM so the UI reports the truth and the confidence bars
        # reflect what actually answered.  The project is edge-first by
        # design — falling back silently but reporting ``cloud_llm`` as
        # the answering route was the single biggest source of "why does
        # my chat not use the SLM?" confusion.
        if (
            route_chosen == "cloud_llm"
            and not self.cloud_client.is_available
        ):
            routing_confidence = {
                "local_slm": 1.0,
                "cloud_llm": 0.0,
            }
            route_chosen = "local_slm"
        logger.debug(
            "Step 6 complete: route=%s, confidence=%s",
            route_chosen,
            routing_confidence,
        )

        # ---- Step 6b: Affect-shift detection (pre-generation) ------------
        # Done BEFORE the response is generated so that subsequent
        # steps (state classifier, accessibility-mode controller,
        # adaptation overrides) all see a consistent, fully-populated
        # affect picture for the turn.  The detector only consumes the
        # embedding + keystroke metrics — it doesn't need the reply
        # text — so it is safe to run here.  The suggestion (if any)
        # is appended after the response is generated.
        with self._trace_stage(_trace, "affect_shift", "Affect-shift detector"):
            affect_shift = self._observe_affect_shift(
                user_id=input.user_id,
                session_id=input.session_id,
                embedding=user_state_embedding,
                input=input,
            )
            self._trace_note(
                _trace, "affect_shift",
                _output={
                    "detected": bool(getattr(affect_shift, "detected", False))
                    if affect_shift is not None else False,
                },
                _notes=(
                    "rolling-embedding cosine + keystroke delta tracker"
                ),
            )

        # ---- Step 6c: Discrete user-state classifier (Live Badge) --------
        with self._trace_stage(_trace, "state_classifier", "State classifier"):
            user_state_label_dict = self._classify_user_state(
                user_id=input.user_id,
                session_id=input.session_id,
                adaptation=adaptation,
                input=input,
                engagement_score=float(getattr(user_model, "engagement_score", 0.0)),
                deviation_from_baseline=float(deviation.current_vs_baseline),
                messages_in_session=(
                    user_model.current_session.message_count
                    if user_model.current_session is not None
                    else 0
                ),
                baseline_established=bool(user_model.baseline_established),
            )
            self._trace_note(
                _trace, "state_classifier",
                _output={
                    "state": str(
                        (user_state_label_dict or {}).get("state", "—")
                    ),
                    "confidence": float(
                        (user_state_label_dict or {}).get("confidence", 0.0) or 0.0
                    ),
                },
                _notes="6-class label — calm/focused/stressed/tired/distracted/warming",
            )

        # ---- Step 6d: Accessibility-mode controller ----------------------
        with self._trace_stage(_trace, "accessibility", "Accessibility controller"):
            accessibility_state_dict = self._observe_accessibility(
                user_id=input.user_id,
                session_id=input.session_id,
                adaptation=adaptation,
                input=input,
            )
            self._trace_note(
                _trace, "accessibility",
                _output={
                    "active": bool(
                        (accessibility_state_dict or {}).get("active", False)
                    ),
                },
                _notes="sticky controller — engages on sustained motor-difficulty",
            )

        # ---- Step 6e (early): Biometric authentication --------------------
        with self._trace_stage(_trace, "biometric", "Biometric Identity Lock"):
            biometric_dict = self._observe_biometric(
                user_id=input.user_id,
                embedding=user_state_embedding,
                input=input,
            )
            _bio_state = str((biometric_dict or {}).get("state", "—"))
            self._trace_note(
                _trace, "biometric",
                _output={
                    "state": _bio_state,
                    "match": float(
                        (biometric_dict or {}).get("match_score", 0.0) or 0.0
                    ),
                },
                _notes=(
                    "Monrose-Rubin / Killourhy-Maxion keystroke template"
                ),
            )

        # ---- Step 6f: Apply per-biometric LoRA personalisation -----------
        with self._trace_stage(_trace, "personalisation", "Personal LoRA adapter"):
            adaptation, personalisation_dict = self._apply_personalisation(
                user_id=input.user_id,
                user_state_embedding=user_state_embedding,
                adaptation=adaptation,
                biometric_dict=biometric_dict,
            )
            self._last_personalisation[input.user_id] = personalisation_dict
            _applied = bool((personalisation_dict or {}).get("applied", False))
            self._trace_note(
                _trace, "personalisation",
                _output={
                    "applied": _applied,
                    "rank": int(
                        (personalisation_dict or {}).get("rank", 4) or 4
                    ),
                    "params": int(
                        (personalisation_dict or {}).get("num_parameters", 0) or 0
                    ),
                },
                _notes=(
                    "8-d residual layered on adaptation"
                    if _applied else "unregistered — base model"
                ),
            )
        self._trace_arrow(
            _trace, "adaptation", "personalisation",
            payload_summary="8-d adaptation",
            size_bytes=32,
        )
        self._trace_arrow(
            _trace, "personalisation", "router",
            payload_summary="8-d adapted vector",
            size_bytes=32,
        )

        # ---- Step 6e: Override adaptation while accessibility-mode is active
        # Done before _generate_response so the adapt_with_log call
        # inside it sees the boosted knobs and the change-log chip
        # strip records the override.
        if accessibility_state_dict and accessibility_state_dict.get("active"):
            adaptation = self._force_accessibility_adaptation(adaptation)

        # ---- Step 6f': Apply Playground tab overrides --------------------
        # When the request carried ``playground_overrides`` (Deliverable
        # 3) we inject explicitly-set fields here so the generation step
        # below sees the operator's choices.  Defaults reproduce normal
        # behaviour exactly — only keys with non-default values are
        # honoured, never silently filled in.
        if playground_overrides is not None:
            try:
                from i3.adaptation.types import AdaptationVector
                if isinstance(playground_overrides.get("adaptation"), dict):
                    adaptation = AdaptationVector.from_dict(
                        playground_overrides["adaptation"]
                    )
                _force_route = str(
                    playground_overrides.get("route") or ""
                ).strip().lower()
                if _force_route == "edge":
                    route_chosen = "local_slm"
                    routing_confidence = {"local_slm": 1.0, "cloud_llm": 0.0}
                elif _force_route == "cloud":
                    route_chosen = "cloud_llm"
                    routing_confidence = {"local_slm": 0.0, "cloud_llm": 1.0}
                if playground_overrides.get("accessibility") is True:
                    adaptation = self._force_accessibility_adaptation(adaptation)
                # Forge a biometric verdict for the response payload
                # without touching the underlying authenticator state.
                _bio_force = str(
                    playground_overrides.get("biometric_state") or ""
                ).strip().lower()
                _bio_map = {
                    "registered": ("registered", 1.0, True),
                    "mismatch":  ("mismatch", 0.15, False),
                    "unregistered": ("unregistered", 0.0, False),
                }
                if _bio_force in _bio_map and biometric_dict is not None:
                    state, sim, owner = _bio_map[_bio_force]
                    biometric_dict["state"] = state
                    biometric_dict["similarity"] = float(sim)
                    biometric_dict["confidence"] = float(sim)
                    biometric_dict["is_owner"] = owner
                    biometric_dict["notes"] = (
                        f"playground override: forced {state}"
                    )
            except Exception:  # pragma: no cover - defensive
                logger.exception("Playground overrides failed to apply")

        # ---- Step 7: Generate response -----------------------------------
        # ``query_for_retrieval`` is the co-reference-resolved form of
        # the user's message (or the raw text when no pronoun was
        # detected).  The retriever's keyword-overlap + entity-tool
        # paths both consume this so a follow-up like "where are
        # they?" can land an entity-fact answer.  The SLM and the
        # response-cleaning step still see the raw user text so we
        # don't echo "huawei" in places the user never said it.
        with self._trace_stage(_trace, "generation", "Generation / retrieval"):
            if safety_refuse_text is not None:
                # Constitutional safety refusal — never even retrieve.
                response_text = safety_refuse_text
                self._last_response_path = "tool:safety"
                self._last_retrieval_score = 0.0
                self._last_adaptation_changes = []
            else:
                response_text = await self._generate_response(
                    route=route_chosen,
                    message=sanitized.sanitized_text,
                    adaptation=adaptation,
                    user_state=user_state_embedding,
                    on_token=on_token,
                    user_id=input.user_id,
                    session_id=input.session_id,
                    query_for_retrieval=query_for_retrieval,
                    topic_prefix_for_embedding=topic_prefix_for_embedding,
                )
                # Iter 51 (2026-04-27): the safety caveat is no longer
                # inlined into response_text — the dashboard surfaces
                # it as a separate chip via PipelineOutput.safety_caveat.
                # Inlining created visible noise on benign topics
                # (e.g. "ok back to transformers" → unnecessary harm
                # warning).  When safety_caveat is set the frontend
                # renders a small "ⓘ moderation note" pill next to
                # the response, but the chat bubble stays clean.
                # safety_caveat captured in local var, propagated
                # via PipelineOutput.safety_caveat below.
            _path = str(getattr(self, "_last_response_path", "unknown"))
            _is_tool = _path.startswith("tool")
            self._trace_note(
                _trace, "generation",
                _output={
                    "path": _path,
                    "words": len(response_text.split()),
                    "retrieval_score": float(
                        getattr(self, "_last_retrieval_score", 0.0) or 0.0
                    ),
                },
                _notes=f"answered via {_path}",
                _is_tool=_is_tool,
            )
            # Critique only fires on the SLM path; record skip otherwise.
            if _path == "slm":
                _crit = getattr(self, "_last_critique", None) or {}
                self._trace_collector.record_skipped(
                    _trace, "critique", "Self-critique",
                    reason=(
                        f"score={float(_crit.get('final_score', 0.0)):.2f} "
                        f"accepted={bool(_crit.get('accepted', False))}"
                    ),
                ) if _trace is not None else None
                # Manually flip critique to fired=True since it ran.
                if _trace is not None:
                    _last = _trace._stage_index.get("critique")
                    if _last is not None:
                        _last.fired = True
                        _last.outputs["score"] = float(
                            _crit.get("final_score", 0.0) or 0.0
                        )
                        _last.outputs["accepted"] = bool(
                            _crit.get("accepted", False)
                        )
            else:
                self._trace_collector.record_skipped(
                    _trace, "critique", "Self-critique",
                    reason=f"skipped — non-SLM path ({_path})",
                ) if _trace is not None else None

            self._trace_collector.record_skipped(
                _trace, "postprocess", "Post-process",
                reason="adaptation rewrites + style mirror",
            ) if _trace is not None else None
            if _trace is not None:
                _post = _trace._stage_index.get("postprocess")
                if _post is not None:
                    _post.fired = True
                    _post.outputs["changes"] = len(
                        list(getattr(self, "_last_adaptation_changes", []) or [])
                    )
        self._trace_arrow(
            _trace, "router", "generation",
            payload_summary=f"route={route_chosen}",
            size_bytes=8,
        )
        self._trace_arrow(
            _trace, "generation", "postprocess",
            payload_summary=f"{len(response_text.split())} words",
            size_bytes=min(4096, len(response_text)),
        )
        logger.debug(
            "Step 7 complete: generated %d-word response via %s",
            len(response_text.split()),
            route_chosen,
        )

        # ---- Step 7a: Sentence-level dedupe (iter 51, 2026-04-27) -------
        # Catches cases where retrieval + SLM regeneration / KG-overview
        # chaining / decomposer composition leaves near-duplicate
        # sentences in the final response.  Symptom prior to this fix:
        # "Python is famous for ... Python was founded by Guido in 1991.
        #  Python was founded in 1991. Python is a high-level ..." — the
        # year sentence and a "Python is" reintroduction would both
        # appear.  Dedupe on a token-set Jaccard ≥ 0.6 fingerprint.
        try:
            response_text = self._dedupe_sentences(response_text)
        except Exception:  # pragma: no cover — never let dedupe crash
            pass

        # ---- Step 7b: Affect-shift check-in is delivered via the
        # PipelineOutput.affect_shift field (NOT inlined into the chat
        # bubble).  Iter 51: rendering moved to a side-channel chip so
        # benign turns no longer get a "your interaction pattern shifted"
        # paragraph appended to the answer.  See web/js/state_panel.js
        # for the chip render path.
        # affect_shift remains in scope and gets propagated to the
        # PipelineOutput at the bottom of this method.

        # ---- Step 7c: Surface the accessibility override on the chip strip
        # The post-processor's per-axis change log is already populated
        # via ``_last_adaptation_changes``; tack on a marker entry so
        # the UI shows that the override was the proximate cause.
        if accessibility_state_dict and accessibility_state_dict.get("active"):
            self._last_adaptation_changes = list(
                getattr(self, "_last_adaptation_changes", []) or []
            )
            self._last_adaptation_changes.append({
                "axis": "accessibility_mode",
                "value": "active",
                "change": (
                    "accessibility mode active → forced cognitive_load=0.85, "
                    "accessibility=0.95, verbosity≤0.25"
                ),
            })

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # ---- Step 8: Log exchange to diary (async, fire-and-forget) ------
        # SEC: Track the task in self._background_tasks so it cannot be
        # garbage-collected mid-flight (PEP 3156 / asyncio docs warning).
        # The done callback removes the task from the set and logs any
        # exception that escaped _log_exchange_safe's own try/except.
        log_task = asyncio.create_task(
            self._log_exchange_safe(
                session_id=input.session_id,
                user_state_embedding=user_state_embedding,
                adaptation=adaptation,
                route_chosen=route_chosen,
                latency_ms=latency_ms,
                user_id=input.user_id,
                message_text=input.message_text,
            )
        )
        self._background_tasks.add(log_task)
        log_task.add_done_callback(self._background_task_done)

        # ---- Step 9: Update engagement tracking --------------------------
        self._last_response_time[input.user_id] = time.time()
        self._last_response_length[input.user_id] = len(response_text.split())
        self._previous_route[input.user_id] = (
            0 if route_chosen == "local_slm" else 1
        )
        # SEC: persist the exact context vector so compute_engagement()
        # can feed it back to bandit.update() instead of zeros.
        self._previous_routing_context[input.user_id] = routing_ctx_vec

        # ---- Step 10: Append the finalised exchange to session history ---
        # Stored *after* postprocessing + adaptation rewrites so the
        # next turn's history reflects what the user actually saw on
        # screen, not the raw SLM/retrieval output. Retrieval-side and
        # SLM-side prompt construction both read from this buffer on
        # the *next* call to process_message().
        try:
            self._append_history_pair(
                user_id=input.user_id,
                session_id=input.session_id,
                user_text=input.message_text,
                assistant_text=response_text,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to append session history for user_id=%s session_id=%s",
                input.user_id,
                input.session_id,
            )

        # ---- Step 10b: Observe entities mentioned this turn -------------
        # Update the per-session entity recency stack from BOTH the
        # user message and the assistant reply.  The next turn's
        # co-reference resolver consumes this stack to anchor pronoun
        # rewrites.  Wrapped in try/except so a tracker bug can never
        # break the response path.
        try:
            # If coref resolved this turn's pronoun to a specific
            # entity, promote that entity to the top of the stack
            # AFTER the assistant response is observed.  Otherwise a
            # rambly response that mentions a different topic
            # ("...PyTorch transformer...") would steal the anaphor
            # priority from the user's actual subject ("python").
            priority_canonical: str | None = None
            try:
                if (
                    coref_resolution is not None
                    and coref_resolution.used_entity is not None
                ):
                    priority_canonical = (
                        coref_resolution.used_entity.canonical
                    )
            except Exception:
                priority_canonical = None
            # Iter 40: when a negation-pivot fired this turn (e.g.
            # "actually not Apple, Microsoft"), promote the new
            # subject over any coref-resolved entity so the NEXT
            # turn's bare follow-up ("their CEO") resolves to it.
            # Also scrub the negated surface from the user_text passed
            # to observe so the negated entity is NOT re-anchored on
            # this turn (otherwise both end up user-anchored at the
            # same turn → clarifier fires on the next message).
            _observe_user_text = sanitized.sanitized_text
            if _neg_anchored:
                priority_canonical = _neg_anchored
                if _neg_negated_surface:
                    try:
                        _observe_user_text = re.sub(
                            r"\b" + re.escape(_neg_negated_surface) + r"\b",
                            "",
                            _observe_user_text,
                            flags=re.I,
                        )
                    except Exception:
                        pass
            self._entity_tracker.observe(
                user_id=input.user_id,
                session_id=input.session_id,
                turn_idx=current_turn,
                user_text=_observe_user_text,
                assistant_text=response_text,
                priority_canonical=priority_canonical,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "EntityTracker.observe failed for user_id=%s session_id=%s",
                input.user_id,
                input.session_id,
            )

        # ---- Hierarchical memory observe (Phase B.5) -------------------
        # Topic stack + user-stated facts + thread summary update.
        # Always wrapped in try/except so a memory failure can never
        # block a turn — single-turn behaviour is the worst-case
        # degradation.
        try:
            if self._hierarchical_memory is not None:
                snap = self._entity_tracker.snapshot(
                    input.user_id, input.session_id,
                )
                fresh_entities: list[str] = []
                seen_canon: set[str] = set()
                for f in snap[:3]:  # top-3 most recent entities
                    if f.kind not in {"org", "topic", "person"}:
                        continue
                    if f.canonical in seen_canon:
                        continue
                    seen_canon.add(f.canonical)
                    fresh_entities.append(f.canonical)
                self._hierarchical_memory.observe(
                    input.user_id,
                    input.session_id,
                    user_message=sanitized.sanitized_text,
                    recent_entities=fresh_entities,
                )
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "HierarchicalMemory.observe failed (non-blocking)",
                exc_info=True,
            )

        # ---- Build output ------------------------------------------------
        session = user_model.current_session
        # SEC: Clamp engagement_score to [0, 1] - the user model's
        # engagement_score property is computed from EMAs that *should*
        # stay in range but a regression upstream must not corrupt the
        # API contract.
        raw_engagement = float(getattr(user_model, "engagement_score", 0.0))
        engagement_score = max(0.0, min(1.0, raw_engagement))

        # SEC: Ensure routing_confidence always contains every arm with
        # numeric values. Defends against future arms being added without
        # the bandit emitting a key.
        full_confidence: dict[str, float] = dict.fromkeys(self.config.router.arms, 0.0)
        for k, v in routing_confidence.items():
            try:
                full_confidence[k] = float(v)
            except (TypeError, ValueError):
                full_confidence[k] = 0.0

        # Update the per-(user_id, session_id) profile aggregator so
        # the Profile tab snapshot reflects this turn.  Wrapped to
        # never block the response path.
        try:
            self._profile_update(
                user_id=input.user_id,
                session_id=input.session_id,
                input=input,
                adaptation_dict=self._adaptation_to_dict(adaptation),
                user_state_label=user_state_label_dict,
                affect_shift=affect_shift,
                biometric=biometric_dict,
                engagement_score=float(engagement_score),
            )
        except Exception:  # pragma: no cover - decorative
            logger.exception(
                "profile_update failed for user_id=%s session_id=%s",
                input.user_id,
                input.session_id,
            )

        # Forward the critique trace ONLY on SLM-generation turns; on
        # retrieval / tool / OOD paths the critic never ran (or its
        # state is stale from a prior turn) and shipping a trace would
        # mislead the UI into showing "self-critique accepted" on a
        # retrieval reply that was never scored.
        last_path = str(getattr(self, "_last_response_path", "") or "")
        critique_dict: dict | None = None
        if last_path == "slm":
            stash = getattr(self, "_last_critique", None) or None
            critique_dict = dict(stash) if isinstance(stash, dict) and stash else None

        # Serialise the co-reference resolution for the WS layer
        # (resolution_to_dict returns None when no entity was used so
        # we don't ship empty noise on every turn).
        try:
            from i3.dialogue.coref import resolution_to_dict
            coref_dict = resolution_to_dict(coref_resolution)
        except Exception:  # pragma: no cover - defensive
            coref_dict = None

        # Finalise the per-turn flow trace.  Wrapped to never block.
        try:
            pipeline_trace_dict: dict | None = (
                self._trace_collector.finalise(_trace)
                if _trace is not None else None
            )
        except Exception:  # pragma: no cover - decorative
            logger.warning("trace.finalise failed", exc_info=True)
            pipeline_trace_dict = None

        # Always populate a privacy-budget snapshot — even on edge-only
        # turns — so the UI's counter stays live and the user can see
        # "0 of 50 calls used" before any cloud call has fired.  The
        # snapshot is JSON-safe.
        try:
            privacy_budget_dict = self.privacy_budget.snapshot(
                input.user_id, input.session_id
            ).to_dict()
            self._last_privacy_budget_snapshot = privacy_budget_dict
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "privacy_budget.snapshot failed; emitting None"
            )
            privacy_budget_dict = None

        # Forward the most recent routing decision dict.  Always
        # populated — even on privacy-override or budget-exhausted
        # turns — so the UI's scatter plot has a fresh data point on
        # every turn.
        routing_decision_dict = (
            dict(self._last_routing_decision)
            if isinstance(self._last_routing_decision, dict)
            else None
        )

        # Hierarchical session memory snapshot (Phase B.5).  Always
        # populated when the memory module is wired in so the UI can
        # render the "user-stated facts on file" chip + thread summary
        # under the response.  None for legacy callers.
        session_memory_dict: dict | None = None
        try:
            if self._hierarchical_memory is not None:
                session_memory_dict = self._hierarchical_memory.to_dict(
                    input.user_id, input.session_id,
                )
        except Exception:  # pragma: no cover - defensive
            session_memory_dict = None

        # Multi-step explain decomposition plan (Phase B.3).  Populated
        # only on turns where the engine ran the decomposer; cleared
        # to None for every other turn.
        explain_plan_dict: dict | None = getattr(
            self, "_last_explain_plan", None,
        )
        # Reset for next turn so the plan never bleeds across turns.
        if hasattr(self, "_last_explain_plan"):
            self._last_explain_plan = None

        # Iter 55: push this turn's latency into the per-cascade-arm
        # rolling tracker so /api/cascade/stats can expose live p50/p95
        # to the dashboard's cascade ribbon.
        try:
            arm = self._classify_cascade_arm(
                getattr(self, "_last_response_path", "unknown"),
                route_chosen,
            )
            self._cascade_arm_latencies[arm].append(float(latency_ms))
        except Exception:  # pragma: no cover - never block on telemetry
            pass

        # Iter 51: snapshot the (user, session) stated-facts dict so the
        # Personal Facts dashboard tab can render live.  Returns a
        # *copy* so the UI can't mutate the engine's authoritative
        # store.  None when nothing has been recorded yet for this
        # session (the common cold-start case).
        try:
            facts_snapshot = self._stated_facts.get(
                (input.user_id, input.session_id),
            )
            personal_facts_dict: dict | None = (
                dict(facts_snapshot) if facts_snapshot else None
            )
        except Exception:  # pragma: no cover - defensive
            personal_facts_dict = None

        # Iter 51: per-turn intent-parse result if the engine ran the
        # parser this turn; None for normal chat.  Stored on
        # ``_last_intent_result`` by the intent-routing path; cleared
        # immediately so it doesn't bleed across turns.
        intent_result_dict: dict | None = getattr(
            self, "_last_intent_result", None,
        )
        if hasattr(self, "_last_intent_result"):
            self._last_intent_result = None

        return PipelineOutput(
            response_text=response_text,
            route_chosen=route_chosen,
            latency_ms=round(latency_ms, 2),
            user_state_embedding_2d=embedding_2d,
            adaptation=adaptation.to_dict(),
            engagement_score=engagement_score,
            deviation_from_baseline=float(deviation.current_vs_baseline),
            routing_confidence=full_confidence,
            messages_in_session=(
                session.message_count if session is not None else 0
            ),
            baseline_established=bool(user_model.baseline_established),
            response_path=getattr(self, "_last_response_path", "unknown"),
            retrieval_score=float(getattr(self, "_last_retrieval_score", 0.0)),
            adaptation_changes=list(
                getattr(self, "_last_adaptation_changes", []) or []
            ),
            affect_shift=affect_shift,
            safety_caveat=safety_caveat or None,
            user_state_label=user_state_label_dict,
            accessibility=accessibility_state_dict,
            biometric=biometric_dict,
            coreference_resolution=coref_dict,
            critique=critique_dict,
            personalisation=self._last_personalisation.get(input.user_id),
            multimodal=multimodal_dict,
            gaze=gaze_output_dict,
            pipeline_trace=pipeline_trace_dict,
            routing_decision=routing_decision_dict,
            privacy_budget=privacy_budget_dict,
            safety=safety_dict,
            session_memory=session_memory_dict,
            explain_plan=explain_plan_dict,
            personal_facts=personal_facts_dict,
            intent_result=intent_result_dict,
        )

    # ------------------------------------------------------------------
    # Pipeline-trace helpers (Flow dashboard)
    # ------------------------------------------------------------------
    # The collector returns a context manager from ``stage()``.  We wrap
    # it in these helpers so the engine call sites stay short and so a
    # ``None`` handle (collector-init-failure path) silently no-ops
    # without forcing every callsite to check.

    def _trace_stage(self, handle, stage_id: str, label: str, *, is_tool: bool = False):
        """Return a context manager that records *stage_id* execution.

        Falls back to a null context manager if *handle* is ``None``
        (collector failed to start) so the engine call sites can stay
        unconditional.
        """
        if handle is None:
            from contextlib import nullcontext
            return nullcontext()
        try:
            return self._trace_collector.stage(
                handle, stage_id, label, is_tool=is_tool,
            )
        except Exception:  # pragma: no cover - decorative
            from contextlib import nullcontext
            return nullcontext()

    def _trace_note(self, handle, stage_id: str, **kv) -> None:
        """Forward to ``PipelineTraceCollector.note`` if handle is live."""
        if handle is None:
            return
        try:
            self._trace_collector.note(handle, stage_id, **kv)
        except Exception:  # pragma: no cover - decorative
            pass

    def _trace_arrow(
        self,
        handle,
        from_id: str,
        to_id: str,
        *,
        payload_summary: str = "",
        size_bytes: int = 0,
    ) -> None:
        """Forward to ``PipelineTraceCollector.arrow`` if handle is live."""
        if handle is None:
            return
        try:
            self._trace_collector.arrow(
                handle, from_id, to_id,
                payload_summary=payload_summary,
                size_bytes=size_bytes,
            )
        except Exception:  # pragma: no cover - decorative
            pass

    # ------------------------------------------------------------------
    # Affect-shift detection helpers
    # ------------------------------------------------------------------

    def _observe_affect_shift(
        self,
        *,
        user_id: str,
        session_id: str,
        embedding: torch.Tensor,
        input: PipelineInput,
    ):
        """Run the affect-shift detector for this turn.

        Wrapped in a try/except so a detector failure can never block
        a response — the feature is decorative.  Returns the
        :class:`AffectShift` (or ``None`` on failure) and stashes it
        on ``self._last_affect_shift`` keyed on
        ``"{user_id}::{session_id}"`` so the WS layer can read it
        when assembling the reasoning trace.
        """
        try:
            iki_mean, iki_std = self._iki_stats(input.keystroke_timings)
            shift = self._shift_detector.observe(
                user_id=user_id,
                session_id=session_id,
                embedding=embedding,
                composition_time_ms=float(input.composition_time_ms),
                edit_count=int(input.edit_count),
                pause_before_send_ms=float(input.pause_before_send_ms),
                keystroke_iki_mean=iki_mean,
                keystroke_iki_std=iki_std,
            )
        except Exception:  # pragma: no cover - decorative; never blocks
            logger.exception(
                "AffectShiftDetector.observe failed for user_id=%s "
                "session_id=%s — continuing without shift",
                user_id,
                session_id,
            )
            return None
        self._last_affect_shift[f"{user_id}::{session_id}"] = shift
        return shift

    @staticmethod
    def _iki_stats(timings: list[float]) -> tuple[float, float]:
        """Mean + std-dev of inter-keystroke intervals (ms).

        ``timings`` may include zeros (the very first keystroke in a
        composition has no interval), which we filter out so the
        statistics describe genuine inter-key gaps.
        """
        if not timings:
            return 0.0, 0.0
        clean = [float(t) for t in timings if isinstance(t, (int, float)) and t > 0.0]
        if not clean:
            return 0.0, 0.0
        mean = sum(clean) / len(clean)
        if len(clean) > 1:
            var = sum((t - mean) ** 2 for t in clean) / len(clean)
            std = var ** 0.5
        else:
            std = 0.0
        return float(mean), float(std)

    def get_last_affect_shift(self, user_id: str, session_id: str):
        """Return the most recent :class:`AffectShift` for *user_id*/*session_id*.

        Returns ``None`` if the detector has not yet observed this
        session, or has been reset via :meth:`end_session`.  The WS
        layer uses this to plumb the shift into the reasoning trace
        without having to thread the value through every callsite.
        """
        return self._last_affect_shift.get(f"{user_id}::{session_id}")

    # ------------------------------------------------------------------
    # State classifier + accessibility helpers
    # ------------------------------------------------------------------

    def _classify_user_state(
        self,
        *,
        user_id: str,
        session_id: str,
        adaptation: Any,
        input: PipelineInput,
        engagement_score: float,
        deviation_from_baseline: float,
        messages_in_session: int,
        baseline_established: bool,
    ) -> dict | None:
        """Run the discrete user-state classifier for this turn.

        Wrapped in try/except — the classifier is purely cosmetic so
        a failure must never block a response.  Caches the result on
        ``self._last_user_state_label`` so the WS layer can read it
        back when emitting a state_badge frame mid-conversation.
        """
        try:
            from i3.affect.state_classifier import classify_user_state

            iki_mean, iki_std = self._iki_stats(input.keystroke_timings)
            adapt_dict = self._adaptation_to_dict(adaptation)
            label = classify_user_state(
                adaptation=adapt_dict,
                composition_time_ms=float(input.composition_time_ms),
                edit_count=int(input.edit_count),
                iki_mean=iki_mean,
                iki_std=iki_std,
                engagement_score=float(engagement_score),
                deviation_from_baseline=float(deviation_from_baseline),
                baseline_established=bool(baseline_established),
                messages_in_session=int(messages_in_session),
            )
            label_dict = label.to_dict()
        except Exception:  # pragma: no cover - decorative
            logger.exception(
                "classify_user_state failed for user_id=%s session_id=%s",
                user_id,
                session_id,
            )
            return None
        self._last_user_state_label[f"{user_id}::{session_id}"] = label_dict
        return label_dict

    def _observe_accessibility(
        self,
        *,
        user_id: str,
        session_id: str,
        adaptation: Any,
        input: PipelineInput,
    ) -> dict | None:
        """Run the accessibility-mode controller for this turn.

        Caches the resulting ``AccessibilityModeState`` dict on
        ``self._last_accessibility_state`` so the toggle endpoint and
        the WS layer can read the post-observation state.
        """
        try:
            iki_mean, iki_std = self._iki_stats(input.keystroke_timings)
            adapt_dict = self._adaptation_to_dict(adaptation)
            state = self._accessibility_controller.observe(
                user_id=user_id,
                session_id=session_id,
                edit_count=float(input.edit_count),
                iki_mean=float(iki_mean),
                iki_std=float(iki_std),
                cognitive_load=float(adapt_dict.get("cognitive_load", 0.5)),
                accessibility_axis=float(adapt_dict.get("accessibility", 0.0)),
            )
            state_dict = state.to_dict()
        except Exception:  # pragma: no cover - decorative
            logger.exception(
                "AccessibilityController.observe failed for user_id=%s session_id=%s",
                user_id,
                session_id,
            )
            return None
        self._last_accessibility_state[f"{user_id}::{session_id}"] = state_dict
        return state_dict

    @staticmethod
    def _force_accessibility_adaptation(adaptation: Any) -> Any:
        """Override adaptation knobs while accessibility mode is active.

        Pushes ``cognitive_load`` and ``accessibility`` toward the
        aggressive end, and clamps the style-mirror verbosity down so
        the post-processor produces a maximally short, simple-vocabulary
        reply.  Mutates the dataclass in place (the AdaptationVector is
        a per-turn object, never shared) and also returns it so the
        caller can chain.
        """
        try:
            adaptation.cognitive_load = max(
                float(getattr(adaptation, "cognitive_load", 0.5)), 0.85
            )
            adaptation.accessibility = max(
                float(getattr(adaptation, "accessibility", 0.0)), 0.95
            )
            style = getattr(adaptation, "style_mirror", None)
            if style is not None and hasattr(style, "verbosity"):
                style.verbosity = min(float(style.verbosity), 0.25)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "_force_accessibility_adaptation failed; leaving adaptation unchanged."
            )
        return adaptation

    def get_last_user_state_label(
        self, user_id: str, session_id: str
    ) -> dict | None:
        """Return the most recent user-state label dict, or ``None``."""
        return self._last_user_state_label.get(f"{user_id}::{session_id}")

    def get_last_accessibility_state(
        self, user_id: str, session_id: str
    ) -> dict | None:
        """Return the most recent accessibility state dict, or ``None``."""
        return self._last_accessibility_state.get(f"{user_id}::{session_id}")

    # ------------------------------------------------------------------
    # Biometric authentication (Identity Lock) helpers
    # ------------------------------------------------------------------

    def _observe_biometric(
        self,
        *,
        user_id: str,
        embedding: torch.Tensor,
        input: PipelineInput,
    ) -> dict | None:
        """Run the keystroke authenticator for this turn.

        Wrapped in try/except -- the authenticator is decorative; a
        failure must never block a response.  Caches the resulting
        :class:`BiometricMatch` dict on ``self._last_biometric`` keyed
        by ``user_id`` so the WS layer / API can read it back.

        See :class:`i3.biometric.KeystrokeAuthenticator` and the
        Monrose-Rubin / Killourhy-Maxion citations therein.
        """
        try:
            iki_mean, iki_std = self._iki_stats(input.keystroke_timings)
            match = self._keystroke_auth.observe(
                user_id=user_id,
                embedding=embedding,
                iki_mean=float(iki_mean),
                iki_std=float(iki_std),
                composition_time_ms=float(input.composition_time_ms),
                edit_count=int(input.edit_count),
            )
            match_dict = match.to_dict()
        except Exception:  # pragma: no cover - decorative
            logger.exception(
                "KeystrokeAuthenticator.observe failed for user_id=%s",
                user_id,
            )
            return None
        self._last_biometric[user_id] = match_dict
        # Track the rising-edge state so the WS layer can fire one-shot
        # biometric_event frames on transitions.
        self._last_biometric_state[user_id] = str(match_dict.get("state", ""))
        return match_dict

    def get_last_biometric(self, user_id: str) -> dict | None:
        """Return the most recent biometric match dict for *user_id*."""
        return self._last_biometric.get(user_id)

    def get_biometric_status(self, user_id: str) -> dict:
        """Read-only status of the typing-biometric auth for *user_id*.

        Returns a JSON-safe dict matching the
        :class:`i3.biometric.BiometricMatch` schema.  Used by
        ``GET /api/biometric/{user_id}/status``.
        """
        try:
            return self._keystroke_auth.status(user_id).to_dict()
        except Exception:  # pragma: no cover
            logger.exception(
                "KeystrokeAuthenticator.status failed for user_id=%s", user_id
            )
            return {
                "state": "unregistered",
                "similarity": 0.0,
                "confidence": 0.0,
                "threshold": 0.65,
                "enrolment_progress": 0,
                "enrolment_target": 5,
                "is_owner": False,
                "drift_alert": False,
                "diverged_signals": [],
                "ewma_iki_mean": 0.0,
                "ewma_iki_std": 0.0,
                "ewma_composition_ms": 0.0,
                "ewma_edit_rate": 0.0,
                "notes": "internal error",
            }

    def reset_biometric_for_user(self, user_id: str) -> dict:
        """Clear the biometric template for *user_id*.

        Idempotent.  Returns the post-reset state (always
        ``unregistered``).  Used by
        ``POST /api/biometric/{user_id}/reset``.
        """
        try:
            result = self._keystroke_auth.reset_for_user(user_id).to_dict()
        except Exception:
            logger.exception(
                "reset_biometric_for_user failed for user_id=%s", user_id
            )
            result = {
                "state": "unregistered",
                "similarity": 0.0,
                "confidence": 0.0,
                "threshold": 0.65,
                "enrolment_progress": 0,
                "enrolment_target": 5,
                "is_owner": False,
                "drift_alert": False,
                "diverged_signals": [],
                "ewma_iki_mean": 0.0,
                "ewma_iki_std": 0.0,
                "ewma_composition_ms": 0.0,
                "ewma_edit_rate": 0.0,
                "notes": "reset failed",
            }
        # Update the cached state so the next response/state_update
        # frame carries the post-reset payload.
        self._last_biometric[user_id] = result
        self._last_biometric_state[user_id] = str(result.get("state", ""))
        return result

    # ------------------------------------------------------------------
    # Per-biometric LoRA personalisation helpers (FLAGSHIP feature)
    # ------------------------------------------------------------------

    def _apply_personalisation(
        self,
        *,
        user_id: str,
        user_state_embedding: torch.Tensor,
        adaptation: Any,
        biometric_dict: dict | None,
    ) -> tuple[Any, dict]:
        """Layer the per-biometric LoRA residual onto the base adaptation.

        Wrapped in try/except so a personalisation glitch never blocks
        a turn.  Returns ``(maybe_personalised_adaptation, status_dict)``.

        Personalisation runs ONLY when the biometric is registered or
        verifying AND ``is_owner == True`` -- this is the killer demo
        line: *your typing identity gates your personalised model*.
        Unregistered users get the neutral base adaptation.

        See :class:`i3.personalisation.PersonalisationManager` for the
        adapter model + persistence design.
        """
        # Default: no personalisation, return base unchanged.
        default_status: dict = {
            "applied": False,
            "drift": {},
            "n_updates": 0,
            "user_key": None,
            "reason": "biometric not registered",
        }
        try:
            if not isinstance(biometric_dict, dict):
                return adaptation, default_status
            state = str(biometric_dict.get("state", ""))
            is_owner = bool(biometric_dict.get("is_owner", False))
            if state not in ("registered", "verifying") or not is_owner:
                default_status["reason"] = (
                    f"biometric state={state!r} is_owner={is_owner}"
                )
                return adaptation, default_status

            tmpl = self._keystroke_auth._templates.get(user_id)
            if tmpl is None or tmpl.template_embedding is None:
                default_status["reason"] = "no template embedding cached"
                return adaptation, default_status

            from i3.adaptation.types import AdaptationVector

            base_tensor = adaptation.to_tensor()
            personalised_tensor, drift = self._personalisation.apply(
                tmpl.template_embedding,
                user_state_embedding,
                base_tensor,
            )
            # Defensive per-axis bound at the call site -- the manager
            # already clips internally, but layering a second clamp
            # here makes the killer demo claim ("residual bounded
            # to ±0.15 per axis") audit-able from this file alone.
            residual = personalised_tensor - base_tensor
            residual_clamped = torch.clamp(residual, -0.15, 0.15)
            adapted_tensor = torch.clamp(
                base_tensor + residual_clamped, 0.0, 1.0
            )
            new_adaptation = AdaptationVector.from_tensor(adapted_tensor)

            adapter = self._personalisation.get_adapter(tmpl.template_embedding)
            user_key = self._personalisation.hash_template(
                tmpl.template_embedding
            )
            return new_adaptation, {
                "applied": True,
                "drift": drift,
                "n_updates": int(adapter.n_updates),
                "user_key": user_key,
                "num_parameters": int(adapter.num_parameters()),
                "rank": int(adapter.rank),
                "reason": "biometric verified — residual applied",
            }
        except Exception:
            logger.exception(
                "Pipeline._apply_personalisation failed for user_id=%s; "
                "falling back to base adaptation",
                user_id,
            )
            default_status["reason"] = "internal error"
            return adaptation, default_status

    def get_last_personalisation(self, user_id: str) -> dict | None:
        """Return the most recent personalisation status dict for *user_id*."""
        return self._last_personalisation.get(user_id)

    def get_personalisation_status(self, user_id: str) -> dict:
        """Return the per-user adapter status for the API endpoint.

        Returns a status dict whether or not the user has a registered
        biometric template; in the unregistered case the dict is a
        zero-filled placeholder so the UI can still render the tile.
        """
        try:
            tmpl = self._keystroke_auth._templates.get(user_id)
            if tmpl is None or tmpl.template_embedding is None:
                return {
                    "user_id": user_id,
                    "applied": False,
                    "biometric_registered": False,
                    "user_key": None,
                    "n_updates": 0,
                    "cumulative_drift": {},
                    "num_parameters": 0,
                    "rank": int(self._personalisation.rank),
                    "active_adapters": int(
                        self._personalisation.global_stats()["active_users"]
                    ),
                    "total_updates": int(
                        self._personalisation.global_stats()["total_updates"]
                    ),
                }
            status = self._personalisation.status(tmpl.template_embedding)
            status["user_id"] = user_id
            status["biometric_registered"] = True
            status["applied"] = status.get("n_updates", 0) > 0
            return status
        except Exception:
            logger.exception(
                "Pipeline.get_personalisation_status failed for user_id=%s",
                user_id,
            )
            return {
                "user_id": user_id,
                "applied": False,
                "biometric_registered": False,
                "user_key": None,
                "n_updates": 0,
                "cumulative_drift": {},
                "num_parameters": 0,
                "rank": int(self._personalisation.rank),
                "active_adapters": 0,
                "total_updates": 0,
            }

    def reset_personalisation_for_user(self, user_id: str) -> dict:
        """Clear the LoRA adapter for *user_id* (idempotent)."""
        try:
            tmpl = self._keystroke_auth._templates.get(user_id)
            if tmpl is None or tmpl.template_embedding is None:
                return {
                    "user_id": user_id,
                    "reset": False,
                    "reason": "no biometric template — nothing to reset",
                }
            result = self._personalisation.reset(tmpl.template_embedding)
            result["user_id"] = user_id
            result["reset"] = True
            self._last_personalisation.pop(user_id, None)
            return result
        except Exception:
            logger.exception(
                "Pipeline.reset_personalisation_for_user failed for "
                "user_id=%s",
                user_id,
            )
            return {
                "user_id": user_id,
                "reset": False,
                "reason": "internal error",
            }

    def get_personalisation_global_stats(self) -> dict:
        """Return system-wide adapter statistics for the admin endpoint."""
        try:
            return self._personalisation.global_stats()
        except Exception:
            logger.exception(
                "Pipeline.get_personalisation_global_stats failed"
            )
            return {
                "active_users": 0,
                "total_updates": 0,
                "max_adapters": 1000,
            }

    # ------------------------------------------------------------------
    # Gaze classifier management (vision fine-tuning flagship)
    # ------------------------------------------------------------------

    def _gaze_user_key(self, user_id: str) -> str:
        """Resolve the per-user storage key for the gaze classifier.

        Symmetric with the personalisation manager: when the user has
        a registered biometric template we use ``sha256(template)``;
        otherwise we fall back to ``sha256("gaze:" + user_id)`` so an
        unregistered user can still calibrate (and later associate
        their head with a registered template).
        """
        import hashlib

        try:
            tmpl = self._keystroke_auth._templates.get(user_id)
            if tmpl is not None and tmpl.template_embedding is not None:
                emb = tmpl.template_embedding
                arr = (
                    emb.detach().cpu().numpy().astype("float32")
                    if hasattr(emb, "detach")
                    else __import__("numpy").asarray(emb, dtype="float32")
                )
                # Quantise to 4dp so jitter doesn't change the key.
                quant = (arr * 10000).astype("int64").tobytes()
                return hashlib.sha256(quant).hexdigest()
        except Exception:  # pragma: no cover - defensive
            pass
        return hashlib.sha256(f"gaze:{user_id}".encode("utf-8")).hexdigest()

    def _gaze_ckpt_path(self, user_key: str):
        """Resolve the on-disk path for a per-user fine-tuned head."""
        from pathlib import Path

        return Path(self._gaze_ckpt_dir) / f"{user_key}.pt"

    def _evict_gaze_classifiers_if_over_cap(self) -> None:
        """LRU-evict cached gaze classifiers above the cap."""
        while len(self._gaze_classifiers) > self._max_gaze_classifiers:
            evicted_key, _ = self._gaze_classifiers.popitem(last=False)
            logger.info(
                "Pipeline evicted gaze classifier user_key=%s (LRU cap)",
                evicted_key[:16],
            )

    def get_gaze_classifier(self, user_id: str):
        """Return (constructing if needed) the user's fine-tuned classifier.

        Lazy: importing :class:`GazeClassifier` triggers the
        torchvision download on first call.  Subsequent calls reuse
        the singleton backbone, so per-user instantiation is cheap
        (~75k head params + a deque buffer).
        """
        from i3.multimodal.gaze_classifier import GazeClassifier

        user_key = self._gaze_user_key(user_id)
        if user_key in self._gaze_classifiers:
            self._gaze_classifiers.move_to_end(user_key)
            return self._gaze_classifiers[user_key]

        clf = GazeClassifier()
        # Hot-load any previously-saved fine-tuned head.
        ckpt = self._gaze_ckpt_path(user_key)
        if ckpt.exists():
            try:
                clf.load(ckpt)
                logger.info(
                    "Pipeline loaded gaze head from %s (user_key=%s)",
                    ckpt, user_key[:16],
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Failed to load gaze checkpoint %s; using fresh head.",
                    ckpt,
                )
        self._gaze_classifiers[user_key] = clf
        self._evict_gaze_classifiers_if_over_cap()
        return clf

    def calibrate_gaze_classifier(
        self,
        user_id: str,
        calibration_frames: dict,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> dict:
        """Fine-tune the user's gaze head and persist it to disk."""
        clf = self.get_gaze_classifier(user_id)
        metrics = clf.calibrate(
            calibration_frames, epochs=int(epochs), lr=float(lr),
        )
        user_key = self._gaze_user_key(user_id)
        ckpt = self._gaze_ckpt_path(user_key)
        try:
            clf.save(ckpt)
        except Exception:  # pragma: no cover - non-fatal
            logger.exception(
                "Failed to save gaze head to %s", ckpt,
            )
        out = dict(metrics)
        out["user_key"] = user_key
        out["success"] = True
        return out

    def get_gaze_status(self, user_id: str) -> dict:
        """Return status of the per-user gaze classifier.

        Always returns a valid dict; the ``calibrated`` field tells the
        caller whether the user has run the calibration flow.
        """
        try:
            clf = self.get_gaze_classifier(user_id)
            status = clf.status()
            user_key = self._gaze_user_key(user_id)
            status["user_id"] = user_id
            status["user_key"] = user_key
            status["checkpoint_exists"] = self._gaze_ckpt_path(user_key).exists()
            return status
        except Exception:
            logger.exception(
                "Pipeline.get_gaze_status failed for user_id=%s", user_id,
            )
            return {
                "user_id": user_id,
                "calibrated": False,
                "calibration_meta": {},
                "backbone": "unknown",
                "labels": [],
                "n_head_params": 0,
                "checkpoint_exists": False,
            }

    def infer_gaze(self, user_id: str, image) -> dict:
        """Run a single inference for *user_id* and return the dict."""
        clf = self.get_gaze_classifier(user_id)
        snap = clf.infer(image)
        return snap.to_dict()

    def cache_preference_offer(
        self,
        user_id: str,
        *,
        response_a_features: list[float],
        response_b_features: list[float],
        prompt: str | None = None,
    ) -> None:
        """Cache the picked / rejected adaptation profiles for the next pref-record call.

        Call from the A/B selector when offering a pair so that when
        the user picks one, the preference-record route knows which
        adaptation profile was rejected and can apply the contrastive
        update.  Bounded LRU at 1000 entries.
        """
        try:
            entry = {
                "response_a_features": list(response_a_features or []),
                "response_b_features": list(response_b_features or []),
                "prompt": str(prompt or "")[:200],
                "ts": time.time(),
            }
            self._preference_offer_cache[user_id] = entry
            self._preference_offer_cache.move_to_end(user_id)
            while len(self._preference_offer_cache) > 1000:
                self._preference_offer_cache.popitem(last=False)
        except Exception:
            logger.exception(
                "Pipeline.cache_preference_offer failed for user_id=%s",
                user_id,
            )

    def get_preference_offer(self, user_id: str) -> dict | None:
        """Return the most recent A/B offer cached for *user_id*."""
        return self._preference_offer_cache.get(user_id)

    def update_personalisation_from_preference(
        self,
        user_id: str,
        *,
        picked_adaptation: list[float],
        rejected_adaptation: list[float],
    ) -> dict | None:
        """Run a contrastive SGD step on the user's LoRA adapter.

        Called from ``POST /api/preference/record`` after the user has
        chosen between two responses with known adaptation profiles.
        Requires that the user's biometric template be registered;
        unregistered users silently receive ``None`` so the existing
        preference-record flow still succeeds.

        Args:
            user_id: User identifier.
            picked_adaptation: 8-d tensor list for the chosen response.
            rejected_adaptation: 8-d tensor list for the rejected response.

        Returns:
            The :class:`LoRAUpdate.to_dict()` for the WS layer / UI
            toast, or ``None`` when no biometric template exists.
        """
        try:
            tmpl = self._keystroke_auth._templates.get(user_id)
            if tmpl is None or tmpl.template_embedding is None:
                return None
            user_state = (
                self._last_user_state_embedding.get(user_id)
                if hasattr(self, "_last_user_state_embedding")
                else None
            )
            if user_state is None:
                # Fall back to a neutral state so the update still moves
                # the adapter — this is the right behaviour for the
                # demo where the user picks before sending another
                # message.
                user_state = torch.zeros(64, dtype=torch.float32)
            picked_t = torch.tensor(
                picked_adaptation[:8] + [0.0] * max(0, 8 - len(picked_adaptation)),
                dtype=torch.float32,
            )[:8]
            rejected_t = torch.tensor(
                rejected_adaptation[:8]
                + [0.0] * max(0, 8 - len(rejected_adaptation)),
                dtype=torch.float32,
            )[:8]
            update = self._personalisation.update(
                tmpl.template_embedding,
                user_state,
                picked_t,
                rejected_t,
            )
            return update.to_dict()
        except Exception:
            logger.exception(
                "Pipeline.update_personalisation_from_preference failed "
                "for user_id=%s",
                user_id,
            )
            return None

    def force_register_biometric(self, user_id: str) -> dict:
        """Force-register the biometric template from recent observations.

        Used by ``POST /api/biometric/{user_id}/force-register`` so the
        demo can stamp a template without typing 5 enrolment messages
        first.  Returns the post-register state.
        """
        try:
            result = self._keystroke_auth.force_register(
                user_id, complete=True
            ).to_dict()
        except Exception:
            logger.exception(
                "force_register_biometric failed for user_id=%s", user_id
            )
            result = self.get_biometric_status(user_id)
        self._last_biometric[user_id] = result
        self._last_biometric_state[user_id] = str(result.get("state", ""))
        return result

    # ------------------------------------------------------------------
    # Per-session cognitive-profile aggregator
    # ------------------------------------------------------------------

    def _profile_key(self, user_id: str, session_id: str) -> str:
        """Compose the OrderedDict key for the profile aggregator map."""
        return f"{user_id}::{session_id}"

    def _profile_update(
        self,
        *,
        user_id: str,
        session_id: str,
        input: PipelineInput,
        adaptation_dict: dict,
        user_state_label: dict | None,
        affect_shift: Any | None,
        biometric: dict | None,
        engagement_score: float,
    ) -> None:
        """Append one turn's worth of stats to the per-session aggregator.

        Bounded LRU at ``self._profile_max_sessions``.  Histories are
        capped at 50 entries so a long-running session doesn't grow
        the map without bound.
        """
        key = self._profile_key(user_id, session_id)
        agg = self._profile_aggregator.get(key)
        if agg is None:
            while len(self._profile_aggregator) >= self._profile_max_sessions:
                evicted_key, _ = self._profile_aggregator.popitem(last=False)
                logger.debug(
                    "profile_aggregator evicted oldest session: %s",
                    evicted_key,
                )
            agg = {
                "session_started_ts": time.time(),
                "messages": 0,
                "iki_mean_history": [],
                "iki_std_history": [],
                "composition_history": [],
                "edit_history": [],
                "cognitive_load_history": [],
                "formality_history": [],
                "verbosity_history": [],
                "accessibility_history": [],
                "state_distribution": {},
                "affect_shift_count": 0,
                "affect_shift_last_ts": None,
                "biometric_drift_count": 0,
            }
            self._profile_aggregator[key] = agg
        else:
            self._profile_aggregator.move_to_end(key)

        iki_mean, iki_std = self._iki_stats(input.keystroke_timings)

        # Bounded history buffers — keep the last 50 turns.
        for hist_key, value in (
            ("iki_mean_history", float(iki_mean)),
            ("iki_std_history", float(iki_std)),
            ("composition_history", float(input.composition_time_ms)),
            ("edit_history", int(input.edit_count)),
            (
                "cognitive_load_history",
                float(adaptation_dict.get("cognitive_load", 0.5)),
            ),
            ("formality_history", float(adaptation_dict.get("formality", 0.5))),
            ("verbosity_history", float(adaptation_dict.get("verbosity", 0.5))),
            (
                "accessibility_history",
                float(adaptation_dict.get("accessibility", 0.0)),
            ),
        ):
            buf = agg[hist_key]
            buf.append(value)
            if len(buf) > 50:
                del buf[: len(buf) - 50]

        # State distribution (counter).
        if isinstance(user_state_label, dict):
            state_name = str(user_state_label.get("state") or "unknown")
            agg["state_distribution"][state_name] = (
                agg["state_distribution"].get(state_name, 0) + 1
            )

        # Affect-shift events.
        if affect_shift is not None:
            try:
                detected = bool(getattr(affect_shift, "detected", False))
            except Exception:
                detected = False
            if detected:
                agg["affect_shift_count"] += 1
                agg["affect_shift_last_ts"] = time.time()

        # Biometric drift count.
        if isinstance(biometric, dict) and biometric.get("drift_alert"):
            agg["biometric_drift_count"] += 1

        agg["messages"] += 1
        agg["last_engagement"] = float(engagement_score)
        agg["last_biometric"] = biometric

    def get_profile_snapshot(
        self, user_id: str, session_id: str
    ) -> dict:
        """Return the aggregated cognitive-profile snapshot for a session.

        Shape matches the ``GET /api/profile/{user}/{session}`` contract.
        Returns an empty-ish skeleton when no data has been recorded yet.
        """
        key = self._profile_key(user_id, session_id)
        agg = self._profile_aggregator.get(key)
        if agg is None:
            return self._empty_profile_snapshot(user_id)
        # LRU touch.
        self._profile_aggregator.move_to_end(key)

        def _mean(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        def _stddev(values: list[float]) -> float:
            if len(values) < 2:
                return 0.0
            mu = sum(values) / len(values)
            var = sum((v - mu) ** 2 for v in values) / len(values)
            return float(math.sqrt(var))

        def _percent_below(values: list[float], threshold: float) -> float:
            if not values:
                return 0.0
            n_below = sum(1 for v in values if v < threshold)
            return float(n_below / len(values))

        # Cognitive-load histogram (5 bins).
        cl_values = list(agg.get("cognitive_load_history", []))
        bins = [0.0] * 5
        for v in cl_values:
            idx = max(0, min(4, int(v * 5)))
            bins[idx] += 1
        total_bins = sum(bins) or 1
        cl_hist = [b / total_bins for b in bins]

        # State distribution → fractions.
        sd = dict(agg.get("state_distribution", {}))
        sd_total = sum(sd.values()) or 1
        state_fractions = {k: v / sd_total for k, v in sd.items()}

        # IKI baseline = first half of history; recent = second half.
        iki_hist = list(agg.get("iki_mean_history", []))
        iki_mean_recent = _mean(iki_hist[-min(10, len(iki_hist)) :]) if iki_hist else 0.0
        iki_mean_baseline = (
            _mean(iki_hist[: max(1, len(iki_hist) // 2)])
            if len(iki_hist) >= 2
            else iki_mean_recent
        )
        if iki_mean_baseline > 0:
            iki_pct = (iki_mean_recent - iki_mean_baseline) / iki_mean_baseline * 100.0
        else:
            iki_pct = 0.0

        edits_history = list(agg.get("edit_history", []))
        edits_mean = _mean([float(e) for e in edits_history])
        edits_peak = float(max(edits_history)) if edits_history else 0.0

        comp_history = list(agg.get("composition_history", []))
        comp_mean = _mean(comp_history)
        fast_threshold = 1500.0
        fast_pct = _percent_below(comp_history, fast_threshold)

        biometric = (
            self._last_biometric.get(user_id)
            or self.get_biometric_status(user_id)
        )

        return {
            "biometric": biometric,
            "iki": {
                "mean": float(iki_mean_recent),
                "std": float(_stddev(iki_hist[-min(10, len(iki_hist)) :])),
                "vs_baseline_pct": float(iki_pct),
                "history": [round(x, 1) for x in iki_hist[-20:]],
            },
            "composition": {
                "mean_ms": float(comp_mean),
                "fast_turn_pct": float(fast_pct),
                "history": [round(x, 0) for x in comp_history[-20:]],
            },
            "edits": {
                "mean_per_turn": float(edits_mean),
                "peak_burst": float(edits_peak),
                "history": [int(e) for e in edits_history[-20:]],
            },
            "cognitive_load": {
                "histogram": cl_hist,
                "mean": float(_mean(cl_values)),
            },
            "style_preferences": {
                "formality_avg": float(_mean(agg.get("formality_history", []))),
                "verbosity_avg": float(_mean(agg.get("verbosity_history", []))),
                "accessibility_avg": float(
                    _mean(agg.get("accessibility_history", []))
                ),
            },
            "state_distribution": state_fractions,
            "affect_shifts": {
                "total": int(agg.get("affect_shift_count", 0)),
                "last_ts": agg.get("affect_shift_last_ts"),
            },
            "biometric_drifts": int(agg.get("biometric_drift_count", 0)),
            "session_messages": int(agg.get("messages", 0)),
            "session_duration_seconds": float(
                max(0.0, time.time() - float(agg.get("session_started_ts", time.time())))
            ),
        }

    def _empty_profile_snapshot(self, user_id: str) -> dict:
        """Return the skeleton snapshot for a session with no data yet."""
        return {
            "biometric": self.get_biometric_status(user_id),
            "iki": {"mean": 0.0, "std": 0.0, "vs_baseline_pct": 0.0, "history": []},
            "composition": {"mean_ms": 0.0, "fast_turn_pct": 0.0, "history": []},
            "edits": {"mean_per_turn": 0.0, "peak_burst": 0.0, "history": []},
            "cognitive_load": {"histogram": [0.0] * 5, "mean": 0.0},
            "style_preferences": {
                "formality_avg": 0.0,
                "verbosity_avg": 0.0,
                "accessibility_avg": 0.0,
            },
            "state_distribution": {},
            "affect_shifts": {"total": 0, "last_ts": None},
            "biometric_drifts": 0,
            "session_messages": 0,
            "session_duration_seconds": 0.0,
        }

    def force_accessibility_mode(
        self, user_id: str, session_id: str, *, force: bool | None
    ) -> dict:
        """Manual override for accessibility mode.

        Used by the ``POST /api/accessibility/{user_id}/toggle``
        endpoint.  ``force=True`` activates immediately, ``False``
        deactivates immediately, ``None`` clears the override and
        resumes auto-evaluation on the next observe().

        Returns:
            The post-toggle :class:`AccessibilityModeState` as a dict.
        """
        state = self._accessibility_controller.force(
            user_id=user_id, session_id=session_id, force=force
        )
        state_dict = state.to_dict()
        self._last_accessibility_state[
            f"{user_id}::{session_id}"
        ] = state_dict
        return state_dict

    # ------------------------------------------------------------------
    # Engagement computation (called between messages)
    # ------------------------------------------------------------------

    def compute_engagement(
        self,
        user_id: str,
        continued: bool,
        response_latency_ms: float,
        user_msg_word_count: int,
        topic_continuity: float = 0.5,
        sentiment_shift: float = 0.0,
    ) -> float:
        """Compute and record the engagement signal from the user's
        response to the AI's previous message.

        This should be called before :meth:`process_message` so that the
        bandit router receives the reward signal from the *previous*
        turn.

        Args:
            user_id: The user whose engagement to compute.
            continued: Whether the user sent a follow-up message.
            response_latency_ms: Time from AI response to user's next
                keystroke (ms).
            user_msg_word_count: Word count of the user's new message.
            topic_continuity: Estimated topic overlap [0, 1].
            sentiment_shift: Sentiment delta [-1, 1].

        Returns:
            The composite engagement score in [0, 1].
        """
        ai_word_count = self._last_response_length.get(user_id, 1)

        signal = self.engagement_estimator.compute(
            continued=continued,
            response_latency_ms=response_latency_ms,
            user_msg_length=user_msg_word_count,
            ai_msg_length=ai_word_count,
            topic_continuity=topic_continuity,
            sentiment_shift=sentiment_shift,
        )

        score = signal.score
        self._previous_engagement[user_id] = score

        # Update the router bandit with the reward from the previous turn.
        # SEC: the contextual Thompson bandit's Laplace posterior is fitted
        # per-arm on the *(context, reward)* pairs accumulated in its
        # history.  Feeding a zero context at every update collapses the
        # context-reward mapping to a degenerate non-contextual bandit.
        # We therefore replay the exact context vector that was handed to
        # ``select_arm`` at the time of the routing decision.
        prev_route = self._previous_route.get(user_id)
        prev_ctx = self._previous_routing_context.get(user_id)
        if prev_route is not None and prev_ctx is not None:
            try:
                self.router.update(prev_route, prev_ctx, reward=score)
            except Exception:
                logger.exception(
                    "Failed to update router bandit for user_id=%s", user_id
                )

        return score

    # ------------------------------------------------------------------
    # Internal: routing
    # ------------------------------------------------------------------

    def _make_routing_decision(
        self,
        user_id: str,
        user_state_embedding: torch.Tensor,
        features: Any,
        query_complexity: float,
        topic_sensitivity: float,
        user_model: Any,
        *,
        message_text: str = "",
        retrieval_top_score: float | None = None,
        session_id: str = "",
    ) -> tuple[str, dict[str, float], np.ndarray]:
        """Run the contextual bandit to choose a generation route.

        Returns:
            A tuple ``(route_name, confidence_dict, context_vector)``
            where ``route_name`` is one of ``"local_slm"`` or
            ``"cloud_llm"``, ``confidence_dict`` maps arm names to
            selection probabilities, and ``context_vector`` is the
            exact numpy vector that was handed to ``select_arm`` — the
            caller must replay it verbatim when later calling
            ``bandit.update()`` so the Laplace posterior sees the same
            context at update time as at selection time.
        """
        from i3.router.types import RoutingContext

        compressed = self._compress_state(user_state_embedding)
        session = user_model.current_session

        routing_context = RoutingContext(
            user_state_compressed=compressed,
            query_complexity=query_complexity,
            topic_sensitivity=topic_sensitivity,
            user_patience=max(0.0, 1.0 - getattr(features, "latency_trend", 0.0)),
            session_progress=getattr(features, "session_progress", 0.0),
            baseline_established=user_model.baseline_established,
            previous_route=self._previous_route.get(user_id, -1),
            previous_engagement=self._previous_engagement.get(user_id, 0.5),
            time_of_day=self._normalized_hour(),
            message_count=session.message_count if session else 0,
            cloud_latency_est=0.5,
            slm_confidence=0.5,
        )
        context_vector = routing_context.to_vector()

        # Compute the per-prompt complexity score (CPU-only, sub-ms).
        # Used as a feature for the bandit and as the dominant signal
        # surfaced on the routing-decision tooltip.
        try:
            prompt_estimate = self._prompt_complexity_estimator.estimate(
                message_text or "",
                retrieval_top_score=retrieval_top_score,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "PromptComplexityEstimator.estimate failed; falling back "
                "to query_complexity scalar."
            )
            from i3.router.complexity_estimator import ComplexityEstimate
            prompt_estimate = ComplexityEstimate(
                score=float(query_complexity),
                factors={
                    "length_factor": 0.0,
                    "rare_token_factor": 0.0,
                    "open_ended_factor": 0.0,
                    "multi_clause_factor": 0.0,
                    "retrieval_miss_factor": 0.0,
                },
                notes="estimator unavailable",
            )

        # Privacy override: force local if topic is sensitive
        privacy_override = (
            self.config.router.privacy_override
            and topic_sensitivity > 0.5
        )

        arms = self.config.router.arms
        confidence: dict[str, float]
        decision_reason: str

        if privacy_override:
            route_chosen = "local_slm"
            confidence = dict.fromkeys(arms, 0.0)
            confidence["local_slm"] = 1.0
            decision_reason = (
                f"privacy override (sensitivity {topic_sensitivity:.2f}"
                f" > 0.5) → edge SLM"
            )
            logger.info(
                "Privacy override triggered (sensitivity=%.2f) -> local_slm",
                topic_sensitivity,
            )
        else:
            arm_index, raw_confidence = self.router.select_arm(context_vector)
            route_chosen = arms[arm_index] if arm_index < len(arms) else "local_slm"

            # Translate arm indices to arm names
            confidence = {}
            for key, value in raw_confidence.items():
                # raw_confidence keys are "arm_0", "arm_1", etc.
                idx = int(key.split("_")[1]) if "_" in key else 0
                name = arms[idx] if idx < len(arms) else f"unknown_{idx}"
                confidence[name] = value

            # Cloud-vs-edge gating ladder.  The cloud arm fires ONLY
            # when:
            #   1. the bandit chose it,
            #   2. the user has explicitly opted in (consent on),
            #   3. the per-session budget allows another call,
            #   4. ``CloudLLMClient.is_available`` says the API key is
            #      configured and the route is not env-disabled.
            # Any failure rewrites the decision to ``local_slm`` and
            # records a clear reason so the UI can surface it.
            if route_chosen == "cloud_llm":
                edge_p = float(confidence.get("local_slm", 0.0))
                cloud_p = float(confidence.get("cloud_llm", 0.0))
                consent_ok = self.privacy_budget.consent(user_id)
                budget_ok, budget_reason = self.privacy_budget.can_call(
                    user_id, session_id
                )
                cloud_avail = bool(
                    getattr(self.cloud_client, "is_available", False)
                )
                if not consent_ok:
                    route_chosen = "local_slm"
                    decision_reason = (
                        f"complexity {prompt_estimate.score:.2f} → "
                        f"bandit picked cloud, but consent off → edge SLM"
                    )
                elif not budget_ok:
                    route_chosen = "local_slm"
                    decision_reason = (
                        f"complexity {prompt_estimate.score:.2f} → "
                        f"bandit picked cloud, but {budget_reason} → "
                        f"edge SLM"
                    )
                elif not cloud_avail:
                    route_chosen = "local_slm"
                    decision_reason = (
                        f"complexity {prompt_estimate.score:.2f} → "
                        f"bandit picked cloud, but cloud client "
                        f"unavailable → edge SLM"
                    )
                else:
                    decision_reason = (
                        f"complexity {prompt_estimate.score:.2f}"
                        f" (cloud {cloud_p:.2f} / edge {edge_p:.2f}) → "
                        f"cloud LLM"
                    )
            else:
                edge_p = float(confidence.get("local_slm", 0.0))
                cloud_p = float(confidence.get("cloud_llm", 0.0))
                decision_reason = (
                    f"complexity {prompt_estimate.score:.2f}"
                    f" (edge {edge_p:.2f} / cloud {cloud_p:.2f}) → "
                    f"edge SLM"
                )

        # Build the structured routing-decision dict for the UI.
        arm_label = "cloud_llm" if route_chosen == "cloud_llm" else "edge_slm"
        decision_dict = {
            "arm": arm_label,
            "route": route_chosen,
            "confidence": float(
                confidence.get(route_chosen, 0.0)
            ),
            "reason": decision_reason,
            "feature_vector": [float(x) for x in context_vector.tolist()],
            "complexity": {
                "score": float(prompt_estimate.score),
                "factors": dict(prompt_estimate.factors),
                "notes": prompt_estimate.notes,
            },
            "retrieval_top_score": (
                float(retrieval_top_score)
                if retrieval_top_score is not None
                else None
            ),
            "consent_required": True,
            "consent_enabled": self.privacy_budget.consent(user_id),
            "timestamp": time.time(),
        }
        self._last_routing_decision = decision_dict
        self._recent_routing_decisions.append(decision_dict)

        return route_chosen, confidence, context_vector

    # ------------------------------------------------------------------
    # Internal: response generation
    # ------------------------------------------------------------------

    async def _generate_response(
        self,
        route: str,
        message: str,
        adaptation: Any,
        user_state: torch.Tensor,
        on_token: Any = None,
        *,
        user_id: str = "",
        session_id: str = "",
        query_for_retrieval: str | None = None,
        topic_prefix_for_embedding: str | None = None,
    ) -> str:
        """Generate a response and apply adaptation-aware rewriting.

        Picks a base response via :meth:`_generate_response_inner` and
        then runs it through
        :meth:`ResponsePostProcessor.adapt_with_log` so the live
        :class:`AdaptationVector` visibly reshapes the text — trimming
        sentences under high cognitive load, expanding contractions
        when formality is high, etc.  The change log is stashed on
        ``self._last_adaptation_changes`` so the websocket layer can
        ship it to the UI as chips.

        ``user_id`` / ``session_id`` are forwarded so the inner method
        can fetch the per-session conversation-history buffer and feed
        prior turns into the retriever's embedding query and the SLM's
        prompt. They default to empty strings, in which case the
        history-aware code paths short-circuit and the call is
        equivalent to the pre-history behaviour.

        ``query_for_retrieval`` (optional) is the co-reference-resolved
        form of the user's message.  When provided it is passed to
        :meth:`ResponseRetriever.best` instead of the raw ``message``
        so the entity-knowledge tool route + the keyword-overlap gate
        both see the rewritten query.  ``None`` (the default)
        preserves single-turn behaviour exactly.
        """
        raw = await self._generate_response_inner(
            route=route,
            message=message,
            adaptation=adaptation,
            user_state=user_state,
            on_token=on_token,
            user_id=user_id,
            session_id=session_id,
            query_for_retrieval=query_for_retrieval,
            topic_prefix_for_embedding=topic_prefix_for_embedding,
        )
        try:
            adapted, change_log = self.postprocessor.adapt_with_log(
                raw, adaptation
            )
        except Exception:
            logger.exception("adapt_with_log failed; using raw response.")
            adapted, change_log = raw, []
        self._last_adaptation_changes = change_log
        return adapted

    async def _generate_response_inner(
        self,
        route: str,
        message: str,
        adaptation: Any,
        user_state: torch.Tensor,
        on_token: Any = None,
        *,
        user_id: str = "",
        session_id: str = "",
        query_for_retrieval: str | None = None,
        topic_prefix_for_embedding: str | None = None,
    ) -> str:
        """Generate the base response via the chosen route.

        Falls back gracefully: cloud -> SLM -> rule-based fallback.

        Args:
            route: ``"cloud_llm"`` or ``"local_slm"``.
            message: Sanitised user message text.
            adaptation: :class:`~src.adaptation.types.AdaptationVector`.
            user_state: 64-dim user-state embedding.
            user_id: Optional user identifier; combined with
                ``session_id`` to look up the per-session conversation
                history buffer. When empty the history-aware code paths
                short-circuit and the call behaves exactly as before.
            session_id: Optional session identifier (see ``user_id``).

        Returns:
            The generated response text.
        """
        # --- Multi-turn history --------------------------------------------
        # Pull the last N-1 ``(user, assistant)`` pairs for this session
        # so the retriever embedding query and the SLM prompt both get
        # contextualised.  We deliberately read N-1 rather than N: the
        # *current* user message is the Nth turn, and the buffer holds
        # only completed exchanges, so the most recent N-1 pairs are
        # the right window.  An empty buffer (first turn of a session
        # or no session_id available) is the no-op case.
        history_pairs = self._get_history_pairs(user_id, session_id)
        # Use up to (max_turns - 1) prior pairs so the contextualised
        # prompt fits inside the same budget the trainer used.
        max_prior = max(0, self._history_max_turns - 1)
        history_pairs = history_pairs[-max_prior:] if max_prior else []
        history_text = self._format_history(history_pairs)
        # Track for the WS reasoning trace ("working from N prior turns").
        if user_id and session_id:
            self._last_history_turns_used[
                self._history_key(user_id, session_id)
            ] = len(history_pairs)

        # --- Cloud route --------------------------------------------------
        # The cloud branch fires only when ``_make_routing_decision``
        # already cleared the consent + budget gates and selected the
        # ``cloud_llm`` arm.  We still re-check ``is_available`` here
        # (defensive: a 4xx in flight could have circuit-broken the
        # client between the routing decision and this call).
        if route == "cloud_llm" and self.cloud_client.is_available:
            try:
                # 1. PII-sanitise the message + the recent history.
                #    The pipeline's caller already passes the
                #    sanitiser's output as ``message`` for the SLM
                #    path — but the cloud path is the *one* place
                #    where bytes actually leave the host, so we
                #    re-sanitise defensively and capture per-category
                #    counts for the privacy budget.
                msg_san = self.sanitizer.sanitize(message)
                # Sanitise every history pair too.  We preserve role
                # ordering but never persist the result.
                sanitised_history: list[dict[str, str]] = []
                hist_redactions = 0
                hist_bytes_redacted = 0
                hist_categories: dict[str, int] = {}
                for u_text, a_text in history_pairs:
                    u_san = self.sanitizer.sanitize(u_text or "")
                    a_san = self.sanitizer.sanitize(a_text or "")
                    sanitised_history.append({
                        "role": "user",
                        "content": u_san.sanitized_text,
                    })
                    sanitised_history.append({
                        "role": "assistant",
                        "content": a_san.sanitized_text,
                    })
                    hist_redactions += (
                        u_san.replacements_made + a_san.replacements_made
                    )
                    hist_bytes_redacted += (
                        getattr(u_san, "bytes_redacted", 0)
                        + getattr(a_san, "bytes_redacted", 0)
                    )
                    for cat, n in (
                        getattr(u_san, "pii_category_counts", {}) or {}
                    ).items():
                        hist_categories[cat] = hist_categories.get(cat, 0) + int(n)
                    for cat, n in (
                        getattr(a_san, "pii_category_counts", {}) or {}
                    ).items():
                        hist_categories[cat] = hist_categories.get(cat, 0) + int(n)
                total_redactions = (
                    msg_san.replacements_made + hist_redactions
                )
                total_bytes_redacted = (
                    getattr(msg_san, "bytes_redacted", 0)
                    + hist_bytes_redacted
                )
                merged_categories: dict[str, int] = dict(hist_categories)
                for cat, n in (
                    getattr(msg_san, "pii_category_counts", {}) or {}
                ).items():
                    merged_categories[cat] = (
                        merged_categories.get(cat, 0) + int(n)
                    )
                # 2. Build the adaptation-conditioned system prompt.
                #    Same builder the cloud client expects, fed the
                #    *live* AdaptationVector so the cloud reply
                #    respects cognitive_load / formality / etc.
                user_summary = self._build_user_summary_for_cloud(
                    user_id=user_id,
                )
                system_prompt = self.prompt_builder.build_system_prompt(
                    adaptation,
                    user_summary=user_summary,
                )
                # 3. Hard timeout 8 s wrapping the cloud call.  The
                #    client itself has retry + backoff with its own
                #    bounded budget; this outer ``wait_for`` is the
                #    final safety net so a stuck upstream cannot wedge
                #    the response path.
                result = await asyncio.wait_for(
                    self.cloud_client.generate(
                        msg_san.sanitized_text,
                        system_prompt,
                        sanitised_history,
                    ),
                    timeout=8.0,
                )
                response = result.get("text", "")
                # 4. Bill the call against the privacy budget.  The
                #    counter tracks both transmitted bytes (after
                #    sanitisation) and bytes-redacted (the visible
                #    "value-add of the sanitiser").
                try:
                    snapshot = self.privacy_budget.record_call(
                        user_id=user_id,
                        session_id=session_id,
                        sanitised_prompt=msg_san.sanitized_text,
                        response_text=response,
                        pii_redactions=total_redactions,
                        pii_categories=merged_categories,
                        bytes_redacted=total_bytes_redacted,
                    )
                    self._last_privacy_budget_snapshot = snapshot.to_dict()
                    # Also stash per-call counters on the routing
                    # decision so the UI's chip can show "pii redacted
                    # · 3" without a separate REST round-trip.
                    if isinstance(self._last_routing_decision, dict):
                        self._last_routing_decision = dict(
                            self._last_routing_decision
                        )
                        self._last_routing_decision["pii_redactions"] = (
                            int(total_redactions)
                        )
                        self._last_routing_decision["bytes_redacted"] = (
                            int(total_bytes_redacted)
                        )
                        self._last_routing_decision["bytes_in"] = (
                            len((msg_san.sanitized_text or "")
                                .encode("utf-8"))
                        )
                        self._last_routing_decision["bytes_out"] = (
                            len((response or "").encode("utf-8"))
                        )
                        self._last_routing_decision["latency_ms"] = (
                            float(result.get("latency_ms", 0.0))
                        )
                        # Replace the ring-buffer tail with the
                        # enriched decision (we appended the basic
                        # one earlier).
                        if self._recent_routing_decisions:
                            self._recent_routing_decisions[-1] = (
                                self._last_routing_decision
                            )
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "PrivacyBudget.record_call failed (counter "
                        "skipped, response unaffected)."
                    )
                if response:
                    self._last_response_path = "cloud_llm"
                    self._last_retrieval_score = 0.0
                    # 5. Post-process via the same adapt_with_log that
                    #    the SLM path uses — the outer
                    #    ``_generate_response`` will call it again, so
                    #    we just return the raw response here.  The
                    #    adapt_with_log applied by the outer wrapper
                    #    fires on every route → adaptation enforcement
                    #    is route-agnostic.
                    return response
                logger.warning("Cloud LLM returned empty response; falling back.")
            except asyncio.TimeoutError:
                logger.warning(
                    "Cloud LLM call exceeded 8s timeout; falling back to SLM."
                )
            except Exception:
                logger.exception(
                    "Cloud LLM generation failed; falling back to SLM."
                )

        # --- Local SLM route: hybrid retrieval + generation ----------------
        # Edge-deployed assistants pair a small transformer with a
        # retrieval index over the training corpus.  The flow is:
        #
        #   1. Exact-match fast path (normalised text key).
        #   2. Embedding-NN retrieval with keyword-overlap veto; if the
        #      score passes the confidence threshold we return the
        #      matched training response directly.
        #   3. Autoregressive generation from the custom transformer,
        #      prompted in the same "history [SEP] response" format
        #      seen at training time.  Used for novel inputs that the
        #      retriever isn't confident on.
        #   4. Borderline-retrieval fall-back for generation that fails
        #      the coherence filter.
        #   5. Honest out-of-distribution disclaimer for everything
        #      else.  This is what real on-device assistants do — a
        #      strict 4 M-param autoregressive generator cannot reliably
        #      carry the full response load by itself.
        retriever = self._slm_retriever
        # Bookkeeping for the UI's pipeline-activity ribbon: record
        # which sub-path of the hybrid stack actually answered.  The
        # caller pulls this out and attaches it to the WebSocket frame.
        self._last_response_path = "none"
        self._last_retrieval_score = 0.0

        # ---- Session recap tool (Iter 27, 2026-04-26) -------------------
        # When the user asks for a recap of the conversation
        # ("what have we discussed", "summarize our conversation",
        # "what topics did we cover"), build a fresh summary from the
        # entity tracker's user-anchored topics.  This is real
        # conversation memory at work — recruiter sees the system
        # "remembering" a multi-turn thread.  Falls through if no
        # anchored topics yet.
        try:
            recap = self._maybe_session_recap(
                message=message,
                user_id=user_id,
                session_id=session_id,
            )
            if recap:
                self._last_response_path = "tool:recap"
                self._last_retrieval_score = 0.99
                return recap
        except Exception:  # pragma: no cover — never block on the helper
            logger.debug(
                "Session-recap helper failed for %r", message[:40],
                exc_info=True,
            )

        # Iter 48: lightweight name-statement / name-recall handler.
        # When the user introduces themselves ("my name is Alex" /
        # "call me Sam" / "I'm Alex" / "I am Alex" / "name's Alex"),
        # store the name in a per-(user, session) dict.  When they
        # later ask "what's my name", recall it.  This gives the
        # system real session-level personal-fact memory (one slot,
        # narrowly scoped — full fact-tracking is iter 49+).
        try:
            name_response = self._maybe_handle_name_statement(
                message=message,
                user_id=user_id,
                session_id=session_id,
            )
            if name_response:
                self._last_response_path = "tool:name"
                self._last_retrieval_score = 0.99
                return name_response
        except Exception:  # pragma: no cover — never block on the helper
            logger.debug(
                "Name handler failed for %r", message[:40], exc_info=True,
            )

        # Iter 49: multi-fact session memory.  Detect declarative
        # statements ("my favourite color is blue", "I work as a
        # nurse", "I live in Berlin", "my hobby is climbing") and
        # recall queries ("what's my favourite color", "where do I
        # live", "what's my hobby").  Stored per-(user, session)
        # in ``self._stated_facts``.
        try:
            fact_response = self._maybe_handle_fact_statement(
                message=message,
                user_id=user_id,
                session_id=session_id,
            )
            if fact_response:
                self._last_response_path = "tool:fact"
                self._last_retrieval_score = 0.99
                return fact_response
        except Exception:
            logger.debug(
                "Fact handler failed for %r", message[:40], exc_info=True,
            )

        # ---- Iter 51: Qwen LoRA HMI command-intent (cascade arm) -------
        # Detect command-shaped utterances ("set timer 5 min", "play
        # jazz", "navigate home") and route them through the Qwen3-1.7B
        # + LoRA + DoRA + NEFTune intent parser (the fine-tune-of-pre-
        # trained leg of the JD's "build models from scratch as well as
        # adapt or fine-tune pre-trained" bullet).  When the parser
        # returns a high-confidence valid action, we short-circuit the
        # SLM and emit a deterministic structured response *plus* the
        # full IntentResult on PipelineOutput so the chat UI renders a
        # green ◆ chip.  Falls through to chat for anything that
        # doesn't look like a command.
        try:
            intent_response = self._maybe_handle_intent_command(
                message=message, user_id=user_id, session_id=session_id,
            )
            if intent_response is not None:
                self._last_response_path = "tool:intent"
                self._last_retrieval_score = 0.99
                return intent_response
        except Exception:  # pragma: no cover — never block on this helper
            logger.debug(
                "Intent handler failed for %r", message[:40], exc_info=True,
            )

        # ---- Bare clarification templates (Phase 14, 2026-04-25) --------
        # When the user types an entity-less question shape (``who
        # founded?``, ``when did he live?``, ``what did she invent?``)
        # AND there is NO entity in the recency stack, we'd rather emit
        # a curated clarification than fall through to OOD.  These are
        # hand-written, deterministic, no SLM calls.  When an entity IS
        # in scope the bare-noun rewriter upstream has already turned
        # ``"who founded?"`` into ``"who founded huawei?"`` so this
        # branch is a no-op for that case.
        try:
            bare_clarification = self._maybe_bare_clarification(
                message=message,
                user_id=user_id,
                session_id=session_id,
                query_for_retrieval=query_for_retrieval,
            )
            if bare_clarification:
                self._last_response_path = "tool:clarify"
                self._last_retrieval_score = 0.99
                return bare_clarification
        except Exception:  # pragma: no cover — never block on the helper
            logger.debug(
                "Bare-clarification helper failed for %r", message[:40],
                exc_info=True,
            )

        # ---- Multi-step explain decomposition (Phase B.3, 2026-04-25) ----
        # Detect "explain X" / "tell me about X" / "describe X" / "how
        # does X work?" patterns and run the structured decomposer.  The
        # decomposer answers each sub-question through the same
        # tool / retrieval stack we run normally, then composes the
        # final 2-3 paragraph response.  Stashed on
        # ``self._last_explain_plan`` so the WS layer can ship it.
        try:
            if not hasattr(self, "_explain_decomposer"):
                from i3.pipeline.explain_decomposer import ExplainDecomposer
                from i3.dialogue.knowledge_graph import get_global_kg
                self._explain_decomposer = ExplainDecomposer(
                    retriever=retriever, kg=get_global_kg(),
                )
            decomposer = self._explain_decomposer
            # Iteration 16 (2026-04-26): use the bare-noun-rewritten
            # query when the rewriter fired (query_for_retrieval set).
            # Without this, "explain in plain english" got decomposed
            # against its raw form (topic="english") even when the
            # upstream rewriter had already produced "explain
            # transformer simply".
            _decomp_input = (
                query_for_retrieval if query_for_retrieval else message
            )
            if decomposer.is_explain_query(_decomp_input):
                plan = decomposer.decompose_and_answer(_decomp_input)
                if plan.composite_answer and len(plan.composite_answer) > 40:
                    self._last_explain_plan = plan.to_dict()
                    self._last_response_path = "explain_decomposed"
                    # Use the composite_answer's first sub-answer
                    # confidence as the retrieval-score chip if there is
                    # one — the UI then renders ``conf 0.93 · explain
                    # decomposed``.
                    if plan.sub_answers:
                        self._last_retrieval_score = max(
                            (a.confidence for a in plan.sub_answers), default=0.0,
                        )
                    return plan.composite_answer
                # Decomposition produced nothing useful — fall through
                # to the regular path.  Don't stash a half-empty plan.
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "Explain decomposer failed for %r; falling through.",
                message[:40] if message else "",
                exc_info=True,
            )
        if retriever is not None:
            try:
                adapt_dict = self._adaptation_to_dict(adaptation)
                # Tightened threshold (was min_score=0.30): the post-
                # rebuild corpus is large + diverse, so cosine matches
                # in the 0.30-0.75 band were almost always Cornell
                # movie subtitles or Wikipedia fragments, not actual
                # answers.  We now require >=0.85 cosine to commit to
                # retrieval and >=0.65 to even keep the candidate as a
                # borderline-fallback option.
                # History-aware retrieval: the keyword/exact-match
                # paths still see the raw ``message`` (history words
                # would pollute the keyword overlap and detune the
                # 0.85+ confidence chip), while the embedding-NN
                # cosine path uses a contextualised query so prompts
                # like "what about animals?" can land on the right
                # topic from the established context.
                # Pick the query string the retriever actually keys
                # on.  When the co-reference resolver fired upstream
                # (e.g. "where are they located?" → "where is huawei
                # located?") we hand the rewritten string in so the
                # entity-knowledge tool route + keyword-overlap gate
                # see the correct entity.  When no resolution
                # happened, ``query_for_retrieval`` is ``None`` and we
                # fall back to the raw user message — preserving
                # single-turn behaviour exactly.
                resolved_message = query_for_retrieval or message
                coref_fired = bool(
                    query_for_retrieval and query_for_retrieval != message
                )
                # Treat a topic-carryover prefix as a "context fired"
                # signal too, so the short-query gate further down
                # doesn't reject a perfectly good "apple most famous
                # product" → Apple-iPhone match just because the raw
                # user text was three tokens.
                context_fired = coref_fired or bool(topic_prefix_for_embedding)
                # Topic-carryover prefix (Fix 3, 2026-04-25): when the
                # user typed a short follow-up with no resolvable
                # pronoun (e.g. "most famous product" right after
                # "tell me about apple") and we have a recent topic
                # entity on the recency stack, prepend it to the
                # *embedding* query only.  Keyword overlap and exact-
                # match still see the raw single-turn text so the
                # carryover can't pollute the 0.85+ confidence chip.
                if topic_prefix_for_embedding:
                    embed_message = (
                        f"{topic_prefix_for_embedding} {resolved_message}"
                    )
                else:
                    embed_message = resolved_message
                # Self-contained query gate (2026-04-26 user-emulation
                # audit): when the current message already carries 3+
                # of its own substantive content keywords AND there's
                # no pronoun coref to resolve, the conversation history
                # almost always pollutes rather than helps the embedding
                # cosine — e.g. asking "what happens if I close this
                # tab" right after a transformer thread would otherwise
                # land on a transformer paragraph.  Skip the history
                # prepend in that case.
                _is_self_contained = False
                try:
                    from i3.slm.retrieval import _keywords as _kw_fn
                    _msg_kw = _kw_fn(resolved_message or "")
                    if (
                        len(_msg_kw) >= 3
                        and not topic_prefix_for_embedding
                        and not coref_fired
                    ):
                        _is_self_contained = True
                except Exception:
                    _is_self_contained = False
                contextualised_query = (
                    f"{history_text}\n[SEP]\n{embed_message}"
                    if (history_text and not _is_self_contained)
                    else (
                        embed_message
                        if topic_prefix_for_embedding
                        else None
                    )
                )
                # Comparison-tool fallback pair: when the user types a
                # short comparison shape with no entities of its own
                # (``"which one is bigger?"``), pick the top two
                # *distinct* org/topic/person entities from the
                # EntityTracker recency stack so the comparison tool
                # has something to compare.  We trust the stack to be
                # already-bounded by the EntityTracker's per-session
                # cap; the recency check is implicit because the stack
                # is ordered most-recent-first.
                compare_fallback_pair: tuple[str, str] | None = None
                try:
                    snap = self._entity_tracker.snapshot(
                        user_id, session_id,
                    )
                    # Iter 35 (2026-04-26): pick the top 2 entities
                    # most likely to be what the user means by "them".
                    # Priority tiers:
                    #   T1: user-anchored ORG (the user explicitly
                    #       named a company)
                    #   T2: user-anchored ORG or TOPIC
                    #   T3: any ORG (in stack order)
                    #   T4: any ORG/TOPIC
                    #   T5: any ORG/TOPIC/PERSON
                    # Each tier preserves stack recency.  Without T1,
                    # incidentally-mentioned ORG entities (Azure,
                    # GitHub) extracted from a Microsoft answer would
                    # outrank user-named topics like Apple.
                    def _pick_two(filter_kinds: set[str], anchored_only: bool) -> list[str]:
                        out: list[str] = []
                        seen2: set[str] = set()
                        for f in snap:
                            if f.kind not in filter_kinds:
                                continue
                            if anchored_only and f.user_anchor_turn is None:
                                continue
                            if f.canonical in seen2:
                                continue
                            seen2.add(f.canonical)
                            out.append(f.canonical)
                            if len(out) >= 2:
                                break
                        return out
                    eligible = _pick_two({"org"}, anchored_only=True)
                    if len(eligible) < 2:
                        eligible = _pick_two({"org", "topic"}, anchored_only=True)
                    if len(eligible) < 2:
                        eligible = _pick_two({"org"}, anchored_only=False)
                    if len(eligible) < 2:
                        eligible = _pick_two({"org", "topic"}, anchored_only=False)
                    if len(eligible) < 2:
                        eligible = _pick_two({"org", "topic", "person"}, anchored_only=False)
                    if len(eligible) >= 2:
                        compare_fallback_pair = (eligible[0], eligible[1])
                except Exception:
                    compare_fallback_pair = None
                match = retriever.best(
                    resolved_message,
                    adaptation=adapt_dict,
                    min_score=0.65,
                    query_for_embedding=contextualised_query,
                    compare_fallback_pair=compare_fallback_pair,
                )
                if match is not None:
                    response_text, score = match
                    self._last_retrieval_score = float(score)
                    # Tool routes (math solver, hostility refusal,
                    # entity facts) get tagged distinctly so the UI
                    # chip reads ``tool: math`` / ``tool: entity``
                    # instead of ``retrieval``.
                    tool_name = getattr(retriever, "_last_tool", None)
                    if tool_name:
                        self._last_response_path = f"tool:{tool_name}"
                        return response_text
                    # ---- Routing tightening (Phase B.1, 2026-04-25) ----
                    # The 2026-04-25 heavy audit found the v2 SLM was
                    # stealing turns where the retrieval candidate was
                    # actually correct (e.g. "tell me about apple" →
                    # SLM word-salad while a 0.85+ cosine match to a
                    # curated apple paragraph sat right there).  The
                    # rules below give retrieval **first refusal** on
                    # any match above the configurable floor, while
                    # keeping the original short-query / sub-0.92 gate
                    # for genuinely ambiguous cases.
                    #
                    # Defaults:
                    #     retrieval_floor = 0.75
                    #
                    # Override-able at runtime via the playground tab —
                    # see :meth:`set_retrieval_floor`.
                    raw_word_count = len((message or "").split())
                    try:
                        from i3.slm.retrieval import _keywords as _kw_fn
                        content_word_count = len(_kw_fn(message or ""))
                    except Exception:  # pragma: no cover - defensive
                        content_word_count = raw_word_count
                    floor = float(getattr(self, "_retrieval_floor", 0.75))

                    # Short-query gate — Fix 3a.  A short user message
                    # (≤4 words) with no co-reference resolution and a
                    # sub-0.92 cosine is almost always a false-confident
                    # retrieval.  Reject and fall through to clarifier /
                    # OOD instead of returning a confident-looking
                    # nonsense match.
                    if (
                        raw_word_count <= 4
                        and not context_fired
                        and score < 0.92
                    ):
                        logger.debug(
                            "Short-query gate rejected retrieval (words=%d, "
                            "score=%.2f, no coref) — falling through.",
                            raw_word_count,
                            score,
                        )
                        self._last_retrieval_candidate = None
                    elif score >= 0.92:
                        # Very high cosine + keyword overlap (the
                        # ``best()`` veto already enforced overlap) →
                        # ALWAYS commit to retrieval, never SLM-steal.
                        self._last_response_path = "retrieval"
                        return response_text
                    elif (
                        score >= floor
                        and content_word_count >= 4
                    ):
                        # Mid-confidence retrieval on a substantive
                        # query (4+ content words means the keyword-
                        # overlap inside ``best()`` is meaningful).
                        # Prefer the retrieved curated text over the
                        # SLM's freshly-generated draft — that's the
                        # exact failure mode the audit caught.
                        self._last_response_path = "retrieval"
                        return response_text
                    elif score >= 0.85:
                        # Retain the original 0.85 commit threshold for
                        # short-query matches that already passed the
                        # short-query gate above.
                        self._last_response_path = "retrieval"
                        return response_text
                    else:
                        # Borderline band (0.65–0.85, including the
                        # 0.65–0.75 sub-band that the new substantive-
                        # query commit floor *could* have caught but
                        # didn't because content_word_count < 4).  Run
                        # the Phase B.6 self-consistency check: when ≥2
                        # of the top-3 candidates agree (token overlap
                        # > 0.4) we promote the borderline match to a
                        # confident retrieval, surfacing the consensus
                        # via ``self._last_consistency`` for the chip.
                        self._last_retrieval_candidate = response_text
                        try:
                            consistency = retriever.consistency_check(
                                resolved_message,
                                adaptation=adapt_dict,
                                query_for_embedding=contextualised_query,
                            )
                            self._last_consistency = consistency
                            if (
                                consistency.get("consistent")
                                and consistency.get("winning_response")
                            ):
                                self._last_response_path = "retrieval_consistent"
                                self._last_retrieval_score = float(
                                    consistency.get("winning_score") or score
                                )
                                return consistency["winning_response"]
                        except Exception:
                            self._last_consistency = None
                else:
                    self._last_retrieval_candidate = None
            except Exception:
                logger.exception(
                    "Retrieval failed; falling through to generator."
                )
                self._last_retrieval_candidate = None
        else:
            self._last_retrieval_candidate = None

        slm_generator = self._slm_generator
        if slm_generator is not None:
            try:
                # PERF (H-3, 2026-04-23 audit): SLM generation is a
                # synchronous PyTorch loop that can take hundreds of ms.
                # Running it inline blocks the event loop for every
                # other coroutine.  Offload to the default executor,
                # mirroring the ``_encode_features`` pattern above.
                loop = asyncio.get_running_loop()
                # At this scale (4.48 M params, ~22 k-sample corpus)
                # sampled decoding produces word salad; greedy with a
                # short budget gives the cleanest, most coherent
                # continuation.  The adaptation conditioning still
                # steers output via the cross-attention projection.
                #
                # History-aware SLM prompt: the trainer formatted each
                # exchange as ``[BOS] history [SEP] response [EOS]``.
                # For multi-turn we extend that with extra ``[SEP]``-
                # joined turns so the decoder sees the same shape.
                # ``dialogue_mode=True`` already wraps the body with
                # ``[BOS] ... [SEP]`` and stops on ``[EOS]``, so we
                # only need to construct the body string here. We
                # truncate to a coarse word budget to leave room for
                # generation past the positional encoding limit.
                # Same self-contained-query gate as the embedding path:
                # when the user's current message has 3+ of its own
                # substantive content keywords AND no pronoun coref
                # to resolve, prior turns more often pollute than help
                # SLM generation — they bias the model toward whatever
                # was just discussed and produce off-topic text.
                if history_text and not _is_self_contained:
                    slm_prompt = self._truncate_history_text(
                        f"{history_text}\n[SEP]\n{message}",
                        max_words=256,
                    )
                else:
                    slm_prompt = message
                if on_token is not None:
                    # Streaming path: run the generator and shuttle each
                    # decoded delta back to the caller via on_token.
                    raw = await self._generate_streaming_async(
                        slm_generator=slm_generator,
                        message=slm_prompt,
                        adaptation=adaptation,
                        user_state=user_state,
                        on_token=on_token,
                    )
                else:
                    raw = await loop.run_in_executor(
                        None,
                        lambda: slm_generator.generate(
                            prompt=slm_prompt,
                            adaptation_vector=adaptation.to_tensor().unsqueeze(0),
                            user_state=user_state.unsqueeze(0),
                            max_new_tokens=40,
                            temperature=0.0,  # greedy
                            top_k=0,
                            top_p=1.0,
                            repetition_penalty=1.3,
                        ),
                    )
                # Echo-stripping in _clean_slm_output uses the LAST
                # turn (the actual user query) as the prompt so the
                # prepended history doesn't get matched as a leading
                # echo and dropped.
                cleaned = self._clean_slm_output(raw, prompt=message)

                # ----- Self-critique loop (Phase 7) ---------------------
                # Score the draft against the rule-based critic.  When
                # the score is below threshold, regenerate ONCE with
                # tighter sampling and keep the better of the two
                # attempts.  Bounded at 2 attempts so the total SLM
                # latency budget stays under ~800 ms.
                #
                # Wrapped in a broad try/except so a critic failure
                # cannot break the existing SLM path — on error we
                # fall through to the original ``_looks_coherent``
                # gate as if the critic hadn't run.
                self._last_critique = {}
                try:
                    critic = self._self_critic
                    adapt_dict = self._adaptation_to_dict(adaptation)
                    attempts: list[dict] = []

                    score = critic.score(
                        prompt=message,
                        response=cleaned,
                        adaptation=adapt_dict,
                    )
                    attempts.append({
                        "text": cleaned,
                        "score": float(round(score.score, 3)),
                        "sub_scores": dict(score.sub_scores),
                        "reasons": list(score.reasons),
                        "sampling_params": {
                            "temperature": 0.0,
                            "rep_pen": 1.3,
                            "decoding": "greedy",
                        },
                    })

                    if not score.accepted and not on_token:
                        # Streaming attempts are skipped: the second
                        # decode would conflict with the
                        # already-rendered token deltas in the UI.  We
                        # still surface the (non-regenerated) critique
                        # below so the chip + trace render.
                        try:
                            raw2 = await loop.run_in_executor(
                                None,
                                lambda: slm_generator.generate(
                                    prompt=slm_prompt,
                                    adaptation_vector=adaptation.to_tensor().unsqueeze(0),
                                    user_state=user_state.unsqueeze(0),
                                    max_new_tokens=40,
                                    temperature=0.4,
                                    top_k=50,
                                    top_p=0.9,
                                    repetition_penalty=1.5,
                                ),
                            )
                            cleaned2 = self._clean_slm_output(
                                raw2, prompt=message
                            )
                            score2 = critic.score(
                                prompt=message,
                                response=cleaned2,
                                adaptation=adapt_dict,
                            )
                            attempts.append({
                                "text": cleaned2,
                                "score": float(round(score2.score, 3)),
                                "sub_scores": dict(score2.sub_scores),
                                "reasons": list(score2.reasons),
                                "sampling_params": {
                                    "temperature": 0.4,
                                    "top_k": 50,
                                    "top_p": 0.9,
                                    "rep_pen": 1.5,
                                    "decoding": "sampled",
                                },
                            })
                            # Keep the better attempt.
                            if score2.score > score.score:
                                cleaned = cleaned2
                                score = score2
                        except Exception:
                            logger.exception(
                                "Self-critique regen attempt failed; "
                                "keeping the original draft."
                            )

                    self._last_critique = {
                        "final_score": float(round(score.score, 3)),
                        "accepted": bool(score.accepted),
                        "regenerated": len(attempts) > 1,
                        "rejected": (len(attempts) > 1) and (not score.accepted),
                        "attempts": attempts,
                        "threshold": float(critic.threshold),
                        "sub_scores": dict(score.sub_scores),
                    }
                except Exception:
                    logger.exception(
                        "Self-critique scoring failed; continuing with "
                        "the original draft (no regen, no critique frame)."
                    )
                    self._last_critique = {}

                if cleaned and _looks_coherent(cleaned):
                    # On-topic guard (2026-04-26 user-emulation audit):
                    # the SLM occasionally produces grammatically clean
                    # but semantically off-topic output — e.g. "what is
                    # your favorite color" → "I like to read. I love
                    # the best."  ``_looks_coherent`` accepts these
                    # because they're well-formed English; they fail
                    # only on the SEMANTIC level.  We require that any
                    # short query with a clear topic keyword have at
                    # least one content keyword in common with the
                    # SLM draft.  When the overlap is zero AND the
                    # response is short, fall through to OOD instead
                    # of showing word salad.
                    try:
                        from i3.slm.retrieval import _keywords as _kw_fn
                        q_kw = _kw_fn(message or "")
                        r_kw = _kw_fn(cleaned)
                        # Drop low-info tokens shared by every short
                        # SLM reply ("i", "you", "the", etc) so the
                        # overlap measures TOPIC, not grammar.  Keep
                        # this aligned with retrieval's _LOW_INFO_KW
                        # so a query stripped to nothing here is the
                        # same as one stripped to nothing there.
                        _filler = {
                            "you", "your", "yours", "me", "my", "im",
                            "want", "like", "need", "think", "know",
                            "feel", "say", "tell", "explain",
                            "describe", "show", "give", "name", "list",
                            "just", "now", "really", "actually", "even",
                            "good", "bad", "great", "nice", "best", "worst",
                            "thing", "things", "stuff", "way", "ways",
                            "kind", "type", "sort", "line", "lines",
                            "one", "two", "three", "four", "five",
                            "six", "seven", "eight", "nine", "ten",
                            "few", "many", "some", "any", "all", "every",
                            "more", "less", "most", "least",
                            "going", "doing", "being", "having",
                            "got", "get", "make", "take", "do", "did",
                            "answer", "question", "reply", "response",
                            "youre", "thats", "whats", "im",
                        }
                        q_topic = q_kw - _filler
                        r_topic = r_kw - _filler
                        slm_words = cleaned.strip().split()
                        on_topic = bool(q_topic & r_topic) or not q_topic
                        # Veto when:
                        #   (a) the response is short and shares no
                        #       topic word with the query, OR
                        #   (b) the query has 2+ substantive topic words
                        #       and ZERO of them appear in the response
                        #       (catches long, authoritative-looking,
                        #       wrong-topic answers — the worst kind of
                        #       failure for a recruiter demo).
                        long_off_topic = (
                            len(q_topic) >= 2
                            and not (q_topic & r_topic)
                        )
                        if (not on_topic and len(slm_words) <= 25) or long_off_topic:
                            logger.debug(
                                "SLM off-topic veto: q_topic=%s r_topic=%s "
                                "draft=%r",
                                sorted(q_topic)[:6],
                                sorted(r_topic)[:6],
                                cleaned[:80],
                            )
                            # Fall through to borderline / OOD branches.
                            cleaned = ""
                    except Exception:
                        # Never let the on-topic check break generation.
                        pass
                if cleaned and _looks_coherent(cleaned):
                    self._last_response_path = "slm"
                    return cleaned
            except Exception:
                logger.exception(
                    "SLM generation failed; using rule-based fallback."
                )

        # --- Borderline retrieval fallback --------------------------------
        borderline = getattr(self, "_last_retrieval_candidate", None)
        if borderline:
            self._last_response_path = "retrieval_borderline"
            return borderline

        # --- Ambiguity-hold / clarification (Phase B.4, 2026-04-25) -------
        # When the retrieval score landed in the 0.55–0.75 borderline
        # band but the user message is short (≤5 content words) we'd
        # rather **ask** than fabricate.  This is the failure mode where
        # the audit caught responses like "Nice. Favourite genre?" — a
        # confident-looking but semantically-empty match.  Asking a
        # short clarifying question keeps the chat honest and surfaces
        # the system's actual uncertainty.
        try:
            last_score = float(getattr(self, "_last_retrieval_score", 0.0) or 0.0)
            from i3.slm.retrieval import _keywords as _kw_fn
            content_kw = _kw_fn(message or "")
            if (
                0.55 <= last_score < 0.75
                and 0 < len(content_kw) <= 5
            ):
                clarifier = self._build_clarifier(
                    message=message,
                    user_id=user_id,
                    session_id=session_id,
                )
                if clarifier:
                    self._last_response_path = "tool:clarify"
                    return clarifier
        except Exception:  # pragma: no cover — never let clarifier crash
            logger.debug("Clarifier build failed; falling through to OOD.")

        # --- Out-of-distribution default ----------------------------------
        # If neither retrieval nor generation produced anything useful,
        # tell the user honestly rather than emit a noisy template.  The
        # response acknowledges the edge-model's limited scope — which
        # is itself part of the "small edge model" pitch.
        ood_options = [
            "I'm a small on-device model and that's outside what I've been trained on — tell me more about what you're looking for?",
            "That's a bit beyond my current corpus. Could you rephrase, or ask about something a bit more everyday?",
            "I don't have a confident answer for that one. Want to try a different angle?",
        ]
        # Pick deterministically by message hash so the same prompt
        # always produces the same fallback (stable demo behaviour).
        idx = sum(ord(c) for c in message) % len(ood_options)
        self._last_response_path = "ood"
        # Iter 25 (2026-04-26): smart OOD recovery — suggest 2-3
        # topics the user could ask about instead.  If the entity
        # tracker has recent topics, prefer those; else fall back to
        # a curated demo list.  Adds the suggestions as a small
        # bullet list at the end of the OOD reply so the user has a
        # concrete next move instead of a dead-end "I don't know".
        suggestions = self._build_ood_suggestions(user_id, session_id)
        base = ood_options[idx]
        if suggestions:
            return (
                f"{base}\n\nI can talk about: "
                + ", ".join(suggestions)
                + "."
            )
        return base

        # --- Rule-based fallback ------------------------------------------
        return self._fallback_response(adaptation)

    # Session-recap detection patterns (iter 27, 2026-04-26).  Each
    # pattern is anchored ^...$ and case-insensitive.  We match the
    # cleaned message AFTER discourse-prefix strip, so "so what have
    # we discussed" hits these too.
    _RECAP_PATTERNS: tuple[re.Pattern[str], ...] = (
        # "summarize our/this/the conversation"
        re.compile(r"^(?:can\s+you\s+)?(?:please\s+)?summari[sz]e\s+(?:our\s+|this\s+|the\s+)?(?:conversation|chat|discussion|talk)\s*\??\s*$", re.I),
        # "summarize what we've talked about / discussed / covered"
        re.compile(r"^(?:can\s+you\s+)?(?:please\s+)?summari[sz]e\s+(?:what\s+)?(?:we['']?ve\s+|we\s+have\s+|have\s+we\s+)?(?:talked\s+about|discussed|covered|been\s+talking\s+about)\s*(?:so\s+far)?\s*\??\s*$", re.I),
        # Bare "summarize" / "summarize everything" / "summarize the chat"
        re.compile(r"^summari[sz]e(?:\s+the\s+(?:chat|discussion))?(?:\s+everything)?\s*\??\s*$", re.I),
        # "recap" alone
        re.compile(r"^(?:can\s+you\s+)?recap\s*(?:our|this|the)?\s*(?:conversation|chat|talk|discussion)?\s*\??\s*$", re.I),
        # Iter 51: "recap what we('ve)? (discussed|talked about|covered)"
        # Drift test T_slow_burn T10 was failing this — without the
        # pattern the SLM took the turn and drifted off-topic.
        re.compile(r"^(?:can\s+you\s+)?(?:please\s+)?recap\s+(?:what\s+)?(?:we['']?ve\s+|we\s+have\s+|have\s+we\s+|we\s+)?(?:talked\s+about|discussed|covered|been\s+talking\s+about)\s*(?:so\s+far)?\s*\??\s*$", re.I),
        # "what have we discussed/talked about/covered (so far)"
        re.compile(r"^what\s+(?:have\s+we|did\s+we|did\s+I|did\s+you|are\s+we)\s*(?:been\s+)?(?:discussed|discussing|talked\s+about|talking\s+about|covered|covering|saying|been\s+saying|been\s+covering)\s*(?:so\s+far|already)?\s*\??\s*$", re.I),
        # "what we've discussed (so far)" / "what we have covered"
        re.compile(r"^what\s+we['']?ve\s+(?:discussed|talked\s+about|covered|been\s+talking\s+about)\s*(?:so\s+far)?\s*\??\s*$", re.I),
        re.compile(r"^what\s+we\s+have\s+(?:discussed|talked\s+about|covered|been\s+talking\s+about)\s*(?:so\s+far)?\s*\??\s*$", re.I),
        # "what topics (have we / did we / are we) (covered / discussed / cover / talk about) (so far)"
        re.compile(r"^what\s+topics?\s*(?:have\s+we|did\s+we|are\s+we)?\s*(?:covered|cover|discussed|discuss|talked\s+about|talk\s+about|covering|discussing|talking\s+about)?\s*(?:so\s+far)?\s*\??\s*$", re.I),
        # "tell me / give me a (quick/brief/short) summary/recap/overview".  The
        # trailing "of conversation" group is optional AND the whitespace
        # before it is optional so "tell me a summary" / "give me a quick
        # recap" (where the message ends right after the noun) still match.
        re.compile(r"^(?:tell\s+me\s+|give\s+me\s+)?(?:a\s+)?(?:quick\s+|brief\s+|short\s+)?(?:summary|recap|overview|tldr|tl;dr)(?:\s+of\s+(?:our\s+|this\s+|the\s+)?(?:conversation|chat|talk|discussion))?\s*\??\s*$", re.I),
        # "remind me (of/about/what) (we've/we have/have we/we) talked about/discussed/covered"
        re.compile(r"^remind\s+me\s+(?:of\s+|about\s+|what\s+)?(?:we['']?ve\s+|we\s+have\s+|have\s+we\s+|we\s+)?(?:talked\s+about|discussed|covered)\s*\??\s*$", re.I),
        # Lazy single phrases
        re.compile(r"^(?:tldr|tl;dr|the\s+gist)\s*\??\s*$", re.I),
        re.compile(r"^so\s+far\s*\??\s*$", re.I),
    )

    # Iter 43 (2026-04-26): patterns that ask specifically about the
    # FIRST topic in the conversation — answered differently from a
    # general recap.  E.g. "what was the first thing I asked" /
    # "what did I first ask" / "the first topic" / "what did we
    # start with".
    # Trailing-context group used by several first-topic patterns:
    # "in this session" / "in our chat" / "of this conversation" / etc.
    _FIRST_TRAIL = (
        r"(?:\s+(?:in\s+)?(?:this|our|the)\s+"
        r"(?:session|chat|conversation|thread|talk|discussion))?"
    )

    _FIRST_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(rf"^what\s+was\s+the\s+first\s+thing\s+(?:i|we)\s+(?:asked|talked\s+about|discussed|said|started\s+with){_FIRST_TRAIL}\s*\??\s*$", re.I),
        re.compile(rf"^what\s+did\s+(?:i|we)\s+(?:first\s+)?ask\s+(?:about\s+)?(?:first)?{_FIRST_TRAIL}\s*\??\s*$", re.I),
        re.compile(rf"^what\s+(?:was|is)\s+the\s+first\s+(?:topic|question|thing){_FIRST_TRAIL}\s*\??\s*$", re.I),
        re.compile(rf"^what\s+did\s+we\s+start\s+with{_FIRST_TRAIL}\s*\??\s*$", re.I),
        re.compile(rf"^what\s+was\s+(?:my|our)\s+first\s+(?:question|topic|message){_FIRST_TRAIL}\s*\??\s*$", re.I),
        re.compile(rf"^the\s+first\s+(?:topic|question|thing){_FIRST_TRAIL}\s*\??\s*$", re.I),
    )

    def _maybe_session_recap(
        self,
        message: str,
        user_id: str,
        session_id: str,
    ) -> str | None:
        """Return a session-recap reply when *message* asks for one.

        Detects "summarize our conversation" / "what have we discussed"
        / "recap" patterns.  Builds the recap from the entity tracker's
        user-anchored ORG/TOPIC frames in stack order (most recent
        first).  If no anchored topics yet, returns None and lets the
        normal flow handle the message.
        """
        if not message:
            return None
        cleaned = message.strip()
        # Discourse-prefix strip so "so what have we discussed" still
        # matches.  Mirrors the strip used by explain_decomposer plus
        # session-end variants ("before I go", "before we wrap",
        # "to wrap up", "one last thing", "quick question") that
        # naturally precede a recap request.
        _DISCOURSE = re.compile(
            r"^\s*(?:"
            r"wait|actually|oh|hmm|sorry|um|uh|ok|okay|so|well|hey|"
            r"listen|you know|i mean|by the way|btw|"
            r"before\s+i\s+go|before\s+we\s+(?:wrap|finish|end|stop)|"
            r"to\s+wrap\s+up|to\s+sum\s+up|"
            r"one\s+last\s+thing|one\s+more\s+thing|"
            r"quick\s+question|just\s+one\s+thing"
            r")\s*[,\-:—]?\s+",
            re.I,
        )
        cleaned = _DISCOURSE.sub("", cleaned, count=1).strip()

        # Iter 43: first-topic-recall — "what was the first thing I
        # asked" returns the EARLIEST user-anchored topic only.
        is_first_topic_query = any(
            p.match(cleaned) for p in self._FIRST_TOPIC_PATTERNS
        )
        if is_first_topic_query:
            try:
                snap = self._entity_tracker.snapshot(user_id, session_id)
            except Exception:
                return None
            anchored = [
                f for f in snap
                if f.first_anchor_turn is not None
                and f.kind in {"org", "topic"}
            ]
            if not anchored:
                return (
                    "We don't have any user-anchored topic on the "
                    "session yet — what would you like to start with?"
                )
            first = min(anchored, key=lambda f: f.first_anchor_turn)
            return (
                f"The first thing we talked about was {first.text}. "
                f"Want me to circle back to {first.text}, or pick up "
                "where we left off?"
            )

        if not any(p.match(cleaned) for p in self._RECAP_PATTERNS):
            return None

        try:
            snap = self._entity_tracker.snapshot(user_id, session_id)
        except Exception:
            return None

        # Pull user-anchored ORG/TOPIC topics in stack order.
        seen: set[str] = set()
        topics: list[tuple[str, str, str]] = []  # (canonical, surface, kind)
        for f in snap:
            if f.canonical in seen:
                continue
            if f.kind not in {"org", "topic"}:
                continue
            if f.user_anchor_turn is None:
                continue
            seen.add(f.canonical)
            topics.append((f.canonical, f.text, f.kind))
            if len(topics) >= 6:
                break

        if not topics:
            return (
                "We've only just started talking — I don't have any "
                "topics on the recency stack yet. Ask me about "
                "something specific and I'll pick up from there."
            )

        # Build a natural-language recap.  Most recent topic last so
        # the sentence reads like a chronological replay.
        topics_chrono = list(reversed(topics))
        if len(topics_chrono) == 1:
            t = topics_chrono[0][1]
            return (
                f"So far we've been talking about {t}. "
                f"Want me to keep going on {t}, or pivot somewhere else?"
            )
        # Format: "first A, then B, and now we're on C."
        first = topics_chrono[0][1]
        last = topics_chrono[-1][1]
        if len(topics_chrono) == 2:
            return (
                f"We started on {first}, and now we're on {last}. "
                f"Want me to dig deeper on {last}, or jump back to "
                f"{first}?"
            )
        middle = ", ".join(t[1] for t in topics_chrono[1:-1])
        return (
            f"We started on {first}, then moved through {middle}, and "
            f"now we're on {last}. Want me to dig deeper on {last}, "
            f"jump back to one of the earlier topics, or pivot to "
            f"something else?"
        )

    # Iter 48: lightweight session-name memory (single slot per
    # (user, session) pair).
    _NAME_STATEMENT_RE = re.compile(
        r"^(?:my\s+name\s+is|i\s+am|i'?m|call\s+me|name'?s|"
        r"please\s+call\s+me|you\s+can\s+call\s+me)\s+"
        r"([A-Za-z][A-Za-z'\-\s]{0,30}?)\s*[.!,?]*\s*$",
        re.I,
    )
    _NAME_RECALL_RE = re.compile(
        r"^(?:what(?:['']?s|\s+is)\s+my\s+name|"
        r"do\s+you\s+(?:remember|know)\s+my\s+name|"
        r"what\s+did\s+i\s+(?:tell\s+you\s+)?my\s+name(?:\s+was)?|"
        r"who\s+am\s+i)\s*\??\s*$",
        re.I,
    )

    def _maybe_handle_name_statement(
        self,
        *,
        message: str,
        user_id: str,
        session_id: str,
    ) -> str | None:
        """Detect "my name is X" / recall "what's my name" patterns.

        Returns a ready-made response string, or None when neither
        pattern matches.  Persists the name in
        ``self._stated_user_name[(user_id, session_id)]`` for the
        lifetime of the session.
        """
        cleaned = (message or "").strip()
        if not cleaned:
            return None
        m = self._NAME_STATEMENT_RE.match(cleaned)
        if m:
            raw_name = m.group(1).strip().rstrip(".,!?")
            tokens = raw_name.split()
            if len(tokens) > 3:
                tokens = tokens[:3]
            name = " ".join(tokens).title() if tokens else None
            if not name:
                return None
            _BLACKLIST = {
                "Tired", "Sad", "Happy", "Hungry", "Sorry", "Sure",
                "Ok", "Okay", "Fine", "Good", "Bad", "Here", "Back",
                "Glad", "Listening", "Ready",
            }
            if name in _BLACKLIST:
                return None
            self._stated_user_name[(user_id, session_id)] = name
            # Iter 50: also write into the multi-fact dict and
            # persist to the diary so cross-session recall works.
            facts = self._stated_facts.setdefault(
                (user_id, session_id), {},
            )
            facts["name"] = name
            try:
                persist_task = asyncio.create_task(
                    self.diary_store.set_user_fact(
                        user_id, "name", name,
                    )
                )
                self._background_tasks.add(persist_task)
                persist_task.add_done_callback(
                    self._background_task_done
                )
            except Exception:  # pragma: no cover — never block
                logger.debug(
                    "Failed to schedule name persistence for %s",
                    user_id, exc_info=True,
                )
            return (
                f"Nice to meet you, {name}! I'll keep that in mind for "
                f"this conversation. What would you like to talk about?"
            )
        if self._NAME_RECALL_RE.match(cleaned):
            stored = self._stated_user_name.get((user_id, session_id))
            if stored:
                return (
                    f"You told me your name is {stored}. (Stored "
                    "encrypted on-device — survives across sessions; "
                    "say 'forget my facts' to wipe it.)"
                )
            return (
                "I don't think you've told me your name yet — what "
                "should I call you?"
            )
        return None

    # Iter 49: multi-fact session memory.  Each tuple is
    # (statement_regex, recall_regex, slot_key, slot_label).  Order
    # matters — first match wins, so put more-specific patterns first.
    _FACT_HANDLERS: tuple[tuple[re.Pattern[str], re.Pattern[str], str, str], ...] = (
        # Favourite color
        (
            re.compile(r"^(?:my\s+)?(?:favou?rite\s+)?colou?r\s+is\s+([A-Za-z][A-Za-z\s\-]{0,30})\s*[.!?]*\s*$", re.I),
            re.compile(r"^(?:what(?:['']?s|\s+is)\s+)?my\s+(?:favou?rite\s+)?colou?r\s*\??\s*$", re.I),
            "favourite_color", "favourite colour",
        ),
        # Favourite food
        (
            re.compile(r"^(?:my\s+)?favou?rite\s+food\s+is\s+([A-Za-z][A-Za-z\s\-]{0,40})\s*[.!?]*\s*$", re.I),
            re.compile(r"^(?:what(?:['']?s|\s+is)\s+)?my\s+favou?rite\s+food\s*\??\s*$", re.I),
            "favourite_food", "favourite food",
        ),
        # Favourite music / band / artist
        (
            re.compile(r"^(?:my\s+)?favou?rite\s+(?:music|band|artist|singer|song)\s+is\s+([A-Za-z][A-Za-z\s\-']{0,40})\s*[.!?]*\s*$", re.I),
            re.compile(r"^(?:what(?:['']?s|\s+is)\s+)?my\s+favou?rite\s+(?:music|band|artist|singer|song)\s*\??\s*$", re.I),
            "favourite_music", "favourite music",
        ),
        # Occupation: "I work as a nurse" / "I'm a nurse" / "I am an engineer"
        (
            re.compile(r"^(?:i\s+work\s+as\s+(?:an?|the)\s+|my\s+job\s+is\s+(?:an?|the)?\s*|i'?m\s+an?\s+|i\s+am\s+an?\s+)([A-Za-z][A-Za-z\s\-]{2,40})\s*[.!?]*\s*$", re.I),
            re.compile(r"^(?:what(?:['']?s|\s+is)\s+)?my\s+(?:job|occupation|profession|work)\s*\??\s*$|^what\s+do\s+i\s+do(?:\s+for\s+(?:work|a\s+living))?\s*\??\s*$", re.I),
            "occupation", "job",
        ),
        # Location: "I live in Berlin" / "I'm from London" / "I'm in NYC"
        (
            re.compile(r"^(?:i\s+live\s+in\s+|i'?m\s+from\s+|i\s+am\s+from\s+|i'?m\s+in\s+|i\s+am\s+in\s+|i'?m\s+based\s+in\s+|i\s+am\s+based\s+in\s+)([A-Za-z][A-Za-z\s\-,]{1,40})\s*[.!?]*\s*$", re.I),
            re.compile(r"^(?:where\s+do\s+i\s+live|where\s+(?:am\s+i|do\s+i\s+come)\s+from|what(?:['']?s|\s+is)\s+my\s+(?:city|location|hometown))\s*\??\s*$", re.I),
            "location", "location",
        ),
        # Hobby: "my hobby is X" / "I love climbing" / "I enjoy painting"
        (
            re.compile(r"^(?:my\s+hobby\s+is\s+|my\s+favou?rite\s+hobby\s+is\s+|i\s+love\s+|i\s+enjoy\s+|i\s+like\s+(?:to\s+)?)([A-Za-z][A-Za-z\s\-]{2,40})\s*[.!?]*\s*$", re.I),
            re.compile(r"^(?:what(?:['']?s|\s+is)\s+)?my\s+(?:hobby|favou?rite\s+hobby|hobbies|favou?rite\s+activity)\s*\??\s*$", re.I),
            "hobby", "hobby",
        ),
        # Age: "I'm 30" / "I am 25 years old"
        (
            re.compile(r"^(?:i'?m\s+|i\s+am\s+)(\d{1,3})(?:\s+years\s+old)?\s*[.!?]*\s*$", re.I),
            re.compile(r"^(?:how\s+old\s+am\s+i|what(?:['']?s|\s+is)\s+my\s+age)\s*\??\s*$", re.I),
            "age", "age",
        ),
        # Pet: "I have a dog called Rex" / "my dog is Rex" / "I have a cat"
        (
            re.compile(r"^(?:i\s+have\s+(?:a|an)\s+|my\s+pet\s+is\s+(?:a|an)?\s*)([A-Za-z][A-Za-z\s\-']{2,40})\s*[.!?]*\s*$", re.I),
            re.compile(r"^(?:what(?:['']?s|\s+is)\s+)?my\s+pet\s*\??\s*$|^what\s+pet\s+do\s+i\s+have\s*\??\s*$", re.I),
            "pet", "pet",
        ),
    )

    # Words that should NOT be captured as fact values (catalog topics
    # / common acks / states that would be misclassified as a person /
    # job / hobby etc).
    _FACT_VALUE_BLACKLIST: frozenset[str] = frozenset({
        "tired", "sad", "happy", "ok", "okay", "fine", "good", "bad",
        "sure", "ready", "back", "here", "there", "fan", "human",
        "wondering", "thinking", "asking", "kidding", "joking",
        "model", "an ai", "ai",
    })

    # Iter 51: command-shaped utterance detector.  Lightweight
    # short-circuit before the full Qwen LoRA parser runs, so we
    # don't pay the LoRA load cost on free-form chat.
    _INTENT_TRIGGER_PATTERNS: tuple = (
        re.compile(r"\b(set|start)\s+(?:a\s+)?(?:\d+\s+)?(?:minute|min|second|sec|hour|hr)\b", re.I),
        # Iter 51 Phase 5: also tolerate "start ..." (polite phrasings
        # like "start a timer" / "could you start a five minute timer
        # please") and an optional adjective word ("a quick timer", "a
        # five minute timer") between the verb and the noun.  Capped
        # at three modifier words so we don't slurp whole sentences.
        re.compile(r"\b(?:set|create|new|start)\s+(?:a\s+)?(?:\w+\s+){0,3}timer\b", re.I),
        re.compile(r"\b(?:set|create|new|start)\s+(?:an?\s+)?(?:\w+\s+){0,3}alarm\b", re.I),
        re.compile(r"\bplay\s+(?:some\s+)?\w+", re.I),
        re.compile(r"\b(?:pause|resume|stop|skip|next|previous|prev)\s*(?:the\s+)?(?:song|music|track|video)?\b", re.I),
        re.compile(r"\b(?:turn|set)\s+(?:the\s+)?volume\s+(?:up|down|to|on|off)\b", re.I),
        # Iter 51: also match bare "volume up" / "volume down" without
        # the leading "turn" / "set".
        re.compile(r"\bvolume\s+(?:up|down|to)\b", re.I),
        re.compile(r"\b(?:mute|unmute)\b", re.I),
        # Iter 51: allow an optional adjective between "the" and the
        # device noun ("turn off the bedroom lamp", "turn on the kitchen lights").
        re.compile(r"\b(?:turn|switch)\s+(?:on|off)\s+(?:the\s+)?(?:\w+\s+)?(?:lights?|lamp|tv|fan|heater|thermostat|switch|outlet|plug|kettle|oven)\b", re.I),
        re.compile(r"\b(?:remind|reminder)\s+me\b", re.I),
        re.compile(r"\b(?:send|text|message)\s+(?:a\s+message\s+to\s+|to\s+)?\w+", re.I),
        re.compile(r"\b(?:call|video\s+call|ring)\s+\w+", re.I),
        re.compile(r"\b(?:open|launch|start)\s+(?:the\s+)?(?:app|application|spotify|netflix|calculator|maps|browser|chrome|safari|firefox)\b", re.I),
        re.compile(r"\b(?:navigate|directions?|drive|go|route)\s+(?:to|home)\b", re.I),
        re.compile(r"\b(?:what'?s|what\s+is)\s+the\s+weather\b", re.I),
        re.compile(r"\bset\s+the\s+thermostat\b", re.I),
        re.compile(r"\b(?:lock|unlock)\s+(?:the\s+)?(?:doors?|front)\b", re.I),
        re.compile(r"\bcancel\s+(?:the\s+)?(?:timer|alarm|reminder)?\b", re.I),
    )

    def _looks_like_command(self, message: str) -> bool:
        """Cheap regex test before invoking the (heavy) Qwen LoRA parser."""
        msg = (message or "").strip()
        if not msg or len(msg) > 200:
            return False
        for pat in self._INTENT_TRIGGER_PATTERNS:
            if pat.search(msg):
                return True
        return False

    def _maybe_handle_intent_command(
        self,
        *,
        message: str,
        user_id: str,
        session_id: str,
    ) -> str | None:
        """Try to parse *message* as an HMI command via the Qwen LoRA.

        Returns a deterministic structured response string when the
        parser returns a high-confidence valid action, ``None``
        otherwise.  The full IntentResult is also stashed on
        ``self._last_intent_result`` so the chat UI's green ◆ chip
        and the WS state_update frame both pick it up.
        """
        if not self._looks_like_command(message):
            return None

        # Iter 66: open an OTel span for the cascade arm B leg so the
        # observability stack (OTel collector + Sentry + Langfuse) can
        # correlate per-arm latency.  Falls back to no-op when OTel
        # isn't configured.
        from i3.observability.tracing import span as _otel_span

        with _otel_span("cascade.arm_b.qwen_intent",
                        i3_user_id=str(user_id)[:32],
                        i3_session_id=str(session_id)[:32],
                        i3_message_chars=len(message or "")):
            # Lazy-load the parser.  Cached on the engine instance so
            # subsequent commands skip the ~30 s base-model load.
            parser = getattr(self, "_intent_parser_qwen", None)
            if parser is None:
                try:
                    with _otel_span("cascade.arm_b.qwen_load"):
                        from i3.intent.qwen_inference import QwenIntentParser
                        parser = QwenIntentParser()
                        self._intent_parser_qwen = parser
                except Exception as exc:  # pragma: no cover - missing deps
                    logger.debug("intent parser load failed: %s", exc)
                    return None

            try:
                with _otel_span("cascade.arm_b.qwen_parse"):
                    result = parser.parse(message)
            except Exception as exc:  # pragma: no cover
                logger.debug("intent parse failed: %s", exc)
                return None

        # Stash result for PipelineOutput regardless of confidence so
        # the dashboard can show "we tried, here's what we got".
        try:
            self._last_intent_result = result.to_dict()
        except Exception:
            self._last_intent_result = None

        # Smart cross-arm fallback: if Qwen returned "unsupported",
        # invalid_action, or invalid_slots, try Gemini as a backup
        # parser before falling through to chat.  Gemini is faster
        # (~900 ms vs Qwen's 7 s) but optional (only fires when the
        # primary parser couldn't produce a valid action).
        # Skipped silently when GEMINI_API_KEY is unset.
        from os import environ as _env
        primary_failed = (
            not getattr(result, "valid_action", False)
            or not getattr(result, "valid_slots", False)
            or getattr(result, "action", None) == "unsupported"
        )
        if primary_failed and _env.get("GEMINI_API_KEY"):
            try:
                with _otel_span("cascade.arm_b.gemini_backup"):
                    backup_parser = getattr(
                        self, "_intent_parser_gemini", None,
                    )
                    if backup_parser is None:
                        from i3.intent.gemini_inference import GeminiIntentParser
                        backup_parser = GeminiIntentParser()
                        self._intent_parser_gemini = backup_parser
                    backup = backup_parser.parse(message)
                if (getattr(backup, "valid_action", False)
                        and getattr(backup, "valid_slots", False)
                        and backup.action != "unsupported"):
                    # Record the backup as the authoritative result so
                    # the dashboard shows the Gemini parse instead.
                    backup.backend = "gemini-backup"
                    self._last_intent_result = backup.to_dict()
                    result = backup
                    logger.info(
                        "intent.cascade: qwen failed → gemini backup "
                        "salvaged action=%s",
                        getattr(backup, "action", "?"),
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("gemini backup parse failed: %s", exc)

        # Only short-circuit when the parser returned something usable.
        if not (getattr(result, "valid_action", False)
                and getattr(result, "valid_slots", False)):
            return None

        action = result.action
        params = result.params or {}
        # Render a deterministic acknowledgement.  We avoid actually
        # *executing* the action (no real timer / no real Spotify) —
        # the demo proves the *parsing* layer; downstream side-effects
        # are out of scope for the assistant.
        if action == "set_timer":
            secs = params.get("duration_seconds") or 0
            mins = secs // 60
            ack = f"Timer for {mins} min set." if mins else "Timer set."
        elif action == "play_music":
            genre = params.get("genre") or params.get("artist") or "music"
            ack = f"Playing {genre}."
        elif action == "send_message":
            who = params.get("recipient", "them")
            ack = f"Message to {who} queued."
        elif action == "navigate":
            dest = params.get("destination", "the destination")
            ack = f"Navigating to {dest}."
        elif action == "set_alarm":
            t = params.get("time", "the requested time")
            ack = f"Alarm set for {t}."
        elif action == "set_volume":
            lvl = params.get("level", "the requested level")
            ack = f"Volume set to {lvl}."
        elif action == "control_device":
            dev = params.get("device", "the device")
            verb = params.get("verb", "toggled")
            ack = f"{dev.capitalize()} {verb}."
        elif action == "weather":
            loc = params.get("location", "your location")
            ack = f"Looking up the weather for {loc}."
        elif action == "remind":
            what = params.get("task", "the reminder")
            ack = f"Reminder set: {what}."
        elif action == "cancel":
            ack = "Cancelled."
        elif action == "unsupported":
            ack = "I can't action that one."
        else:
            ack = f"Action '{action}' acknowledged."

        # Attach a parser-confidence chip so the chat UI shows the
        # green ◆ pill via _appendSideChips.
        return ack

    def _maybe_handle_fact_statement(
        self,
        *,
        message: str,
        user_id: str,
        session_id: str,
    ) -> str | None:
        """Detect "my favourite X is Y" / "I work as Y" / etc.  Or
        recall "what's my favourite X".  Returns a polite response,
        or None when no pattern matched.
        """
        cleaned = (message or "").strip()
        if not cleaned:
            return None
        facts = self._stated_facts.setdefault((user_id, session_id), {})
        # 1) Statement: store the value.  Walk handlers in order.
        for stmt_pat, recall_pat, slot, label in self._FACT_HANDLERS:
            m = stmt_pat.match(cleaned)
            if m:
                value = m.group(1).strip().rstrip(".,!?")
                # Reject obvious non-values.
                if value.lower() in self._FACT_VALUE_BLACKLIST:
                    continue
                if not value or len(value) < 2:
                    continue
                # For multi-word values that look like a question /
                # nonsense, reject.
                if "?" in value:
                    continue
                facts[slot] = value
                # Iter 50: cross-session persistence.  Schedule a
                # fire-and-forget DB write so recall works in a NEW
                # session for the same user.  Encrypted at rest when
                # an encryptor is configured (standard for I3).
                try:
                    persist_task = asyncio.create_task(
                        self.diary_store.set_user_fact(
                            user_id, slot, value,
                        )
                    )
                    # Track to avoid GC; same pattern as
                    # ``_log_exchange_safe``'s background task.
                    self._background_tasks.add(persist_task)
                    persist_task.add_done_callback(
                        self._background_task_done
                    )
                    # Mirror name into the iter-48 single-name slot.
                    if slot == "name":
                        self._stated_user_name[(user_id, session_id)] = value
                except Exception:  # pragma: no cover — never block
                    logger.debug(
                        "Failed to schedule fact persistence for "
                        "(%s, %s)", user_id, slot, exc_info=True,
                    )
                # Acknowledgement phrasing varies per slot.
                if slot == "occupation":
                    return f"Got it — you work as {value}. I'll keep that in mind for this conversation."
                if slot == "location":
                    return f"Got it — you're in {value}. Anything you'd like to talk about?"
                if slot == "favourite_color":
                    return f"Noted — your favourite colour is {value}. Anything else on your mind?"
                if slot == "favourite_food":
                    return f"Noted — your favourite food is {value}. Anything else on your mind?"
                if slot == "favourite_music":
                    return f"Noted — your favourite music is {value}. Anything else?"
                if slot == "hobby":
                    return f"Got it — your hobby is {value}. Sounds great. What would you like to chat about?"
                if slot == "age":
                    return f"Got it — you're {value}. What would you like to talk about?"
                if slot == "pet":
                    return f"Lovely — you have {value}. What would you like to chat about?"
                return f"Noted: your {label} is {value}."
        # 2) Recall: look up the slot.
        for stmt_pat, recall_pat, slot, label in self._FACT_HANDLERS:
            if recall_pat.match(cleaned):
                stored = facts.get(slot)
                if stored:
                    return (
                        f"You told me your {label} is {stored}. "
                        "(Stored encrypted on-device — survives "
                        "across sessions; say 'forget my facts' to "
                        "wipe it.)"
                    )
                return (
                    f"I don't think you've told me your {label} yet "
                    "— happy to know if you want to share."
                )
        # Iter 50: "forget my facts" / "wipe my data" — clear all
        # in-memory + persisted facts for this user.
        _FORGET_RE = re.compile(
            r"^(?:please\s+)?(?:forget|delete|wipe|clear|erase)\s+"
            r"(?:my|all\s+(?:my|the))\s+"
            r"(?:facts|data|information|info|memory|details|"
            r"personal\s+(?:facts|data|info|details))\s*[.!?]*\s*$",
            re.I,
        )
        if _FORGET_RE.match(cleaned):
            facts.clear()
            self._stated_user_name.pop((user_id, session_id), None)
            try:
                forget_task = asyncio.create_task(
                    self.diary_store.forget_user_facts(user_id)
                )
                self._background_tasks.add(forget_task)
                forget_task.add_done_callback(
                    self._background_task_done
                )
            except Exception:  # pragma: no cover
                logger.debug(
                    "Failed to schedule forget for %s",
                    user_id, exc_info=True,
                )
            return (
                "Done — I've cleared every personal fact you'd told "
                "me (in-memory + the encrypted on-device store). "
                "Fresh start whenever you're ready."
            )
        return None

    def _build_ood_suggestions(
        self,
        user_id: str,
        session_id: str,
        max_suggestions: int = 3,
    ) -> list[str]:
        """Build a list of topic suggestions for the OOD reply.

        Strategy:
            1. Walk the entity tracker stack and pull the top-N
               distinct ORG/TOPIC entities from the active session.
               These are things the user has shown interest in.
            2. If we don't have enough, top up with a curated demo
               list (broadly-known topics the system handles well).

        Returns at most ``max_suggestions`` distinct topics, in
        recency order (most-recent first), formatted with their
        canonical surface form.
        """
        out: list[str] = []
        seen_canon: set[str] = set()
        try:
            snap = self._entity_tracker.snapshot(user_id, session_id)
            for f in snap:
                if len(out) >= max_suggestions:
                    break
                if f.canonical in seen_canon:
                    continue
                if f.kind not in {"org", "topic", "place"}:
                    continue
                seen_canon.add(f.canonical)
                # Capitalise the first letter so the inline list reads
                # cleanly in the OOD reply.
                surf = f.text.strip()
                if surf:
                    out.append(surf[0].upper() + surf[1:])
        except Exception:  # pragma: no cover - defensive
            pass
        # Top up with curated demo seeds if we still have room.
        _DEMO_SEEDS = [
            "transformers", "photosynthesis", "gravity", "Apple",
            "the Huawei pitch", "how I adapt to your typing",
        ]
        for s in _DEMO_SEEDS:
            if len(out) >= max_suggestions:
                break
            canon = s.lower().split()[-1]
            if canon in seen_canon:
                continue
            seen_canon.add(canon)
            out.append(s)
        return out[:max_suggestions]

    async def _generate_streaming_async(
        self,
        slm_generator: Any,
        message: str,
        adaptation: Any,
        user_state: torch.Tensor,
        on_token: Any,
    ) -> str:
        """Run :meth:`SLMGenerator.generate_streaming` off-thread and
        shuttle each decoded token delta back to ``on_token``.

        The generator is synchronous PyTorch code.  We push each yielded
        delta onto a bounded ``asyncio.Queue`` from the worker thread via
        :meth:`loop.call_soon_threadsafe`, then the coroutine drains the
        queue and awaits ``on_token`` for each delta.  This preserves
        back-pressure (the executor blocks briefly if the queue fills)
        and guarantees ordered delivery.

        Returns the final raw text the generator produced (matching what
        :meth:`SLMGenerator.generate` returns).
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        # Sentinel used to signal "no more deltas".
        _DONE = object()

        def _worker() -> str:
            """Thread-pool worker: iterate the sync generator, push to
            the queue, return the final text.
            """
            final_text = ""
            try:
                for item in slm_generator.generate_streaming(
                    prompt=message,
                    adaptation_vector=adaptation.to_tensor().unsqueeze(0),
                    user_state=user_state.unsqueeze(0),
                    max_new_tokens=40,
                    temperature=0.0,  # greedy
                    top_k=0,
                    top_p=1.0,
                    repetition_penalty=1.3,
                ):
                    if isinstance(item, tuple) and item and item[0] == "final":
                        final_text = str(item[1]) if len(item) > 1 else ""
                    else:
                        # Push the delta onto the queue (thread-safe).
                        loop.call_soon_threadsafe(queue.put_nowait, item)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)
            return final_text

        # Kick the worker; drain the queue until we see the sentinel.
        worker_future = loop.run_in_executor(None, _worker)

        try:
            while True:
                delta = await queue.get()
                if delta is _DONE:
                    break
                try:
                    result = on_token(delta)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # pragma: no cover - defensive
                    logger.exception("on_token callback raised; continuing.")
        finally:
            # Ensure the worker is fully done before returning so we
            # never leak a running thread (and so final_text is bound).
            pass

        final_text = await worker_future
        return final_text

    def _build_user_summary_for_cloud(
        self, *, user_id: str
    ) -> dict[str, Any] | None:
        """Aggregate non-sensitive user metadata for the cloud system prompt.

        The PromptBuilder accepts a ``user_summary`` dict with fields
        like ``session_count``, ``relationship_strength``,
        ``preferred_topics``.  We populate it from the existing user
        model when available; never raw text, only metadata.
        """
        try:
            user_model = self.user_models.get(user_id)
        except Exception:
            user_model = None
        if user_model is None:
            return None
        summary: dict[str, Any] = {}
        try:
            session_count = int(getattr(user_model, "session_count", 0) or 0)
            summary["session_count"] = session_count
        except (TypeError, ValueError):
            pass
        try:
            engagement = float(
                getattr(user_model, "engagement_score", 0.5) or 0.5
            )
            summary["avg_engagement"] = max(0.0, min(1.0, engagement))
        except (TypeError, ValueError):
            pass
        try:
            baseline_strength = float(
                getattr(user_model, "baseline_strength", 0.0) or 0.0
            )
            summary["relationship_strength"] = max(
                0.0, min(1.0, baseline_strength)
            )
        except (TypeError, ValueError):
            pass
        return summary or None

    @staticmethod
    def _adaptation_to_dict(adaptation: Any) -> dict[str, float]:
        """Flatten an ``AdaptationVector`` into a scalar dict.

        The retrieval layer only needs the scalar axes (``formality``,
        ``verbosity``, ``cognitive_load``, ``emotional_tone``,
        ``accessibility``) to bias ranking, so we hoist the
        ``StyleVector`` sub-fields into the same flat namespace the
        WebSocket state-update frame uses.
        """
        out: dict[str, float] = {}
        for k in ("cognitive_load", "emotional_tone", "accessibility"):
            try:
                out[k] = float(getattr(adaptation, k, 0.5))
            except (TypeError, ValueError):
                out[k] = 0.5
        style = getattr(adaptation, "style_mirror", None)
        if style is not None:
            for k in ("formality", "verbosity", "emotionality", "directness"):
                try:
                    out[k] = float(getattr(style, k, 0.5))
                except (TypeError, ValueError):
                    out[k] = 0.5
        return out

    @staticmethod
    def _clean_slm_output(raw: str, *, prompt: str = "") -> str:
        """Turn the word-level SLM's raw token stream into human-readable text.

        The word-level tokenizer stores punctuation and literal ``[``,
        ``sep``, ``]`` tokens as separate vocab entries, then joins them
        with single spaces on decode.  That leaves three classes of
        artefacts we have to scrub before showing the output to a user:

        1. **Literal separators.**  The training corpus embeds ``[SEP]``
           between user / assistant turns, so the model happily emits
           "[ sep ]" in the middle of a reply.  We split on that
           pattern and keep only the first turn.
        2. **Echoed prompt.**  The generator returns ``prompt +
           continuation`` (per its docstring) — the UI already shows
           the user's message, so we trim a leading copy of it.
        3. **Whitespace around punctuation / apostrophes.**  ``that ' s``
           and ``hello !`` are cosmetic: a few targeted regexes restore
           idiomatic spacing without touching semantics.
        """
        import re

        if not raw:
            return ""

        text = raw

        # (1) Split on the turn-boundary marker.  The training format is
        # "[BOS] history [SEP] response [EOS]" and the word-level
        # tokenizer decomposes ``[SEP]`` into the three word tokens ``[``,
        # ``sep``, ``]``.  With dialogue-mode prompting, the decoded
        # sequence looks like:
        #
        #     "<echoed prompt> [ sep ] <response> [ sep ] <continuation>"
        #
        # so we keep everything AFTER the FIRST separator (the response)
        # and stop at the NEXT separator (end of the first response turn).
        sep_pat = r"\[?\s*(?:sep|eos|bos|pad|unk)\s*\]?|\n\s*(?:user|assistant)\s*:"
        parts = re.split(sep_pat, text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            # parts[0] is the echoed prompt; parts[1] is the response.
            text = parts[1]
        else:
            # No separator found — the model likely never emitted a SEP.
            # Fall back to the whole text and rely on the prompt-echo
            # strip below.
            text = parts[0] if parts else ""

        # Drop any surviving bare ``[``/``]`` that leaked through.
        text = re.sub(r"[\[\]]", " ", text)

        # (2) Strip a leading echo of the prompt (still useful in the
        # fallback branch above, or when the model echoes a partial
        # prompt between separators).
        if prompt:
            lower_text = text.lstrip().lower()
            lower_prompt = prompt.strip().lower()
            if lower_text.startswith(lower_prompt):
                text = text.lstrip()[len(lower_prompt):]

        # (3) Fix spacing around punctuation, apostrophes, and quotes.
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"\s+'\s*", "'", text)  # "that ' s" -> "that's"
        text = re.sub(r"(?<=\w)'(?=\w)", "'", text)  # keep contractions tight
        text = re.sub(r"\s+", " ", text).strip()

        # (4) Keep only the first 1–2 sentences to avoid the "tail-continues-
        # with-random-training-phrase" behaviour that a template-trained SLM
        # defaults to.  Split on sentence terminators, then keep up to two
        # non-empty sentences, always keeping the terminator so the reply
        # reads naturally.
        sentences: list[str] = []
        tail = text
        while tail and len(sentences) < 2:
            match = re.search(r"[.!?](?:\s|$)", tail)
            if not match:
                # No terminator — fall back to taking the whole rest.
                sentences.append(tail.strip())
                break
            end = match.end()
            sentence = tail[:end].strip()
            if sentence:
                sentences.append(sentence)
            tail = tail[end:]
        if sentences:
            text = " ".join(sentences).strip()
        else:
            text = text.strip()

        # (5) Capitalise the first letter for a more polished look.
        if text:
            text = text[0].upper() + text[1:]

        # (6) Ensure a terminating punctuation mark — stops the reply
        # from looking mid-sentence when the model bailed without one.
        if text and text[-1] not in ".!?":
            text = text + "."

        # (7) Bail to rule-based fallback if the scrub left us with
        # essentially nothing or just noise.
        if len(text) < 2 or not any(c.isalpha() for c in text):
            return ""
        return text

    @staticmethod
    def _fallback_response(adaptation: Any) -> str:
        """Produce a simple rule-based response when no model is available.

        Selects a response template based on the adaptation vector's
        accessibility and emotional-tone dimensions.

        Args:
            adaptation: :class:`~src.adaptation.types.AdaptationVector`.

        Returns:
            A short, safe response string.
        """
        if adaptation.accessibility > 0.5:
            return "I'm here. How can I help?"
        if adaptation.emotional_tone < 0.3:
            return "I hear you. Would you like to talk about it?"
        return "That's interesting! Tell me more."

    # ----------------------------------------------------------------
    # Safety-classifier whitelist helpers (Phase B fix, 2026-04-25)
    # ----------------------------------------------------------------
    # The char-CNN safety classifier was occasionally firing on benign
    # factoid follow-ups ("who proposed it?", "now back to apple — who
    # founded it?", "when did it fall?"); the audit traced these to
    # 0.66+ unsafe-class probability that crossed the REFUSE_THRESHOLD.
    # Rather than retrain, we add a narrow whitelist over query shapes
    # that are *certainly* benign factoid asks AND a small list of
    # actual safety-trigger keywords.  When the user's text matches
    # the benign pattern AND has no trigger keywords, we downgrade
    # refuse/review back to safe.
    _BENIGN_FACTOID_PATTERNS = (
        re.compile(r"^(?:who|what|where|when|why|how)\b", re.I),
        re.compile(r"\b(?:tell\s+me\s+about|describe|explain)\b", re.I),
        re.compile(r"\b(?:founded|founder|invented|discovered|proposed|created|started)\b", re.I),
        re.compile(r"\b(?:headquartered|based|located)\b", re.I),
        re.compile(r"\b(?:competitors?|owns|acquired|bought)\b", re.I),
        re.compile(r"\b(?:vs|versus|compare)\b", re.I),
        re.compile(r"\b(?:fall|fell|win|won|started|ended)\b", re.I),
        # Phase 14 (2026-04-25) — arithmetic word-form expressions.
        # "two to the power of three" / "99 squared" / "eleven cubed"
        # all match a math pattern, with "power" as a non-trigger word
        # in this context (the safety classifier mis-flags it).
        re.compile(r"\b(?:squared|cubed|to\s+the\s+power)\b", re.I),
        re.compile(r"^\s*\d+\s*[\+\-\*/x×÷%]\s*\d+", re.I),
    )

    _SAFETY_TRIGGER_WORDS = frozenset({
        # physical-harm
        "kill", "murder", "suicide", "self-harm", "stab", "shoot",
        "weapon", "bomb", "explosive", "poison", "harm",
        # illegal
        "hack", "exploit", "phish", "fraud", "evade", "scam",
        # medical
        "dosage", "overdose", "diagnose", "diagnosis", "prescribe",
        "prescription",
    })

    @classmethod
    def _is_benign_factoid_query(cls, text: str) -> bool:
        if not text:
            return False
        cleaned = text.strip().lower()
        # Check at least one benign pattern matches.  Most factoid
        # queries match more than one (e.g. "who founded apple" hits
        # both the WH-question and the founded-keyword pattern).
        matches = sum(
            1 for pat in cls._BENIGN_FACTOID_PATTERNS if pat.search(cleaned)
        )
        return matches >= 1

    @classmethod
    def _has_safety_trigger_word(cls, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        # word-boundary check
        for word in cls._SAFETY_TRIGGER_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", lower):
                return True
        return False

    # Bare-clarification patterns (Phase 14, 2026-04-25).  Each tuple
    # is ``(regex, response_template)``.  Templates are hand-written,
    # deterministic, no SLM calls.  Fired only when there is no entity
    # in the recency stack — i.e. the user's question is genuinely
    # ambiguous in scope.
    _BARE_CLARIFICATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (
            re.compile(
                r"^\s*who\s+(?:founded|created|started|invented|made|built|wrote)\s*\??\s*$",
                re.I,
            ),
            (
                "Founded what? Could you tell me which company, organisation, "
                "or project you mean? I know about Apple, Microsoft, Huawei, "
                "Google, OpenAI, and a handful of others."
            ),
        ),
        (
            re.compile(
                r"^\s*when\s+did\s+(?:it|he|she|they)\s+"
                r"(?:live|happen|start|end|begin|fall|die|begin|rise)\s*\??\s*$",
                re.I,
            ),
            (
                "When did what happen? Which person, country, or event are you "
                "asking about? Once you name it I can give you a date range."
            ),
        ),
        (
            re.compile(
                r"^\s*what\s+did\s+(?:he|she|they|it)\s+"
                r"(?:do|invent|create|discover|make|write|compose|build|find)\s*\??\s*$",
                re.I,
            ),
            (
                "What did who do? I lost the thread — could you name the person "
                "or thing you have in mind?"
            ),
        ),
        (
            re.compile(
                r"^\s*where\s+(?:is|was)\s+(?:it|he|she|they)\s*\??\s*$",
                re.I,
            ),
            (
                "Where is who or what? Tell me which person, place, or "
                "company you mean and I'll point you at it."
            ),
        ),
        (
            re.compile(
                r"^\s*why\s+did\s+(?:it|he|she|they)\s+"
                r"(?:do|happen|fail|fall|matter|leave|come|go)\s*\??\s*$",
                re.I,
            ),
            (
                "Why did what happen? Could you name the person, country, or "
                "event so I know which thread to pick up?"
            ),
        ),
    )

    def _maybe_bare_clarification(
        self,
        *,
        message: str,
        user_id: str,
        session_id: str,
        query_for_retrieval: str | None,
    ) -> str | None:
        """Return a curated clarification template, or None.

        Fires when:
          - The raw user message matches one of the bare-clarification
            patterns (``who founded?``, ``when did he live?``, …).
          - There is NO entity on the recency stack (the upstream
            bare-noun rewriter would otherwise have rewritten it).
          - The coref resolver did not bind a pronoun (``query_for_
            retrieval`` is None — i.e. nothing in scope).

        Returns ``None`` when the question is not bare-clarification-
        shaped or when context is in scope.
        """
        if not message:
            return None
        # If coref / bare-noun rewrite already produced an enriched
        # query, we have an entity in scope and should not clarify.
        if query_for_retrieval and query_for_retrieval.strip() != message.strip():
            return None
        cleaned = message.strip()
        # Match against the curated bare-clarification patterns.
        for pattern, template in self._BARE_CLARIFICATION_PATTERNS:
            if pattern.match(cleaned):
                # Final guard: do we have any entity on the recency
                # stack?  If yes, we shouldn't clarify — the engine's
                # other paths can answer.
                try:
                    snap = self._entity_tracker.snapshot(user_id, session_id)
                except Exception:
                    snap = []
                if snap:
                    # An entity is in scope; the upstream rewriter
                    # should have already enriched the query.  If it
                    # didn't, this clarification template is still
                    # the wrong response — let downstream handle it.
                    return None
                return template
        return None

    @staticmethod
    def _dedupe_sentences(text: str) -> str:
        """Drop near-duplicate sentences from *text*.

        Iter 51 (2026-04-27).  Final-response polish: catches
        retrieval-then-SLM concatenations and KG-overview chains that
        leave a sentence twice (e.g. "Python was founded by Guido in
        1991.  Python was founded in 1991.").  Algorithm:

        1. Split on sentence terminators while preserving them.
        2. For each sentence, compute a content-token bag (lowercase,
           strip stopwords/punctuation, length > 1).
        3. Skip a sentence whose Jaccard overlap with any previously-
           kept sentence is ≥ 0.6 — strong signal of restatement.
        4. Reassemble preserving order.

        Returns *text* unchanged when only one sentence is present,
        when the input is empty, or when no near-duplicates exist.
        """
        import re as _re
        if not text or len(text) < 40:
            return text
        # Split on sentence terminators while preserving them.  We
        # avoid the heavier ``nltk.sent_tokenize`` because it pulls in
        # a model download and adds latency on the hot path.
        parts = _re.split(r"([.!?]\s+)", text)
        # Re-pair each sentence with its terminator.
        sentences: list[str] = []
        i = 0
        while i < len(parts):
            seg = parts[i]
            term = parts[i + 1] if i + 1 < len(parts) else ""
            full = (seg + term).strip()
            if full:
                sentences.append(full)
            i += 2
        if len(sentences) < 2:
            return text
        _STOP = {
            "is", "the", "a", "an", "and", "of", "in", "by", "to",
            "for", "with", "on", "was", "were", "are", "be", "as", "at",
            "from", "or", "it", "its", "that", "this", "these", "those",
            "i", "you", "he", "she", "they", "we", "but", "not", "no",
            "do", "does", "did", "have", "has", "had", "will", "would",
            "can", "could", "should", "may", "might", "must", "into",
            "than", "then", "if", "so", "what", "which", "who", "whom",
        }

        def _tokens(s: str) -> set[str]:
            toks = _re.findall(r"[a-zA-Z][a-zA-Z]+|\d+", s.lower())
            return {t for t in toks if t not in _STOP and len(t) > 1}

        kept: list[str] = []
        kept_token_sets: list[set[str]] = []
        for sent in sentences:
            new_tok = _tokens(sent)
            if not new_tok:
                kept.append(sent)
                kept_token_sets.append(set())
                continue
            duplicate = False
            for prev_tok in kept_token_sets:
                if not prev_tok:
                    continue
                inter = len(new_tok & prev_tok)
                union = len(new_tok | prev_tok)
                jaccard = inter / union if union else 0.0
                # Asymmetric overlap also catches "B is a subset of A".
                cover = inter / max(len(new_tok), 1)
                if jaccard >= 0.6 or cover >= 0.75:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(sent)
                kept_token_sets.append(new_tok)
        if len(kept) == len(sentences):
            return text
        return " ".join(kept)

    def _build_clarifier(
        self,
        *,
        message: str,
        user_id: str,
        session_id: str,
    ) -> str | None:
        """Build a short, polite clarifying question (Phase B.4).

        Returns ``None`` when no useful clarifier can be composed; the
        caller falls through to OOD.  The clarifier prefers, in order:

        1. **Two-entity disambiguation.**  If the entity tracker has
           two recent entities, ask which one the user meant.
        2. **Bare-context disambiguation.**  If exactly one recent
           entity is on the stack, ask whether the user meant that.
        3. **Generic "say more".**  When no entities are available,
           ask the user to expand.
        """
        cleaned = (message or "").strip()
        if not cleaned:
            return None
        try:
            snap = self._entity_tracker.snapshot(user_id, session_id)
        except Exception:
            snap = []
        canonicals: list[str] = []
        seen: set[str] = set()
        for f in snap:
            if f.kind not in {"org", "topic", "person"}:
                continue
            if f.canonical in seen:
                continue
            seen.add(f.canonical)
            canonicals.append(f.canonical)
            if len(canonicals) >= 2:
                break

        # Title-case the entity names for display, falling back to the
        # KG override map when available.
        try:
            from i3.dialogue.knowledge_graph import KnowledgeGraph
            display_map = KnowledgeGraph._DISPLAY_OVERRIDES
        except Exception:
            display_map = {}
        def _disp(c: str) -> str:
            return display_map.get(c, c.title())

        if len(canonicals) >= 2:
            return (
                f"I'm not sure if you mean {_disp(canonicals[0])} or "
                f"{_disp(canonicals[1])} — could you clarify?"
            )
        if len(canonicals) == 1:
            return (
                f"Did you mean {_disp(canonicals[0])}? Could you say a "
                "bit more about what you're looking for?"
            )
        return (
            "I'm not quite sure what you're after — could you say a bit "
            "more about what you'd like to know?"
        )

    # ------------------------------------------------------------------
    # Internal: diary logging (fire-and-forget)
    # ------------------------------------------------------------------

    async def _log_exchange_safe(
        self,
        session_id: str,
        user_state_embedding: torch.Tensor,
        adaptation: Any,
        route_chosen: str,
        latency_ms: float,
        user_id: str,
        message_text: str,
    ) -> None:
        """Log an exchange to the diary store, swallowing any errors.

        This method is designed to be scheduled via
        :func:`asyncio.create_task` so that diary writes do not block the
        response path.  Errors are logged but never propagated to the
        caller.
        """
        try:
            # SEC: Route the embedding through the versioned encryption
            # envelope so that when a ModelEncryptor is configured, the
            # per-exchange state embedding is Fernet-encrypted at rest.
            # Falls back to a plaintext (still versioned) envelope when no
            # encryptor is attached.
            from i3.diary.store import encrypt_embedding_envelope

            embedding_bytes = encrypt_embedding_envelope(
                user_state_embedding, self._encryptor
            )
            adaptation_dict = adaptation.to_dict()

            # Extract lightweight topic keywords (no raw text stored)
            topics = self._extract_topics(message_text)

            engagement = self._previous_engagement.get(user_id, 0.5)

            await self.diary_store.log_exchange(
                session_id=session_id,
                user_state_embedding=embedding_bytes,
                adaptation_vector=adaptation_dict,
                route_chosen=route_chosen,
                response_latency_ms=int(latency_ms),
                engagement_signal=engagement,
                topics=topics,
            )
            logger.debug(
                "Exchange logged for session_id=%s (route=%s, latency=%dms)",
                session_id,
                route_chosen,
                int(latency_ms),
            )
        except Exception:
            logger.exception(
                "Failed to log exchange for session_id=%s (non-fatal).",
                session_id,
            )

    # ------------------------------------------------------------------
    # Internal: TCN encoder helpers
    # ------------------------------------------------------------------

    def _try_load_encoder(self) -> None:
        """Attempt to load the TCN encoder from the default checkpoint path.

        Accepts both the historical ``models/encoder/checkpoint.pt``
        location (pre-refactor) and the current ``checkpoints/encoder/
        best_model.pt`` path emitted by ``training/train_encoder.py``.
        If neither exists the encoder remains ``None`` and
        :meth:`_encode_features` uses a zero embedding.
        """
        from pathlib import Path

        candidates = [
            Path("checkpoints/encoder/best_model.pt"),
            Path("checkpoints/encoder/final_model.pt"),
            Path("models/encoder/checkpoint.pt"),
        ]
        checkpoint_path: Path | None = next((p for p in candidates if p.is_file()), None)
        if checkpoint_path is None:
            logger.info(
                "No encoder checkpoint found (tried %s); using zero embeddings.",
                [str(p) for p in candidates],
            )
            return

        try:
            from i3.encoder.inference import EncoderInference

            self._encoder = EncoderInference(
                checkpoint_path=str(checkpoint_path),
                window_size=self.config.interaction.feature_window,
                input_dim=self._encoder_config.input_dim,
                hidden_dims=self._encoder_config.hidden_dims,
                kernel_size=self._encoder_config.kernel_size,
                dilations=self._encoder_config.dilations,
                embedding_dim=self._encoder_config.embedding_dim,
                dropout=self._encoder_config.dropout,
            )
            logger.info("TCN encoder loaded from %s", checkpoint_path)
        except Exception:
            logger.exception("Failed to load TCN encoder; using zero embeddings.")
            self._encoder = None

    def _build_retriever(self, model: Any, tokenizer: Any) -> None:
        """Build the retrieval index from the training triples.

        The retriever is a lightweight cosine-similarity search over the
        SLM's token embeddings — it lets the pipeline return a known-
        good, grammatical response for common prompts without falling
        through to a 4 M-param autoregressive generator that would
        produce word salad at this scale.  The autoregressive path
        still runs for novel prompts the retriever doesn't match well.
        """
        import json
        from pathlib import Path

        from i3.slm.retrieval import ResponseRetriever

        for candidate in (
            Path("data/processed/dialogue/triples.json"),
            Path("data/dialogue/triples.json"),
        ):
            if candidate.is_file():
                try:
                    triples = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    logger.exception(
                        "Failed to parse retrieval corpus at %s", candidate
                    )
                    return
                # Curated-overlay merge (2026-04-26): load any
                # high-priority Q→A pairs from a small companion file
                # and prepend them so they're indexed alongside the
                # main corpus.  Used to add small-talk + identity
                # answers that the main corpus doesn't cover well
                # ("favorite color", "name three planets", "how do you
                # handle privacy", etc.).  Failures are logged and
                # ignored — the overlay is non-critical.
                overlay_path = candidate.parent / "triples_curated_overlay.json"
                if overlay_path.is_file():
                    try:
                        overlay = json.loads(
                            overlay_path.read_text(encoding="utf-8")
                        )
                        if isinstance(overlay, list) and overlay:
                            triples = list(overlay) + list(triples)
                            logger.info(
                                "Merged %d curated overlay entries from %s",
                                len(overlay), overlay_path,
                            )
                    except Exception:
                        logger.exception(
                            "Failed to merge curated overlay from %s",
                            overlay_path,
                        )
                try:
                    self._slm_retriever = ResponseRetriever(
                        tokenizer=tokenizer,
                        model=model,
                        triples=triples,
                    )
                    logger.info(
                        "Retrieval index built from %s (%d triples)",
                        candidate,
                        len(triples),
                    )
                except Exception:
                    logger.exception(
                        "Failed to build retrieval index from %s", candidate
                    )
                return

        logger.info(
            "No retrieval corpus found — hybrid retrieval disabled. "
            "(Looked for data/processed/dialogue/triples.json.)"
        )

    def _try_load_slm(self) -> None:
        """Attempt to load the trained SLM, preferring v2 over v1.

        Order of preference:

        1. **v2** — :class:`AdaptiveTransformerV2` (204 M, MoE+ACT, BPE
           tokenizer) under ``checkpoints/slm_v2/``.  This is the
           current-generation model and ships with the v2 BPE
           tokenizer at ``checkpoints/slm/tokenizer_bpe.json``.
        2. **v1** — :class:`AdaptiveSLM` (4.5 M params, word-level
           tokenizer) under ``checkpoints/slm/`` — kept as a graceful
           fallback so an unhealthy v2 checkpoint never bricks the
           server.

        If neither set is available the SLM stays unloaded and
        ``_generate_response`` falls back to the rule-based responder.
        """
        from pathlib import Path

        # --- v2 first -------------------------------------------------
        v2_paths = [
            Path("checkpoints/slm_v2/best_model.pt"),
            Path("checkpoints/slm_v2/final_model.pt"),
        ]
        v2_tokenizer_path = Path("checkpoints/slm/tokenizer_bpe.json")
        if v2_tokenizer_path.is_file():
            for p in v2_paths:
                if not p.is_file():
                    continue
                try:
                    self._load_slm_v2(p, v2_tokenizer_path)
                except Exception:
                    logger.exception(
                        "v2 SLM load failed at %s — trying next v2 candidate.",
                        p,
                    )
                    self._slm_generator = None
                    continue
                if self._slm_generator is not None:
                    return  # v2 loaded successfully
        else:
            logger.info(
                "v2 BPE tokenizer not found at %s — skipping v2 path.",
                v2_tokenizer_path,
            )

        # --- v1 fallback ----------------------------------------------
        self._try_load_slm_v1()

    def _try_load_slm_v1(self) -> None:
        """Legacy v1 (AdaptiveSLM + word-level tokenizer) loader."""
        from pathlib import Path

        model_candidates = [
            Path("checkpoints/slm/best_model.pt"),
            Path("checkpoints/slm/final_model.pt"),
        ]
        tokenizer_path = Path("checkpoints/slm/tokenizer.json")

        model_path = next((p for p in model_candidates if p.is_file()), None)
        if model_path is None:
            logger.info(
                "No SLM checkpoint found (tried %s); "
                "responses will use the rule-based fallback until training runs.",
                [str(p) for p in model_candidates],
            )
            return
        if not tokenizer_path.is_file():
            logger.warning(
                "SLM checkpoint at %s present but tokenizer %s is missing; "
                "responses will use the rule-based fallback.",
                model_path,
                tokenizer_path,
            )
            return

        # ``load_slm`` handles its own exceptions and logs a Traceback
        # into the server log if the architecture fails to match the
        # checkpoint, so the caller does not need a second try/except.
        self.load_slm(str(model_path), str(tokenizer_path))

    # ------------------------------------------------------------------
    # v2 SLM loader
    # ------------------------------------------------------------------

    def _load_slm_v2(self, checkpoint_path: Any, tokenizer_path: Any) -> None:
        """Load the v2 :class:`AdaptiveTransformerV2` from a v2 checkpoint.

        v2 specifics handled here (vs the v1 ``load_slm``):

        * Tokenizer: ``BPETokenizer.load(path)`` — byte-level BPE with
          a 32 k vocab and the special-id layout PAD=0, UNK=1, BOS=2,
          EOS=3, SEP=4.  The :class:`SLMGenerator` already detects this
          flavour and dispatches encode/SEP injection accordingly.
        * Architecture: pulled from the checkpoint's
          ``config['model']`` block (matches the
          :class:`AdaptiveTransformerV2Config` schema), so the
          server's static config is *ignored* — the saved
          dimensions are authoritative.
        * State-dict prefix: the trainer wraps the model in
          :class:`_GradCkptTransformerV2`, but ``state_dict()``
          produces flat keys (because the wrapper does not register
          extra submodules), so no prefix-stripping is required for
          checkpoints saved by the v2 trainer.  The defensive strip
          below covers any hand-edited dump that might add one.
        * Device: GPU if available.  v2 fits comfortably under fp32 +
          inference (~1.6 GB VRAM at d_model=768 / n_layers=12).
        * Retrieval: rebuilt + cached to
          ``checkpoints/slm_v2/retrieval_embeddings.pt`` to amortise
          the ~2 min CPU-side build over restarts.
        """
        from pathlib import Path

        from i3.slm.adaptive_transformer_v2 import (
            AdaptiveTransformerV2,
            AdaptiveTransformerV2Config,
        )
        from i3.slm.bpe_tokenizer import BPETokenizer
        from i3.slm.generate import SLMGenerator

        checkpoint_path = Path(checkpoint_path)
        tokenizer_path = Path(tokenizer_path)

        # --- tokenizer ------------------------------------------------
        tokenizer = BPETokenizer.load(tokenizer_path)
        logger.info(
            "v2 BPE tokenizer loaded: %d tokens, %d merges from %s",
            len(tokenizer), len(tokenizer.merges), tokenizer_path,
        )

        # --- checkpoint -----------------------------------------------
        # weights_only=False is necessary because v2 checkpoints store
        # an ``optimizer_state_dict`` and other non-tensor fields, but
        # we restrict the load surface by only reading ``config`` and
        # ``model_state_dict`` from it.  In production the checkpoint
        # path is local-only (server boot only ever loads files we
        # produced ourselves), so the pickled-eval risk is negligible.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )

        cfg_block = (checkpoint.get("config") or {}).get("model") or {}
        if not cfg_block:
            raise ValueError(
                f"{checkpoint_path}: missing config.model block — not a v2 checkpoint?"
            )
        # Drop any extra fields not part of the v2 config dataclass so
        # an old checkpoint with an extra field doesn't crash __init__.
        valid_keys = set(AdaptiveTransformerV2Config().__dict__.keys())
        cfg_kwargs = {k: v for k, v in cfg_block.items() if k in valid_keys}
        model_cfg = AdaptiveTransformerV2Config(**cfg_kwargs)

        model = AdaptiveTransformerV2(config=model_cfg)

        # --- load weights (with optional wrapper-prefix strip) --------
        sd = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in sd):
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
        # Defensive: tied output_projection weights are sometimes saved
        # twice in newer torch versions; if missing, skip + tie below.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            logger.warning(
                "v2 load: %d unexpected keys (sample=%s)",
                len(unexpected), unexpected[:3],
            )
        # Re-tie weights if load wiped the alias (PyTorch quirk).
        if model_cfg.tie_weights:
            model.output_projection.weight = (
                model.embedding.token_embedding.embedding.weight
            )
        if missing:
            critical = [k for k in missing if "output_projection" not in k]
            if critical:
                raise RuntimeError(
                    f"v2 load: {len(critical)} missing keys (sample={critical[:3]})"
                )

        model = model.to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        eval_loss = float(checkpoint.get("eval_loss") or float("nan"))
        eval_ppl = math.exp(min(eval_loss, 20.0)) if eval_loss == eval_loss else float("nan")
        step = int(checkpoint.get("step") or 0)
        logger.info(
            "SLM v2 loaded: %d params (%.1f M), step=%d, eval_loss=%.4f, eval_ppl=%.2f, "
            "device=%s, checkpoint=%s",
            n_params, n_params / 1e6, step, eval_loss, eval_ppl, device, checkpoint_path,
        )

        # --- generator -------------------------------------------------
        # ``device="auto"`` lets pick_device promote to CUDA; we pass it
        # the explicit device string so behaviour is deterministic.
        self._slm_generator = SLMGenerator(
            model, tokenizer, device=str(device),
        )
        self._slm_version = "v2"

        # --- retrieval (with on-disk embeddings cache) -----------------
        self._build_retriever_v2(model, tokenizer, checkpoint_path.parent)

        # --- prompt-complexity tokenizer ------------------------------
        try:
            self._prompt_complexity_estimator.tokenizer = tokenizer
        except Exception:
            logger.debug(
                "Prompt-complexity estimator does not accept tokenizer assignment; skipping."
            )

    def _build_retriever_v2(
        self, model: Any, tokenizer: Any, checkpoint_dir: Any,
    ) -> None:
        """Build the v2 retrieval index, with an embedding-matrix cache.

        Mirrors :meth:`_build_retriever` but threads through an explicit
        ``embeddings_cache_path`` argument so subsequent restarts skip
        the ~2 min CPU rebuild of the [N, 768] embedding matrix.
        """
        import json
        from pathlib import Path

        from i3.slm.retrieval import ResponseRetriever

        cache_path = Path(checkpoint_dir) / "retrieval_embeddings.pt"

        for candidate in (
            Path("data/processed/dialogue/triples.json"),
            Path("data/dialogue/triples.json"),
        ):
            if not candidate.is_file():
                continue
            try:
                triples = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                logger.exception(
                    "v2 retrieval: failed to parse %s", candidate,
                )
                return
            # Curated-overlay merge (2026-04-26): high-priority Q→A
            # pairs that the main corpus doesn't cover well — small-
            # talk, identity, system-meta queries.  Failures are
            # logged and ignored.
            overlay_path = candidate.parent / "triples_curated_overlay.json"
            if overlay_path.is_file():
                try:
                    overlay = json.loads(
                        overlay_path.read_text(encoding="utf-8")
                    )
                    if isinstance(overlay, list) and overlay:
                        triples = list(overlay) + list(triples)
                        logger.info(
                            "v2 retrieval: merged %d curated overlay entries "
                            "from %s",
                            len(overlay), overlay_path,
                        )
                except Exception:
                    logger.exception(
                        "v2 retrieval: failed to merge overlay %s",
                        overlay_path,
                    )
            try:
                self._slm_retriever = ResponseRetriever(
                    tokenizer=tokenizer,
                    model=model,
                    triples=triples,
                    embeddings_cache_path=cache_path,
                )
                logger.info(
                    "v2 retrieval index built from %s (%d triples, cache=%s)",
                    candidate, len(triples), cache_path,
                )
            except Exception:
                logger.exception(
                    "v2 retrieval: failed to build from %s", candidate,
                )
            return

        logger.info(
            "v2 retrieval: no corpus found — hybrid retrieval disabled."
        )

    def _encode_features(
        self, feature_window: list[Any]
    ) -> torch.Tensor:
        """Encode the feature window into a 64-dim user-state embedding.

        If the encoder is not available (no checkpoint loaded), returns a
        zero tensor as a graceful fallback.

        Args:
            feature_window: List of
                :class:`~src.interaction.types.InteractionFeatureVector`
                instances.

        Returns:
            A 1-D ``torch.Tensor`` of shape ``[64]``.
        """
        if self._encoder is not None and feature_window:
            try:
                return self._encoder.encode(feature_window)
            except Exception:
                logger.exception(
                    "Encoder inference failed; returning zero embedding."
                )

        return torch.zeros(self._encoder_config.embedding_dim)

    def _encode_multimodal_features(
        self,
        key_emb: torch.Tensor,
        prosody_payload: dict | None,
        gaze_payload: dict | None = None,
    ) -> dict:
        """Fuse the keystroke embedding with optional voice-prosody +
        gaze features into a 128-d multimodal user-state vector.

        Pure-CPU forward pass (the prosody MLP + gaze MLP + fusion are
        <25k params combined; the heavy MobileNetV3 backbone runs at
        inference time inside :class:`GazeClassifier`, not here).
        Always returns a dict so the WS layer can ship it verbatim:

        * ``prosody_active`` — ``True`` when a validated prosody
          payload was supplied and fused.
        * ``gaze_active`` — ``True`` when a validated gaze payload
          was supplied and fused.
        * ``fused_dim`` — the 128-d embedding dim, surfaced for the
          UI chip text (or 96-d if the legacy two-modality fusion is
          configured).
        * ``samples_count`` / ``captured_seconds`` — bookkeeping from
          the JS extractor.
        * ``feature_summary`` — prosody scalars rounded to 3dp.
        * ``gaze_summary`` — gaze scalars (top label + confidence)
          for the reasoning trace.

        Wrapped in a top-level try/except so a fusion failure can
        never block the response.
        """
        try:
            from i3.multimodal.prosody import (
                validate_gaze_payload,
                validate_prosody_payload,
            )

            feats = validate_prosody_payload(prosody_payload)
            gaze_dict = validate_gaze_payload(gaze_payload)

            with torch.no_grad():
                prosody_emb = (
                    self._prosody_encoder(feats.to_tensor())
                    if feats is not None
                    else None
                )
                if gaze_dict is not None:
                    gaze_tensor = torch.tensor(
                        [
                            gaze_dict["p_at_screen"],
                            gaze_dict["p_away_left"],
                            gaze_dict["p_away_right"],
                            gaze_dict["p_away_other"],
                            gaze_dict["presence"],
                            gaze_dict["blink_rate_norm"],
                            gaze_dict["head_stability"],
                            gaze_dict["confidence"],
                        ],
                        dtype=torch.float32,
                    )
                    gaze_emb = self._gaze_encoder(gaze_tensor)
                else:
                    gaze_emb = None

                fused = self._multimodal_fusion(
                    key_emb.detach(), prosody_emb, gaze_emb,
                )

            result: dict = {
                "prosody_active": feats is not None,
                "gaze_active": gaze_dict is not None,
                "fused_dim": int(fused.shape[-1]),
                "samples_count": int(feats.samples_count) if feats else 0,
                "captured_seconds": (
                    float(feats.captured_seconds) if feats else 0.0
                ),
                "feature_summary": (
                    {
                        "speech_rate_wpm_norm": round(
                            feats.speech_rate_wpm_norm, 3
                        ),
                        "pitch_mean_norm": round(feats.pitch_mean_norm, 3),
                        "pitch_variance_norm": round(
                            feats.pitch_variance_norm, 3
                        ),
                        "energy_mean_norm": round(feats.energy_mean_norm, 3),
                        "energy_variance_norm": round(
                            feats.energy_variance_norm, 3
                        ),
                        "voiced_ratio": round(feats.voiced_ratio, 3),
                        "pause_density": round(feats.pause_density, 3),
                        "spectral_centroid_norm": round(
                            feats.spectral_centroid_norm, 3
                        ),
                    }
                    if feats is not None
                    else {}
                ),
            }
            if gaze_dict is not None:
                # Pull the top label + confidence from the source
                # payload (the validator dropped the strings).
                top_label = ""
                top_conf = 0.0
                if isinstance(gaze_payload, dict):
                    top_label = str(gaze_payload.get("label", "") or "")
                    try:
                        top_conf = float(gaze_payload.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        top_conf = 0.0
                result["gaze_summary"] = {
                    "label": top_label,
                    "confidence": round(top_conf, 3),
                    "presence": bool(gaze_dict["presence"] >= 0.5),
                    "blink_rate_norm": round(gaze_dict["blink_rate_norm"], 3),
                    "head_stability": round(gaze_dict["head_stability"], 3),
                }
            return result
        except Exception:  # pragma: no cover - decorative
            logger.exception(
                "Multimodal fusion failed; falling back to "
                "keystroke-only embedding."
            )
            return {
                "prosody_active": False,
                "gaze_active": False,
                "fused_dim": int(self._encoder_config.embedding_dim) + 32 + 32,
                "samples_count": 0,
                "captured_seconds": 0.0,
                "feature_summary": {},
            }

    def _project_2d(
        self, embedding: torch.Tensor
    ) -> tuple[float, float]:
        """Project a 64-dim embedding to 2-D for visualisation.

        Delegates to the encoder's :meth:`project_2d` method if
        available, otherwise uses a deterministic random projection.

        Args:
            embedding: 1-D tensor of shape ``[64]``.

        Returns:
            A ``(x, y)`` coordinate tuple.
        """
        if self._encoder is not None:
            try:
                return self._encoder.project_2d(embedding)
            except Exception:
                logger.exception("2D projection failed; using default.")

        # Fallback: first two components
        return (float(embedding[0]), float(embedding[1]))

    # ------------------------------------------------------------------
    # Internal: user model management
    # ------------------------------------------------------------------

    def _get_or_create_user_model(self, user_id: str) -> Any:
        """Retrieve or create the :class:`~src.user_model.model.UserModel`
        for the given user.

        SEC: This sync variant is retained for non-concurrent call sites
        (start_session/end_session, which run inside a single FastAPI
        request and are not racing themselves). The async-safe variant
        :meth:`_aget_or_create_user_model` is used by
        :meth:`process_message`, where two concurrent first-time messages
        for the same new user could otherwise create duplicate models.

        Args:
            user_id: Unique user identifier.

        Returns:
            The user's :class:`UserModel` instance.
        """
        if user_id not in self.user_models:
            from i3.user_model.model import UserModel

            self.user_models[user_id] = UserModel(
                user_id=user_id,
                config=self.config.user_model,
            )
            self._evict_user_models_if_over_cap()
            logger.debug("Created new UserModel for user_id=%s", user_id)
        else:
            # LRU touch so the model stays at the "recently used" end.
            self.user_models.move_to_end(user_id)

        return self.user_models[user_id]

    # ------------------------------------------------------------------
    # Internal: session conversation history
    # ------------------------------------------------------------------

    @staticmethod
    def _history_key(user_id: str, session_id: str) -> str:
        """Compose the ``OrderedDict`` key for the session history map."""
        return f"{user_id}::{session_id}"

    def _get_history_pairs(
        self, user_id: str, session_id: str
    ) -> list[tuple[str, str]]:
        """Return the stored ``(user, assistant)`` pairs for a session.

        LRU-touches the entry on access (so freshly-used sessions are not
        the first to be evicted under pressure).  Returns an empty list
        when no history has been recorded yet — that branch is the
        common case for the very first message of a session.
        """
        if not user_id or not session_id:
            return []
        key = self._history_key(user_id, session_id)
        pairs = self._session_histories.get(key)
        if pairs is None:
            return []
        # LRU touch.
        self._session_histories.move_to_end(key)
        return pairs

    def _format_history(self, pairs: list[tuple[str, str]]) -> str:
        """Render history pairs into the trainer's ``[SEP]``-joined format.

        The on-device SLM was trained on ``history [SEP] response [EOS]``
        triples, so prepending older turns in the same shape keeps the
        decoder inside its training distribution. The retriever's
        embedding head reuses the same string for the cosine path so
        contextual queries ("what about animals?") match against the
        topic vocabulary already in play.
        """
        if not pairs:
            return ""
        chunks: list[str] = []
        for user_text, assistant_text in pairs:
            u = (user_text or "").strip()
            a = (assistant_text or "").strip()
            if u and a:
                chunks.append(f"{u}\n[SEP]\n{a}")
            elif u:
                chunks.append(u)
        return "\n[SEP]\n".join(chunks)

    @staticmethod
    def _truncate_history_text(text: str, max_words: int = 256) -> str:
        """Cap the formatted history to ``max_words`` whitespace tokens.

        The SLM's positional encoding caps total context length, so we
        keep a coarse word budget for the prepended history. Truncation
        is from the *front* (oldest first) so the most recent turn is
        always preserved verbatim when it fits.
        """
        if not text:
            return text
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[-max_words:])

    def _append_history_pair(
        self, user_id: str, session_id: str, user_text: str, assistant_text: str
    ) -> None:
        """Append a finalised exchange to the session's history buffer.

        Called *after* all post-processing so the stored assistant turn
        matches what the user actually saw on screen. Trims the buffer
        to ``self._history_max_turns`` entries and evicts oldest sessions
        if the total session-count would exceed the LRU cap.
        """
        if not user_id or not session_id:
            return
        if not (user_text or assistant_text):
            return
        key = self._history_key(user_id, session_id)
        pairs = self._session_histories.get(key)
        if pairs is None:
            pairs = []
            self._session_histories[key] = pairs
        else:
            self._session_histories.move_to_end(key)
        pairs.append((user_text or "", assistant_text or ""))
        if len(pairs) > self._history_max_turns:
            # Drop the oldest pair(s) so we never exceed the cap.
            del pairs[: len(pairs) - self._history_max_turns]
        self._evict_session_histories_if_over_cap()

    def _evict_session_histories_if_over_cap(self) -> None:
        """LRU-evict oldest session-history entries past the cap."""
        while len(self._session_histories) > self._max_sessions_tracked:
            evicted_key, _ = self._session_histories.popitem(last=False)
            self._last_history_turns_used.pop(evicted_key, None)

    def get_session_history_length(self, user_id: str, session_id: str) -> int:
        """Return the number of stored exchanges for a session.

        Used by the WS layer to surface "working from N prior turns of
        context" in the reasoning trace. Returns 0 when no history has
        been recorded yet (first turn of a session, or after
        ``end_session`` cleared the buffer).
        """
        return len(self._get_history_pairs(user_id, session_id))

    def get_last_history_turns_used(
        self, user_id: str, session_id: str
    ) -> int:
        """Return the history depth that fed the most recent turn.

        Differs from :meth:`get_session_history_length` after a turn has
        completed — that method counts what's stored *now* (including
        the just-finished exchange), while this returns what was
        *consumed* by the retriever / SLM on the last call. The WS
        handler reads this so the reasoning trace narrates the
        contextual prompt accurately.
        """
        if not user_id or not session_id:
            return 0
        key = self._history_key(user_id, session_id)
        return int(self._last_history_turns_used.get(key, 0))

    def get_last_coreference(
        self, user_id: str, session_id: str
    ) -> dict | None:
        """Return the most-recent co-reference resolution dict, or ``None``.

        Wraps the stashed :class:`ResolutionResult` via
        :func:`i3.dialogue.coref.resolution_to_dict` so the caller
        gets a JSON-safe dict (or ``None`` when the last turn had no
        pronoun).  Used by the WS layer to surface the
        ``coref · they → huawei`` chip and to narrate the resolution
        in the reasoning trace.
        """
        if not user_id or not session_id:
            return None
        try:
            from i3.dialogue.coref import resolution_to_dict
            res = self._last_coref.get(self._history_key(user_id, session_id))
            return resolution_to_dict(res)
        except Exception:  # pragma: no cover - defensive
            return None

    # ----------------------------------------------------------------
    # Retrieval-floor knob (Phase B.1) — exposed so the playground tab
    # can demonstrate routing sensitivity.  Bounded to [0, 1] so a
    # caller can't silently disable the floor by passing a negative.
    # ----------------------------------------------------------------
    def set_retrieval_floor(self, floor: float) -> float:
        """Update the retrieval-vs-SLM commit floor at runtime.

        Returns the clamped value actually applied.  ``floor`` is the
        cosine threshold above which retrieval wins on a substantive
        (≥4-content-word) query.  Defaults are calibrated as:
            0.92  always-commit (any query length)
            0.85  short-query commit
            0.75  substantive-query commit floor (this knob)
        """
        try:
            f = float(floor)
        except (TypeError, ValueError):
            return float(self._retrieval_floor)
        f = max(0.0, min(1.0, f))
        self._retrieval_floor = f
        logger.info("set_retrieval_floor: floor=%.2f", f)
        return f

    def get_retrieval_floor(self) -> float:
        return float(getattr(self, "_retrieval_floor", 0.75))

    def get_last_critique(self) -> dict:
        """Return the most recent self-critique trace, or ``{}`` if none.

        Populated by :meth:`_generate_response_inner` on every SLM-path
        turn.  The dict shape is documented on
        :attr:`PipelineOutput.critique`.  Returns an empty dict (not
        ``None``) when the last turn never invoked the critic
        (retrieval / tool / OOD paths) or when the critic raised — the
        caller is expected to treat falsy as "no trace available".
        """
        return dict(getattr(self, "_last_critique", {}) or {})

    def _drop_session_history(self, user_id: str, session_id: str) -> None:
        """Forget a session's history buffer (called on ``end_session``)."""
        if not user_id or not session_id:
            return
        key = self._history_key(user_id, session_id)
        self._session_histories.pop(key, None)
        self._last_history_turns_used.pop(key, None)

    def _evict_user_models_if_over_cap(self) -> None:
        """Drop oldest ``UserModel`` entries if the cap is exceeded.

        See :data:`_max_users` / ``I3_MAX_TRACKED_USERS`` — evicting is
        O(1) per entry via ``OrderedDict.popitem``.  Also clears the
        matching engagement dicts so the user's footprint is fully
        collected.
        """
        while len(self.user_models) > self._max_users:
            evicted_id, _ = self.user_models.popitem(last=False)
            self._last_response_time.pop(evicted_id, None)
            self._last_response_length.pop(evicted_id, None)
            self._previous_engagement.pop(evicted_id, None)
            self._previous_route.pop(evicted_id, None)
            self._previous_routing_context.pop(evicted_id, None)
            logger.info(
                "user_model.evicted",
                extra={"event": "user_model_evicted", "user_id": evicted_id},
            )

    async def _aget_or_create_user_model(self, user_id: str) -> Any:
        """Async-locked variant of :meth:`_get_or_create_user_model`.

        SEC: Guards self.user_models mutation with self._user_models_lock
        so concurrent process_message() calls for a brand-new user cannot
        race and clobber each other's UserModel instance.
        """
        # Fast path: model exists, no lock needed (dict reads are
        # atomic under the GIL).
        existing = self.user_models.get(user_id)
        if existing is not None:
            return existing

        async with self._user_models_lock:
            # Re-check inside the lock to avoid the lost-update race.
            existing = self.user_models.get(user_id)
            if existing is not None:
                return existing

            from i3.user_model.model import UserModel

            user_model = UserModel(
                user_id=user_id,
                config=self.config.user_model,
            )
            self.user_models[user_id] = user_model
            self._evict_user_models_if_over_cap()
            logger.debug("Created new UserModel for user_id=%s", user_id)
            return user_model

    def _background_task_done(self, task: asyncio.Task[Any]) -> None:
        """Done-callback for fire-and-forget background tasks.

        SEC: Removes the task from the tracking set (so it can be GC'd)
        and logs any exception that escaped the task body. Without this
        callback, exceptions in unawaited tasks become silent because
        asyncio's default ``Task.exception()`` is only consulted when the
        task is awaited or its result is requested.
        """
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Background pipeline task failed: %s",
                exc,
                exc_info=exc,
            )

    def _build_error_output(
        self, latency_ms: float, exc: Exception
    ) -> PipelineOutput:
        """Build a degraded :class:`PipelineOutput` after an exception.

        SEC: Used by the outer try/except in :meth:`process_message` to
        guarantee that callers always receive a serialisable response,
        never ``None`` and never an unhandled exception. The error string
        is the exception class name only -- not the message -- to avoid
        accidentally leaking sanitiser internals or PII.
        """
        arms = self.config.router.arms
        confidence = dict.fromkeys(arms, 0.0)
        if "local_slm" in confidence:
            confidence["local_slm"] = 1.0
        return PipelineOutput(
            response_text=self._fallback_response_for_error(),
            route_chosen="error_fallback",
            latency_ms=round(latency_ms, 2),
            user_state_embedding_2d=(0.0, 0.0),
            adaptation={
                "cognitive_load": 0.5,
                "emotional_tone": 0.5,
                "accessibility": 0.0,
                # SEC (H-7, 2026-04-23 audit): never leak the Python
                # exception class name to the wire.  Keep the identity
                # in the structured log instead (already emitted by the
                # caller via logger.exception).
                "error": "pipeline_error",
            },
            engagement_score=0.0,
            deviation_from_baseline=0.0,
            routing_confidence=confidence,
            messages_in_session=0,
            baseline_established=False,
        )

    @staticmethod
    def _fallback_response_for_error() -> str:
        """Static safe response shown when the pipeline hits an unexpected error."""
        return (
            "I'm having trouble processing that just now. "
            "Could you try again in a moment?"
        )

    # ------------------------------------------------------------------
    # Internal: utility helpers
    # ------------------------------------------------------------------

    def _compress_state(self, embedding: torch.Tensor) -> list[float]:
        """Compress a 64-dim embedding to 4-dim for the routing context.

        Uses the first four components as a simple dimensionality
        reduction.  For production, replace with PCA fitted on a
        reference set of user-state embeddings.

        SEC: Always returns *exactly* 4 floats. If the input embedding
        is shorter than 4 (e.g. mocked or degenerate fallback), the
        result is right-padded with zeros so RoutingContext.to_vector()
        sees a stable shape.

        Args:
            embedding: 1-D tensor of shape ``[64]``.

        Returns:
            A 4-element list of floats.
        """
        try:
            values = embedding[:4].tolist()
        except Exception:
            logger.exception(
                "_compress_state failed to slice embedding; using zeros."
            )
            values = []
        if len(values) < 4:
            values = values + [0.0] * (4 - len(values))
        result: list[float] = [float(v) for v in values[:4]]
        return result

    @staticmethod
    def _normalized_hour() -> float:
        """Return the current hour of day normalised to [0, 1).

        ``0.0`` corresponds to midnight, ``0.5`` to noon.
        """
        return datetime.now(timezone.utc).hour / 24.0

    @staticmethod
    def _extract_topics(text: str, max_topics: int = 5) -> list[str]:
        """Extract lightweight topic keywords from message text.

        Uses a simple heuristic: take the longest unique words that are
        at least 4 characters.  No raw text is stored -- only keywords.

        Args:
            text: The user's message text.
            max_topics: Maximum number of topic keywords to return.

        Returns:
            A list of keyword strings (lowercase, deduplicated).
        """
        words = text.lower().split()
        # Filter to content words (length >= 4, alphabetic only)
        content_words = [
            w.strip(".,;:!?()[]{}\"'")
            for w in words
            if len(w.strip(".,;:!?()[]{}\"'")) >= 4
            and w.strip(".,;:!?()[]{}\"'").isalpha()
        ]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for w in content_words:
            if w not in seen:
                seen.add(w)
                unique.append(w)

        # SEC: Sort by (-length, word) to make ordering deterministic
        # for words of equal length. Tests rely on stable topic output.
        unique.sort(key=lambda w: (-len(w), w))
        return unique[:max_topics]

    @staticmethod
    def _build_fallback_summary(session_summary: dict[str, Any]) -> str:
        """Build a simple template-based session summary.

        Used when the cloud LLM is unavailable for summary generation.

        Args:
            session_summary: Dict from
                :meth:`~src.user_model.model.UserModel.end_session`.

        Returns:
            A one-sentence natural-language summary string.
        """
        msg_count = session_summary.get("message_count", 0)
        duration = session_summary.get("duration_seconds", 0)
        engagement = session_summary.get("mean_engagement", 0.0)

        minutes = max(1, int(duration / 60))

        if engagement > 0.7:
            tone = "highly engaged"
        elif engagement > 0.4:
            tone = "moderately engaged"
        else:
            tone = "briefly engaged"

        return (
            f"A {tone} session lasting {minutes} minute(s) with "
            f"{msg_count} message(s) exchanged."
        )

    def _ensure_initialized(self) -> None:
        """Raise :class:`RuntimeError` if the pipeline is not initialised."""
        if not self._initialized:
            raise RuntimeError(
                "Pipeline has not been initialised. "
                "Call `await pipeline.initialize()` first."
            )

    # ------------------------------------------------------------------
    # SLM management (optional)
    # ------------------------------------------------------------------

    def load_slm(self, model_path: str, tokenizer_path: str) -> None:
        """Load a trained SLM checkpoint for local generation.

        This is optional -- if no SLM is loaded, the pipeline will use
        the cloud LLM or fall back to rule-based responses.

        Args:
            model_path: Path to the SLM model checkpoint (``.pt``).
            tokenizer_path: Path to the tokenizer vocabulary file
                (``.json``).
        """
        try:
            from i3.slm.generate import SLMGenerator
            from i3.slm.model import AdaptiveSLM
            from i3.slm.tokenizer import SimpleTokenizer

            tokenizer = SimpleTokenizer.load(tokenizer_path)
            # Security: weights_only=True prevents pickled-object code
            # execution during checkpoint deserialization (CVE class:
            # insecure torch.load).  Inference-time loads must be safe.
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

            # Architecture must match what the checkpoint was saved with,
            # not what the current config says.  The training pipeline
            # prunes vocab_size to the corpus's actual unique-token count
            # (typically << the config ceiling), so instantiating the
            # model with config.slm.vocab_size then load_state_dict()
            # raises a shape mismatch on the token embedding tables.
            # Prefer the checkpoint's own "configs" block when available,
            # then fall back to introspecting the embedding weight shape,
            # then to the static config.
            ckpt_cfg = checkpoint.get("configs") or checkpoint.get("config") or {}
            sd = checkpoint["model_state_dict"]
            # The embedding tensor has moved around between refactors — try
            # a few canonical locations before giving up and trusting the
            # checkpoint's own config block.
            emb_w = None
            for key in (
                "embedding.token_embedding.embedding.weight",
                "token_embedding.embedding.weight",
                "embedding.token_embedding.weight",
                "token_embedding.weight",
            ):
                if key in sd:
                    emb_w = sd[key]
                    break
            if emb_w is not None:
                ckpt_vocab = int(emb_w.shape[0])
                ckpt_d_model = int(emb_w.shape[1])
            else:
                ckpt_vocab = int(ckpt_cfg.get("vocab_size", self.config.slm.vocab_size))
                ckpt_d_model = int(ckpt_cfg.get("d_model", self.config.slm.d_model))

            # Keep tokenizer in sync with the checkpoint's vocab so
            # indices produced by the tokenizer stay within the
            # embedding table's row count.
            if tokenizer.vocab_size != ckpt_vocab:
                logger.info(
                    "Aligning tokenizer.vocab_size %d -> %d (checkpoint)",
                    tokenizer.vocab_size,
                    ckpt_vocab,
                )
                tokenizer.vocab_size = ckpt_vocab

            model = AdaptiveSLM(
                vocab_size=ckpt_vocab,
                max_seq_len=int(ckpt_cfg.get("max_seq_len", self.config.slm.max_seq_len)),
                d_model=ckpt_d_model,
                n_heads=int(ckpt_cfg.get("n_heads", self.config.slm.n_heads)),
                n_layers=int(ckpt_cfg.get("n_layers", self.config.slm.n_layers)),
                d_ff=int(ckpt_cfg.get("d_ff", self.config.slm.d_ff)),
                dropout=float(ckpt_cfg.get("dropout", self.config.slm.dropout)),
                conditioning_dim=int(
                    ckpt_cfg.get("conditioning_dim", self.config.slm.conditioning_dim)
                ),
                adaptation_dim=int(
                    ckpt_cfg.get("adaptation_dim", self.config.slm.adaptation_dim)
                ),
            )
            model.load_state_dict(sd)

            self._slm_generator = SLMGenerator(model, tokenizer)
            self._slm_version = "v1"
            logger.info("SLM v1 loaded from %s", model_path)

            # Build the retrieval index from the training triples so
            # the pipeline can answer common prompts with the exact,
            # known-good response from the corpus before falling back
            # to autoregressive generation.
            self._build_retriever(model, tokenizer)
        except Exception:
            logger.exception("Failed to load SLM from %s", model_path)
            self._slm_generator = None
            self._slm_version = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict[str, Any]:
        """Return a snapshot of pipeline health metrics.

        Useful for monitoring dashboards and debugging.

        Returns:
            A dict with keys: ``initialized``, ``active_users``,
            ``encoder_loaded``, ``slm_loaded``, ``cloud_available``,
            ``diary_ready``, ``router_stats``.
        """
        return {
            "initialized": self._initialized,
            "active_users": len(self.user_models),
            "encoder_loaded": self._encoder is not None,
            "slm_loaded": self._slm_generator is not None,
            "cloud_available": self.cloud_client.is_available,
            "diary_ready": self.diary_store._initialized,
            "router_stats": self.router.get_arm_stats(),
            "cloud_usage": self.cloud_client.usage_stats,
        }
