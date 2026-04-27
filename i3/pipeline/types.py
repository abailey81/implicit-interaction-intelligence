"""Core data types for the Pipeline Orchestration Engine.

Defines the input/output contracts for the pipeline and the
:class:`EngagementSignal` used to compute reward signals for the
contextual bandit router.

All types are plain dataclasses to keep the pipeline layer free of heavy
framework dependencies.  The :class:`PipelineOutput` carries everything
the frontend dashboard needs to render user-state visualisations, routing
confidence bars, and adaptation gauges.

Privacy note
~~~~~~~~~~~~
:class:`PipelineInput` *does* carry raw ``message_text`` because the
pipeline needs it for feature extraction, routing, and generation.
However, raw text is **never persisted** -- only abstract
representations (embeddings, scalar metrics, topic keywords) reach the
diary store.  The pipeline enforces this guarantee at every write
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Pipeline input
# ---------------------------------------------------------------------------

@dataclass
class PipelineInput:
    """Input to the pipeline from a single user message.

    Attributes:
        user_id: Unique user identifier (opaque string).
        session_id: Identifier for the current conversation session.
        message_text: Raw message text submitted by the user.
        timestamp: Unix epoch seconds when the message was submitted.
        composition_time_ms: Total time the user spent composing the
            message (milliseconds), as reported by the client.
        edit_count: Number of edits (cut/paste, undo, etc.) observed
            during composition.
        pause_before_send_ms: Hesitation time between the last keystroke
            and pressing "send" (milliseconds).
        keystroke_timings: List of inter-key interval durations in
            milliseconds, captured from the client WebSocket stream.
    """

    user_id: str
    session_id: str
    message_text: str
    timestamp: float
    composition_time_ms: float
    edit_count: int
    pause_before_send_ms: float
    keystroke_timings: list[float] = field(default_factory=list)

    # Optional voice-prosody features for the multimodal fusion path.
    # When non-``None``, the dict carries the eight scalars listed in
    # :data:`i3.multimodal.PROSODY_FEATURE_KEYS` plus two metadata fields
    # (``samples_count``, ``captured_seconds``).  The browser-side
    # ``VoiceProsodyMonitor`` (``web/js/voice_prosody.js``) extracts these
    # via the WebAudio API and ships them on the ``message`` frame; the
    # WS layer validates the payload via
    # :func:`i3.multimodal.validate_prosody_payload` before constructing
    # the input.  ``None`` means the user did not enable the mic this
    # turn (the default) — the engine then runs its keystroke-only path.
    #
    # Privacy contract: the raw audio buffer NEVER leaves the browser.
    # See the top-of-file docstring of ``i3/multimodal/prosody.py``.
    prosody_features: dict | None = None

    # Optional gaze-classifier features for the vision multimodal path.
    # The browser-side ``GazeCapture`` monitor (``web/js/gaze_capture.js``)
    # captures one webcam frame per ~250 ms, downsamples to a 64×48
    # grayscale fingerprint, and ships the aggregated dict on the
    # ``message`` frame.  Shape:
    # ``{label, confidence, label_probs, presence, blink_rate_norm,
    # head_stability, captured_seconds, samples_count}``.  Validated
    # by :func:`i3.multimodal.validate_gaze_payload`.  ``None`` means
    # the camera was off this turn (the default).
    #
    # Privacy contract: the raw image NEVER leaves the browser as a
    # full-resolution frame.  Only the 64×48 grayscale fingerprint
    # (3072 bytes) is shipped to the server's gaze classifier; that
    # in turn is fine-tuned per-user via in-session calibration.
    # See ``i3/multimodal/gaze_classifier.py`` for the full contract.
    gaze_features: dict | None = None

    # Optional per-turn playground overrides.  When non-``None``, the
    # pipeline applies the supplied keys to bypass / override the
    # corresponding stages (adaptation, biometric, accessibility,
    # route, critique, coref).  Defaults reproduce the normal flow
    # exactly; only the keys the operator explicitly set are honoured.
    # Hard-capped to 100 requests per session by the playground
    # endpoint.  Shape:
    #   {"adaptation": {<AdaptationVector dict>} | None,
    #    "biometric_state": "registered" | "mismatch" | "unregistered",
    #    "accessibility": True | False,
    #    "route": "edge" | "cloud",
    #    "critique": True | False,
    #    "coref": True | False,
    #    "safety": True | False}
    # Never persisted; lives only for the duration of the turn.
    playground_overrides: dict | None = None


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    """Output from the pipeline after processing a single user message.

    Contains the generated response text, the routing decision, timing
    information, and a snapshot of all user-state metrics that the
    frontend dashboard needs for real-time visualisation.

    Attributes:
        response_text: The AI-generated response to the user's message.
        route_chosen: Which generation backend was used -- ``"local_slm"``
            or ``"cloud_llm"``.
        latency_ms: End-to-end pipeline latency in milliseconds.
        user_state_embedding_2d: 2-D projection of the 64-dim user-state
            embedding, suitable for scatter-plot visualisation.
        adaptation: Serialised :class:`~src.adaptation.types.AdaptationVector`
            as a nested dict.
        engagement_score: Current engagement estimate in [0, 1].
        deviation_from_baseline: Cosine distance between the user's
            current state embedding and their long-term baseline.
        routing_confidence: Per-arm selection probabilities from the
            contextual bandit (keys: ``"local_slm"``, ``"cloud_llm"``).
        messages_in_session: Running message count for the active session.
        baseline_established: Whether the user model has accumulated
            enough observations for a reliable baseline.
        diary_entry: Optional diary entry dict, included only when a
            significant event is detected during the exchange.
    """

    response_text: str
    route_chosen: str
    latency_ms: float

    # State update for frontend dashboards
    user_state_embedding_2d: tuple[float, float]
    adaptation: dict
    engagement_score: float
    deviation_from_baseline: float
    routing_confidence: dict[str, float]
    messages_in_session: int
    baseline_established: bool

    # Diary entry (optional, only on significant events)
    diary_entry: dict | None = None

    # Which sub-path of the hybrid local-SLM stack carried the turn.
    # One of ``"retrieval"`` / ``"retrieval_borderline"`` / ``"slm"`` /
    # ``"ood"`` / ``"none"``.  Used by the UI's pipeline-activity ribbon
    # to highlight the correct component.  Optional so legacy callers
    # that construct PipelineOutput directly aren't broken.
    response_path: str = "unknown"

    # Cosine-similarity score from the retrieval layer (0.0–1.0), only
    # meaningful when response_path starts with "retrieval".  The UI
    # shows this as a "confidence" chip next to the reply.
    retrieval_score: float = 0.0

    # Iter 51 phase 9: structured routing decision so the chip can
    # show the user EXACTLY which arm answered and why.
    # Shape::
    #     {
    #         "arm":         "slm+retrieval" | "qwen-lora" |
    #                        "gemini-backup" | "gemini-chat" |
    #                        "diary" | "hostility-guard" | "ood",
    #         "model":       "AdaptiveTransformerV2 (204M)" / etc.,
    #         "query_class": "command" | "system_intro" |
    #                        "cascade_meta" | "world_chat" |
    #                        "default_chat" | "fact" | "hostility",
    #         "reason":      one-sentence human explanation,
    #         "threshold":   "retrieval_score 0.92 ≥ 0.85",
    #     }
    # Empty dict means the pipeline didn't tag the decision.
    route_decision: dict = field(default_factory=dict)

    # Per-axis adaptation rewrites applied to the visible response.
    # Each entry is ``{"axis": ..., "value": ..., "change": ...}`` —
    # e.g. ``{"axis": "formality", "value": "0.74",
    # "change": "expanded contractions"}``.  Empty when adaptation was
    # neutral.  The UI surfaces these as chips beneath the reply so the
    # user can see exactly how their typing reshaped the answer.
    adaptation_changes: list = field(default_factory=list)

    # Mid-conversation affect-shift detection result.  ``None`` when
    # the detector hasn't been wired in (legacy callers) or when the
    # turn produced no shift signal.  When non-``None`` and
    # ``detected=True`` the engine has already appended the
    # ``suggestion`` to the visible ``response_text``.  See
    # :class:`i3.affect.AffectShift` for the full contract.
    affect_shift: Any | None = None

    # Iter 51 (2026-04-27): safety-classifier soft caveat surfaced via
    # a side-channel chip rather than appended to the chat bubble.
    # ``None`` when the safety classifier did not emit a caveat for
    # this turn (the common case).  When a string is set, the
    # frontend renders a small "ⓘ moderation note" pill next to the
    # response without polluting the chat text.  The caveat is the
    # full sentence (e.g., "Do not provide assistance with self-harm,
    # violence, or weapon construction (Constitutional principle:
    # physical-harm).").
    safety_caveat: str | None = None

    # Discrete user-state label produced by the state classifier
    # (Live State Badge feature).  Populated as a JSON-safe dict
    # with keys ``state``, ``confidence``, ``secondary_state``,
    # ``contributing_signals``.  ``None`` when the classifier is
    # not wired into the pipeline (legacy callers).  See
    # :func:`i3.affect.classify_user_state` for the full contract.
    user_state_label: dict | None = None

    # Accessibility-mode auto-switch state produced by the
    # :class:`i3.affect.AccessibilityController` (Accessibility Mode
    # feature).  Populated as a JSON-safe dict with the
    # :class:`AccessibilityModeState` field set; ``None`` when the
    # controller isn't wired in.  When ``active=True`` the engine has
    # already pushed the adaptation knobs harder so the post-processor
    # produces a maximally-trimmed, simple-vocabulary reply.
    accessibility: dict | None = None

    # Continuous typing-biometric authentication result produced by
    # :class:`i3.biometric.KeystrokeAuthenticator` (the I3 headline
    # feature -- Identity Lock).  Serialised as a plain dict with the
    # :class:`BiometricMatch` fields set; ``None`` when the
    # authenticator isn't wired in (legacy callers).  The WS layer
    # ships this on every response/state_update frame so the front-end
    # Identity Lock badge can paint live.
    biometric: dict | None = None

    # Co-reference resolution result produced by
    # :class:`i3.dialogue.EntityTracker` on every turn.  ``None`` when
    # the resolver isn't wired in (legacy callers) OR when the user's
    # message contained no pronoun / referring expression that could
    # be resolved.  When non-``None`` the WS layer surfaces the chip
    # ``coref · they → huawei`` and the reasoning-trace narrates
    # ``"You said 'where are they located?'; I resolved 'they' to
    # 'Huawei' (most recent ORG mentioned 1 turn ago) before
    # retrieval."``  Shape: ``{"original_query": str,
    # "resolved_query": str, "used_entity": {"text", "canonical",
    # "kind", "last_turn_idx"}, "used_pronoun": str,
    # "confidence": float, "reasoning": str}``.
    coreference_resolution: dict | None = None

    # Per-biometric LoRA personalisation status produced by
    # :class:`i3.personalisation.PersonalisationManager` (the FLAGSHIP
    # novelty -- see i3.personalisation.lora_adapter for the design).
    # Shape: ``{"applied": bool, "drift": {axis: float}, "n_updates":
    # int, "user_key": str | None, "num_parameters": int, "rank": int,
    # "reason": str}``.  ``applied=False`` means the user's
    # biometric template isn't registered yet (or the residual is
    # exactly zero on a fresh adapter); ``applied=True`` means the
    # personalised residual was layered onto the base AdaptationVector
    # before downstream rewriting.
    personalisation: dict | None = None

    # Self-critique loop trace produced by :class:`i3.critique.SelfCritic`
    # on the SLM-generation path (Phase 7 HMI pitch piece).  Populated
    # only when ``response_path == "slm"``; ``None`` for retrieval /
    # tool / OOD turns since the critic doesn't run there.  Shape:
    # ``{"final_score": float, "accepted": bool, "regenerated": bool,
    # "rejected": bool, "threshold": float,
    # "attempts": [{"text": str, "score": float, "sub_scores": {..},
    # "reasons": [..], "sampling_params": {..}}, ...]}``.  The WS layer
    # ships this on every response/response_done frame so the chat UI
    # can render a "self-critique" chip + an expandable trace under the
    # message.
    critique: dict | None = None

    # Voice-prosody multimodal fusion status produced by the
    # :class:`i3.multimodal.MultimodalFusion` head when the browser
    # ships ``prosody_features`` on a turn.  Shape:
    # ``{"prosody_active": bool, "gaze_active": bool, "fused_dim": int,
    # "samples_count": int, "captured_seconds": float,
    # "feature_summary": {<key>: float, ...}}``.  ``None`` when the
    # multimodal head wasn't wired in OR the user did not enable the
    # mic on this turn (the default).  When ``prosody_active=True``,
    # the WS layer surfaces a ``voice prosody · active`` chip on the
    # response and the reasoning trace narrates the fusion sentence.
    multimodal: dict | None = None

    # Vision-gaze classifier output (third multimodal flagship —
    # fine-tuned MobileNetV3-small backbone + per-user fine-tuned head).
    # Shape: ``{"label": str, "confidence": float, "label_probs": {...},
    # "presence": bool, "blink_rate_norm": float, "head_stability": float,
    # "captured_seconds": float, "samples_count": int,
    # "gaze_aware_note": str | None}``.  ``None`` when the camera was
    # off (the default) or the classifier wasn't wired in.  When
    # ``presence=False`` the engine has annotated ``gaze_aware_note``
    # with the gaze-conditioned response-timing message.
    gaze: dict | None = None

    # Per-turn pipeline trace produced by
    # :class:`i3.observability.pipeline_trace.PipelineTraceCollector`
    # (third flagship surface — the live Flow dashboard).  Shape:
    # ``{"turn_id": str, "user_id": str, "session_id": str,
    # "started_at_ms": float, "ended_at_ms": float,
    # "total_latency_ms": float, "stages": [<StageRecord>, ...],
    # "arrow_flows": [<arrow>, ...]}``.  ``None`` when the collector
    # wasn't wired in (legacy callers / build_error_output).  The WS
    # layer ships this on every response/response_done frame so
    # ``web/js/flow_dashboard.js`` can animate every stage box.
    pipeline_trace: dict | None = None

    # Detailed routing-decision audit trail produced by the LinUCB
    # bandit + complexity / privacy gates on every turn.  Shape:
    # ``{"arm": "edge_slm" | "cloud_llm",
    #    "confidence": float,
    #    "reason": str,
    #    "feature_vector": list[float],
    #    "complexity": {"score": float, "factors": dict, "notes": str},
    #    "consent_required": bool}``.  ``None`` only on legacy
    # callers / build_error_output.  Used by the Routing tab's
    # scatter plot, the cloud-route reasoning paragraph, and the
    # ``GET /api/routing/decision/recent`` endpoint.
    routing_decision: dict | None = None

    # Privacy-budget snapshot for the cloud LLM hybrid route.
    # Populated on every turn (whether or not the cloud arm fired)
    # so the UI can render the "X of 50 calls used" counter live.
    # Shape mirrors :class:`i3.privacy.budget.PrivacyBudgetSnapshot`.
    # ``None`` only on legacy callers / build_error_output.
    privacy_budget: dict | None = None

    # Constitutional safety verdict produced by the char-CNN classifier
    # in :mod:`i3.safety.classifier`.  Shape mirrors
    # :class:`SafetyVerdict.to_dict`:
    #   {"verdict": "safe"|"review"|"refuse",
    #    "confidence": float,
    #    "reasons": [..],
    #    "constitutional_principle": str,
    #    "suggested_response": str,
    #    "scores": {<label>: float, ...}}
    # ``None`` only on legacy callers / build_error_output / when the
    # classifier failed to load.  When ``verdict == "refuse"`` the
    # engine has already short-circuited the response_text to the
    # canonical refusal and set ``response_path = "tool:safety"``.
    safety: dict | None = None

    # Hierarchical session memory snapshot (Phase B.5, 2026-04-25).
    # Shape: ``{"user_facts": [{"predicate", "object", "confidence",
    # "decay"}, ...], "topic_stack": [{"canonical", "weight"}, ...],
    # "turn_count": int, "thread_summary": str}``.  ``None`` when the
    # memory module isn't wired in.  Surfaced in the reasoning trace
    # as ``"User-stated facts on file: 'developer', 'prefers
    # bullets'.  Session topic thread: 'X'."``
    session_memory: dict | None = None

    # Multi-step explain decomposition trace (Phase B.3, 2026-04-25).
    # Populated only on turns where the engine detected an "explain X"
    # / "tell me about X" / "describe X" query and ran the
    # :class:`ExplainDecomposer`.  Shape: ``{"topic": str,
    # "sub_questions": [str], "sub_answers": [{"question", "source",
    # "text", "confidence"}], "composite_answer": str}``.  ``None`` for
    # every other turn.  Surfaced in the chat UI as a collapsible
    # ``<details>`` "Reasoning chain" element under the response.
    explain_plan: dict | None = None

    # Iter 51 (2026-04-27): per-(user, session) stated facts dict
    # captured by the pipeline's fact-handler regex chain
    # (``Pipeline._stated_facts``).  Shape: ``{<slot>: <value>, ...}``
    # with slots like ``name``, ``favourite_color``, ``occupation``,
    # ``location``, ``hobby``, ``age``, ``pet``, ``language``.
    # Surfaced live by the Personal Facts dashboard tab via the
    # ``i3:state_update`` browser CustomEvent (see
    # ``web/js/huawei_tabs.js:wireFactsTab``).  ``None`` when no facts
    # have been stated yet this session OR on legacy callers that
    # construct PipelineOutput directly.  Empty dict means the user
    # has visited the recall path but no facts are on file.
    personal_facts: dict | None = None

    # Iter 51 (2026-04-27): structured-output intent-parse result for
    # turns that hit the ``/api/intent`` HTTP endpoint OR for turns
    # where the engine detected a command-shaped utterance ("set timer
    # for 10 minutes", "play jazz").  Shape mirrors
    # :class:`i3.intent.IntentResult.to_dict`:
    #   {"action": str, "params": dict,
    #    "valid_json": bool, "valid_action": bool, "valid_slots": bool,
    #    "confidence": float, "raw_text": str, "backend": str,
    #    "latency_ms": float}
    # ``None`` for normal chat turns.  Surfaced in the chat UI as a
    # green ``intent · play_music · genre=jazz`` chip.
    intent_result: dict | None = None


# ---------------------------------------------------------------------------
# Engagement signal
# ---------------------------------------------------------------------------

@dataclass
class EngagementSignal:
    """Engagement signal derived from user behaviour AFTER receiving a response.

    The five sub-signals capture orthogonal aspects of engagement:

    1. **Continued conversation** -- Did the user send another message?
    2. **Response latency** -- How quickly did the user respond?
    3. **Response length ratio** -- Is the user's next message proportional
       to the AI's response?
    4. **Topic continuity** -- Did the user stay on topic?
    5. **Sentiment shift** -- Did the user's sentiment change after the
       AI's response?

    The composite :attr:`score` is the arithmetic mean of these five
    sub-signals, each normalised to [0, 1].

    Attributes:
        continued_conversation: ``True`` if the user sent a follow-up
            message within the session timeout window.
        response_latency_ms: Milliseconds between the AI response being
            delivered and the user starting their next message.
        response_length_ratio: Ratio of the user's next message length
            (in words) to the AI's response length (in words).
        topic_continuity: Estimated topic overlap between the AI's
            response and the user's follow-up, in [0, 1].
        sentiment_shift: Change in sentiment valence between the user's
            previous and current messages, in [-1, 1].
    """

    continued_conversation: bool
    response_latency_ms: float
    response_length_ratio: float
    topic_continuity: float
    sentiment_shift: float

    @property
    def score(self) -> float:
        """Compute an overall engagement score in [0, 1].

        Returns the arithmetic mean of five normalised sub-signals:

        - **Continuation**: 1.0 if the user continued, else 0.0.
        - **Latency**: Faster responses indicate higher engagement.
          Saturates at 30 s (score drops to 0.0 beyond that).
        - **Length ratio**: Longer replies indicate engagement. Capped
          at 1.0.
        - **Topic continuity**: Passed through directly (already [0, 1]).
        - **Sentiment shift**: Mapped from [-1, 1] to [0, 1].
        """
        # SEC: Each sub-signal is clamped to [0, 1] before averaging so a
        # bad upstream value (e.g. negative latency from a clock skew, or
        # topic_continuity > 1) cannot push the composite outside [0, 1].
        signals = [
            1.0 if self.continued_conversation else 0.0,
            max(0.0, min(1.0, 1.0 - self.response_latency_ms / 30_000.0)),
            max(0.0, min(1.0, self.response_length_ratio)),
            max(0.0, min(1.0, self.topic_continuity)),
            max(0.0, min(1.0, (self.sentiment_shift + 1.0) / 2.0)),
        ]
        return float(max(0.0, min(1.0, np.mean(signals))))


# ---------------------------------------------------------------------------
# Engagement estimator (stateless helper)
# ---------------------------------------------------------------------------

class EngagementEstimator:
    """Stateless utility that computes :class:`EngagementSignal` from raw metrics.

    This class does not maintain per-user state -- it simply packages the
    raw metrics into an :class:`EngagementSignal` and returns the composite
    score.  Per-user tracking (previous response time, previous response
    length, etc.) is handled by the :class:`~src.pipeline.engine.Pipeline`.

    Example::

        estimator = EngagementEstimator()
        signal = estimator.compute(
            continued=True,
            response_latency_ms=2500.0,
            user_msg_length=12,
            ai_msg_length=25,
            topic_continuity=0.7,
            sentiment_shift=0.1,
        )
        print(signal.score)  # ~0.72
    """

    def compute(
        self,
        continued: bool,
        response_latency_ms: float,
        user_msg_length: int,
        ai_msg_length: int,
        topic_continuity: float = 0.5,
        sentiment_shift: float = 0.0,
    ) -> EngagementSignal:
        """Build an :class:`EngagementSignal` from raw interaction metrics.

        Args:
            continued: Whether the user sent a follow-up message.
            response_latency_ms: Time between AI response delivery and
                the user's next keystroke (milliseconds).
            user_msg_length: Word count of the user's follow-up message.
            ai_msg_length: Word count of the AI's most recent response.
            topic_continuity: Estimated topic overlap [0, 1].
            sentiment_shift: Sentiment delta [-1, 1].

        Returns:
            A fully populated :class:`EngagementSignal`.
        """
        length_ratio = (
            user_msg_length / max(1, ai_msg_length)
        )
        return EngagementSignal(
            continued_conversation=continued,
            response_latency_ms=response_latency_ms,
            response_length_ratio=length_ratio,
            topic_continuity=max(0.0, min(1.0, topic_continuity)),
            sentiment_shift=max(-1.0, min(1.0, sentiment_shift)),
        )
