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
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import torch

from i3.config import Config
from i3.pipeline.types import (
    EngagementEstimator,
    EngagementSignal,
    PipelineInput,
    PipelineOutput,
)

logger = logging.getLogger(__name__)


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
        from i3.interaction.monitor import InteractionMonitor
        from i3.encoder.inference import EncoderInference
        from i3.adaptation.controller import AdaptationController
        from i3.router.bandit import ContextualThompsonBandit
        from i3.router.complexity import QueryComplexityEstimator
        from i3.router.sensitivity import TopicSensitivityDetector
        from i3.cloud.client import CloudLLMClient
        from i3.cloud.prompt_builder import PromptBuilder
        from i3.cloud.postprocess import ResponsePostProcessor
        from i3.diary.store import DiaryStore
        from i3.privacy.sanitizer import PrivacySanitizer

        # ---- Interaction monitoring --------------------------------------
        self.monitor = InteractionMonitor(
            feature_window=config.interaction.feature_window,
            baseline_warmup=config.user_model.baseline_warmup,
        )

        # ---- TCN encoder (lazy -- needs checkpoint path) -----------------
        self._encoder: Optional[EncoderInference] = None
        self._encoder_config = config.encoder

        # ---- Per-user models ---------------------------------------------
        self.user_models: dict[str, Any] = {}  # str -> UserModel

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

        # ---- Response generation -----------------------------------------
        self._slm_generator: Optional[Any] = None  # Lazy init
        self.cloud_client = CloudLLMClient(config)
        self.prompt_builder = PromptBuilder()
        self.postprocessor = ResponsePostProcessor()

        # ---- Diary -------------------------------------------------------
        self.diary_store = DiaryStore(config.diary.db_path)
        self._diary_logger: Optional[Any] = None  # After store init

        # ---- Privacy -----------------------------------------------------
        self.sanitizer = PrivacySanitizer(config.privacy.strip_pii)

        # ---- Engagement tracking (per-user) ------------------------------
        self._last_response_time: dict[str, float] = {}
        self._last_response_length: dict[str, int] = {}
        self._previous_engagement: dict[str, float] = {}
        self._previous_route: dict[str, int] = {}
        self.engagement_estimator = EngagementEstimator()

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

        self._initialized = True
        logger.info("Pipeline initialised successfully.")

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

    async def start_session(self, user_id: str) -> str:
        """Start a new interaction session for the given user.

        Creates the per-user :class:`~src.user_model.model.UserModel` if
        it does not already exist, initialises a new session in the user
        model, records the session in the diary store, and resets the
        adaptation controller.

        Args:
            user_id: Unique user identifier.

        Returns:
            A new UUID4 session identifier.
        """
        self._ensure_initialized()

        session_id = str(uuid.uuid4())
        user_model = self._get_or_create_user_model(user_id)
        user_model.start_session()

        await self.diary_store.create_session(session_id, user_id)
        self.adaptation.reset()

        # Clear per-user route tracking for the new session
        self._previous_route.pop(user_id, None)

        logger.info(
            "Session started: session_id=%s, user_id=%s", session_id, user_id
        )
        return session_id

    async def end_session(
        self, user_id: str, session_id: str
    ) -> Optional[dict[str, Any]]:
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
                summary_text = await self.cloud_client.generate_session_summary(
                    session_summary
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

        logger.info("Session ended: session_id=%s, user_id=%s", session_id, user_id)
        return {"summary": summary_text, **session_summary}

    # ------------------------------------------------------------------
    # Core message processing
    # ------------------------------------------------------------------

    async def process_message(self, input: PipelineInput) -> PipelineOutput:
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

        Returns:
            A :class:`PipelineOutput` with the AI response, routing
            decision, adaptation vector, and user-state metrics.

        Raises:
            RuntimeError: If the pipeline has not been initialised.
        """
        self._ensure_initialized()
        start_time = time.perf_counter()

        try:
            return await self._process_message_inner(input, start_time)
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
        self, input: PipelineInput, start_time: float
    ) -> PipelineOutput:
        """Inner pipeline implementation, wrapped by :meth:`process_message`."""
        # ---- Step 1: Extract interaction features ------------------------
        features = await self.monitor.process_message(
            user_id=input.user_id,
            text=input.message_text,
            composition_time_ms=input.composition_time_ms,
            edit_count=input.edit_count,
            pause_before_send_ms=input.pause_before_send_ms,
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
        user_state_embedding = await loop.run_in_executor(
            None, self._encode_features, feature_window
        )
        embedding_2d = self._project_2d(user_state_embedding)
        logger.debug(
            "Step 2 complete: 64-dim embedding -> 2D projection (%.3f, %.3f)",
            embedding_2d[0],
            embedding_2d[1],
        )

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
        adaptation = self.adaptation.compute(features, deviation)
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

        # ---- Step 6: Route decision -------------------------------------
        query_complexity = self.complexity_estimator.estimate(
            input.message_text
        )
        topic_sensitivity = self.sensitivity_detector.detect(
            input.message_text
        )

        route_chosen, routing_confidence = self._make_routing_decision(
            user_id=input.user_id,
            user_state_embedding=user_state_embedding,
            features=features,
            query_complexity=query_complexity,
            topic_sensitivity=topic_sensitivity,
            user_model=user_model,
        )
        logger.debug(
            "Step 6 complete: route=%s, confidence=%s",
            route_chosen,
            routing_confidence,
        )

        # ---- Step 7: Generate response -----------------------------------
        response_text = await self._generate_response(
            route=route_chosen,
            message=sanitized.sanitized_text,
            adaptation=adaptation,
            user_state=user_state_embedding,
        )
        logger.debug(
            "Step 7 complete: generated %d-word response via %s",
            len(response_text.split()),
            route_chosen,
        )

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
        full_confidence: dict[str, float] = {
            name: 0.0 for name in self.config.router.arms
        }
        for k, v in routing_confidence.items():
            try:
                full_confidence[k] = float(v)
            except (TypeError, ValueError):
                full_confidence[k] = 0.0

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
        )

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

        # Update the router bandit with the reward from the previous turn
        prev_route = self._previous_route.get(user_id)
        if prev_route is not None:
            try:
                # Build a dummy context for the update (the bandit just
                # needs the arm index and reward)
                ctx = np.zeros(self.config.router.context_dim, dtype=np.float64)
                self.router.update(prev_route, ctx, reward=score)
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
    ) -> tuple[str, dict[str, float]]:
        """Run the contextual bandit to choose a generation route.

        Returns:
            A tuple ``(route_name, confidence_dict)`` where ``route_name``
            is one of ``"local_slm"`` or ``"cloud_llm"`` and
            ``confidence_dict`` maps arm names to selection probabilities.
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

        # Privacy override: force local if topic is sensitive
        privacy_override = (
            self.config.router.privacy_override
            and topic_sensitivity > 0.5
        )

        arms = self.config.router.arms

        if privacy_override:
            route_chosen = "local_slm"
            confidence = {name: 0.0 for name in arms}
            confidence["local_slm"] = 1.0
            logger.info(
                "Privacy override triggered (sensitivity=%.2f) -> local_slm",
                topic_sensitivity,
            )
        else:
            arm_index, raw_confidence = self.router.select_arm(
                routing_context.to_vector()
            )
            route_chosen = arms[arm_index] if arm_index < len(arms) else "local_slm"

            # Translate arm indices to arm names
            confidence: dict[str, float] = {}
            for key, value in raw_confidence.items():
                # raw_confidence keys are "arm_0", "arm_1", etc.
                idx = int(key.split("_")[1]) if "_" in key else 0
                name = arms[idx] if idx < len(arms) else f"unknown_{idx}"
                confidence[name] = value

        return route_chosen, confidence

    # ------------------------------------------------------------------
    # Internal: response generation
    # ------------------------------------------------------------------

    async def _generate_response(
        self,
        route: str,
        message: str,
        adaptation: Any,
        user_state: torch.Tensor,
    ) -> str:
        """Generate a response via the chosen route.

        Falls back gracefully: cloud -> SLM -> rule-based fallback.

        Args:
            route: ``"cloud_llm"`` or ``"local_slm"``.
            message: Sanitised user message text.
            adaptation: :class:`~src.adaptation.types.AdaptationVector`.
            user_state: 64-dim user-state embedding.

        Returns:
            The generated response text.
        """
        # --- Cloud route --------------------------------------------------
        if route == "cloud_llm" and self.cloud_client.is_available:
            try:
                system_prompt = self.prompt_builder.build_system_prompt(
                    adaptation
                )
                result = await self.cloud_client.generate(message, system_prompt)
                response = result.get("text", "")
                if response:
                    return self.postprocessor.process(response, adaptation)
                logger.warning("Cloud LLM returned empty response; falling back.")
            except Exception:
                logger.exception(
                    "Cloud LLM generation failed; falling back to SLM."
                )

        # --- Local SLM route ----------------------------------------------
        if self._slm_generator is not None:
            try:
                return self._slm_generator.generate(
                    prompt=message,
                    adaptation_vector=adaptation.to_tensor().unsqueeze(0),
                    user_state=user_state.unsqueeze(0),
                )
            except Exception:
                logger.exception(
                    "SLM generation failed; using rule-based fallback."
                )

        # --- Rule-based fallback ------------------------------------------
        return self._fallback_response(adaptation)

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
            embedding_bytes = user_state_embedding.detach().cpu().numpy().tobytes()
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

        If the checkpoint file does not exist, the encoder remains
        ``None`` and :meth:`_encode_features` will use a zero embedding.
        """
        from pathlib import Path

        checkpoint_path = Path("models/encoder/checkpoint.pt")
        if not checkpoint_path.is_file():
            logger.info(
                "No encoder checkpoint found at %s; using zero embeddings.",
                checkpoint_path,
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
            logger.debug("Created new UserModel for user_id=%s", user_id)

        return self.user_models[user_id]

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
            logger.debug("Created new UserModel for user_id=%s", user_id)
            return user_model

    def _background_task_done(self, task: "asyncio.Task[Any]") -> None:
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
        confidence = {name: 0.0 for name in arms}
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
                "error": type(exc).__name__,
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
        return values[:4]

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
            from i3.slm.model import AdaptiveSLM
            from i3.slm.tokenizer import SimpleTokenizer
            from i3.slm.generate import SLMGenerator

            tokenizer = SimpleTokenizer.load(tokenizer_path)
            model = AdaptiveSLM(
                vocab_size=self.config.slm.vocab_size,
                max_seq_len=self.config.slm.max_seq_len,
                d_model=self.config.slm.d_model,
                n_heads=self.config.slm.n_heads,
                n_layers=self.config.slm.n_layers,
                d_ff=self.config.slm.d_ff,
                dropout=self.config.slm.dropout,
                conditioning_dim=self.config.slm.conditioning_dim,
                adaptation_dim=self.config.slm.adaptation_dim,
            )
            # Security: weights_only=True prevents pickled-object code
            # execution during checkpoint deserialization (CVE class:
            # insecure torch.load).  Inference-time loads must be safe.
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])

            self._slm_generator = SLMGenerator(model, tokenizer)
            logger.info("SLM loaded from %s", model_path)
        except Exception:
            logger.exception("Failed to load SLM from %s", model_path)
            self._slm_generator = None

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
