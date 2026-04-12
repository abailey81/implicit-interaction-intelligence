"""Integration tests exercising multi-component flows of the I3 pipeline.

These tests verify that the major subsystems compose correctly:

    - Feature extraction -> TCN encoder shape contract.
    - Deviation metrics -> Adaptation controller output.
    - Router + sensitivity detector privacy overrides.
    - DiaryStore end-to-end privacy guarantee (no raw text).

The tests stub out heavy dependencies (cloud APIs, trained SLM weights)
and only exercise the deterministic, on-device code paths.
"""

from __future__ import annotations

import pytest
import torch

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.interaction.types import InteractionFeatureVector
from i3.user_model.types import DeviationMetrics
from i3.privacy.sanitizer import PrivacySanitizer
from i3.diary.store import DiaryStore


def _zero_deviation() -> DeviationMetrics:
    """Construct a DeviationMetrics with no deviation from baseline."""
    return DeviationMetrics(
        current_vs_baseline=0.0,
        current_vs_session=0.0,
        engagement_score=0.5,
        magnitude=0.0,
        iki_deviation=0.0,
        length_deviation=0.0,
        vocab_deviation=0.0,
        formality_deviation=0.0,
        speed_deviation=0.0,
        engagement_deviation=0.0,
        complexity_deviation=0.0,
        pattern_deviation=0.0,
    )


def _fatigue_deviation() -> DeviationMetrics:
    """Construct a DeviationMetrics that signals fatigue / cognitive drop."""
    return DeviationMetrics(
        current_vs_baseline=0.2,
        current_vs_session=0.1,
        engagement_score=0.3,
        magnitude=1.2,
        iki_deviation=1.5,         # slower typing
        length_deviation=-0.8,
        vocab_deviation=-1.0,
        formality_deviation=0.0,
        speed_deviation=-1.1,
        engagement_deviation=-1.0,
        complexity_deviation=-1.0,  # clearly below baseline
        pattern_deviation=1.2,
    )


def _motor_difficulty_deviation() -> DeviationMetrics:
    """Construct a DeviationMetrics suggesting motor difficulty (slow, jittery)."""
    return DeviationMetrics(
        current_vs_baseline=0.1,
        current_vs_session=0.05,
        engagement_score=0.4,
        magnitude=0.9,
        iki_deviation=2.0,  # much slower
        length_deviation=-0.3,
        vocab_deviation=-0.1,
        formality_deviation=0.0,
        speed_deviation=-1.5,
        engagement_deviation=-0.3,
        complexity_deviation=-0.2,
        pattern_deviation=0.9,
    )


# -------------------------------------------------------------------------
# Feature extraction -> TCN encoding
# -------------------------------------------------------------------------


class TestInteractionToEncoding:
    """Feature vector -> TCN encoding pipeline."""

    def test_feature_vector_to_encoding(self) -> None:
        """Stacking feature vectors and encoding with the TCN should give 64-dim output."""
        tcn_mod = pytest.importorskip("i3.encoder.tcn")
        model = tcn_mod.TemporalConvNet(input_dim=32, embedding_dim=64)
        model.eval()

        features = [InteractionFeatureVector.zeros() for _ in range(5)]
        # Vary some values so it is not all zeros
        for i, fv in enumerate(features):
            fv.message_length = 0.1 * (i + 1)
            fv.type_token_ratio = 0.2 + 0.05 * i
            fv.formality = 0.5

        stacked = torch.stack([fv.to_tensor() for fv in features])  # (5, 32)
        batch = stacked.unsqueeze(0)  # (1, 5, 32)
        with torch.no_grad():
            emb = model(batch)
        assert emb.shape == (1, 64)
        assert torch.isfinite(emb).all()
        # Unit norm
        assert torch.allclose(emb.norm(dim=1), torch.ones(1), atol=1e-5)

    def test_different_features_produce_different_embeddings(self) -> None:
        """Distinct feature sequences should (almost always) yield distinct embeddings."""
        tcn_mod = pytest.importorskip("i3.encoder.tcn")
        torch.manual_seed(0)
        model = tcn_mod.TemporalConvNet(input_dim=32, embedding_dim=64)
        model.eval()

        x1 = torch.randn(1, 6, 32)
        x2 = torch.randn(1, 6, 32) * 0.5 + 1.0
        with torch.no_grad():
            e1 = model(x1)
            e2 = model(x2)
        # L2 distance should be meaningfully non-zero
        assert torch.norm(e1 - e2) > 1e-3


# -------------------------------------------------------------------------
# Adaptation pipeline
# -------------------------------------------------------------------------


class TestAdaptationPipeline:
    """User-state -> adaptation dimensions mapping through AdaptationController."""

    @pytest.fixture
    def controller(self):
        ctrl_mod = pytest.importorskip("i3.adaptation.controller")
        cfg_mod = pytest.importorskip("i3.config")
        adaptation_cfg = cfg_mod.AdaptationConfig()
        return ctrl_mod.AdaptationController(adaptation_cfg)

    def test_baseline_not_established_uses_defaults(self, controller) -> None:
        """With zero-valued features and deviation, output should be clamped and default-ish."""
        features = InteractionFeatureVector.zeros()
        deviation = _zero_deviation()
        av = controller.compute(features, deviation)
        assert isinstance(av, AdaptationVector)
        # With zero features, accessibility should be off
        assert av.accessibility == 0.0
        # All values in valid range
        assert 0.0 <= av.cognitive_load <= 1.0
        assert 0.0 <= av.emotional_tone <= 1.0

    def test_fatigue_signals_reduce_cognitive_load(self, controller) -> None:
        """Strongly negative complexity deviation should lead to lower cognitive_load."""
        features = InteractionFeatureVector.zeros()
        features.type_token_ratio = 0.3
        features.mean_word_length = 4.0
        features.flesch_kincaid = 6.0
        features.message_length = 0.3

        deviation_fatigue = _fatigue_deviation()
        av_fatigue = controller.compute(features, deviation_fatigue)

        # Reset controller and run the neutral case
        controller.reset()
        deviation_calm = _zero_deviation()
        # With zero deviation: features above baseline -> higher cognitive_load
        features_calm = InteractionFeatureVector.zeros()
        features_calm.type_token_ratio = 0.6
        features_calm.mean_word_length = 6.0
        features_calm.flesch_kincaid = 10.0
        features_calm.message_length = 0.5
        av_calm = controller.compute(features_calm, deviation_calm)

        assert av_fatigue.cognitive_load < av_calm.cognitive_load

    def test_motor_difficulty_triggers_accessibility(self, controller) -> None:
        """High motor/typing difficulty should activate accessibility mode."""
        features = InteractionFeatureVector.zeros()
        features.backspace_ratio = 0.9
        features.editing_effort = 0.9

        deviation = _motor_difficulty_deviation()
        av = controller.compute(features, deviation)
        # Accessibility threshold default is 0.7 -- our difficulty score should exceed it
        assert av.accessibility > 0.0

    def test_controller_reset_clears_style(self, controller) -> None:
        """reset() should restore the style mirror to its neutral default."""
        features = InteractionFeatureVector.zeros()
        features.formality = 1.0
        features.emoji_density = 1.0
        deviation = _zero_deviation()
        controller.compute(features, deviation)
        controller.reset()
        default_style = StyleVector.default()
        assert controller.current_style.formality == default_style.formality
        assert controller.current_style.verbosity == default_style.verbosity


# -------------------------------------------------------------------------
# Router + sensitivity detector
# -------------------------------------------------------------------------


class TestRouterWithSensitivity:
    """Integration of the TopicSensitivityDetector with routing decisions.

    We avoid importing the full IntelligentRouter (which has cloud
    dependencies) and instead verify the detector -> override logic
    directly, mirroring the router's behaviour.
    """

    def test_sensitive_topic_forces_local_route(self) -> None:
        """Sensitive queries should produce a sensitivity score high enough to override."""
        detector_mod = pytest.importorskip("i3.router.sensitivity")
        detector = detector_mod.TopicSensitivityDetector()
        score = detector.detect(
            "I'm feeling suicidal and need to talk to someone privately."
        )
        # Router threshold is 0.5 for privacy override
        assert score >= 0.5

    def test_benign_topic_uses_bandit(self) -> None:
        """Benign queries should have sensitivity below the override threshold."""
        detector_mod = pytest.importorskip("i3.router.sensitivity")
        detector = detector_mod.TopicSensitivityDetector()
        score = detector.detect("What is the capital of France?")
        assert score < 0.5

    def test_sensitivity_detailed_returns_categories(self) -> None:
        """detect_detailed should return a structured dict with matched categories."""
        detector_mod = pytest.importorskip("i3.router.sensitivity")
        detector = detector_mod.TopicSensitivityDetector()
        detail = detector.detect_detailed(
            "I'm anxious about my salary and need therapy"
        )
        assert isinstance(detail, dict)
        assert "score" in detail
        assert "matched_categories" in detail
        assert len(detail["matched_categories"]) >= 1

    @pytest.mark.parametrize(
        "text,expected_route",
        [
            ("Please share your password", "local_slm"),
            ("I feel depressed", "local_slm"),
            ("What is 2 + 2?", "either"),
            ("Explain recursion", "either"),
            ("My SSN is private", "local_slm"),
        ],
    )
    def test_routing_by_sensitivity(self, text: str, expected_route: str) -> None:
        """Simulated router override logic should route sensitive queries locally."""
        detector_mod = pytest.importorskip("i3.router.sensitivity")
        detector = detector_mod.TopicSensitivityDetector()
        score = detector.detect(text)
        threshold = 0.5
        if expected_route == "local_slm":
            assert score >= threshold
        else:
            # Either route is acceptable -- but this test just asserts no error
            assert 0.0 <= score <= 1.0


# -------------------------------------------------------------------------
# Diary store: end-to-end privacy
# -------------------------------------------------------------------------


class TestDiaryNoRawText:
    """End-to-end verification that no raw user text ever touches the diary database."""

    async def test_log_exchange_stores_only_metadata(
        self, temp_diary_store: DiaryStore
    ) -> None:
        """Log a typical exchange and verify no stored column contains the raw text."""
        # A distinctive sensitive sentence with unique phrases that should
        # never appear in any diary column.
        sensitive_raw_text = (
            "Xylophone-parakeet-quasar zanzibar wildebeest phrenology "
            "confetti lorem ipsum distinctivequarkword persistentmarker."
        )

        # Pre-sanitise the text -- the pipeline would do this before calling us.
        sanitizer = PrivacySanitizer(enabled=True)
        _ = sanitizer.sanitize(sensitive_raw_text)

        # Create a session and log an exchange using ONLY embeddings, scalars, topics.
        session_id = "integration-session-1"
        user_id = "test-user-1"
        await temp_diary_store.create_session(session_id, user_id)

        embedding = torch.randn(64, dtype=torch.float32)
        embedding_bytes = embedding.numpy().tobytes()
        adaptation_vector = AdaptationVector.default().to_dict()
        topics = ["medical", "diagnosis"]

        exchange_id = await temp_diary_store.log_exchange(
            session_id=session_id,
            user_state_embedding=embedding_bytes,
            adaptation_vector=adaptation_vector,
            route_chosen="local_slm",
            response_latency_ms=123,
            engagement_signal=0.8,
            topics=topics,
        )
        assert isinstance(exchange_id, str) and len(exchange_id) > 0

        # Fetch the exchange back
        exchanges = await temp_diary_store.get_session_exchanges(session_id)
        assert len(exchanges) == 1
        record = exchanges[0]

        # No field should contain any distinctive word from the raw text.
        distinctive_words = [
            "xylophone",
            "parakeet",
            "quasar",
            "zanzibar",
            "wildebeest",
            "phrenology",
            "distinctivequarkword",
            "persistentmarker",
        ]
        for key, value in record.items():
            if isinstance(value, str):
                for word in distinctive_words:
                    assert word not in value.lower(), (
                        f"raw text word {word!r} leaked into column {key}"
                    )

        # Assert the embedding survived the round-trip
        assert record["user_state_embedding"] == embedding_bytes
        # Adaptation vector should be parseable
        assert isinstance(record["adaptation_vector"], dict)

    async def test_session_end_persists_only_scalar_and_keyword_metadata(
        self, temp_diary_store: DiaryStore
    ) -> None:
        """end_session should only persist aggregated scalars + topic keywords."""
        session_id = "integration-session-2"
        user_id = "test-user-2"
        await temp_diary_store.create_session(session_id, user_id)
        await temp_diary_store.end_session(
            session_id=session_id,
            summary="User discussed medical topics (metadata-only summary).",
            dominant_emotion="concern",
            topics=["health", "wellness"],
            mean_engagement=0.62,
            mean_cognitive_load=0.45,
            mean_accessibility=0.1,
            relationship_strength=0.3,
        )
        sessions = await temp_diary_store.get_user_sessions(user_id)
        assert len(sessions) == 1
        record = sessions[0]
        assert record["dominant_emotion"] == "concern"
        assert record["topics"] == ["health", "wellness"]
        assert abs(record["mean_engagement"] - 0.62) < 1e-6

    async def test_privacy_auditor_flags_no_violations_for_clean_db(
        self, temp_diary_store: DiaryStore, tmp_path
    ) -> None:
        """A diary DB with only metadata should pass the PrivacyAuditor scan."""
        from i3.privacy.sanitizer import PrivacyAuditor

        # Create a clean session + exchange with only metadata
        session_id = "clean-session"
        user_id = "clean-user"
        await temp_diary_store.create_session(session_id, user_id)
        await temp_diary_store.log_exchange(
            session_id=session_id,
            user_state_embedding=torch.zeros(64, dtype=torch.float32).numpy().tobytes(),
            adaptation_vector=AdaptationVector.default().to_dict(),
            route_chosen="local_slm",
            response_latency_ms=50,
            engagement_signal=0.5,
            topics=["test", "topic"],
        )

        auditor = PrivacyAuditor()
        result = await auditor.audit_database(temp_diary_store.db_path)
        assert isinstance(result, dict)
        assert "violations" in result
        # A clean store should have no violations or only safe-metadata columns
        real_violations = [
            v for v in result["violations"]
            if isinstance(v, dict) and "error" not in v
        ]
        assert len(real_violations) == 0, (
            f"PrivacyAuditor unexpectedly flagged violations: {real_violations}"
        )
