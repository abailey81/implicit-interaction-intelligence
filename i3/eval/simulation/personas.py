"""HCI persona library for closed-loop simulation.

This module defines eight canonical :class:`HCIPersona` instances that
capture distributional typing and linguistic signatures grounded in
published HCI literature. Each persona bundles:

1. A :class:`TypingProfile` -- (mean, std) tuples for inter-key interval,
   burst ratio, pause ratio, correction rate, and typing speed in
   characters per minute (cpm). All fields are *distributions*, not
   scalars, so the simulator can sample plausible message-level
   variability.
2. A :class:`LinguisticProfile` -- target Flesch-Kincaid reading grade,
   formality target, verbosity mean, and sentiment baseline.
3. An :class:`~i3.adaptation.types.AdaptationVector` representing the
   ground-truth adaptation the system *should* converge to for this
   persona.
4. An optional drift schedule -- a list of
   ``(time_fraction_in_session, parameter_override)`` pairs that model
   within-session drift such as fatigue or warm-up.

The eight personas span the dimensions that an on-device assistant is
expected to adapt to: baseline fresh user, fatigue, accessibility,
second-language use, cognitive load, dyslexia, high energy, and low
vision. They were designed so that the ground-truth adaptation vectors
are pairwise distinguishable (L2 distance > 0.2 between any two) while
individual sampled messages remain realistically overlapping.

Citations
~~~~~~~~~
* Epp, C., Lippold, M., & Mandryk, R. L. (2011). **Identifying
  emotional states using keystroke dynamics.** Proceedings of the
  SIGCHI Conference on Human Factors in Computing Systems (CHI '11),
  715-724.
* Vizer, L. M., Zhou, L., & Sears, A. (2009). **Automated stress
  detection using keystroke and linguistic features: An exploratory
  study.** International Journal of Human-Computer Studies, 67(10),
  870-886.
* Zimmermann, M., Chappelier, J.-C., & Bunke, H. (2014). **Applying
  dynamic time warping for the analysis of keystroke dynamics data.**
  Pattern Recognition Letters, 34(13), 1478-1486.
* Hart, S. G., & Staveland, L. E. (1988). **Development of NASA-TLX
  (Task Load Index): Results of empirical and theoretical research.**
  Advances in Psychology, 52, 139-183.
* Trewin, S. (2000). **Configuration agents, control and privacy.**
  Proceedings of the 2000 Conference on Universal Usability, 9-16. (low
  vision and motor impairment distributional ranges)
* Hornbæk, K., & Oulasvirta, A. (2017). **What is Interaction?** CHI
  '17, 5040-5052. (on adaptation targets as HCI ground truth)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from i3.adaptation.types import AdaptationVector, StyleVector

# ---------------------------------------------------------------------------
# Profile sub-models
# ---------------------------------------------------------------------------


class TypingProfile(BaseModel):
    """Distributional parameters describing a persona's typing behaviour.

    Each field is a ``(mean, std)`` tuple representing the per-message
    Gaussian from which the simulator draws samples. The documented
    ranges below reflect the HCI literature cited in the module
    docstring; values outside the range are truncated by the simulator.

    Attributes:
        inter_key_interval_ms: Mean and std of inter-key intervals, in
            milliseconds. Typical adults: mean 120-220 ms (Epp 2011).
            Motor-impaired users can exceed 600 ms (Trewin 2000).
        burst_ratio: Mean and std of burst-to-total-key ratio in
            ``[0, 1]``. Higher values indicate long uninterrupted runs.
        pause_ratio: Mean and std of pause-to-total-time ratio in
            ``[0, 1]``. Second-language speakers and high-load users
            elevate this.
        correction_rate: Mean and std of backspace rate in ``[0, 1]``.
            Dyslexic and motor-impaired users have elevated values.
        typing_speed_cpm: Mean and std of characters per minute. A
            typical adult averages 180-240 cpm (Karat 1999).
    """

    model_config = ConfigDict(frozen=True)

    inter_key_interval_ms: tuple[float, float]
    burst_ratio: tuple[float, float]
    pause_ratio: tuple[float, float]
    correction_rate: tuple[float, float]
    typing_speed_cpm: tuple[float, float]


class LinguisticProfile(BaseModel):
    """Distributional parameters describing linguistic style.

    Attributes:
        flesch_kincaid_target: Target FK reading grade (0 = very simple,
            20 = graduate-level). Simpler ranges suit dyslexia and
            accessibility targets.
        formality_target: Target formality in ``[0, 1]``.
        verbosity_mean: Mean words per message (raw, not normalised).
            Used by the simulator to pick canonical sample texts.
        sentiment_baseline: Baseline sentiment valence in ``[-1, 1]``.
    """

    model_config = ConfigDict(frozen=True)

    flesch_kincaid_target: float = Field(ge=0.0, le=20.0)
    formality_target: float = Field(ge=0.0, le=1.0)
    verbosity_mean: float = Field(ge=0.0, le=500.0)
    sentiment_baseline: float = Field(ge=-1.0, le=1.0)


# ---------------------------------------------------------------------------
# HCIPersona
# ---------------------------------------------------------------------------


class HCIPersona(BaseModel):
    """A reproducible, literature-grounded synthetic user.

    A persona bundles everything the :class:`UserSimulator` needs to
    synthesise a plausible message stream plus the ground-truth
    :class:`~i3.adaptation.types.AdaptationVector` the closed-loop
    evaluator compares against.

    Attributes:
        name: Short machine-friendly identifier (e.g.
            ``"fatigued_developer"``).
        description: One-sentence natural-language description for
            reporting.
        typing_profile: Distributional typing parameters.
        linguistic_profile: Distributional linguistic parameters.
        expected_adaptation: The AdaptationVector the pipeline should
            converge to for this persona. This is the ground truth the
            closed-loop evaluator scores against.
        drift_schedule: Optional list of
            ``(time_fraction, parameter_override)`` pairs capturing
            within-session drift. ``time_fraction`` is in ``[0, 1]`` and
            must be monotonically non-decreasing.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    typing_profile: TypingProfile
    linguistic_profile: LinguisticProfile
    expected_adaptation: AdaptationVector
    drift_schedule: list[tuple[float, dict[str, Any]]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical persona instances
# ---------------------------------------------------------------------------


FRESH_USER: HCIPersona = HCIPersona(
    name="fresh_user",
    description=(
        "Typical adult at the start of a focused work session; "
        "balanced typing rhythm and neutral style."
    ),
    typing_profile=TypingProfile(
        inter_key_interval_ms=(155.0, 35.0),
        burst_ratio=(0.72, 0.08),
        pause_ratio=(0.15, 0.05),
        correction_rate=(0.05, 0.02),
        typing_speed_cpm=(210.0, 40.0),
    ),
    linguistic_profile=LinguisticProfile(
        flesch_kincaid_target=8.0,
        formality_target=0.5,
        verbosity_mean=18.0,
        sentiment_baseline=0.05,
    ),
    expected_adaptation=AdaptationVector(
        cognitive_load=0.45,
        style_mirror=StyleVector(
            formality=0.5, verbosity=0.5, emotionality=0.5, directness=0.55
        ),
        emotional_tone=0.5,
        accessibility=0.05,
    ),
    drift_schedule=[],
)


FATIGUED_DEVELOPER: HCIPersona = HCIPersona(
    name="fatigued_developer",
    description=(
        "Experienced developer late in a long coding session: "
        "slower typing, longer pauses, elevated corrections, terse style."
    ),
    typing_profile=TypingProfile(
        inter_key_interval_ms=(290.0, 75.0),
        burst_ratio=(0.55, 0.12),
        pause_ratio=(0.32, 0.10),
        correction_rate=(0.11, 0.04),
        typing_speed_cpm=(150.0, 45.0),
    ),
    linguistic_profile=LinguisticProfile(
        flesch_kincaid_target=9.5,
        formality_target=0.35,
        verbosity_mean=14.0,
        sentiment_baseline=-0.10,
    ),
    expected_adaptation=AdaptationVector(
        cognitive_load=0.70,
        style_mirror=StyleVector(
            formality=0.4, verbosity=0.4, emotionality=0.3, directness=0.7
        ),
        emotional_tone=0.35,
        accessibility=0.20,
    ),
    drift_schedule=[
        # As the session progresses, fatigue deepens: IKI rises further,
        # pause_ratio rises, correction_rate rises.
        (
            0.5,
            {
                "inter_key_interval_ms": (320.0, 80.0),
                "pause_ratio": (0.36, 0.11),
            },
        ),
        (
            0.8,
            {
                "inter_key_interval_ms": (355.0, 90.0),
                "correction_rate": (0.14, 0.04),
            },
        ),
    ],
)


MOTOR_IMPAIRED_USER: HCIPersona = HCIPersona(
    name="motor_impaired_user",
    description=(
        "User with a motor impairment: very slow keystrokes, short "
        "bursts, many corrections, simple vocabulary (Trewin 2000)."
    ),
    typing_profile=TypingProfile(
        inter_key_interval_ms=(620.0, 140.0),
        burst_ratio=(0.35, 0.10),
        pause_ratio=(0.45, 0.12),
        correction_rate=(0.22, 0.06),
        typing_speed_cpm=(75.0, 20.0),
    ),
    linguistic_profile=LinguisticProfile(
        flesch_kincaid_target=6.0,
        formality_target=0.45,
        verbosity_mean=10.0,
        sentiment_baseline=0.0,
    ),
    expected_adaptation=AdaptationVector(
        cognitive_load=0.25,
        style_mirror=StyleVector(
            formality=0.45, verbosity=0.35, emotionality=0.55, directness=0.65
        ),
        emotional_tone=0.30,
        accessibility=0.75,
    ),
    drift_schedule=[],
)


SECOND_LANGUAGE_SPEAKER: HCIPersona = HCIPersona(
    name="second_language_speaker",
    description=(
        "Non-native English speaker: moderate typing speed, elevated "
        "pauses for word retrieval, lower formality variance (Vizer 2009)."
    ),
    typing_profile=TypingProfile(
        inter_key_interval_ms=(210.0, 55.0),
        burst_ratio=(0.50, 0.10),
        pause_ratio=(0.40, 0.12),
        correction_rate=(0.09, 0.03),
        typing_speed_cpm=(165.0, 35.0),
    ),
    linguistic_profile=LinguisticProfile(
        flesch_kincaid_target=7.0,
        formality_target=0.55,
        verbosity_mean=12.0,
        sentiment_baseline=0.05,
    ),
    expected_adaptation=AdaptationVector(
        cognitive_load=0.40,
        style_mirror=StyleVector(
            formality=0.55, verbosity=0.40, emotionality=0.5, directness=0.5
        ),
        emotional_tone=0.45,
        accessibility=0.40,
    ),
    drift_schedule=[
        # Warm-up effect: second-language users often become faster as
        # the conversation progresses.
        (
            0.3,
            {"inter_key_interval_ms": (195.0, 50.0), "pause_ratio": (0.36, 0.11)},
        ),
    ],
)


HIGH_LOAD_USER: HCIPersona = HCIPersona(
    name="high_load_user",
    description=(
        "User under high cognitive load: scattered bursts, uneven rhythm, "
        "frequent edits, sporadic complex passages (Hart & Staveland 1988)."
    ),
    typing_profile=TypingProfile(
        inter_key_interval_ms=(245.0, 110.0),
        burst_ratio=(0.58, 0.18),
        pause_ratio=(0.30, 0.14),
        correction_rate=(0.13, 0.05),
        typing_speed_cpm=(180.0, 70.0),
    ),
    linguistic_profile=LinguisticProfile(
        flesch_kincaid_target=11.0,
        formality_target=0.55,
        verbosity_mean=22.0,
        sentiment_baseline=-0.05,
    ),
    expected_adaptation=AdaptationVector(
        cognitive_load=0.75,
        style_mirror=StyleVector(
            formality=0.55, verbosity=0.55, emotionality=0.35, directness=0.65
        ),
        emotional_tone=0.45,
        accessibility=0.15,
    ),
    drift_schedule=[
        (
            0.4,
            {
                "inter_key_interval_ms": (265.0, 120.0),
                "correction_rate": (0.16, 0.05),
            },
        ),
    ],
)


DYSLEXIC_USER: HCIPersona = HCIPersona(
    name="dyslexic_user",
    description=(
        "User with dyslexia: elevated correction_rate around specific "
        "letter clusters, slower but consistent rhythm, shorter words."
    ),
    typing_profile=TypingProfile(
        inter_key_interval_ms=(245.0, 55.0),
        burst_ratio=(0.45, 0.08),
        pause_ratio=(0.28, 0.08),
        correction_rate=(0.18, 0.05),
        typing_speed_cpm=(140.0, 28.0),
    ),
    linguistic_profile=LinguisticProfile(
        flesch_kincaid_target=5.0,
        formality_target=0.40,
        verbosity_mean=11.0,
        sentiment_baseline=0.0,
    ),
    expected_adaptation=AdaptationVector(
        cognitive_load=0.30,
        style_mirror=StyleVector(
            formality=0.40, verbosity=0.40, emotionality=0.50, directness=0.55
        ),
        emotional_tone=0.40,
        accessibility=0.60,
    ),
    drift_schedule=[],
)


ENERGETIC_USER: HCIPersona = HCIPersona(
    name="energetic_user",
    description=(
        "Fast, informal user: short bursts of rapid typing, exclamations, "
        "warm sentiment, high emoji density."
    ),
    typing_profile=TypingProfile(
        inter_key_interval_ms=(95.0, 25.0),
        burst_ratio=(0.85, 0.07),
        pause_ratio=(0.08, 0.03),
        correction_rate=(0.04, 0.02),
        typing_speed_cpm=(320.0, 55.0),
    ),
    linguistic_profile=LinguisticProfile(
        flesch_kincaid_target=7.0,
        formality_target=0.20,
        verbosity_mean=16.0,
        sentiment_baseline=0.45,
    ),
    expected_adaptation=AdaptationVector(
        cognitive_load=0.40,
        style_mirror=StyleVector(
            formality=0.20, verbosity=0.50, emotionality=0.80, directness=0.55
        ),
        emotional_tone=0.25,
        accessibility=0.05,
    ),
    drift_schedule=[],
)


LOW_VISION_USER: HCIPersona = HCIPersona(
    name="low_vision_user",
    description=(
        "Low-vision user relying on screen reader: slower rhythm, "
        "occasional bursts after landmark navigation, more deliberate "
        "formality, lower verbosity (Trewin 2000)."
    ),
    typing_profile=TypingProfile(
        inter_key_interval_ms=(355.0, 95.0),
        burst_ratio=(0.40, 0.12),
        pause_ratio=(0.42, 0.15),
        correction_rate=(0.10, 0.04),
        typing_speed_cpm=(115.0, 30.0),
    ),
    linguistic_profile=LinguisticProfile(
        flesch_kincaid_target=8.0,
        formality_target=0.65,
        verbosity_mean=10.0,
        sentiment_baseline=0.05,
    ),
    expected_adaptation=AdaptationVector(
        cognitive_load=0.40,
        style_mirror=StyleVector(
            formality=0.65, verbosity=0.35, emotionality=0.45, directness=0.65
        ),
        emotional_tone=0.40,
        accessibility=0.70,
    ),
    drift_schedule=[],
)


ALL_PERSONAS: list[HCIPersona] = [
    FRESH_USER,
    FATIGUED_DEVELOPER,
    MOTOR_IMPAIRED_USER,
    SECOND_LANGUAGE_SPEAKER,
    HIGH_LOAD_USER,
    DYSLEXIC_USER,
    ENERGETIC_USER,
    LOW_VISION_USER,
]
"""All eight canonical personas, in a fixed order used by reports."""
