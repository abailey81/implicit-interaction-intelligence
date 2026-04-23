"""Inference-time ablation wrapper for :class:`AdaptationController`.

An *ablation mode* short-circuits one or more dimensions of the
adaptation controller to their neutral default value without touching
the controller itself.  This is the mechanism behind the "encoder off",
"user-model off", "router override", and "style mirror off" toggles in
the demo UI -- a tool for interviewer-facing explainability
("here is what the system does *without* the TCN encoder, in real
time").

The ablation is performed at the *controller output* layer: the wrapped
controller still runs its full pipeline, but every adaptation dimension
is replaced with a neutral default before the vector is returned.  This
keeps the internal state (style-mirror EMA, etc.) consistent with the
interactive session and means toggling ablation back on is a zero-cost,
no-data-loss operation.

References
----------
- Morris, M. R. & Hopkins, C. G. (2014). *The Ablation Protocol for
  Usability Experiments*.  CHI Workshops.  (Definition of dimension-
  wise ablation used here.)
- ``docs/architecture/full-reference.md`` §8.5 -- Conditioning Sensitivity Test.  The
  ablation mode is the counterfactual that powers the test.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from i3.adaptation.types import AdaptationVector, StyleVector

if TYPE_CHECKING:
    from i3.adaptation.controller import AdaptationController
    from i3.interaction.types import InteractionFeatureVector
    from i3.user_model.types import DeviationMetrics


# ---------------------------------------------------------------------------
# AblationMode dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationMode:
    """Immutable specification of which dimensions to ablate.

    Attributes:
        encoder: When True, the TCN-derived cognitive-load component is
            disabled (replaced with the neutral 0.5).  Corresponds to
            the "encoder off" interview toggle.
        user_model: When True, the user-state-driven accessibility
            dimension is disabled.
        router_override: When True, the emotional-tone dimension is
            reset (simulating a cloud router that cannot honour tone).
        style_mirror: When True, the four StyleVector sub-dimensions
            are all reset to neutral (0.5).
    """

    encoder: bool = False
    user_model: bool = False
    router_override: bool = False
    style_mirror: bool = False

    @property
    def any_active(self) -> bool:
        """True if at least one ablation flag is set."""
        return (
            self.encoder
            or self.user_model
            or self.router_override
            or self.style_mirror
        )

    def apply(self, vector: AdaptationVector) -> AdaptationVector:
        """Return a new :class:`AdaptationVector` with ablated dims reset.

        The original ``vector`` is not modified.  Dimensions whose
        corresponding flag is ``False`` pass through unchanged.

        Args:
            vector: The :class:`AdaptationVector` computed by the
                underlying :class:`AdaptationController`.

        Returns:
            A new :class:`AdaptationVector` with ablated dimensions
            replaced by their neutral defaults.
        """
        if not self.any_active:
            return vector

        cognitive_load = 0.5 if self.encoder else vector.cognitive_load
        accessibility = 0.0 if self.user_model else vector.accessibility
        emotional_tone = 0.5 if self.router_override else vector.emotional_tone
        if self.style_mirror:
            style = StyleVector.default()
        else:
            style = replace(vector.style_mirror)

        return AdaptationVector(
            cognitive_load=cognitive_load,
            style_mirror=style,
            emotional_tone=emotional_tone,
            accessibility=accessibility,
        )


# ---------------------------------------------------------------------------
# ControllerView -- lightweight adapter returned by with_ablation()
# ---------------------------------------------------------------------------


class ControllerView:
    """Callable wrapper that applies an :class:`AblationMode` on every compute.

    ``ControllerView`` is what :meth:`AblationController.with_ablation`
    returns.  It mimics the public surface of
    :class:`~i3.adaptation.controller.AdaptationController` (``compute``,
    ``reset``, ``current_style``) so it can be substituted at call sites
    without further conditionals.

    Parameters
    ----------
    inner : AdaptationController
        The underlying controller.  NOT owned by the view -- the view
        does not mutate it beyond calling its public methods.
    mode : AblationMode
        The ablation specification to apply to every ``compute()`` call.

    Attributes
    ----------
    mode : AblationMode
        The currently-configured ablation mode (read-only).
    """

    def __init__(
        self,
        inner: "AdaptationController",
        mode: AblationMode,
    ) -> None:
        self._inner = inner
        self._mode = mode

    @property
    def mode(self) -> AblationMode:
        """The ablation mode in force for this view."""
        return self._mode

    @property
    def current_style(self) -> StyleVector:
        """Passthrough to :attr:`AdaptationController.current_style`.

        Note that if ``style_mirror`` is ablated, the *computed*
        adaptation vector will have a neutral style but the underlying
        controller's mutable ``_current_style`` is still being updated
        in the background.  This property returns the underlying value
        so diagnostics can tell how far the two have drifted.
        """
        return self._inner.current_style

    def compute(
        self,
        features: "InteractionFeatureVector",
        deviation: "DeviationMetrics",
    ) -> AdaptationVector:
        """Run the underlying controller and ablate dimensions.

        Args:
            features: The 32-dim interaction feature vector.
            deviation: The deviation metrics produced by the user model.

        Returns:
            The :class:`AdaptationVector` with every ablated dimension
            reset to its neutral default.
        """
        raw = self._inner.compute(features, deviation)
        return self._mode.apply(raw)

    def reset(self) -> None:
        """Passthrough to the underlying controller's :meth:`reset`."""
        self._inner.reset()


# ---------------------------------------------------------------------------
# AblationController entry point
# ---------------------------------------------------------------------------


class AblationController:
    """Factory that produces ablated :class:`ControllerView` s.

    This class does *not* inherit from or modify
    :class:`~i3.adaptation.controller.AdaptationController`.  It wraps
    the controller by composition so the ablation capability can be
    toggled on and off at inference time without affecting any other
    consumer of the same controller instance.

    Parameters
    ----------
    controller : AdaptationController
        The underlying controller.  Must outlive any views produced by
        this ablation controller.

    Example
    -------
    >>> ablation = AblationController(controller)
    >>> view = ablation.with_ablation(encoder=True, style_mirror=True)
    >>> vec = view.compute(features, deviation)
    # vec.cognitive_load == 0.5 and vec.style_mirror == StyleVector.default()
    """

    def __init__(self, controller: "AdaptationController") -> None:
        if controller is None:
            raise TypeError("controller must not be None")
        self._controller = controller

    def with_ablation(
        self,
        encoder: bool = True,
        user_model: bool = True,
        router_override: bool = True,
        style_mirror: bool = True,
    ) -> ControllerView:
        """Produce a :class:`ControllerView` with the given dimensions ablated.

        Each flag *enables* the ablation of its dimension.  Passing all
        ``False`` returns a pass-through view that behaves identically
        to the underlying controller.

        Args:
            encoder: Ablate the encoder-driven cognitive-load dim.
            user_model: Ablate the accessibility dim.
            router_override: Ablate the emotional-tone dim.
            style_mirror: Ablate all four StyleVector sub-dimensions.

        Returns:
            A :class:`ControllerView` that applies the specified
            ablation to every ``compute()`` call.
        """
        mode = AblationMode(
            encoder=bool(encoder),
            user_model=bool(user_model),
            router_override=bool(router_override),
            style_mirror=bool(style_mirror),
        )
        return ControllerView(self._controller, mode)

    def passthrough(self) -> ControllerView:
        """Return a view with *no* dimensions ablated.

        Convenience for the "ablation off" state of the UI toggle.
        """
        return ControllerView(self._controller, AblationMode())

    @property
    def inner(self) -> "AdaptationController":
        """The wrapped underlying controller (read-only access)."""
        return self._controller


__all__ = ["AblationController", "AblationMode", "ControllerView"]
