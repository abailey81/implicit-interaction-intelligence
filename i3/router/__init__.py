"""Intelligent Router for Implicit Interaction Intelligence (I3).

Implements a Contextual Thompson Sampling multi-armed bandit that decides
whether to route each user query to a local Small Language Model (fast,
private, edge-deployable) or a cloud LLM (high quality, higher latency).

Built entirely from scratch -- no bandit libraries.

Public API:
    ContextualThompsonBandit  -- the core contextual bandit
    RoutingDecision           -- dataclass describing a routing decision
    RoutingContext            -- 12-dimensional context vector
    QueryComplexityEstimator  -- heuristic query complexity scorer
    TopicSensitivityDetector  -- privacy-sensitive topic detector
    IntelligentRouter         -- high-level router combining all components
"""

from i3.router.bandit import ContextualThompsonBandit
from i3.router.complexity import QueryComplexityEstimator
from i3.router.router import IntelligentRouter
from i3.router.sensitivity import TopicSensitivityDetector
from i3.router.types import RouteChoice, RoutingContext, RoutingDecision

__all__ = [
    "ContextualThompsonBandit",
    "IntelligentRouter",
    "QueryComplexityEstimator",
    "RouteChoice",
    "RoutingContext",
    "RoutingDecision",
    "TopicSensitivityDetector",
]
