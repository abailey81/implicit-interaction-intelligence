"""CLI demo for the PDDL-grounded privacy-safety planner.

Given a :class:`~i3.safety.pddl_planner.SafetyContext` specified by
command-line flags, the script produces the corresponding
:class:`~i3.safety.pddl_planner.SafetyPlan`, verifies it, and prints
both the ordered actions and the YAML-serialised
:class:`~i3.safety.certificates.SafetyCertificate`.

Examples
--------
Sensitive-topic request with PII (should redact + route local)::

    python scripts/run_safety_planner_demo.py \\
        --sensitive --network --auth --keyed --pii

Rate-limited caller (should deny early)::

    python scripts/run_safety_planner_demo.py --rate-limited

Non-sensitive, no-PII, network available (route to cloud)::

    python scripts/run_safety_planner_demo.py --network --auth --keyed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script directly from a fresh checkout.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from i3.safety import (  # noqa: E402  (sys.path mutation above)
    PrivacySafetyPlanner,
    SafetyContext,
    certificate_to_yaml,
)


logger = logging.getLogger("run_safety_planner_demo")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The populated :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the PDDL-grounded privacy-safety planner over a "
            "caller-specified context and print the certificate."
        )
    )
    parser.add_argument(
        "--sensitive",
        action="store_true",
        help="Mark the topic as sensitive (forces local routing).",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Mark the network as available.",
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Mark the caller as authenticated.",
    )
    parser.add_argument(
        "--keyed",
        action="store_true",
        help="Mark the encryption key as loaded.",
    )
    parser.add_argument(
        "--rate-limited",
        action="store_true",
        help="Mark the caller as currently rate-limited.",
    )
    parser.add_argument(
        "--pii",
        action="store_true",
        help="Mark the request as still containing PII.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the demo and return a shell exit code.

    Returns:
        0 on success, non-zero on failure.
    """
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ctx = SafetyContext(
        sensitive_topic=args.sensitive,
        network_available=args.network,
        authenticated_user=args.auth,
        encryption_key_loaded=args.keyed,
        rate_limited=args.rate_limited,
        contains_pii=args.pii,
    )
    logger.info("Context: %s", ctx)

    planner = PrivacySafetyPlanner()
    plan = planner.plan(ctx)
    logger.info("Plan: %s", " -> ".join(plan.actions) or "(empty)")

    certificate = planner.certify(plan, context=ctx)
    yaml_text = certificate_to_yaml(certificate)

    print("=" * 68)
    print(f"SAFETY PLAN: {certificate.summary()}")
    print("=" * 68)
    print(yaml_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
