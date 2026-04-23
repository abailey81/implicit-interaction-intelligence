"""Offline GDPR-style export of a user's persisted I3 state.

Writes ``reports/gdpr_export_<user_id>_<date>.json`` with everything
the system has ever stored about the user, keyed by the same user id
the live server uses:

* User profile (baseline embedding base64-encoded, feature statistics,
  long-term style, relationship strength).
* Full diary (session summaries + per-exchange metadata; embeddings
  base64-encoded).
* Bandit state (arm stats + aggregate pulls / rewards).

This is the parallel of :func:`server.routes_admin.admin_export`: the
REST endpoint is for live-server operation; this CLI runs offline against
the SQLite files directly, so the interview laptop can produce an export
even if the FastAPI process is down.

Usage::

    python scripts/export_user_gdpr.py --user-id demo_user
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("i3.gdpr.export")


async def _export_user(
    user_id: str,
    config_path: str,
    output_dir: Path,
) -> Path:
    """Gather everything stored about *user_id* and write it to disk.

    Args:
        user_id: The target user id (must pass the REST regex, but we
            don't enforce it here — the CLI is for operator use only).
        config_path: Path to the YAML config whose DB paths we should use.
        output_dir: Directory for the output JSON.

    Returns:
        The :class:`Path` to the written file.
    """
    from i3.config import load_config
    from i3.diary.store import DiaryStore
    from i3.user_model.store import UserModelStore

    config = load_config(config_path)

    # ---- Profile --------------------------------------------------------
    profile_payload: dict[str, Any] | None = None
    user_db = str(getattr(config.user_model, "db_path", "data/user_model.db"))
    try:
        async with UserModelStore(user_db) as store:
            profile = await store.load_profile(user_id)
    except (RuntimeError, OSError) as exc:
        logger.warning(
            "gdpr.profile.load_failed",
            extra={"event": "gdpr_export", "err": type(exc).__name__},
        )
        profile = None
    if profile is not None:
        embedding_b64: str | None = None
        if profile.baseline_embedding is not None:
            raw = profile.baseline_embedding.detach().cpu().numpy().tobytes()
            embedding_b64 = base64.b64encode(raw).decode("ascii")
        profile_payload = {
            "user_id": profile.user_id,
            "baseline_embedding_b64": embedding_b64,
            "baseline_embedding_dim": (
                int(profile.baseline_embedding.numel())
                if profile.baseline_embedding is not None
                else None
            ),
            "baseline_features_mean": profile.baseline_features_mean,
            "baseline_features_std": profile.baseline_features_std,
            "total_sessions": profile.total_sessions,
            "total_messages": profile.total_messages,
            "relationship_strength": profile.relationship_strength,
            "long_term_style": profile.long_term_style,
            "baseline_established": profile.baseline_established,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
        }

    # ---- Diary ----------------------------------------------------------
    diary_payload: list[dict[str, Any]] = []
    diary = DiaryStore(str(getattr(config.diary, "db_path", "data/diary.db")))
    try:
        await diary.initialize()
        sessions = await diary.get_user_sessions(user_id, limit=10_000)
        for session in sessions:
            exchanges = await diary.get_session_exchanges(session["session_id"])
            safe_exchanges: list[dict[str, Any]] = []
            for ex in exchanges:
                blob = ex.get("user_state_embedding")
                embedding_b64 = None
                if isinstance(blob, (bytes, bytearray)):
                    embedding_b64 = base64.b64encode(bytes(blob)).decode("ascii")
                safe_exchanges.append(
                    {
                        "exchange_id": ex.get("exchange_id"),
                        "timestamp": ex.get("timestamp"),
                        "route_chosen": ex.get("route_chosen"),
                        "response_latency_ms": ex.get("response_latency_ms"),
                        "engagement_signal": ex.get("engagement_signal"),
                        "topics": ex.get("topics"),
                        "adaptation_vector": ex.get("adaptation_vector"),
                        "user_state_embedding_b64": embedding_b64,
                    }
                )
            diary_payload.append(
                {
                    "session_id": session.get("session_id"),
                    "start_time": session.get("start_time"),
                    "end_time": session.get("end_time"),
                    "message_count": session.get("message_count"),
                    "summary": session.get("summary"),
                    "dominant_emotion": session.get("dominant_emotion"),
                    "topics": session.get("topics"),
                    "mean_engagement": session.get("mean_engagement"),
                    "mean_cognitive_load": session.get("mean_cognitive_load"),
                    "mean_accessibility": session.get("mean_accessibility"),
                    "relationship_strength": session.get("relationship_strength"),
                    "exchanges": safe_exchanges,
                }
            )
    except (RuntimeError, OSError) as exc:
        logger.warning(
            "gdpr.diary.read_failed",
            extra={"event": "gdpr_export", "err": type(exc).__name__},
        )

    # ---- Bandit state (best-effort) ------------------------------------
    bandit_payload: dict[str, Any] = {}
    bandit_path = Path("data/router_bandit_state.json")
    if bandit_path.is_file():
        try:
            bandit_payload = json.loads(bandit_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "gdpr.bandit.load_failed",
                extra={"event": "gdpr_export", "err": type(exc).__name__},
            )

    # ---- Write file -----------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    safe_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
    out_path = output_dir / f"gdpr_export_{safe_id}_{today}.json"

    body = {
        "user_id": user_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "privacy_note": (
            "Raw user text is never persisted by I3; this export contains "
            "embeddings (base64), metadata, and scalar statistics only."
        ),
        "profile": profile_payload,
        "diary": diary_payload,
        "bandit": bandit_payload,
    }
    out_path.write_text(
        json.dumps(body, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        "gdpr.export.written",
        extra={
            "event": "gdpr_export",
            "user_id": user_id,
            "path": str(out_path),
            "diary_sessions": len(diary_payload),
            "profile_present": profile_payload is not None,
        },
    )
    return out_path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CLI."""
    parser = argparse.ArgumentParser(
        description="Write a GDPR-style JSON export of an I3 user's data."
    )
    parser.add_argument("--user-id", default="demo_user", type=str)
    parser.add_argument(
        "--config", default="configs/default.yaml", type=str
    )
    parser.add_argument(
        "--output-dir", default="reports", type=str,
        help="Target directory (default: reports/).",
    )
    parser.add_argument("--log-level", default="INFO", type=str)
    return parser.parse_args()


def main() -> int:
    """Synchronous CLI entry point — returns the process exit code."""
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        path = asyncio.run(
            _export_user(
                user_id=args.user_id,
                config_path=args.config,
                output_dir=Path(args.output_dir),
            )
        )
    except (RuntimeError, OSError, ValueError) as exc:
        logger.error(
            "gdpr.export.failed",
            extra={"event": "gdpr_export", "err": type(exc).__name__, "detail": str(exc)},
        )
        return 1
    print(f"wrote {path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
