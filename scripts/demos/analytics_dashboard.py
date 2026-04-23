#!/usr/bin/env python
"""CLI: run a one-shot I3 analytics report.

The script attaches the SQLite diary via DuckDB, runs the core
analytics queries, writes a Parquet snapshot plus a Markdown summary
(with inline matplotlib plots when the package is installed), and
exits.

Usage::

    python scripts/run_analytics_dashboard.py \\
        --diary data/diary.db \\
        --out-dir reports/

The output files are named ``analytics_<YYYY-MM-DD>.{parquet,md}`` and
any generated PNG plots are placed alongside them under
``reports/plots/analytics_<date>_<name>.png``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import Any

# Allow running the script from a source checkout without install.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from i3.analytics.arrow_interop import write_parquet  # noqa: E402
from i3.analytics.duckdb_engine import DuckDBAnalytics  # noqa: E402

logger = logging.getLogger("i3.analytics.dashboard")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _maybe_matplotlib() -> Any | None:
    """Return the ``matplotlib.pyplot`` module if available, else ``None``."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # no X11
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _polars_to_markdown(df: Any, max_rows: int = 20) -> str:
    """Render the first ``max_rows`` of a Polars DataFrame as a Markdown table."""
    if df.is_empty():
        return "_(no rows)_\n"
    limited = df.head(max_rows)
    headers = limited.columns
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = []
    for row in limited.iter_rows():
        cells = []
        for v in row:
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        body_rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body_rows]) + "\n"


def _save_barplot(
    plt: Any, df: Any, x: str, y: str, title: str, out_path: Path
) -> Path | None:
    """Save a simple bar plot of one column against another."""
    if df.is_empty():
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([str(v) for v in df[x].to_list()], df[y].to_list())
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------


def run_report(diary_path: Path, out_dir: Path, user_id: str | None) -> Path:
    """Generate the Markdown + Parquet report.  Returns the Markdown path."""
    import polars as pl  # local import — heavy dep

    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    md_path = out_dir / f"analytics_{today}.md"
    parquet_path = out_dir / f"analytics_{today}.parquet"
    plots_dir = out_dir / "plots"
    plt = _maybe_matplotlib()

    with DuckDBAnalytics(diary_path) as ddb:
        route_df = ddb.route_distribution(user_id)
        adapt_df = ddb.adaptation_distribution(user_id)
        latency_df = ddb.latency_percentiles(user_id)
        heatmap_df = (
            ddb.session_heatmap_by_hour(user_id)
            if user_id is not None
            else pl.DataFrame({"weekday": [], "hour": [], "sessions": [], "mean_engagement": []})
        )
        rolling_df = (
            ddb.rolling_engagement(user_id)
            if user_id is not None
            else pl.DataFrame(
                {"day": [], "sessions_in_day": [], "rolling_mean_engagement": []}
            )
        )

        # Per-user route/latency summaries are always useful.
        with_user_route = ddb._sql_to_polars(  # noqa: SLF001 - deliberate
            """
            SELECT s.user_id, e.route_chosen AS route, COUNT(*) AS count
            FROM diary.exchanges e JOIN diary.sessions s USING (session_id)
            GROUP BY s.user_id, e.route_chosen
            ORDER BY s.user_id, count DESC
            """
        )
        users_count = ddb._sql_to_polars(  # noqa: SLF001
            "SELECT COUNT(DISTINCT user_id) AS n_users FROM diary.sessions"
        )
        sessions_count = ddb._sql_to_polars(  # noqa: SLF001
            "SELECT COUNT(*) AS n_sessions FROM diary.sessions"
        )
        exchanges_count = ddb._sql_to_polars(  # noqa: SLF001
            "SELECT COUNT(*) AS n_exchanges FROM diary.exchanges"
        )

    # --- Parquet snapshot -------------------------------------------------
    # Combine the tabular artefacts into a tagged Parquet with one
    # table per analytic.  We store them as columns of a single table
    # encoded as JSON for portability.
    parquet_table = pl.DataFrame(
        {
            "analytic": [
                "route_distribution",
                "adaptation_distribution",
                "latency_percentiles",
                "rolling_engagement",
                "per_user_route",
            ],
            "rows_json": [
                route_df.write_json(),
                adapt_df.write_json(),
                latency_df.write_json(),
                rolling_df.write_json(),
                with_user_route.write_json(),
            ],
        }
    )
    write_parquet(parquet_table.to_arrow(), parquet_path, compression="zstd")

    # --- Plots ------------------------------------------------------------
    plot_links: dict[str, Path | None] = {}
    if plt is not None:
        plot_links["route"] = _save_barplot(
            plt,
            route_df,
            x="route",
            y="count",
            title="Route distribution",
            out_path=plots_dir / f"analytics_{today}_routes.png",
        )
        plot_links["latency"] = _save_barplot(
            plt,
            latency_df,
            x="route",
            y="p95_ms",
            title="P95 latency per route (ms)",
            out_path=plots_dir / f"analytics_{today}_latency.png",
        )

    # --- Markdown summary -------------------------------------------------
    md_lines: list[str] = []
    md_lines.append(f"# I3 Analytics Report — {today}")
    md_lines.append("")
    md_lines.append(f"- Diary file: `{diary_path}`")
    if user_id:
        md_lines.append(f"- User filter: **{user_id}**")
    md_lines.append(f"- Users:     {users_count[0, 0]}")
    md_lines.append(f"- Sessions:  {sessions_count[0, 0]}")
    md_lines.append(f"- Exchanges: {exchanges_count[0, 0]}")
    md_lines.append("")

    md_lines.append("## 1. Route distribution")
    md_lines.append(_polars_to_markdown(route_df))
    if plot_links.get("route"):
        md_lines.append(f"![routes](plots/{plot_links['route'].name})\n")

    md_lines.append("## 2. Latency percentiles (per route, ms)")
    md_lines.append(_polars_to_markdown(latency_df))
    if plot_links.get("latency"):
        md_lines.append(f"![latency](plots/{plot_links['latency'].name})\n")

    md_lines.append("## 3. Adaptation distribution (per dimension)")
    md_lines.append(_polars_to_markdown(adapt_df))

    md_lines.append("## 4. Rolling engagement (7d)")
    md_lines.append(_polars_to_markdown(rolling_df))

    md_lines.append("## 5. Session heatmap (weekday × hour)")
    md_lines.append(_polars_to_markdown(heatmap_df))

    md_lines.append("## 6. Per-user route counts")
    md_lines.append(_polars_to_markdown(with_user_route))

    md_lines.append("## 7. Aggregate counts")
    md_lines.append(
        _polars_to_markdown(
            pl.concat(
                [users_count, sessions_count, exchanges_count], how="horizontal"
            )
        )
    )

    md_lines.append("## 8. Snapshot")
    md_lines.append(
        f"- Parquet snapshot: `{parquet_path.name}` "
        f"(compression=zstd, {parquet_path.stat().st_size} bytes)\n"
    )

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Analytics report written: %s", md_path)
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--diary",
        type=Path,
        default=Path("data/diary.db"),
        help="Path to the SQLite diary (default: data/diary.db).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory (default: reports/).",
    )
    p.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Optional user to filter by (default: all users).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    md = run_report(args.diary, args.out_dir, args.user_id)
    print(f"Report written to: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
