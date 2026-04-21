#!/usr/bin/env python
"""CLI: export the SQLite diary to a DuckDB-optimised Parquet snapshot.

The diary contains no raw text by construction, so this export is
automatically privacy-safe — it simply re-encodes the on-disk data in
the columnar Parquet format for fast downstream analytics.

Usage::

    python scripts/export_diary_to_parquet.py \\
        --diary data/diary.db --out-dir reports/

Produces ``reports/diary_<YYYY-MM-DD>.parquet`` containing a
``sessions`` partition and an ``exchanges`` partition.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from i3.analytics.arrow_interop import (  # noqa: E402
    arrow_table_from_diary,
    write_parquet,
)

logger = logging.getLogger("i3.analytics.export")


def export(diary_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Export ``sessions`` and ``exchanges`` tables to two Parquet files.

    Args:
        diary_path: SQLite diary path.
        out_dir: Output directory.  Created if missing.

    Returns:
        Tuple ``(sessions_path, exchanges_path)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()

    sessions_tbl = arrow_table_from_diary(diary_path, table="sessions")
    exchanges_tbl = arrow_table_from_diary(diary_path, table="exchanges")

    sessions_path = write_parquet(
        sessions_tbl, out_dir / f"diary_{today}_sessions.parquet"
    )
    exchanges_path = write_parquet(
        exchanges_tbl, out_dir / f"diary_{today}_exchanges.parquet"
    )
    logger.info(
        "Diary export complete: %d sessions, %d exchanges",
        sessions_tbl.num_rows,
        exchanges_tbl.num_rows,
    )
    return sessions_path, exchanges_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--diary",
        type=Path,
        default=Path("data/diary.db"),
        help="SQLite diary path (default: data/diary.db).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory (default: reports/).",
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
    s, e = export(args.diary, args.out_dir)
    print(f"Sessions:  {s}")
    print(f"Exchanges: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
