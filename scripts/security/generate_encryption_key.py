#!/usr/bin/env python3
"""Generate a new Fernet encryption key for I3_ENCRYPTION_KEY.

Usage:
    python scripts/generate_encryption_key.py
    python scripts/generate_encryption_key.py --update-env
    python scripts/generate_encryption_key.py --update-env --env-file path/to/.env
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# SEC: Force UTF-8 on stdout/stderr so the "I³" glyph and the colour
# escape codes render cleanly on Windows consoles that default to
# cp1251 / cp437 / cp1252.  Without this the first ``print()`` crashes
# with ``UnicodeEncodeError: 'charmap' codec can't encode character
# '\\xb3'``.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:  # pragma: no cover - pre-3.7 fallback, kept for robustness
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )


# ── ANSI colors (no external deps) ──────────────────────────────────────
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def generate_key() -> str:
    """Generate a fresh Fernet symmetric encryption key."""
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        print(
            f"{RED}Error:{RESET} `cryptography` is not installed.\n"
            f"  Install with: {CYAN}pip install cryptography{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)
    return Fernet.generate_key().decode("utf-8")


def update_env_file(env_path: Path, new_key: str) -> None:
    """Update (or insert) the I3_ENCRYPTION_KEY line in a .env file in place.

    Preserves all other variables and comments exactly as-is.
    """
    if not env_path.exists():
        print(
            f"{RED}Error:{RESET} {env_path} does not exist.\n"
            f"  Create it first with: {CYAN}cp .env.example .env{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)

    lines = env_path.read_text().splitlines(keepends=False)
    new_lines: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("I3_ENCRYPTION_KEY=") and not stripped.startswith("#"):
            new_lines.append(f"I3_ENCRYPTION_KEY={new_key}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        # Append if not present
        if new_lines and new_lines[-1] != "":
            new_lines.append("")
        new_lines.append(f"I3_ENCRYPTION_KEY={new_key}")

    env_path.write_text("\n".join(new_lines) + "\n")
    action = "Updated" if replaced else "Added"
    print(f"  {GREEN}✓{RESET} {action} I3_ENCRYPTION_KEY in {CYAN}{env_path}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Fernet encryption key for I³.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--update-env",
        action="store_true",
        help="Write the key into the .env file (preserves other variables).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to the .env file (default: ./.env).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Print only the raw key (useful for piping / scripts).",
    )
    args = parser.parse_args()

    key = generate_key()

    if args.quiet:
        print(key)
        return

    print()
    print(f"{BOLD}{BLUE}  I³ Fernet Encryption Key Generator{RESET}")
    print(f"{BLUE}  ──────────────────────────────────────{RESET}")
    print()
    print(f"  {DIM}Generated key:{RESET}")
    print(f"  {CYAN}{BOLD}{key}{RESET}")
    print()

    if args.update_env:
        update_env_file(args.env_file, key)
    else:
        print(f"  {DIM}To use this key, add it to your .env file:{RESET}")
        print(f"    {YELLOW}I3_ENCRYPTION_KEY={key}{RESET}")
        print()
        print(
            f"  {DIM}Or run with {RESET}{CYAN}--update-env{RESET}"
            f"{DIM} to write it automatically.{RESET}"
        )
    print()


if __name__ == "__main__":
    main()
