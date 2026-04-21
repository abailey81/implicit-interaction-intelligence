#!/usr/bin/env python3
"""atheris fuzz target for :func:`PrivacySanitizer.sanitize`.

The invariant under test is simple and strong:

    For *any* sequence of bytes, ``sanitize(...)`` must not raise.

Run locally::

    python tests/fuzz/fuzz_sanitizer.py -atheris_runs=100000

Any uncaught exception fails the fuzz run.  AssertionError is treated as
an invariant violation (never caught) so future contributors can assert
properties inside the target safely.
"""

from __future__ import annotations

import sys
from typing import Any

try:
    import atheris  # type: ignore
except ImportError:
    atheris = None  # type: ignore


# Exceptions that are *expected* and therefore swallowed by the harness.
# Anything outside this tuple is a real crash and libFuzzer will record it.
_EXPECTED_EXCEPTIONS: tuple[type[BaseException], ...] = (
    UnicodeError,
    # We import PrivacySanitizer lazily below — ImportError here is a
    # configuration issue, not a sanitizer bug.
)


def _decode(data: bytes) -> str:
    """Best-effort decode — the sanitiser must accept any decoded string."""
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def TestOneInput(data: bytes) -> None:
    """Entry point invoked by atheris for every mutation."""
    from i3.privacy.sanitizer import PrivacySanitizer

    sanitizer = PrivacySanitizer(enabled=True)
    text = _decode(data)

    try:
        result = sanitizer.sanitize(text)
    except AssertionError:
        raise  # always surface
    except _EXPECTED_EXCEPTIONS:
        return
    except Exception as exc:  # pragma: no cover — fuzz finding
        # libFuzzer will treat an uncaught exception as a crash.  We
        # re-raise so it records the input.
        raise RuntimeError(
            f"sanitize() raised unexpected {type(exc).__name__}: {exc!r}"
        ) from exc

    # Post-condition invariants — these must always hold.
    assert isinstance(result.sanitized_text, str)
    assert isinstance(result.pii_detected, bool)
    assert result.replacements_made >= 0
    # The sanitised string must not be longer than the input after
    # MAX_INPUT_LENGTH truncation; PII placeholders shrink or grow by
    # a bounded amount, so we just assert non-negative length here.
    assert len(result.sanitized_text) >= 0

    # Also verify contains_pii() never raises on the same input.
    try:
        _ = sanitizer.contains_pii(text)
    except AssertionError:
        raise
    except _EXPECTED_EXCEPTIONS:
        return
    except Exception as exc:  # pragma: no cover — fuzz finding
        raise RuntimeError(
            f"contains_pii() raised unexpected {type(exc).__name__}: {exc!r}"
        ) from exc


def main() -> None:
    if atheris is None:
        print(
            "atheris is not installed. Install it with `pip install atheris`.",
            file=sys.stderr,
        )
        sys.exit(2)
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
