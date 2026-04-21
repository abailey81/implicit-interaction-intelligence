#!/usr/bin/env python3
"""atheris fuzz target for the YAML config loader.

``i3.config.load_config`` parses YAML and validates the result through
Pydantic.  Both layers can raise *expected* errors (``ConfigError``,
``yaml.YAMLError``, ``pydantic.ValidationError``) on malformed input —
those are catalogued below.  Anything else is a real crash.

Run locally::

    python tests/fuzz/fuzz_config_loader.py -atheris_runs=50000
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

try:
    import atheris  # type: ignore
except ImportError:
    atheris = None  # type: ignore


# Collect the expected-exception classes lazily so the harness still
# builds on an install that lacks pydantic.
def _expected_exceptions() -> tuple[type[BaseException], ...]:
    exc: list[type[BaseException]] = [ValueError, TypeError, OSError]
    try:
        import yaml  # type: ignore
        exc.append(yaml.YAMLError)
    except Exception:
        pass
    try:
        from pydantic import ValidationError  # type: ignore
        exc.append(ValidationError)
    except Exception:
        pass
    try:
        from i3.config import ConfigError  # type: ignore
        exc.append(ConfigError)
    except Exception:
        pass
    return tuple(exc)


_EXPECTED = _expected_exceptions()


def TestOneInput(data: bytes) -> None:
    """Entry point invoked by atheris for every mutation."""
    from i3.config import load_config

    # Write the fuzzer bytes to a temp YAML file so we exercise the
    # real production code path (file open -> yaml.safe_load -> pydantic).
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, mode="wb"
    ) as fh:
        fh.write(data)
        path = Path(fh.name)

    try:
        try:
            load_config(path, set_seeds=False)
        except AssertionError:
            raise
        except _EXPECTED:
            return  # all expected / well-formed failures
        except Exception as exc:  # pragma: no cover — fuzz finding
            raise RuntimeError(
                f"load_config raised unexpected {type(exc).__name__}: {exc!r}"
            ) from exc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


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
