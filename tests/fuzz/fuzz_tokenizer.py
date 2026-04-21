#!/usr/bin/env python3
"""atheris fuzz target for :meth:`SimpleTokenizer.encode`.

Invariant: ``encode()`` must not raise on arbitrary unicode input, and
must terminate in bounded time (i.e., no infinite loop inside the
punctuation / whitespace regex).

Run locally::

    python tests/fuzz/fuzz_tokenizer.py -atheris_runs=100000
"""

from __future__ import annotations

import sys

try:
    import atheris  # type: ignore
except ImportError:
    atheris = None  # type: ignore


_EXPECTED_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    UnicodeError,
)


# Build a small vocabulary once at startup so the fuzz loop stays fast.
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from i3.slm.tokenizer import SimpleTokenizer

        tok = SimpleTokenizer(vocab_size=256)
        tok.build_vocab(
            [
                "the quick brown fox jumps over the lazy dog",
                "hello world how are you",
                "the model learns from text",
            ]
        )
        _TOKENIZER = tok
    return _TOKENIZER


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def TestOneInput(data: bytes) -> None:
    """atheris entry point."""
    tok = _get_tokenizer()
    text = _decode(data)

    try:
        ids = tok.encode(text, add_special=True)
    except AssertionError:
        raise
    except _EXPECTED_EXCEPTIONS:
        return
    except Exception as exc:  # pragma: no cover — fuzz finding
        raise RuntimeError(
            f"encode() raised unexpected {type(exc).__name__}: {exc!r}"
        ) from exc

    # Post-conditions that must always hold.
    assert isinstance(ids, list)
    for tid in ids:
        assert isinstance(tid, int)
        assert 0 <= tid < tok.vocab_size

    # Round-trip back — decode must also never raise.
    try:
        text_back = tok.decode(ids, skip_special=True)
    except AssertionError:
        raise
    except _EXPECTED_EXCEPTIONS:
        return
    except Exception as exc:  # pragma: no cover — fuzz finding
        raise RuntimeError(
            f"decode() raised unexpected {type(exc).__name__}: {exc!r}"
        ) from exc
    assert isinstance(text_back, str)


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
