"""PersonaChat corpus loader.

Facebook AI 2018 / ParlAI ConvAI2 mirror — ~11k persona-grounded dialogues,
~65k (utterance, reply) pairs once flattened.

Format (ParlAI ``train_self_original.txt`` / ``train_both_original.txt``)::

    1 your persona: i love painting.
    1 partner's persona: i am a teacher.
    1 hi how are you ?\ti am good thank you !\tcand1|cand2|...
    2 ...

Each line begins with a turn number. Lines whose body starts with
``"your persona:"`` or ``"partner's persona:"`` are persona descriptions and
are skipped. Other numbered lines are
``<N> <text>\\t<reply>\\t<cand1>|<cand2>|...`` — we keep the first two
tab-separated fields after the leading number.

Output records::

    {"history": str, "response": str, "kind": "qa"}

Standard library only + ``requests``. The archive is a ``.tar.gz``; we parse
with ``tarfile``.
"""
from __future__ import annotations

import logging
import tarfile
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SOURCE_NAME = "persona_chat"
ARCHIVE_URLS = [
    # ParlAI's official ConvAI2 archive (preferred — contains the
    # canonical train_self_original.txt and train_both_original.txt).
    "http://parl.ai/downloads/convai2/convai2_fix_723.tgz",
    # Older PersonaChat-only archive — same format, fewer dialogues.
    "http://parl.ai/downloads/personachat/personachat.tgz",
]
HTTP_TIMEOUT = 120
MAX_RESPONSE_LEN = 2000
GZIP_MAGIC = b"\x1f\x8b"
TARGET_FILES = ("train_self_original.txt", "train_both_original.txt")


def _download(cache_dir: Path) -> Path | None:
    """Download a ParlAI tarball into ``cache_dir``.

    Returns the path to the local ``.tgz`` file, or ``None`` if all URLs
    failed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "personachat_convai2.tgz"
    if archive_path.exists() and archive_path.stat().st_size > 0:
        logger.info("PersonaChat archive already cached at %s (%d bytes)",
                    archive_path, archive_path.stat().st_size)
        return archive_path

    for url in ARCHIVE_URLS:
        try:
            logger.info("PersonaChat: downloading %s", url)
            t0 = time.time()
            resp = requests.get(url, timeout=HTTP_TIMEOUT, stream=True,
                                allow_redirects=True)
            if resp.status_code != 200:
                logger.warning("PersonaChat: %s returned HTTP %d",
                               url, resp.status_code)
                continue
            total = 0
            with open(archive_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        fh.write(chunk)
                        total += len(chunk)
            elapsed = time.time() - t0
            with open(archive_path, "rb") as fh:
                magic = fh.read(2)
            if magic != GZIP_MAGIC:
                logger.warning(
                    "PersonaChat: %s returned non-gzip content (%d bytes, magic=%r); trying next URL",
                    url, total, magic,
                )
                try:
                    archive_path.unlink()
                except OSError:
                    pass
                continue
            logger.info("PersonaChat: downloaded %d bytes from %s in %.1fs",
                        total, url, elapsed)
            return archive_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("PersonaChat: download from %s failed: %s", url, exc)
            continue
    logger.error("PersonaChat: all download URLs failed")
    return None


def _extract_target(archive_path: Path) -> str | None:
    """Pull out ``train_self_original.txt`` or ``train_both_original.txt``.

    Tries ``train_self_original.txt`` first (more pairs); falls back to
    ``train_both_original.txt``. Returns the file content as a string, or
    ``None`` if neither could be found.
    """
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            members = tf.getmembers()
            # First pass: prefer train_self_original.txt
            for target_name in TARGET_FILES:
                for m in members:
                    if not m.isfile():
                        continue
                    if m.name.endswith("/" + target_name) or m.name == target_name:
                        logger.info("PersonaChat: extracting %s from archive", m.name)
                        fh = tf.extractfile(m)
                        if fh is None:
                            continue
                        return fh.read().decode("utf-8", errors="replace")
            # Diagnostic: log a few member names so we can iterate if the
            # layout is unexpected.
            sample = [m.name for m in members[:10]]
            logger.error(
                "PersonaChat: target files %s not found in archive; first members: %s",
                list(TARGET_FILES), sample,
            )
            return None
    except Exception as exc:  # noqa: BLE001
        logger.error("PersonaChat: could not read archive %s: %s",
                     archive_path, exc)
        return None


def _strip_leading_index(text: str) -> tuple[int | None, str]:
    """Strip the leading turn-number token.

    Returns ``(turn_int_or_None, remainder)``. If the line does not start
    with a digit token, returns ``(None, original_text)``.
    """
    parts = text.split(" ", 1)
    if len(parts) == 2 and parts[0].isdigit():
        try:
            return int(parts[0]), parts[1]
        except ValueError:
            return None, text
    return None, text


def load_persona_chat(cache_dir: Path) -> list[dict]:
    """Load PersonaChat (utterance, reply) pairs.

    Idempotent via ``persona_chat.done`` marker. Returns ``[]`` on any error.
    """
    cache_dir = Path(cache_dir)
    done_marker = cache_dir / f"{SOURCE_NAME}.done"
    pairs_cache = cache_dir / f"{SOURCE_NAME}.pairs.txt"

    try:
        if done_marker.exists() and pairs_cache.exists():
            logger.info("PersonaChat: cached pairs found, reloading from %s", pairs_cache)
            pairs: list[dict] = []
            with open(pairs_cache, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    history, response = parts
                    if not history.strip() or not response.strip():
                        continue
                    if len(response) >= MAX_RESPONSE_LEN:
                        continue
                    pairs.append({"history": history, "response": response, "kind": "qa"})
            logger.info("PersonaChat: %d pairs reloaded from cache", len(pairs))
            return pairs

        t0 = time.time()
        archive_path = _download(cache_dir)
        if archive_path is None:
            return []

        text = _extract_target(archive_path)
        if text is None:
            return []

        pairs = []
        n_persona = 0
        n_dialog = 0
        n_skipped = 0
        for raw in text.splitlines():
            if not raw.strip():
                continue
            turn, body = _strip_leading_index(raw)
            if turn is None:
                # Line did not start with a turn number — skip.
                n_skipped += 1
                continue
            lowered = body.lower().lstrip()
            if lowered.startswith("your persona:") or lowered.startswith("partner's persona:"):
                n_persona += 1
                continue
            if "\t" not in body:
                # Some non-persona, non-dialogue lines may exist; skip safely.
                n_skipped += 1
                continue
            n_dialog += 1
            fields = body.split("\t")
            history = fields[0].strip()
            response = fields[1].strip() if len(fields) >= 2 else ""
            if not history or not response:
                continue
            if len(response) >= MAX_RESPONSE_LEN:
                continue
            pairs.append({"history": history, "response": response, "kind": "qa"})

        elapsed = time.time() - t0
        logger.info(
            "PersonaChat: %d persona lines skipped, %d dialog lines parsed, %d other lines skipped -> %d pairs in %.1fs",
            n_persona, n_dialog, n_skipped, len(pairs), elapsed,
        )

        try:
            with open(pairs_cache, "w", encoding="utf-8") as fh:
                for p in pairs:
                    h = p["history"].replace("\t", " ").replace("\n", " ")
                    r = p["response"].replace("\t", " ").replace("\n", " ")
                    fh.write(f"{h}\t{r}\n")
            done_marker.write_text("ok", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("PersonaChat: could not write cache: %s", exc)

        return pairs
    except Exception as exc:  # noqa: BLE001
        logger.exception("PersonaChat: unexpected failure: %s", exc)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = load_persona_chat(Path("D:/caches/corpus"))
    print(f"PersonaChat pair count: {len(out)}")
