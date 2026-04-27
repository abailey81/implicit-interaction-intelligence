"""WikiText-103 raw corpus loader.

Stephen Merity et al. — 103M tokens of curated Wikipedia articles. Used as
filler / general-language-modelling material to broaden the SLM's vocabulary
beyond conversational dialogue.

We download the raw zip (~200 MB), extract ``wiki.train.raw``, split into
paragraphs on blank lines, skip section-heading paragraphs (those that start
with ``" = "``), and emit the first two sentences of each paragraph as a
``(history, response)`` pair labelled ``"filler"``.

Output records::

    {"history": str, "response": str, "kind": "filler"}

Standard library only + ``requests``. The download is streamed so we never
hold the whole file in memory; progress is logged every 10 MB.
"""
from __future__ import annotations

import logging
import time
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SOURCE_NAME = "wikitext_103"
ZIP_URLS = [
    # Stephen Merity's own CDN (the original author's mirror) — hands out
    # the raw v1 zip with HTTP Range support.
    "https://wikitext.smerity.com/wikitext-103-raw-v1.zip",
    # Original Salesforce S3 location — left as a fallback in case the
    # bucket policy is updated.
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
    # HuggingFace's raw CDN as a static file host — NOT using the `datasets`
    # library, just downloading a public zip.
    "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1.zip",
]
HTTP_TIMEOUT = 300
MAX_RESPONSE_LEN = 2000
MIN_SENTENCE_LEN = 20
MAX_SENTENCE_LEN = 400
LOG_EVERY_BYTES = 10 * 1024 * 1024  # log progress every 10 MB
ZIP_MAGIC = b"PK\x03\x04"
TRAIN_MEMBER_SUFFIX = "wiki.train.raw"


def _download(cache_dir: Path) -> Path | None:
    """Stream-download WikiText-103 zip, logging every 10 MB."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "wikitext-103-raw-v1.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        logger.info("WikiText: zip already cached at %s (%d bytes)",
                    zip_path, zip_path.stat().st_size)
        return zip_path

    for url in ZIP_URLS:
        try:
            logger.info("WikiText: downloading %s", url)
            t0 = time.time()
            resp = requests.get(url, timeout=HTTP_TIMEOUT, stream=True,
                                allow_redirects=True)
            if resp.status_code != 200:
                logger.warning("WikiText: %s returned HTTP %d", url, resp.status_code)
                continue
            total = 0
            next_log = LOG_EVERY_BYTES
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    total += len(chunk)
                    if total >= next_log:
                        elapsed = time.time() - t0
                        mb = total / (1024 * 1024)
                        rate = mb / elapsed if elapsed > 0 else 0
                        logger.info(
                            "WikiText: downloaded %.1f MB (%.1f MB/s)",
                            mb, rate,
                        )
                        next_log += LOG_EVERY_BYTES
            elapsed = time.time() - t0
            with open(zip_path, "rb") as fh:
                magic = fh.read(4)
            if magic != ZIP_MAGIC:
                logger.warning(
                    "WikiText: %s returned non-zip content (%d bytes, magic=%r); trying next URL",
                    url, total, magic,
                )
                try:
                    zip_path.unlink()
                except OSError:
                    pass
                continue
            logger.info(
                "WikiText: downloaded %d bytes (%.1f MB) from %s in %.1fs",
                total, total / (1024 * 1024), url, elapsed,
            )
            return zip_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("WikiText: download from %s failed: %s", url, exc)
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except OSError:
                pass
            continue
    logger.error("WikiText: all download URLs failed")
    return None


def _read_train_text(zip_path: Path) -> str | None:
    """Pull ``wiki.train.raw`` out of the zip."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            target_name = None
            for name in zf.namelist():
                if name.lower().endswith(TRAIN_MEMBER_SUFFIX):
                    target_name = name
                    break
            if target_name is None:
                logger.error(
                    "WikiText: %s not found in zip; members include: %s",
                    TRAIN_MEMBER_SUFFIX, zf.namelist()[:10],
                )
                return None
            logger.info("WikiText: extracting %s", target_name)
            with zf.open(target_name) as fh:
                return fh.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        logger.error("WikiText: could not read zip %s: %s", zip_path, exc)
        return None


def _split_first_two_sentences(paragraph: str) -> tuple[str, str] | None:
    """Split ``paragraph`` on ``". "`` and return the first two sentences.

    Returns ``None`` if fewer than two sentences are present after the split.
    The ``". "`` separator is dropped during split, so we re-append a ``"."``
    to each kept sentence so it reads as a complete clause.
    """
    parts = paragraph.split(". ")
    if len(parts) < 2:
        return None
    s1 = parts[0].strip()
    s2 = parts[1].strip()
    if not s1 or not s2:
        return None
    # Re-append a period if the original split removed it.
    if not s1.endswith("."):
        s1 = s1 + "."
    if not s2.endswith(".") and not s2.endswith("?") and not s2.endswith("!"):
        s2 = s2 + "."
    return s1, s2


def load_wikitext(cache_dir: Path) -> list[dict]:
    """Load WikiText-103 paragraph -> (sentence1, sentence2) pairs.

    Idempotent via ``wikitext_103.done`` marker. Returns ``[]`` on any error.
    """
    cache_dir = Path(cache_dir)
    done_marker = cache_dir / f"{SOURCE_NAME}.done"
    pairs_cache = cache_dir / f"{SOURCE_NAME}.pairs.txt"

    try:
        if done_marker.exists() and pairs_cache.exists():
            logger.info("WikiText: cached pairs found, reloading from %s", pairs_cache)
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
                    pairs.append({"history": history, "response": response, "kind": "filler"})
            logger.info("WikiText: %d pairs reloaded from cache", len(pairs))
            return pairs

        t0 = time.time()
        zip_path = _download(cache_dir)
        if zip_path is None:
            return []

        text = _read_train_text(zip_path)
        if text is None:
            return []

        # Paragraphs are separated by blank lines.
        pairs: list[dict] = []
        n_para = 0
        n_heading_skipped = 0
        n_short_skipped = 0
        # Iterate without holding the entire splitlines list in memory by
        # building up paragraphs incrementally.
        current: list[str] = []

        def flush(buf: list[str]) -> None:
            nonlocal n_para, n_heading_skipped, n_short_skipped
            if not buf:
                return
            paragraph = " ".join(s.strip() for s in buf if s.strip())
            if not paragraph:
                return
            n_para += 1
            # Section headings in WikiText raw look like " = Title = " or
            # " = = Subtitle = = ". Skip them.
            if paragraph.lstrip().startswith("= "):
                n_heading_skipped += 1
                return
            split = _split_first_two_sentences(paragraph)
            if split is None:
                n_short_skipped += 1
                return
            s1, s2 = split
            if len(s1) < MIN_SENTENCE_LEN or len(s1) > MAX_SENTENCE_LEN:
                n_short_skipped += 1
                return
            if len(s2) < MIN_SENTENCE_LEN or len(s2) > MAX_SENTENCE_LEN:
                n_short_skipped += 1
                return
            if len(s2) >= MAX_RESPONSE_LEN:
                return
            pairs.append({"history": s1, "response": s2, "kind": "filler"})

        for line in text.splitlines():
            if line.strip() == "":
                flush(current)
                current = []
            else:
                current.append(line)
        flush(current)

        elapsed = time.time() - t0
        logger.info(
            "WikiText: %d paragraphs, %d heading skipped, %d too-short/too-long skipped -> %d pairs in %.1fs",
            n_para, n_heading_skipped, n_short_skipped, len(pairs), elapsed,
        )

        try:
            with open(pairs_cache, "w", encoding="utf-8") as fh:
                for p in pairs:
                    h = p["history"].replace("\t", " ").replace("\n", " ")
                    r = p["response"].replace("\t", " ").replace("\n", " ")
                    fh.write(f"{h}\t{r}\n")
            done_marker.write_text("ok", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("WikiText: could not write cache: %s", exc)

        return pairs
    except Exception as exc:  # noqa: BLE001
        logger.exception("WikiText: unexpected failure: %s", exc)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = load_wikitext(Path("D:/caches/corpus"))
    print(f"WikiText-103 pair count: {len(out)}")
