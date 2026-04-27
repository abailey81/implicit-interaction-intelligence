"""OpenSubtitles English mono corpus loader (sliced).

Tiedemann 2016 — movie/TV subtitle lines. Excellent for casual spoken
English. The full mono English archive is ~1.5 GB compressed; we only need
a small slice, so we request the first ~20 MB via an HTTP ``Range`` header,
decompress it, and emit adjacent-line pairs.

If the server doesn't honor ``Range``, we still cap the on-disk write at
``MAX_DOWNLOAD_BYTES`` and the decompressed read at ``MAX_DECOMPRESSED_LINES``.

Output records::

    {"history": str, "response": str, "kind": "qa"}

Standard library only + ``requests``.
"""
from __future__ import annotations

import gzip
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SOURCE_NAME = "open_subtitles_en"
GZ_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz"
HTTP_TIMEOUT = 300
MAX_RESPONSE_LEN = 2000
MAX_LINE_LEN = 200
MAX_DOWNLOAD_BYTES = 20_000_000   # ~20 MB compressed
MAX_DECOMPRESSED_LINES = 500_000  # cap decompressed read
GZIP_MAGIC = b"\x1f\x8b"


def _download_slice(cache_dir: Path) -> Path | None:
    """Download the first ~20 MB of the OpenSubtitles English gzip.

    Sends a ``Range: bytes=0-MAX_DOWNLOAD_BYTES`` header. Servers that
    honour the range will return HTTP 206 with the slice; servers that
    don't may return 200 with the full file — in which case we still cap
    the local write at ``MAX_DOWNLOAD_BYTES``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / "opensubtitles_en_slice.txt.gz"
    if gz_path.exists() and gz_path.stat().st_size > 0:
        logger.info("OpenSubtitles: slice already cached at %s (%d bytes)",
                    gz_path, gz_path.stat().st_size)
        return gz_path

    headers = {"Range": f"bytes=0-{MAX_DOWNLOAD_BYTES}"}
    try:
        logger.info("OpenSubtitles: downloading first ~%d MB of %s",
                    MAX_DOWNLOAD_BYTES // (1024 * 1024), GZ_URL)
        t0 = time.time()
        resp = requests.get(GZ_URL, timeout=HTTP_TIMEOUT, stream=True,
                            allow_redirects=True, headers=headers)
        if resp.status_code not in (200, 206):
            logger.error("OpenSubtitles: HTTP %d from %s",
                         resp.status_code, GZ_URL)
            return None
        if resp.status_code == 200:
            logger.warning(
                "OpenSubtitles: server returned 200 (Range header not honored); "
                "capping local write at %d bytes",
                MAX_DOWNLOAD_BYTES,
            )
        total = 0
        with open(gz_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                # Hard cap to avoid pulling 1.5 GB if Range is ignored.
                remaining = MAX_DOWNLOAD_BYTES - total
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                fh.write(chunk)
                total += len(chunk)
                if total >= MAX_DOWNLOAD_BYTES:
                    break
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass
        elapsed = time.time() - t0
        with open(gz_path, "rb") as fh:
            magic = fh.read(2)
        if magic != GZIP_MAGIC:
            logger.error(
                "OpenSubtitles: download is not gzip (%d bytes, magic=%r)",
                total, magic,
            )
            try:
                gz_path.unlink()
            except OSError:
                pass
            return None
        logger.info("OpenSubtitles: downloaded %d bytes in %.1fs",
                    total, elapsed)
        return gz_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenSubtitles: download failed: %s", exc)
        try:
            if gz_path.exists():
                gz_path.unlink()
        except OSError:
            pass
        return None


def _read_lines(gz_path: Path) -> list[str]:
    """Decompress the gzip slice line-by-line, capped at ``MAX_DECOMPRESSED_LINES``.

    Trailing partial gzip data (because we sliced mid-stream) raises
    ``EOFError`` / ``BadGzipFile``; we catch and return whatever we got.
    """
    out: list[str] = []
    try:
        with gzip.open(gz_path, "rb") as fh:
            for i, raw in enumerate(fh):
                if i >= MAX_DECOMPRESSED_LINES:
                    break
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                except Exception:  # noqa: BLE001
                    continue
                out.append(line)
    except (EOFError, OSError, gzip.BadGzipFile) as exc:
        # Expected for a sliced stream — last block is truncated.
        logger.info("OpenSubtitles: decompression stopped at line %d (%s) — "
                    "this is expected for a sliced download",
                    len(out), type(exc).__name__)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenSubtitles: unexpected decompression error after %d lines: %s",
                       len(out), exc)
    return out


def load_open_subtitles(cache_dir: Path) -> list[dict]:
    """Load adjacent-line pairs from a slice of OpenSubtitles English.

    Idempotent via ``open_subtitles_en.done`` marker. Returns ``[]`` on
    any error.
    """
    cache_dir = Path(cache_dir)
    done_marker = cache_dir / f"{SOURCE_NAME}.done"
    pairs_cache = cache_dir / f"{SOURCE_NAME}.pairs.txt"

    try:
        if done_marker.exists() and pairs_cache.exists():
            logger.info("OpenSubtitles: cached pairs found, reloading from %s", pairs_cache)
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
            logger.info("OpenSubtitles: %d pairs reloaded from cache", len(pairs))
            return pairs

        t0 = time.time()
        gz_path = _download_slice(cache_dir)
        if gz_path is None:
            return []

        lines = _read_lines(gz_path)
        logger.info("OpenSubtitles: decompressed %d candidate lines", len(lines))

        pairs = []
        for i in range(len(lines) - 1):
            h = lines[i].strip()
            r = lines[i + 1].strip()
            if not h or not r:
                continue
            if len(h) >= MAX_LINE_LEN or len(r) >= MAX_LINE_LEN:
                continue
            if len(r) >= MAX_RESPONSE_LEN:
                continue
            pairs.append({"history": h, "response": r, "kind": "qa"})

        elapsed = time.time() - t0
        logger.info("OpenSubtitles: %d pairs from %d lines in %.1fs",
                    len(pairs), len(lines), elapsed)

        try:
            with open(pairs_cache, "w", encoding="utf-8") as fh:
                for p in pairs:
                    h = p["history"].replace("\t", " ").replace("\n", " ")
                    r = p["response"].replace("\t", " ").replace("\n", " ")
                    fh.write(f"{h}\t{r}\n")
            done_marker.write_text("ok", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenSubtitles: could not write cache: %s", exc)

        return pairs
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenSubtitles: unexpected failure: %s", exc)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = load_open_subtitles(Path("D:/caches/corpus"))
    print(f"OpenSubtitles EN pair count: {len(out)}")
