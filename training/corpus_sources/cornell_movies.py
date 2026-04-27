"""Cornell Movie-Dialogs corpus loader.

Danescu-Niculescu-Mizil & Lee 2011 — ~220k conversational exchanges from movie
scripts.

Output records::

    {"history": str, "response": str, "kind": "qa"}

Standard library only + ``requests``.
"""
from __future__ import annotations

import ast
import logging
import time
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SOURCE_NAME = "cornell_movies"
ZIP_URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
HTTP_TIMEOUT = 60
MAX_LINE_LEN = 200       # drop lines longer than this character count
MAX_RESPONSE_LEN = 2000
SEP = " +++$+++ "


def _download(cache_dir: Path) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "cornell_movie_dialogs_corpus.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        logger.info("Cornell zip already cached at %s (%d bytes)",
                    zip_path, zip_path.stat().st_size)
        return zip_path

    try:
        logger.info("Cornell: downloading %s", ZIP_URL)
        t0 = time.time()
        resp = requests.get(ZIP_URL, timeout=HTTP_TIMEOUT, stream=True)
        if resp.status_code != 200:
            logger.error("Cornell: HTTP %d from %s", resp.status_code, ZIP_URL)
            return None
        total = 0
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
        elapsed = time.time() - t0
        logger.info("Cornell: downloaded %d bytes in %.1fs", total, elapsed)
        return zip_path
    except Exception as exc:  # noqa: BLE001
        logger.error("Cornell: download failed: %s", exc)
        return None


def _read_member(zf: zipfile.ZipFile, suffix: str) -> str | None:
    for name in zf.namelist():
        if name.lower().endswith(suffix.lower()):
            with zf.open(name) as fh:
                return fh.read().decode("latin-1", errors="replace")
    return None


def load_cornell_movies(cache_dir: Path) -> list[dict]:
    """Load Cornell Movie-Dialogs adjacent-line pairs.

    Idempotent via ``cornell_movies.done`` marker. Returns ``[]`` on any error.
    """
    cache_dir = Path(cache_dir)
    done_marker = cache_dir / f"{SOURCE_NAME}.done"
    pairs_cache = cache_dir / f"{SOURCE_NAME}.pairs.txt"

    try:
        if done_marker.exists() and pairs_cache.exists():
            logger.info("Cornell: cached pairs found, reloading from %s", pairs_cache)
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
            logger.info("Cornell: %d pairs reloaded from cache", len(pairs))
            return pairs

        t0 = time.time()
        zip_path = _download(cache_dir)
        if zip_path is None:
            return []

        with zipfile.ZipFile(zip_path) as zf:
            lines_text = _read_member(zf, "movie_lines.txt")
            convs_text = _read_member(zf, "movie_conversations.txt")

        if lines_text is None or convs_text is None:
            logger.error("Cornell: missing movie_lines.txt or movie_conversations.txt")
            return []

        # Parse movie_lines.txt -> dict[line_id, text]
        line_text: dict[str, str] = {}
        for raw in lines_text.splitlines():
            if not raw:
                continue
            parts = raw.split(SEP)
            if len(parts) < 5:
                continue
            line_id = parts[0].strip()
            text = parts[4].strip()
            if not line_id or not text:
                continue
            if len(text) > MAX_LINE_LEN:
                continue
            line_text[line_id] = text
        logger.info("Cornell: parsed %d candidate lines (after length filter)",
                    len(line_text))

        pairs = []
        n_convs = 0
        for raw in convs_text.splitlines():
            if not raw:
                continue
            parts = raw.split(SEP)
            if len(parts) < 4:
                continue
            id_list_str = parts[3].strip()
            try:
                id_list = ast.literal_eval(id_list_str)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(id_list, list) or len(id_list) < 2:
                continue
            n_convs += 1
            for i in range(len(id_list) - 1):
                history = line_text.get(id_list[i])
                response = line_text.get(id_list[i + 1])
                if not history or not response:
                    continue
                if not history.strip() or not response.strip():
                    continue
                if len(response) >= MAX_RESPONSE_LEN:
                    continue
                pairs.append({"history": history, "response": response, "kind": "qa"})

        elapsed = time.time() - t0
        logger.info(
            "Cornell: parsed %d conversations -> %d pairs in %.1fs",
            n_convs, len(pairs), elapsed,
        )

        try:
            with open(pairs_cache, "w", encoding="utf-8") as fh:
                for p in pairs:
                    h = p["history"].replace("\t", " ").replace("\n", " ")
                    r = p["response"].replace("\t", " ").replace("\n", " ")
                    fh.write(f"{h}\t{r}\n")
            done_marker.write_text("ok", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cornell: could not write cache: %s", exc)

        return pairs
    except Exception as exc:  # noqa: BLE001
        logger.exception("Cornell: unexpected failure: %s", exc)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = load_cornell_movies(Path("D:/caches/corpus"))
    print(f"Cornell Movies pair count: {len(out)}")
