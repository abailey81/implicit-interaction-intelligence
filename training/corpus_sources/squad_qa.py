"""SQuAD 1.1 corpus loader.

Stanford Question Answering Dataset — ~100k Wikipedia-derived Q&A items. We
flatten the article -> paragraph -> qas hierarchy, taking the question as the
``history`` and the first listed answer as the ``response``.

Output records::

    {"history": str, "response": str, "kind": "qa"}
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SOURCE_NAME = "squad_qa"
JSON_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
HTTP_TIMEOUT = 60
MAX_RESPONSE_LEN = 2000


def _download(cache_dir: Path) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / "train-v1.1.json"
    if json_path.exists() and json_path.stat().st_size > 0:
        logger.info("SQuAD JSON already cached at %s (%d bytes)",
                    json_path, json_path.stat().st_size)
        return json_path

    try:
        logger.info("SQuAD: downloading %s", JSON_URL)
        t0 = time.time()
        resp = requests.get(JSON_URL, timeout=HTTP_TIMEOUT, stream=True)
        if resp.status_code != 200:
            logger.error("SQuAD: HTTP %d from %s", resp.status_code, JSON_URL)
            return None
        total = 0
        with open(json_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
        elapsed = time.time() - t0
        logger.info("SQuAD: downloaded %d bytes in %.1fs", total, elapsed)
        return json_path
    except Exception as exc:  # noqa: BLE001
        logger.error("SQuAD: download failed: %s", exc)
        return None


def load_squad_qa(cache_dir: Path) -> list[dict]:
    """Load SQuAD 1.1 (question, answer) pairs.

    Idempotent via ``squad_qa.done`` marker. Returns ``[]`` on any error.
    """
    cache_dir = Path(cache_dir)
    done_marker = cache_dir / f"{SOURCE_NAME}.done"
    pairs_cache = cache_dir / f"{SOURCE_NAME}.pairs.txt"

    try:
        if done_marker.exists() and pairs_cache.exists():
            logger.info("SQuAD: cached pairs found, reloading from %s", pairs_cache)
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
            logger.info("SQuAD: %d pairs reloaded from cache", len(pairs))
            return pairs

        t0 = time.time()
        json_path = _download(cache_dir)
        if json_path is None:
            return []

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        pairs = []
        articles = data.get("data", []) or []
        for article in articles:
            for paragraph in article.get("paragraphs", []) or []:
                for qa in paragraph.get("qas", []) or []:
                    question = (qa.get("question") or "").strip()
                    answers = qa.get("answers") or []
                    if not question or not answers:
                        continue
                    answer = (answers[0].get("text") or "").strip()
                    if not answer:
                        continue
                    if len(answer) >= MAX_RESPONSE_LEN:
                        continue
                    pairs.append({"history": question, "response": answer, "kind": "qa"})

        elapsed = time.time() - t0
        logger.info("SQuAD: parsed %d articles -> %d pairs in %.1fs",
                    len(articles), len(pairs), elapsed)

        try:
            with open(pairs_cache, "w", encoding="utf-8") as fh:
                for p in pairs:
                    h = p["history"].replace("\t", " ").replace("\n", " ")
                    r = p["response"].replace("\t", " ").replace("\n", " ")
                    fh.write(f"{h}\t{r}\n")
            done_marker.write_text("ok", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SQuAD: could not write cache: %s", exc)

        return pairs
    except Exception as exc:  # noqa: BLE001
        logger.exception("SQuAD: unexpected failure: %s", exc)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = load_squad_qa(Path("D:/caches/corpus"))
    print(f"SQuAD pair count: {len(out)}")
