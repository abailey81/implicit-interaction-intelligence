"""DailyDialog corpus loader.

Li et al. 2017 — ~13k human-written multi-turn dialogues. Each dialogue is one
line in ``dialogues_text.txt`` with turns separated by the ``__eou__`` token.

Output records follow the project-wide dialogue format::

    {"history": str, "response": str, "kind": "qa"}

Standard library only + ``requests``. No HuggingFace, no pretrained anything.
"""
from __future__ import annotations

import logging
import time
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SOURCE_NAME = "daily_dialog"
ZIP_URLS = [
    # Official ACL Anthology mirror — supplementary attachment of the EMNLP
    # release of the dataset (paper I17-1099). Contains
    # ``EMNLP_dataset/dialogues_text.txt``.
    "https://aclanthology.org/attachments/I17-1099.Datasets.zip",
    # Earlier mirror candidates we tried — keep as fallbacks in case the
    # ACL Anthology URL ever moves.
    "https://github.com/Sanghoon94/DailyDialogue-Parser/raw/master/ijcnlp_dailydialog.zip",
    "https://github.com/TIXFeniks/dailydialog/raw/master/ijcnlp_dailydialog.zip",
    "http://yanran.li/files/ijcnlp_dailydialog.zip",
]
HTTP_TIMEOUT = 60
MAX_RESPONSE_LEN = 2000
EOU = "__eou__"
ZIP_MAGIC = b"PK\x03\x04"


def _download(cache_dir: Path) -> Path | None:
    """Download the zip into ``cache_dir`` if not already present.

    Returns the path to the zip file, or ``None`` if all URLs failed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ijcnlp_dailydialog.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        logger.info("DailyDialog zip already cached at %s (%d bytes)",
                    zip_path, zip_path.stat().st_size)
        return zip_path

    for url in ZIP_URLS:
        try:
            logger.info("DailyDialog: downloading %s", url)
            t0 = time.time()
            resp = requests.get(url, timeout=HTTP_TIMEOUT, stream=True,
                                allow_redirects=True)
            if resp.status_code != 200:
                logger.warning("DailyDialog: %s returned HTTP %d", url, resp.status_code)
                continue
            total = 0
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        fh.write(chunk)
                        total += len(chunk)
            elapsed = time.time() - t0
            # Validate that we actually got a zip (some hosts return HTML
            # captchas or redirect pages with HTTP 200).
            with open(zip_path, "rb") as fh:
                magic = fh.read(4)
            if magic != ZIP_MAGIC:
                logger.warning(
                    "DailyDialog: %s returned non-zip content (%d bytes, magic=%r); trying next URL",
                    url, total, magic,
                )
                try:
                    zip_path.unlink()
                except OSError:
                    pass
                continue
            logger.info("DailyDialog: downloaded %d bytes from %s in %.1fs",
                        total, url, elapsed)
            return zip_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("DailyDialog: download from %s failed: %s", url, exc)
            continue
    logger.error("DailyDialog: all download URLs failed")
    return None


def _read_dialogues_text(zip_path: Path) -> str | None:
    """Pull the ``dialogues_text.txt`` file out of the (possibly nested) zip."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            # Find the inner file. The archive has either a top-level
            # dialogues_text.txt or it is nested inside ijcnlp_dailydialog/.
            target_name = None
            for name in zf.namelist():
                lower = name.lower()
                if lower.endswith("dialogues_text.txt"):
                    target_name = name
                    break
            if target_name is None:
                # Some mirrors ship a nested zip. Try once.
                inner_zip_name = None
                for name in zf.namelist():
                    if name.lower().endswith(".zip"):
                        inner_zip_name = name
                        break
                if inner_zip_name is None:
                    logger.error("DailyDialog: dialogues_text.txt not found in zip")
                    return None
                with zf.open(inner_zip_name) as inner_fh:
                    inner_bytes = inner_fh.read()
                inner_path = zip_path.parent / "_inner_dailydialog.zip"
                inner_path.write_bytes(inner_bytes)
                with zipfile.ZipFile(inner_path) as inner_zf:
                    for name in inner_zf.namelist():
                        if name.lower().endswith("dialogues_text.txt"):
                            with inner_zf.open(name) as fh:
                                return fh.read().decode("utf-8", errors="replace")
                logger.error("DailyDialog: dialogues_text.txt not found in inner zip")
                return None
            with zf.open(target_name) as fh:
                return fh.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        logger.error("DailyDialog: could not read zip %s: %s", zip_path, exc)
        return None


def load_daily_dialog(cache_dir: Path) -> list[dict]:
    """Load DailyDialog as adjacent-turn (history, response) pairs.

    Idempotent via a ``daily_dialog.done`` marker in ``cache_dir``.

    Returns an empty list on any failure — never raises.
    """
    cache_dir = Path(cache_dir)
    done_marker = cache_dir / f"{SOURCE_NAME}.done"
    pairs_cache = cache_dir / f"{SOURCE_NAME}.pairs.txt"

    try:
        if done_marker.exists() and pairs_cache.exists():
            logger.info("DailyDialog: cached pairs found, reloading from %s", pairs_cache)
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
            logger.info("DailyDialog: %d pairs reloaded from cache", len(pairs))
            return pairs

        t0 = time.time()
        zip_path = _download(cache_dir)
        if zip_path is None:
            return []
        text = _read_dialogues_text(zip_path)
        if text is None:
            return []

        pairs = []
        n_dialogues = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            n_dialogues += 1
            turns = [t.strip() for t in line.split(EOU) if t.strip()]
            for i in range(len(turns) - 1):
                history = turns[i]
                response = turns[i + 1]
                if not history or not response:
                    continue
                if len(response) >= MAX_RESPONSE_LEN:
                    continue
                pairs.append({"history": history, "response": response, "kind": "qa"})

        elapsed = time.time() - t0
        logger.info(
            "DailyDialog: parsed %d dialogues -> %d pairs in %.1fs",
            n_dialogues, len(pairs), elapsed,
        )

        # Cache pairs as TSV for fast reload.
        try:
            with open(pairs_cache, "w", encoding="utf-8") as fh:
                for p in pairs:
                    h = p["history"].replace("\t", " ").replace("\n", " ")
                    r = p["response"].replace("\t", " ").replace("\n", " ")
                    fh.write(f"{h}\t{r}\n")
            done_marker.write_text("ok", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("DailyDialog: could not write cache: %s", exc)

        return pairs
    except Exception as exc:  # noqa: BLE001
        logger.exception("DailyDialog: unexpected failure: %s", exc)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = load_daily_dialog(Path("D:/caches/corpus"))
    print(f"DailyDialog pair count: {len(out)}")
