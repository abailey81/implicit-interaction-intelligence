"""Source adapters for real-world dialogue corpora.

Each adapter converts a vendor-specific layout into a uniform stream of
:class:`~i3.data.lineage.RecordSchema` instances.  Every record carries
a :class:`~i3.data.lineage.Lineage` tagged with the source format so
the downstream auditor can reconstruct provenance.

Built-in adapters:

- :class:`JSONLSource` — one record per line, ``{"text": "...",
  "label": "...", "speaker": "...", "conv_id": "..."}`` keys.
- :class:`CSVSource` — column-mapped CSV.
- :class:`PlainTextSource` — every non-empty line becomes a record.
- :class:`DailyDialogSource` — parses the DailyDialog research corpus
  layout (one dialogue per line, turns separated by ``__eou__``,
  emotion labels in ``dialogues_emotion.txt``).
- :class:`EmpatheticDialoguesSource` — parses the FAIR
  EmpatheticDialogues CSV layout.

Users who need a new format subclass :class:`SourceAdapter` and
implement :meth:`iter_records`; every rule in
:mod:`i3.data.quality` and the deduplicator in :mod:`i3.data.dedup`
compose over whichever adapter is plugged in.
"""

from __future__ import annotations

import csv
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional

from i3.data.lineage import Lineage, RecordSchema


def _original_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SourceAdapter(ABC):
    """Abstract base for every source adapter."""

    source_format: str = "unknown"

    def __init__(self, source_uri: str) -> None:
        self.source_uri = source_uri

    @abstractmethod
    def iter_records(self) -> Iterator[RecordSchema]:
        """Yield :class:`RecordSchema` from the underlying source."""


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


class JSONLSource(SourceAdapter):
    """One JSON object per line; keys map 1:1 onto :class:`RecordSchema`.

    Recognised keys (all optional except ``text``): ``text``,
    ``label``, ``speaker``, ``conv_id``, ``extra``.
    """

    source_format = "jsonl"

    def __init__(
        self,
        path: str | Path,
        *,
        text_key: str = "text",
        label_key: str = "label",
        speaker_key: str = "speaker",
        conv_id_key: str = "conv_id",
    ) -> None:
        super().__init__(str(path))
        self._path = Path(path)
        self._text_key = text_key
        self._label_key = label_key
        self._speaker_key = speaker_key
        self._conv_id_key = conv_id_key

    def iter_records(self) -> Iterator[RecordSchema]:
        with self._path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines silently
                if not isinstance(obj, dict):
                    continue
                text = obj.get(self._text_key)
                if not isinstance(text, str) or not text.strip():
                    continue
                yield RecordSchema(
                    text=text,
                    label=_str_or_none(obj.get(self._label_key)),
                    speaker=_speaker_or_none(obj.get(self._speaker_key)),
                    conv_id=_str_or_none(obj.get(self._conv_id_key)),
                    lineage=Lineage(
                        source_uri=f"{self.source_uri}#L{line_no}",
                        source_format=self.source_format,
                        original_hash=_original_hash(text),
                    ),
                    extra=_extras(obj, {
                        self._text_key, self._label_key,
                        self._speaker_key, self._conv_id_key,
                    }),
                )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CSVColumnMap:
    """Mapping of CSV column names to :class:`RecordSchema` fields."""

    text: str
    label: Optional[str] = None
    speaker: Optional[str] = None
    conv_id: Optional[str] = None


class CSVSource(SourceAdapter):
    """Column-mapped CSV source."""

    source_format = "csv"

    def __init__(
        self,
        path: str | Path,
        columns: CSVColumnMap,
        *,
        delimiter: str = ",",
    ) -> None:
        super().__init__(str(path))
        self._path = Path(path)
        self._columns = columns
        self._delimiter = delimiter

    def iter_records(self) -> Iterator[RecordSchema]:
        with self._path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=self._delimiter)
            for row_no, row in enumerate(reader, start=2):  # header is row 1
                text = row.get(self._columns.text, "") or ""
                if not text.strip():
                    continue
                yield RecordSchema(
                    text=text,
                    label=_column(row, self._columns.label),
                    speaker=_speaker_or_none(_column(row, self._columns.speaker)),
                    conv_id=_column(row, self._columns.conv_id),
                    lineage=Lineage(
                        source_uri=f"{self.source_uri}#R{row_no}",
                        source_format=self.source_format,
                        original_hash=_original_hash(text),
                    ),
                )


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


class PlainTextSource(SourceAdapter):
    """One record per non-empty line.

    Useful for raw-text dumps, chat logs, or small bundled fixtures.
    """

    source_format = "txt"

    def __init__(self, path: str | Path, *, label: str | None = None) -> None:
        super().__init__(str(path))
        self._path = Path(path)
        self._label = label

    def iter_records(self) -> Iterator[RecordSchema]:
        with self._path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                yield RecordSchema(
                    text=text,
                    label=self._label,
                    lineage=Lineage(
                        source_uri=f"{self.source_uri}#L{line_no}",
                        source_format=self.source_format,
                        original_hash=_original_hash(text),
                    ),
                )


# ---------------------------------------------------------------------------
# DailyDialog research corpus
# ---------------------------------------------------------------------------


class DailyDialogSource(SourceAdapter):
    """Parser for the DailyDialog corpus (Li et al. 2017).

    Expected layout inside the given directory:

    ::

        dialogues_text.txt     one dialogue per line, turns separated
                               by " __eou__ " (with spaces)
        dialogues_emotion.txt  one line per dialogue, space-separated
                               integer emotion labels per turn
                               (0=no emotion, 1=anger, 2=disgust,
                                3=fear, 4=happiness, 5=sadness,
                                6=surprise)
        dialogues_act.txt      (optional) dialogue act per turn

    Each *turn* becomes a :class:`RecordSchema` with ``conv_id`` set
    to the dialogue number so split logic can keep a dialogue in one
    split.
    """

    source_format = "dailydialog"
    _EOU = "__eou__"
    _EMOTIONS = {
        0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
        4: "happiness", 5: "sadness", 6: "surprise",
    }

    def __init__(self, directory: str | Path) -> None:
        super().__init__(str(directory))
        self._dir = Path(directory)
        self._text_file = self._dir / "dialogues_text.txt"
        self._emo_file = self._dir / "dialogues_emotion.txt"

    def iter_records(self) -> Iterator[RecordSchema]:
        if not self._text_file.exists():
            raise FileNotFoundError(
                f"DailyDialog text file not found: {self._text_file}"
            )
        with self._text_file.open("r", encoding="utf-8") as tf:
            text_lines = tf.readlines()
        if self._emo_file.exists():
            with self._emo_file.open("r", encoding="utf-8") as ef:
                emo_lines: list[str | None] = list(ef.readlines())
        else:
            emo_lines = [None] * len(text_lines)
        for dialog_no, (text_line, emo_line) in enumerate(
            zip(text_lines, emo_lines), start=1
        ):
            turns = [
                t.strip() for t in text_line.strip().split(self._EOU)
                if t.strip()
            ]
            emo_ids = (
                [int(x) for x in (emo_line or "").split()]
                if emo_line else [0] * len(turns)
            )
            for turn_no, turn in enumerate(turns):
                emo_id = emo_ids[turn_no] if turn_no < len(emo_ids) else 0
                label = self._EMOTIONS.get(emo_id, "neutral")
                speaker = "user" if turn_no % 2 == 0 else "assistant"
                yield RecordSchema(
                    text=turn,
                    label=label,
                    speaker=speaker,
                    conv_id=f"dd-{dialog_no:06d}",
                    lineage=Lineage(
                        source_uri=f"{self.source_uri}#{dialog_no}/{turn_no}",
                        source_format=self.source_format,
                        original_hash=_original_hash(turn),
                    ),
                    extra={"emotion_id": emo_id, "turn_index": turn_no},
                )


# ---------------------------------------------------------------------------
# EmpatheticDialogues
# ---------------------------------------------------------------------------


class EmpatheticDialoguesSource(SourceAdapter):
    """Parser for the FAIR EmpatheticDialogues dataset.

    Expected layout: a single CSV with header
    ``conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags``.

    The ``context`` column carries an emotion label (e.g. ``"excited"``);
    we surface that as the record ``label``.
    """

    source_format = "empathetic_dialogues"

    def __init__(self, csv_path: str | Path) -> None:
        super().__init__(str(csv_path))
        self._path = Path(csv_path)

    def iter_records(self) -> Iterator[RecordSchema]:
        with self._path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_no, row in enumerate(reader, start=2):
                text = (row.get("utterance") or "").strip()
                if not text:
                    continue
                speaker_idx = row.get("speaker_idx", "")
                speaker: Optional[str] = None
                if speaker_idx == "1":
                    speaker = "user"
                elif speaker_idx == "2":
                    speaker = "assistant"
                yield RecordSchema(
                    text=text,
                    label=_str_or_none(row.get("context")),
                    speaker=_speaker_or_none(speaker),
                    conv_id=_str_or_none(row.get("conv_id")),
                    lineage=Lineage(
                        source_uri=f"{self.source_uri}#R{row_no}",
                        source_format=self.source_format,
                        original_hash=_original_hash(text),
                    ),
                    extra={
                        "utterance_idx": _int_or_zero(row.get("utterance_idx")),
                    },
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_or_none(v: object) -> Optional[str]:
    if isinstance(v, str) and v.strip():
        return v
    return None


def _speaker_or_none(v: object) -> Optional[str]:
    if isinstance(v, str) and v in {"user", "assistant", "system", "narrator"}:
        return v  # type: ignore[return-value]
    return None


def _int_or_zero(v: object) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _column(row: Mapping[str, str], key: Optional[str]) -> Optional[str]:
    if key is None:
        return None
    v = row.get(key)
    return v if isinstance(v, str) and v.strip() else None


def _extras(obj: dict, consumed: set[str]) -> dict:
    allowed = (int, float, str, bool)
    return {
        k: v for k, v in obj.items()
        if k not in consumed and isinstance(v, allowed)
    }


__all__ = [
    "CSVColumnMap",
    "CSVSource",
    "DailyDialogSource",
    "EmpatheticDialoguesSource",
    "JSONLSource",
    "PlainTextSource",
    "SourceAdapter",
]
