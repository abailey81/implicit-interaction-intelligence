"""Provenance metadata and Pydantic schema for every pipeline record.

The :class:`Lineage` metadata travels with each record through every
pipeline stage so downstream auditors (including the data card generator
in :mod:`docs.responsible_ai`) can reconstruct exactly where a sample
came from and which cleaning passes touched it.

The :class:`RecordSchema` is the Pydantic v2 contract every record must
satisfy at the boundary between the cleaner and the deduper — it
catches malformed inputs before they poison downstream stages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The pipeline's own schema version.  Bump when you change the
#: :class:`RecordSchema` contract in a backwards-incompatible way so
#: persisted artefacts can be re-parsed with the matching version.
SCHEMA_VERSION: Literal["1.1.0"] = "1.1.0"


class Lineage(BaseModel):
    """Provenance metadata for a single pipeline record.

    Attributes:
        source_uri: The original file path, URL, or dataset identifier.
            Never includes credentials.
        source_format: Short tag for the original layout (``"dailydialog"``,
            ``"empathetic"``, ``"jsonl"``, ``"csv"``, …).
        ingested_at: UTC timestamp of when the record entered the
            pipeline.  Used for deterministic split keying.
        applied_transforms: Ordered list of the transforms that have
            been applied so far.  Each entry is the transform's
            ``__class__.__name__`` or a named function identifier.
        schema_version: The :data:`SCHEMA_VERSION` at ingest time.
            Kept on the record so later loaders can migrate if needed.
        original_hash: Content hash of the *raw* text before any
            cleaning, so auditors can recover pre-normalised form.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_uri: str = Field(..., min_length=1, max_length=1024)
    source_format: str = Field(..., min_length=1, max_length=64)
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    applied_transforms: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    original_hash: str = Field(..., min_length=1, max_length=128)

    def with_transform(self, name: str) -> "Lineage":
        """Return a new :class:`Lineage` with ``name`` appended."""
        return self.model_copy(
            update={"applied_transforms": self.applied_transforms + (name,)}
        )


class RecordSchema(BaseModel):
    """Canonical record that flows through every pipeline stage.

    Attributes:
        text: Cleaned text.  Always NFKC-normalised, whitespace-collapsed,
            zero-width characters stripped.
        label: Optional class label carried by the source
            (e.g. emotion tag in DailyDialog).
        speaker: Optional speaker role if the source is dialogue
            (``"user"``, ``"assistant"``, ``"system"``).
        conv_id: Optional conversation identifier — two records with
            the same ``conv_id`` must end up in the SAME split to
            prevent data leakage.
        lineage: Provenance metadata.
        extra: Free-form attributes carried through but not validated
            (useful for source-specific annotations).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., min_length=1, max_length=100_000)
    label: Optional[str] = Field(default=None, max_length=64)
    speaker: Optional[Literal["user", "assistant", "system", "narrator"]] = None
    conv_id: Optional[str] = Field(default=None, max_length=128)
    lineage: Lineage
    extra: dict[str, int | float | str | bool] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_is_not_just_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace content")
        return v


__all__ = ["Lineage", "RecordSchema", "SCHEMA_VERSION"]
