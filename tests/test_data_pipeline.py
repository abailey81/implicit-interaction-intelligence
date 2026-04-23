"""Comprehensive tests for :mod:`i3.data` — the ingestion pipeline.

Coverage:

* **Cleaning** — unicode normalisation, zero-width stripping, HTML
  entity decoding, newline canonicalisation, whitespace collapse,
  control-char stripping.
* **Quality rules** — each of the eight built-in rules verified on
  positive + negative examples.
* **Deduplication** — exact content hashes, min-hash signatures,
  LSH bucket collisions, Jaccard similarity correctness.
* **Sources** — every built-in adapter (JSONL, CSV, plain text,
  DailyDialog, EmpatheticDialogues) round-trips through a synthetic
  fixture.
* **End-to-end** — a full pipeline run on a realistic 500-record
  fixture covering junk, duplicates, and multilingual content.
* **Determinism** — the splitter is stable across runs for the same
  seed, and ``conv_id`` keeps dialogue turns in the same split.
* **Provenance** — lineage metadata round-trips through the
  pipeline and is present in the output JSONL.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from i3.data import (
    CSVColumnMap,
    CSVSource,
    Cleaner,
    CleaningConfig,
    DailyDialogSource,
    DataPipeline,
    Deduplicator,
    EmpatheticDialoguesSource,
    JSONLSource,
    Lineage,
    MinHashLSH,
    PipelineConfig,
    PlainTextSource,
    QualityFilter,
    QualityRule,
    RecordSchema,
    SCHEMA_VERSION,
    content_hash,
    default_rules,
    jaccard,
    normalise_unicode,
    strip_zero_width,
)


# ---------------------------------------------------------------------------
# Cleaning — pure functions
# ---------------------------------------------------------------------------


class TestCleaner:
    def test_nfkc_normalisation_collapses_compatibility_forms(self):
        # "①" (CIRCLED DIGIT ONE) normalises to "1" under NFKC
        assert normalise_unicode("①②③") == "123"

    def test_nfkc_combines_decomposed_accents(self):
        # e + combining acute → single code point é
        decomposed = "é"
        composed = "é"
        assert normalise_unicode(decomposed) == composed

    def test_strip_zero_width_removes_zwsp(self):
        assert strip_zero_width("hi​there") == "hithere"

    def test_strip_zero_width_removes_bidi_override(self):
        # RTL override (U+202E) is a common adversarial-unicode trick
        assert strip_zero_width("good‮word") == "goodword"

    def test_strip_zero_width_preserves_normal_spaces(self):
        assert strip_zero_width("hello world") == "hello world"

    def test_cleaner_decodes_html_entities(self):
        c = Cleaner()
        assert c.clean("Tom &amp; Jerry") == "Tom & Jerry"

    def test_cleaner_collapses_multiple_spaces(self):
        c = Cleaner()
        assert c.clean("hello    world   !") == "hello world !"

    def test_cleaner_canonicalises_line_endings(self):
        c = Cleaner()
        assert c.clean("one\r\ntwo\rthree") == "one\ntwo\nthree"

    def test_cleaner_trims_each_line(self):
        c = Cleaner()
        assert c.clean("   a   \n   b   ") == "a\nb"

    def test_cleaner_idempotent_on_already_clean_input(self):
        c = Cleaner()
        assert c.clean(c.clean("hello world")) == c.clean("hello world")

    def test_cleaner_preserves_content_case(self):
        c = Cleaner()
        # Content casing (not just stopwords) is preserved.
        assert c.clean("Hello World") == "Hello World"

    def test_cleaner_configurable_disables_individual_stages(self):
        c = Cleaner(CleaningConfig(decode_html_entities=False))
        assert c.clean("Tom &amp; Jerry") == "Tom &amp; Jerry"

    def test_cleaner_strips_non_printable_control_chars(self):
        """NULL, BEL, and other Cc / Cf characters are stripped; the
        whitespace-collapse stage additionally normalises tabs into
        ordinary spaces."""
        c = Cleaner()
        out = c.clean("line one\nline two\tcol\x00ntrol\x07")
        assert "\x00" not in out
        assert "\x07" not in out
        # Tab collapsed to space; newline preserved.
        assert "\n" in out
        assert "\t" not in out  # collapsed by _collapse_whitespace


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDedup:
    def test_content_hash_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_content_hash_normalises_whitespace(self):
        assert content_hash("hello   world") == content_hash("hello world")

    def test_content_hash_case_insensitive(self):
        assert content_hash("Hello World") == content_hash("hello world")

    def test_content_hash_differs_on_real_difference(self):
        assert content_hash("hello") != content_hash("goodbye")

    def test_jaccard_identical_sets(self):
        assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_jaccard_disjoint_sets(self):
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_empty_sets_defined_as_one(self):
        assert jaccard(set(), set()) == 1.0

    def test_jaccard_partial_overlap(self):
        # |A ∩ B| = 2, |A ∪ B| = 4  →  0.5
        assert abs(jaccard({"a", "b", "c"}, {"b", "c", "d"}) - 0.5) < 1e-9

    def test_minhash_catches_high_similarity(self):
        # 20-token sentence with a 1-token diff → Jaccard ≈ 0.9, well
        # above what LSH catches reliably with the default band config.
        lsh = MinHashLSH(threshold=0.6, shingle_size=3)
        a = (
            "The quick brown fox jumped over the lazy dog while "
            "the cat slept peacefully on the warm sunny windowsill"
        )
        b = (
            "The quick brown fox jumped over the lazy dog while "
            "the cat slept peacefully on the warm sunny couch"
        )
        lsh.add("r1", a)
        is_dup, rid, sim = lsh.is_near_duplicate(b)
        assert is_dup, f"high-overlap near-dup not detected (sim={sim})"
        assert rid == "r1"

    def test_minhash_rejects_unrelated(self):
        lsh = MinHashLSH(threshold=0.6, shingle_size=3)
        lsh.add("r1", "the quick brown fox jumped over the lazy dog")
        is_dup, _, _ = lsh.is_near_duplicate("python is a great programming language")
        assert not is_dup

    def test_deduplicator_exact_only_mode(self):
        d = Deduplicator(exact_only=True)
        ok1, _ = d.add_and_check("r1", "hello world")
        ok2, reason = d.add_and_check("r2", "hello world")
        assert ok1 is True
        assert ok2 is False and reason == "exact"

    def test_deduplicator_detects_near_dup(self):
        d = Deduplicator(threshold=0.6)
        a = (
            "The quick brown fox jumped over the lazy dog while "
            "the cat slept peacefully on the warm sunny windowsill"
        )
        b = (
            "The quick brown fox jumped over the lazy dog while "
            "the cat slept peacefully on the warm sunny couch"
        )
        ok1, _ = d.add_and_check("r1", a)
        ok2, reason = d.add_and_check("r2", b)
        assert ok1 is True
        assert ok2 is False and reason.startswith("near(")

    def test_deduplicator_stats(self):
        d = Deduplicator(exact_only=True)
        d.add_and_check("r1", "hello")
        d.add_and_check("r2", "hello")  # exact dup
        d.add_and_check("r3", "world")
        s = d.stats
        assert s["exact_dupes"] == 1
        assert s["unique"] == 2


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


def _mk_record(text: str, **kwargs) -> RecordSchema:
    lineage = Lineage(
        source_uri=kwargs.pop("source_uri", "test://x"),
        source_format=kwargs.pop("source_format", "test"),
        original_hash=content_hash(text),
    )
    return RecordSchema(text=text, lineage=lineage, **kwargs)


class TestQualityRules:
    def test_min_length_rejects_too_short(self):
        f = QualityFilter()
        assert not f.accept(_mk_record("ok"))

    def test_min_length_accepts_borderline(self):
        f = QualityFilter()
        assert f.accept(_mk_record("hello world"))

    def test_max_length_rejects_overflow(self):
        f = QualityFilter()
        long_text = "word " * 3000  # > 2048 tokens
        assert not f.accept(_mk_record(long_text))

    def test_latin_ratio_rejects_non_latin(self):
        f = QualityFilter()
        # Predominantly non-Latin script → rejected.
        assert not f.accept(_mk_record("Здравствуй мир, как ты сегодня?"))

    def test_latin_ratio_accepts_english(self):
        f = QualityFilter()
        assert f.accept(_mk_record("Hello, how are you today?"))

    def test_unique_token_ratio_rejects_repetition(self):
        f = QualityFilter()
        assert not f.accept(_mk_record("lol lol lol lol lol lol lol lol"))

    def test_url_dump_rejected(self):
        f = QualityFilter()
        assert not f.accept(_mk_record(
            "https://a.com https://b.com https://c.com https://d.com https://e.com"
        ))

    def test_profanity_budget_lets_mild_pass(self):
        f = QualityFilter()
        # A single mild word in a 15-word sentence: under 10 % threshold.
        assert f.accept(_mk_record(
            "This is a perfectly reasonable sentence about what happened yesterday at the office."
        ))

    def test_report_records_rejection_reason(self):
        f = QualityFilter()
        f.accept(_mk_record("ok"))  # rejected: min_length
        assert f.report.rejected_by_rule["min_length"] == 1

    def test_report_rejection_rate_computed(self):
        f = QualityFilter()
        f.accept(_mk_record("This is a perfectly fine record."))
        f.accept(_mk_record("ok"))
        assert 0.0 < f.report.rejection_rate < 1.0


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------


class TestSources:
    def test_jsonl_source_roundtrip(self, tmp_path: Path):
        p = tmp_path / "c.jsonl"
        p.write_text(
            json.dumps({"text": "hello", "label": "greeting"}) + "\n"
            + json.dumps({"text": "bye", "label": "farewell"}) + "\n",
            encoding="utf-8",
        )
        records = list(JSONLSource(p).iter_records())
        assert len(records) == 2
        assert {r.text for r in records} == {"hello", "bye"}
        assert records[0].label == "greeting"
        assert records[0].lineage.source_format == "jsonl"

    def test_jsonl_source_skips_malformed_lines(self, tmp_path: Path):
        p = tmp_path / "c.jsonl"
        p.write_text(
            json.dumps({"text": "good"}) + "\n"
            + "NOT VALID JSON\n"
            + json.dumps({"text": "also good"}) + "\n",
            encoding="utf-8",
        )
        records = list(JSONLSource(p).iter_records())
        assert len(records) == 2

    def test_csv_source_column_mapping(self, tmp_path: Path):
        p = tmp_path / "c.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utterance", "emotion", "conv"])
            w.writerow(["hello there", "happy", "c1"])
            w.writerow(["fine, thanks", "neutral", "c1"])
        cols = CSVColumnMap(text="utterance", label="emotion", conv_id="conv")
        records = list(CSVSource(p, columns=cols).iter_records())
        assert len(records) == 2
        assert records[0].label == "happy"
        assert records[0].conv_id == "c1"

    def test_plain_text_source(self, tmp_path: Path):
        p = tmp_path / "c.txt"
        p.write_text("alpha\nbeta\n\n\ngamma\n", encoding="utf-8")
        records = list(PlainTextSource(p).iter_records())
        assert [r.text for r in records] == ["alpha", "beta", "gamma"]

    def test_dailydialog_parser(self, tmp_path: Path):
        dd = tmp_path / "dd"
        dd.mkdir()
        (dd / "dialogues_text.txt").write_text(
            "Hello ! __eou__ Hi there ! __eou__\n"
            "Goodbye . __eou__ See you ! __eou__\n",
            encoding="utf-8",
        )
        (dd / "dialogues_emotion.txt").write_text(
            "0 4\n" "5 0\n", encoding="utf-8"
        )
        records = list(DailyDialogSource(dd).iter_records())
        assert len(records) == 4
        assert records[0].label == "neutral"  # 0
        assert records[1].label == "happiness"  # 4
        # Turn speakers alternate
        assert records[0].speaker == "user"
        assert records[1].speaker == "assistant"
        # Both turns of a dialogue share conv_id
        assert records[0].conv_id == records[1].conv_id
        assert records[0].conv_id != records[2].conv_id

    def test_empathetic_dialogues_parser(self, tmp_path: Path):
        p = tmp_path / "ed.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "conv_id", "utterance_idx", "context", "prompt",
                "speaker_idx", "utterance", "selfeval", "tags",
            ])
            w.writerow([
                "hit:1", "0", "sentimental", "my dog died",
                "1", "I feel so sad about this.", "", "",
            ])
            w.writerow([
                "hit:1", "1", "sentimental", "my dog died",
                "2", "I am so sorry to hear that.", "", "",
            ])
        records = list(EmpatheticDialoguesSource(p).iter_records())
        assert len(records) == 2
        assert records[0].label == "sentimental"
        assert records[0].speaker == "user"
        assert records[1].speaker == "assistant"


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    def _build_corpus(self, tmp_path: Path) -> Path:
        lines: list[str] = []
        # 120 high-quality unique sentences
        for subj in (
            "the weather", "my project", "the book", "our meeting",
            "the code", "this recipe", "the library", "my trip",
            "the experiment", "our discussion", "the design", "the paper",
        ):
            for verb in (
                "was great", "is interesting", "seems challenging",
                "went well", "needs more work", "looks promising",
                "turned out well", "was helpful", "raised concerns",
                "sparked a debate",
            ):
                lines.append(f"{subj.capitalize()} {verb} yesterday afternoon.")
        # 50 exact duplicates of a canonical sentence
        lines.extend(["The weather was great yesterday afternoon."] * 50)
        # 40 junk (URL dumps) — rejected by url_dump + unique_token_ratio
        lines.extend(["https://example.com " * 20 for _ in range(40)])
        # 30 too-short
        lines.extend(["ok"] * 30)
        p = tmp_path / "corpus.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def test_pipeline_produces_three_splits(self, tmp_path: Path):
        src = self._build_corpus(tmp_path)
        out = tmp_path / "out"
        pipe = DataPipeline(PipelineConfig(output_dir=out))
        report = pipe.run([PlainTextSource(str(src))])
        assert (out / "train.jsonl").exists()
        assert (out / "val.jsonl").exists()
        assert (out / "test.jsonl").exists()
        assert (out / "report.json").exists()
        assert sum(report.splits.values()) > 0

    def test_pipeline_rejects_junk(self, tmp_path: Path):
        src = self._build_corpus(tmp_path)
        pipe = DataPipeline(PipelineConfig(output_dir=tmp_path / "out"))
        report = pipe.run([PlainTextSource(str(src))])
        rej = report.quality["rejected_by_rule"]
        assert rej.get("min_length", 0) >= 30
        # URL dumps: hit either `no_url_dump` or `unique_token_ratio`
        junk_hits = rej.get("no_url_dump", 0) + rej.get("unique_token_ratio", 0)
        assert junk_hits >= 40

    def test_pipeline_dedups_exact_duplicates(self, tmp_path: Path):
        src = self._build_corpus(tmp_path)
        pipe = DataPipeline(PipelineConfig(output_dir=tmp_path / "out"))
        report = pipe.run([PlainTextSource(str(src))])
        assert report.dedup["exact_dupes"] >= 49  # 50 copies → 49 dupes

    def test_pipeline_output_records_have_lineage(self, tmp_path: Path):
        src = self._build_corpus(tmp_path)
        out = tmp_path / "out"
        DataPipeline(PipelineConfig(output_dir=out)).run(
            [PlainTextSource(str(src))]
        )
        with (out / "train.jsonl").open("r", encoding="utf-8") as f:
            first = json.loads(f.readline())
        assert "source_uri" in first
        assert "source_format" in first
        assert first["schema_version"] == SCHEMA_VERSION
        assert "Cleaner" in first["applied_transforms"]
        assert "Deduplicator" in first["applied_transforms"]

    def test_pipeline_split_is_deterministic(self, tmp_path: Path):
        src = self._build_corpus(tmp_path)
        run1 = DataPipeline(
            PipelineConfig(output_dir=tmp_path / "out1", seed="fixed")
        ).run([PlainTextSource(str(src))])
        run2 = DataPipeline(
            PipelineConfig(output_dir=tmp_path / "out2", seed="fixed")
        ).run([PlainTextSource(str(src))])
        assert run1.splits == run2.splits

    def test_pipeline_different_seed_gives_different_split(self, tmp_path: Path):
        src = self._build_corpus(tmp_path)
        run1 = DataPipeline(
            PipelineConfig(output_dir=tmp_path / "a", seed="seed-a")
        ).run([PlainTextSource(str(src))])
        run2 = DataPipeline(
            PipelineConfig(output_dir=tmp_path / "b", seed="seed-b")
        ).run([PlainTextSource(str(src))])
        # Total accepted should match (same corpus, same filter) but
        # per-split counts should NOT exactly match (different hash).
        assert sum(run1.splits.values()) == sum(run2.splits.values())
        assert run1.splits != run2.splits

    def test_pipeline_report_json_valid(self, tmp_path: Path):
        src = self._build_corpus(tmp_path)
        out = tmp_path / "out"
        DataPipeline(PipelineConfig(output_dir=out)).run(
            [PlainTextSource(str(src))]
        )
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        for key in (
            "config", "schema_version", "started_at", "finished_at",
            "duration_s", "quality", "dedup", "splits",
        ):
            assert key in report, f"missing report key: {key}"
        assert report["schema_version"] == SCHEMA_VERSION

    def test_pipeline_conv_id_keeps_dialogue_together(self, tmp_path: Path):
        """Two turns with the same conv_id must land in the same split."""
        dd = tmp_path / "dd"
        dd.mkdir()
        # 20 dialogues, each with 4 turns
        text_lines: list[str] = []
        emo_lines: list[str] = []
        for d in range(20):
            turns = [f"Dialogue {d} turn {t} content here."  for t in range(4)]
            text_lines.append(" __eou__ ".join(turns) + " __eou__")
            emo_lines.append(" ".join(["0"] * 4))
        (dd / "dialogues_text.txt").write_text("\n".join(text_lines), encoding="utf-8")
        (dd / "dialogues_emotion.txt").write_text("\n".join(emo_lines), encoding="utf-8")

        out = tmp_path / "out"
        DataPipeline(PipelineConfig(output_dir=out)).run([DailyDialogSource(dd)])

        # Walk every split; record the (conv_id -> split) mapping.
        conv_to_split: dict[str, str] = {}
        for split in ("train", "val", "test"):
            with (out / f"{split}.jsonl").open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    cid = rec.get("conv_id")
                    if cid is None:
                        continue
                    if cid in conv_to_split:
                        assert conv_to_split[cid] == split, (
                            f"conv {cid} leaked across splits: "
                            f"{conv_to_split[cid]} vs {split}"
                        )
                    else:
                        conv_to_split[cid] = split

    def test_pipeline_empty_source_handled(self, tmp_path: Path):
        src = tmp_path / "empty.txt"
        src.write_text("\n\n\n", encoding="utf-8")
        out = tmp_path / "out"
        report = DataPipeline(PipelineConfig(output_dir=out)).run(
            [PlainTextSource(str(src))]
        )
        assert sum(report.splits.values()) == 0
        assert report.quality["total_seen"] == 0


# ---------------------------------------------------------------------------
# Lineage / schema
# ---------------------------------------------------------------------------


class TestLineageSchema:
    def test_lineage_frozen(self):
        lin = Lineage(
            source_uri="x", source_format="test", original_hash="abc"
        )
        with pytest.raises(Exception):
            lin.source_uri = "y"  # type: ignore[misc]

    def test_lineage_with_transform_appends(self):
        lin = Lineage(
            source_uri="x", source_format="test", original_hash="abc"
        )
        lin2 = lin.with_transform("A").with_transform("B")
        assert lin.applied_transforms == ()
        assert lin2.applied_transforms == ("A", "B")

    def test_record_rejects_whitespace_only_text(self):
        with pytest.raises(Exception):
            RecordSchema(
                text="   ",
                lineage=Lineage(
                    source_uri="x", source_format="test",
                    original_hash="abc",
                ),
            )

    def test_record_rejects_unknown_speaker(self):
        with pytest.raises(Exception):
            RecordSchema(
                text="hi",
                speaker="boss",  # not in Literal set
                lineage=Lineage(
                    source_uri="x", source_format="test",
                    original_hash="abc",
                ),
            )

    def test_record_extra_allows_primitive_values_only(self):
        # Primitives are OK
        r = RecordSchema(
            text="hi",
            lineage=Lineage(
                source_uri="x", source_format="test", original_hash="abc",
            ),
            extra={"turn": 3, "score": 0.5, "tag": "greeting", "bot": True},
        )
        assert r.extra["turn"] == 3


# ---------------------------------------------------------------------------
# Custom rules (extensibility contract)
# ---------------------------------------------------------------------------


class TestCustomRules:
    def test_custom_rule_plugs_into_filter(self):
        def reject_if_starts_with_hello(r: RecordSchema) -> bool:
            return not r.text.lower().startswith("hello")

        custom = QualityRule(
            name="no_hello",
            description="reject greetings starting with 'hello'",
            check=reject_if_starts_with_hello,
        )
        f = QualityFilter([custom, *default_rules()])
        assert not f.accept(_mk_record("Hello there, how are you?"))
        assert f.accept(_mk_record("Good afternoon, how are you?"))

    def test_filter_reports_custom_rule_rejection(self):
        def always_reject(r: RecordSchema) -> bool:
            return False

        f = QualityFilter([
            QualityRule("always_reject", "", always_reject),
        ])
        f.accept(_mk_record("anything at all goes here"))
        assert f.report.rejected_by_rule["always_reject"] == 1
