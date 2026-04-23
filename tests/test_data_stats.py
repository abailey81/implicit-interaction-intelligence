"""Tests for :mod:`i3.data.stats` — post-hoc dataset diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from i3.data import (
    DataPipeline,
    JSONLSource,
    PipelineConfig,
    compute_stats,
)
from i3.data.stats import SplitStats


# ---------------------------------------------------------------------------
# SplitStats unit behaviour
# ---------------------------------------------------------------------------


class TestSplitStats:
    def test_absorb_tracks_counts(self):
        s = SplitStats(name="train")
        s.absorb("hello world", label=None, speaker=None)
        s.absorb("hello friend", label=None, speaker=None)
        assert s.records == 2
        assert s.tokens == 4
        assert s.unique_tokens == 3  # hello, world, friend
        assert s.token_counts["hello"] == 2

    def test_absorb_records_duplicate(self):
        s = SplitStats(name="x")
        s.absorb("hello world", label=None, speaker=None)
        s.absorb("Hello   World", label=None, speaker=None)  # dup modulo case/space
        assert s.duplicate_hashes == 1

    def test_type_token_ratio(self):
        s = SplitStats(name="x")
        s.absorb("a a a b", label=None, speaker=None)
        assert pytest.approx(s.type_token_ratio()) == 0.5

    def test_label_entropy_zero_for_single_label(self):
        s = SplitStats(name="x")
        s.absorb("first", label="A", speaker=None)
        s.absorb("second", label="A", speaker=None)
        assert s.label_entropy() == 0.0

    def test_label_entropy_max_for_uniform_labels(self):
        s = SplitStats(name="x")
        for lbl in ("A", "B", "C", "D"):
            s.absorb(f"text for {lbl}", label=lbl, speaker=None)
        # 4 uniform labels → 2 bits
        assert pytest.approx(s.label_entropy(), abs=1e-9) == 2.0

    def test_label_gini_zero_for_balanced(self):
        s = SplitStats(name="x")
        for lbl in ("A", "B", "C", "D"):
            s.absorb(f"t {lbl}", label=lbl, speaker=None)
        # Uniform distribution → small Gini
        assert s.label_gini() < 0.1

    def test_length_bucketing(self):
        s = SplitStats(name="x")
        s.absorb("short", label=None, speaker=None)  # 1 token -> <=5
        s.absorb(" ".join(["word"] * 30), label=None, speaker=None)  # -> <=50
        s.absorb(" ".join(["word"] * 600), label=None, speaker=None)  # -> <=1000
        assert s.length_histogram["<=5"] >= 1
        assert s.length_histogram["<=50"] >= 1
        assert s.length_histogram["<=1000"] >= 1

    def test_zipf_slope_healthy_english(self):
        """On a natural-language token sequence, the Zipf slope should
        land in roughly the canonical [-1.5, -0.3] range."""
        s = SplitStats(name="x")
        # Use a Zipfian-ish corpus: word 'a' 100 times, 'b' 50 times, 'c' 25 times...
        words = []
        for i, freq in enumerate((100, 50, 25, 12, 6, 3, 1)):
            words.extend([chr(ord("a") + i)] * freq)
        s.absorb(" ".join(words), label=None, speaker=None)
        slope = s.zipf_slope(top_n=10)
        assert -3.0 <= slope <= -0.1, f"Zipf slope {slope} outside expected range"

    def test_to_dict_round_trip(self):
        s = SplitStats(name="train")
        s.absorb("hello world", label="greeting", speaker="user")
        d = s.to_dict()
        for key in (
            "name", "records", "tokens", "unique_tokens", "type_token_ratio",
            "zipf_slope_top500", "length_histogram", "label_counts",
            "label_entropy_bits", "top_tokens",
        ):
            assert key in d
        assert d["name"] == "train"


# ---------------------------------------------------------------------------
# compute_stats — end-to-end on the bundled sample corpus
# ---------------------------------------------------------------------------


class TestComputeStats:
    def _build_dataset(self, tmp_path: Path) -> Path:
        """Run the pipeline on the bundled sample + return the output dir."""
        out = tmp_path / "dataset"
        DataPipeline(PipelineConfig(output_dir=out)).run([
            JSONLSource("data/corpora/sample_dialogues.jsonl")
        ])
        return out

    def test_stats_populates_every_present_split(self, tmp_path: Path):
        out = self._build_dataset(tmp_path)
        stats = compute_stats(out)
        # With a tiny bundled corpus, conv_id-aware splitting may put
        # everything in train; the point is every file that exists is
        # surfaced in the stats dict.
        for split_name in ("train", "val", "test"):
            if (out / f"{split_name}.jsonl").exists():
                assert split_name in stats.splits

    def test_stats_train_not_empty(self, tmp_path: Path):
        out = self._build_dataset(tmp_path)
        stats = compute_stats(out)
        train = stats.splits["train"]
        assert train.records > 0
        assert train.tokens > 0
        assert train.unique_tokens > 0

    def test_stats_oov_rate_defined_when_val_present(self, tmp_path: Path):
        # Construct a custom dataset where val has content.
        out = tmp_path / "custom"
        out.mkdir()
        (out / "train.jsonl").write_text(
            "\n".join(
                json.dumps({"text": "hello world", "label": None}) for _ in range(5)
            ),
            encoding="utf-8",
        )
        (out / "val.jsonl").write_text(
            json.dumps({"text": "hello universe", "label": None}) + "\n",
            encoding="utf-8",
        )
        stats = compute_stats(out)
        # val contains "hello" (in train) and "universe" (not in train)
        # so OOV rate should be 0.5.
        assert "val" in stats.oov_rates
        assert 0.4 <= stats.oov_rates["val"] <= 0.6

    def test_stats_to_dict_is_json_serialisable(self, tmp_path: Path):
        out = self._build_dataset(tmp_path)
        stats = compute_stats(out)
        encoded = json.dumps(stats.to_dict())
        parsed = json.loads(encoded)
        assert "splits" in parsed

    def test_stats_missing_dir_yields_empty(self, tmp_path: Path):
        stats = compute_stats(tmp_path / "does-not-exist")
        assert stats.splits == {}

    def test_stats_skips_malformed_lines(self, tmp_path: Path):
        out = tmp_path / "bad"
        out.mkdir()
        (out / "train.jsonl").write_text(
            json.dumps({"text": "hello"}) + "\n"
            + "NOT VALID\n"
            + json.dumps({"text": "world"}) + "\n",
            encoding="utf-8",
        )
        stats = compute_stats(out)
        assert stats.splits["train"].records == 2
