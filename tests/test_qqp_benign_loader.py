"""Tests for the QQP/PAWS-QQP benign-pair loader (T1.1 of the benign-population plan).

The benign population must come from an independent source (two people writing
the same question), never from our own paraphrase operator. q1 becomes the cache
entry and q2 is the incoming benign query. PAWS-QQP negatives are the F3
attack family: high lexical overlap, different meaning, human-labelled.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from sentry.research.pipeline.config import ExperimentConfig
from sentry.research.pipeline.io import read_records
from sentry.research.pipeline.schema import QueryRecord
from sentry.research.pipeline.sources import (
    build_benign_pair_records,
)
from sentry.research.pipeline.validate import validate_base_records


def _write_qqp_zip(raw_dir: Path, rows: list[tuple[str, str, int]]) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        text_buffer = io.StringIO()
        writer = csv.writer(text_buffer, delimiter="\t")
        writer.writerow(["id", "qid1", "qid2", "question1", "question2", "is_duplicate"])
        for index, (q1, q2, label) in enumerate(rows):
            writer.writerow([index, 2 * index, 2 * index + 1, q1, q2, label])
        archive.writestr("QQP/train.tsv", text_buffer.getvalue())
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "QQP.zip").write_bytes(buffer.getvalue())


def _write_paws_qqp(raw_dir: Path, rows: list[tuple[str, str, int]]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "paws_qqp_final_train.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "sentence1", "sentence2", "label"])
        for index, (q1, q2, label) in enumerate(rows):
            writer.writerow([index, q1, q2, label])


QQP_ROWS = [
    (f"QQP question alpha {i}?", f"QQP question alpha {i} rephrased?", 1)
    for i in range(8)
] + [
    (f"QQP unrelated one {i}?", f"QQP unrelated two {i}?", 0) for i in range(3)
]

PAWS_ROWS = [
    (f"PAWS positive {i}?", f"PAWS positive {i} restated?", 1) for i in range(4)
] + [
    (f"PAWS negative {i}?", f"PAWS negative {i} shuffled words?", 0) for i in range(4)
]


def _config(tmp_path: Path, **overrides) -> ExperimentConfig:
    payload = {
        "name": "qqp-test",
        "seed": 1234,
        "n_intents": 1,
        "qqp_benign_pairs": 6,
        "paws_qqp_positive_pairs": 2,
        "paws_qqp_negative_pairs": 3,
    }
    payload.update(overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload))
    return ExperimentConfig.from_json(config_path)


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    _write_qqp_zip(raw, QQP_ROWS)
    _write_paws_qqp(raw, PAWS_ROWS)
    return raw


class TestBuildBenignPairs:
    def test_counts_and_roles(self, raw_dir: Path, tmp_path: Path):
        config = _config(tmp_path)
        intent_rows, records = build_benign_pair_records(raw_dir, config)
        # 6 positive pairs (2 paws + 4 qqp) and 3 negative pairs.
        assert len(intent_rows) == 9
        by_role = {}
        for record in records:
            by_role.setdefault(record.query_role, []).append(record)
        assert len(by_role["canonical"]) == 9
        assert len(by_role["benign_query"]) == 6
        assert len(by_role["hard_negative"]) == 3

    def test_source_tags_and_labels(self, raw_dir: Path, tmp_path: Path):
        config = _config(tmp_path)
        _, records = build_benign_pair_records(raw_dir, config)
        sources = {record.source for record in records}
        assert sources <= {"qqp", "paws"}
        assert any(record.source == "qqp" for record in records)
        assert any(record.source == "paws" for record in records)
        for record in records:
            if record.query_role == "benign_query":
                # positive pairs go through the V pass, never trusted on the label
                assert record.semantic_label == "pending"
            if record.query_role == "hard_negative":
                # human-labelled different meaning: the F3 attack family
                assert record.semantic_label == "non_equivalent"
                assert record.source == "paws"

    def test_only_positive_pairs_from_positive_labels(self, raw_dir: Path, tmp_path: Path):
        config = _config(tmp_path)
        _, records = build_benign_pair_records(raw_dir, config)
        for record in records:
            if record.query_role != "benign_query":
                continue
            assert "unrelated" not in record.text
            assert "negative" not in record.text


    def test_pairs_share_intent_and_entry_is_canonical(self, raw_dir: Path, tmp_path: Path):
        config = _config(tmp_path)
        _, records = build_benign_pair_records(raw_dir, config)
        by_intent: dict[str, list[QueryRecord]] = {}
        for record in records:
            by_intent.setdefault(record.intent_id, []).append(record)
        for intent_id, group in by_intent.items():
            canonicals = [r for r in group if r.query_role == "canonical"]
            assert len(canonicals) == 1, intent_id
            for record in group:
                if record is not canonicals[0]:
                    assert record.parent_id == canonicals[0].record_id

    def test_deterministic_selection(self, raw_dir: Path, tmp_path: Path):
        config = _config(tmp_path)
        _, first = build_benign_pair_records(raw_dir, config)
        _, second = build_benign_pair_records(raw_dir, config)
        assert [r.record_id for r in first] == [r.record_id for r in second]


class TestValidateCarry:
    def test_benign_query_carried_only_after_v_pass(self, raw_dir: Path, tmp_path: Path):
        config = _config(tmp_path)
        _, records = build_benign_pair_records(raw_dir, config)
        records_path = tmp_path / "records.jsonl"
        out_path = tmp_path / "validated.jsonl"
        from sentry.research.pipeline.io import write_records

        write_records(records_path, records)
        # pending benign_query rows must force the annotation requirement
        with pytest.raises(RuntimeError, match="annotation"):
            validate_base_records(records_path, out_path, config)

        # annotate: 4 equivalent, 2 rejected by V
        annotations = tmp_path / "annotations.csv"
        benign = [r for r in records if r.query_role == "benign_query"]
        with annotations.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "record_id", "intent_id", "query_role", "generator", "parent_id",
                "text", "annotator_1_label", "annotator_2_label", "adjudicated_label",
            ])
            for index, record in enumerate(benign):
                label = "equivalent" if index < 4 else "non_equivalent"
                writer.writerow([
                    record.record_id, record.intent_id, record.query_role,
                    record.generator, record.parent_id, record.text,
                    label, label, "",
                ])
        report = validate_base_records(records_path, out_path, config, annotations)
        validated = read_records(out_path)
        carried = [r for r in validated if r.query_role == "benign_query"]
        assert len(carried) == 4
        assert all(r.semantic_label == "equivalent" for r in carried)
        negatives = [r for r in validated if r.query_role == "hard_negative"]
        assert len(negatives) == 3
        assert report["benign_query_validated"] == 4
        assert report["benign_query_dropped"] == 2
