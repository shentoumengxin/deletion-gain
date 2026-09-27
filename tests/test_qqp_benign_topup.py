"""Top-up semantics for the benign-pair loader (V-pass rejection follow-up)."""

from __future__ import annotations

import json
from pathlib import Path

from sentry.research.pipeline.sources import (
    append_benign_pair_records,
    build_benign_pair_records,
)
from tests.test_qqp_benign_loader import (
    PAWS_ROWS,
    QQP_ROWS,
    _config,
    _write_paws_qqp,
    _write_qqp_zip,
)


def test_topup_skips_existing_texts(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_qqp_zip(raw, QQP_ROWS)
    _write_paws_qqp(raw, PAWS_ROWS)

    config = _config(tmp_path)  # 6 positive (2 paws + 4 qqp), 3 negative
    dataset = tmp_path / "datasets"
    dataset.mkdir()
    # Simulate an existing build: first run wrote these records.
    _, first = build_benign_pair_records(raw, config)
    existing_texts = {r.text for r in first[:3]}  # pretend a few already exist
    from sentry.research.pipeline.io import write_records

    write_records(dataset / "records.jsonl", first[:3])
    (dataset / "intents.jsonl").write_text("{}\n")
    (dataset / "build_manifest.json").write_text(json.dumps({"config": {}}))

    stats = append_benign_pair_records(dataset, raw, config)
    from sentry.research.pipeline.io import read_records

    final = read_records(dataset / "records.jsonl")
    texts = [r.text for r in final]
    assert len(texts) == len(set(texts)), "top-up must not duplicate texts"
    skipped = sum(1 for t in existing_texts if t in texts)
    assert skipped == 3
    assert stats["positive_pairs"] >= 0
