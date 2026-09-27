from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from sentry.research.pipeline.cli import run_smoke
from sentry.research.pipeline.config import ExperimentConfig
from sentry.research.pipeline.schema import QueryRecord
from sentry.research.pipeline.sources import (
    build_smoke_records,
    deterministic_split,
)


def test_deterministic_split_is_stable_and_intent_scoped():
    config = ExperimentConfig(name="test", seed=17, n_intents=3)
    first = deterministic_split("intent-a", config)
    assert first == deterministic_split("intent-a", config)
    assert first in {"train", "val", "test"}


def test_query_record_schema_rejects_invalid_role():
    record = QueryRecord(
        record_id="r1",
        intent_id="i1",
        source="test",
        split="train",
        query_role="not-a-role",
        text="hello",
        generator="test",
        seed=1,
    )
    try:
        record.validate()
    except ValueError as exc:
        assert "query_role" in str(exc)
    else:
        raise AssertionError("invalid query role was accepted")


def test_smoke_build_has_no_intent_split_leakage(tmp_path):
    config = ExperimentConfig(name="test", seed=17, n_intents=3)
    records = build_smoke_records(tmp_path, config)
    split_by_intent = {}
    for record in records:
        split_by_intent.setdefault(record.intent_id, set()).add(record.split)
    assert split_by_intent
    assert all(len(splits) == 1 for splits in split_by_intent.values())


def test_generate_ndss_builds_fluent_collision_records(tmp_path):
    from sentry.research.pipeline.generate import NDSS_BUILDERS, generate_ndss
    from sentry.research.pipeline.io import read_records, write_records

    config = ExperimentConfig(name="test", seed=17, n_intents=2)
    # Two canonicals with entity answers so _wrong_payloads has a same-type pool.
    canonicals = [
        QueryRecord(
            record_id="c1",
            intent_id="i1",
            source="test",
            split="train",
            query_role="canonical",
            text="What is the capital of France?",
            generator="test",
            seed=1,
            canonical_answer=["Paris"],
        ),
        QueryRecord(
            record_id="c2",
            intent_id="i2",
            source="test",
            split="train",
            query_role="canonical",
            text="What is the capital of Japan?",
            generator="test",
            seed=1,
            canonical_answer=["Tokyo"],
        ),
    ]
    records_path = tmp_path / "datasets" / "records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    write_records(records_path, canonicals)

    added = generate_ndss(records_path, config)
    assert added == len(canonicals) * len(NDSS_BUILDERS)
    records = read_records(records_path)
    ndss = [r for r in records if r.query_role == "ndss"]
    assert len(ndss) == added
    styles = {r.generator for r in ndss}
    assert styles == set(NDSS_BUILDERS)
    # Each Qadv keeps Qtarget verbatim (NDSS Eq.2) and carries a wrong poison target.
    by_style = {}
    for r in ndss:
        by_style.setdefault(r.generator, []).append(r)
    for style, group in by_style.items():
        assert all(r.metadata.get("ndss_style") == style for r in group)
        assert all(r.metadata.get("poison_target") for r in group)
        assert all(r.semantic_label == "non_equivalent" for r in group)
    # Qtarget substring present -> high-cosine collision.
    for r in ndss:
        assert "France" in r.text or "Japan" in r.text


def test_three_intent_smoke_pipeline(tmp_path, monkeypatch):
    from sentry.research.pipeline import cli
    from sentry.research.pipeline.io import read_records

    def deny_generation(*args, **kwargs):
        raise AssertionError("The corpus smoke must not generate candidates or attacks")

    for name in (
        "generate_legal_candidates", "generate_wu", "generate_mock_gcg",
        "generate_ndss", "generate_ndss_matched", "generate_long_legit",
    ):
        monkeypatch.setattr(cli, name, deny_generation)
    config = ExperimentConfig.from_json(
        REPO_ROOT / "experiments/paper/configs/corpus/smoke.json"
    )
    result = run_smoke(tmp_path, config)
    assert result["smoke"] is True
    assert result["validation"]["intent_count"] == 3
    assert result["validation"]["complete_legal_intents"] == 3
    records = read_records(tmp_path / "datasets" / "embedded_records.jsonl")
    assert {record.query_role for record in records} == {"canonical", "legal"}
    assert all(record.semantic_label == "equivalent" for record in records)
    with np.load(tmp_path / "embeddings" / "query_embeddings.npz", allow_pickle=False) as vectors:
        assert set(vectors.files) == {"record_ids", "embeddings", "model_name"}
        assert vectors["record_ids"].tolist() == [record.record_id for record in records]
        assert vectors["embeddings"].shape == (9, 256)
        np.testing.assert_allclose(np.linalg.norm(vectors["embeddings"], axis=1), 1, atol=1e-6)
    manifest = json.loads((tmp_path / "embeddings" / "embedding_manifest.json").read_text())
    assert manifest == result["embedding"]
    assert manifest["record_count"] == len(records)
    assert not (tmp_path / "reports" / "REPORT.md").exists()
    assert not (tmp_path / "metrics" / "pair_metrics.csv").exists()


@pytest.mark.parametrize("option", ["--work-root", "--workspace"])
def test_corpus_cli_work_root_alias(option, tmp_path):
    from sentry.research.pipeline.cli import build_parser

    args = build_parser().parse_args([
        "--config", "fixture.json", option, str(tmp_path), "embed",
        "--records", "legacy/final_records.jsonl", "--smoke",
    ])
    assert args.workspace == str(tmp_path)
    assert args.records == "legacy/final_records.jsonl"


def test_retained_configs_are_loadable():
    from sentry.research.pipeline.instruction_benign import InstructionExperimentConfig

    paths = sorted((REPO_ROOT / "experiments/paper/configs").rglob("*.json"))
    loaded = []
    for path in paths:
        payload = json.loads(path.read_text())
        if "n_intents" not in payload:
            continue
        cls = InstructionExperimentConfig if "instruction_benign" in path.name else ExperimentConfig
        config = cls.from_json(path)
        assert config.name == payload["name"]
        loaded.append(path)
    assert len(loaded) >= 10


def test_adaptive_injection_families_defeat_the_structural_decomposer():
    """P3: the two ablation families must reduce to a single clause.

    The decomposition prototype splits only on ``[.?!]`` and its coverage
    statistic equals base_cosine exactly whenever it returns one clause, so a
    single-clause variant drives its incremental signal to exactly zero rather
    than merely degrading it.
    """
    import re

    from sentry.research.pipeline.generate import (
        NDSS_ADAPTIVE_BUILDERS,
        NDSS_BUILDERS,
    )

    def clauses(text):
        return [p for p in re.split(r"(?<=[.?!])\s+", text.strip()) if len(p) >= 3]

    question, payload = "Who wrote Hamlet?", "Paris"
    for builder in NDSS_BUILDERS.values():
        assert len(clauses(builder(question, payload))) == 2
    for builder in NDSS_ADAPTIVE_BUILDERS.values():
        text = builder(question, payload)
        assert len(clauses(text)) == 1
        assert payload in text


def _injection_fixture(tmp_path, n_intents=3):
    """Canonical + validated legitimate paraphrases + one long GCG attack each."""
    from sentry.research.pipeline.io import write_records

    records = []
    for index in range(n_intents):
        intent_id = f"i{index}"
        canonical = QueryRecord(
            record_id=f"c{index}",
            intent_id=intent_id,
            source="t",
            split="test",
            query_role="canonical",
            text=f"Who won the {index}th championship in the modern era?",
            generator="t",
            seed=1,
            semantic_label="equivalent",
            canonical_answer=["1999"],
        )
        records.append(canonical)
        for j in range(3):
            records.append(
                QueryRecord(
                    record_id=f"l{index}{j}",
                    intent_id=intent_id,
                    source="t",
                    split="test",
                    query_role="legal",
                    text=f"Which team took the {index}th modern title?" + " x" * j,
                    generator="api",
                    seed=1,
                    semantic_label="equivalent",
                    parent_id=canonical.record_id,
                    canonical_answer=["1999"],
                )
            )
        records.append(
            QueryRecord(
                record_id=f"g{index}",
                intent_id=intent_id,
                source="t",
                split="test",
                query_role="gcg",
                text=canonical.text + " " + "describing.\\ + similarly Now" * 6,
                generator="gcg",
                seed=1,
                semantic_label="non_equivalent",
                parent_id=canonical.record_id,
                canonical_answer=["1999"],
            )
        )
    path = tmp_path / "records.jsonl"
    write_records(path, records)
    return path


def test_matched_injection_respects_the_length_budget(tmp_path):
    from sentry.research.pipeline.generate import generate_ndss_matched
    from sentry.research.pipeline.io import read_records

    path = _injection_fixture(tmp_path)
    config = ExperimentConfig(
        name="t",
        seed=7,
        n_intents=3,
        matched_attempts_per_intent=9,
        matched_length_tolerance=0.25,
    )
    result = generate_ndss_matched(path, config, smoke=True)
    assert result["attempts"] == 27
    # Every attempt is logged, so a yield denominator exists.
    log = [
        json.loads(line)
        for line in (tmp_path / "matched_attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(log) == result["attempts"]
    assert {row["strategy"] for row in log} == {"compress_append", "blend", "fuse"}
    promoted = [
        record
        for record in read_records(path)
        if record.generator.startswith("ndss_matched")
    ]
    assert len(promoted) == result["records_added"]
    # C2 holds for everything promoted, and no promoted candidate carries the
    # sentence boundary the templated family relies on.
    for record in promoted:
        median = record.metadata["legit_median_chars"]
        assert abs(len(record.text) - median) <= 0.25 * median + 1
        assert len(re.findall(r"[.?!]\s+\S", record.text)) == 0


def test_matched_injection_requires_validated_legitimate_paraphrases(tmp_path):
    import pytest

    from sentry.research.pipeline.generate import generate_ndss_matched
    from sentry.research.pipeline.io import read_records, write_records

    path = _injection_fixture(tmp_path)
    # Strip the validated legitimate records: the budget has no reference left.
    write_records(
        path, [r for r in read_records(path) if r.query_role != "legal"]
    )
    with pytest.raises(ValueError, match="validated legitimate"):
        generate_ndss_matched(path, ExperimentConfig(name="t", seed=7, n_intents=3))


def test_long_legit_bridges_the_gap_legitimate_paraphrases_leave(tmp_path):
    from sentry.research.pipeline.generate import generate_long_legit
    from sentry.research.pipeline.io import read_records

    path = _injection_fixture(tmp_path)
    config = ExperimentConfig(name="t", seed=7, n_intents=3, long_legit_per_intent=4)
    result = generate_long_legit(
        path, config, smoke=True, attack_generators=("gcg",)
    )
    assert result["records_added"] > 0
    records = read_records(path)
    added = [r for r in records if r.generator == "long_legit"]
    # V decides equivalence, so these arrive as pending candidates.
    assert {r.query_role for r in added} == {"legal_candidate"}
    assert {r.semantic_label for r in added} == {"pending"}
    assert len({r.text for r in added}) == len(added)
    for intent_id in {r.intent_id for r in added}:
        legit_max = max(
            len(r.text)
            for r in records
            if r.query_role == "legal" and r.intent_id == intent_id
        )
        attack_len = max(
            len(r.text)
            for r in records
            if r.query_role == "gcg" and r.intent_id == intent_id
        )
        lengths = sorted(len(r.text) for r in added if r.intent_id == intent_id)
        assert all(legit_max < value <= attack_len for value in lengths)
        # Spread across the gap, not clustered at one end: that is what makes
        # mid-range length strata usable.
        assert lengths[-1] - lengths[0] > 0.3 * (attack_len - legit_max)


def test_long_legit_is_idempotent_and_family_scoped(tmp_path):
    from sentry.research.pipeline.generate import (
        generate_long_legit,
        generate_ndss_matched,
    )

    path = _injection_fixture(tmp_path)
    config = ExperimentConfig(
        name="t",
        seed=7,
        n_intents=3,
        long_legit_per_intent=4,
        matched_attempts_per_intent=9,
        matched_length_tolerance=0.25,
    )
    first = generate_long_legit(path, config, smoke=True, attack_generators=("gcg",))
    assert first["records_added"] > 0
    # Short matched injections must not drag the GCG target band downward.
    generate_ndss_matched(path, config, smoke=True)
    again = generate_long_legit(path, config, smoke=True, attack_generators=("gcg",))
    assert again["candidates_built"] == first["candidates_built"]
    assert again["records_added"] == 0


def test_config_rejects_out_of_range_new_thresholds():
    import pytest

    base = dict(name="t", seed=1, n_intents=1)
    for bad in [
        {"matched_length_tolerance": 0.0},
        {"matched_attempts_per_intent": 0},
        {"fluency_percentile": 1.5},
        {"min_feasible_attacks": 0},
        {"long_legit_per_intent": -1},
    ]:
        with pytest.raises(ValueError):
            ExperimentConfig(**base, **bad).validate()

