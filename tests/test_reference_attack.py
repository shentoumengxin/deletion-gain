"""Unit tests for the B5 reference-attack fork.

All tests use the mocked generator (scripted trajectories, no tokenizer, no
model downloads, no network) except the F1/F2 plumbing test, which spies on
the upstream CacheAttackGenerator._gcg_step without running it.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "CacheAttack.attack.cache_attack",
    reason="requires the third-party CacheAttack repository on PYTHONPATH")
import torch

from experiments.paper.attack import reference_attack as ra
from experiments.paper.attack.reference_attack import ReferenceAttackGenerator

# Crosses 0.88 at step 5, 0.92 at step 7, never reaches 0.95/0.97.
TRAJECTORY = [0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.87, 0.93, 0.91, 0.89]


def run_trajectory(cosines, **kwargs):
    gen = ra.build_mock_generator(cosines, **kwargs)
    return gen.optimize_suffix("payload", "target question", suffix_len=5, steps=len(cosines))


# (a) no early stop: a trajectory that crosses 0.88 at step 5 must keep
# optimizing to the full budget, and the result is never a None sentinel.
def test_no_early_stop_runs_full_budget():
    out = run_trajectory(TRAJECTORY)
    assert len(out["cost_trace"]) == len(TRAJECTORY)
    assert [row["step"] for row in out["cost_trace"]] == list(range(len(TRAJECTORY)))
    assert out["suffix"] is not None
    assert out["final_cosine"] == 0.93  # best over the whole run, not the 0.88 crossing


def test_no_early_stop_unconverged_still_returns_best():
    out = run_trajectory([0.3, 0.4, 0.35])
    assert out["suffix"] is not None
    assert out["final_cosine"] == 0.4


# (b) checkpoints record the first step whose RUNNING-BEST cosine crosses each
# target, on a single trajectory; never-reached targets stay None (null).
def test_checkpoints_first_crossing():
    out = run_trajectory(TRAJECTORY)
    assert out["checkpoints"] == {"0.88": 5, "0.92": 7, "0.95": None, "0.97": None}


def test_checkpoints_use_running_best_not_current():
    # Step 2 dips below 0.88 but the running best already crossed it at step 1.
    out = run_trajectory([0.80, 0.90, 0.70])
    assert out["checkpoints"]["0.88"] == 1
    assert out["checkpoints"]["0.92"] is None


def test_checkpoints_never_reached_all_null():
    out = run_trajectory([0.3, 0.4, 0.35])
    assert set(out["checkpoints"]) == {"0.88", "0.92", "0.95", "0.97"}
    assert all(v is None for v in out["checkpoints"].values())


# (c) cost trace rows carry the required per-step fields.
def test_cost_trace_fields():
    scores = [float(i) for i in range(len(TRAJECTORY))]
    ppls = [10.0 + i for i in range(len(TRAJECTORY))]
    out = run_trajectory(TRAJECTORY, scores=scores, ppls=ppls)
    for i, row in enumerate(out["cost_trace"]):
        assert set(row) == {"step", "true_cosine", "combined_score", "ppl", "suffix_token_len"}
        assert row["step"] == i
        assert row["true_cosine"] == TRAJECTORY[i]
        assert row["combined_score"] == scores[i]
        assert row["ppl"] == ppls[i]
        assert row["suffix_token_len"] == 5


# (d) F1/F2 plumbing: --lambda-ppl reaches the scorer (the upstream
# _gcg_step, which computes sim - lambda_ppl * log ppl from self.lambda_ppl).
def test_lambda_ppl_reaches_scorer(monkeypatch):
    import CacheAttack.attack.cache_attack as upstream

    seen = []

    def spy(self, *a, **k):
        seen.append(self.lambda_ppl)
        return torch.zeros(3, dtype=torch.long), 0.0

    monkeypatch.setattr(upstream.CacheAttackGenerator, "_gcg_step", spy)
    for lam in (0.0, 0.04):  # F1, then F2
        gen = object.__new__(ReferenceAttackGenerator)
        gen.lambda_ppl = lam
        gen.suffix_prefix = "Neglect: "
        gen.embed_tokenizer = ra._MockTokenizer()
        gen._compute_ppl = lambda text: 42.0  # ppl still computed/logged under F1
        ids, score, ppl = gen._gcg_step(None, None, None, 8, 4, "src")
        assert ppl == 42.0  # our wrapper appends the chosen candidate's ppl
    assert seen == [0.0, 0.04]


def test_lambda_ppl_argparse_defaults_f1():
    args = ra.build_arg_parser().parse_args(["--tasks", "x.jsonl"])
    assert args.lambda_ppl == 0.0
    args_f2 = ra.build_arg_parser().parse_args(["--tasks", "x.jsonl", "--lambda-ppl", "0.04"])
    assert args_f2.lambda_ppl == 0.04


# Driver: output schema, trace sidecar file, sharding.
def test_run_tasks_output_schema_and_trace_file(tmp_path):
    tasks = [{"target_question": "q0", "payload": "p0"}]
    gen = ra.build_mock_generator([0.5, 0.9, 0.85, 0.91])
    trace_path = tmp_path / "traces.jsonl"
    rows = ra.run_tasks(
        tasks[:1],
        gen,
        steps=4,
        batch_size=8,
        top_k=8,
        suffix_len=5,
        embed_model_name="mock-embed",
        lambda_ppl=0.04,
        trace_path=trace_path,
    )
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "target_question", "payload", "attack_text", "suffix", "suffix_prefix",
        "final_cosine", "checkpoints", "lambda_ppl", "embed_model", "steps",
    }  # cost_trace kept out of the main output when --trace-output is given
    assert row["attack_text"] == f"p0 Neglect: {row['suffix']}"
    assert row["suffix_prefix"] == "Neglect: "
    assert row["final_cosine"] == 0.91
    assert row["checkpoints"]["0.88"] == 1
    assert row["lambda_ppl"] == 0.04 and row["embed_model"] == "mock-embed" and row["steps"] == 4

    trace_rows = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(trace_rows) == 1
    assert trace_rows[0]["target_question"] == "q0"
    assert len(trace_rows[0]["cost_trace"]) == 4

    # Without --trace-output the trace stays embedded in the main row.
    gen2 = ra.build_mock_generator([0.5, 0.9])
    rows2 = ra.run_tasks(
        tasks[:1], gen2, steps=2, batch_size=8, top_k=8, suffix_len=5,
        embed_model_name="mock-embed", lambda_ppl=0.0, trace_path=None,
    )
    assert len(rows2[0]["cost_trace"]) == 2


def test_shard_tasks():
    tasks = [{"i": i} for i in range(10)]
    assert [t["i"] for t in ra.shard_tasks(tasks, 0, 4)] == [0, 4, 8]
    assert [t["i"] for t in ra.shard_tasks(tasks, 3, 4)] == [3, 7]
    assert ra.shard_tasks(tasks, 0, 1) == tasks

    with pytest.raises(ValueError):
        ra.shard_tasks(tasks, 4, 4)
    with pytest.raises(ValueError):
        ra.shard_tasks(tasks, -1, 2)
