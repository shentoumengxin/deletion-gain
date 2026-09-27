"""Recompute main-table BR/FPR on common, disjoint intent holdouts.

Consumes frozen scores only. Each split fits on benign calibration intents and
evaluates both arms on the other intents, using the same fitted rule for BR and
FPR. Split means are descriptive averages, not independent experimental repeats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.paper.paths import REPO_ROOT, data_root
from experiments.paper.rq1_detection.rq1_operating_points import intent_holdout
from experiments.paper.rq1_detection.v3_detect import DumpedColumns, load_dumped_rows
from sentry.cache.defense.calibrate import blocks_under, fit_rule_fence, joint_fence
from sentry.cache.defense.fence import CalibrationRow

POLICY = "multi[count:4+width:2:cap16]/runs"
BUDGET = 0.05
SET_NAMES = (("CAP", "cap", "lmp"), ("SCP", "scp", "scp"), ("KCA", "kca", "kca"))
BASELINES = {
    "Cosine": "cosine threshold",
    "Perplexity": "cond. perplexity",
    "Multi-model": "multi-encoder min-cos",
    "Key salting": "key salting",
    "LLM judge": "LLM judge (vote)",
    "Erase-and-check": "erase-and-check",
}


def split_masks(benign_intents, attack_intents, seed):
    """An intent has the same assignment on both arms and for every method."""
    held = intent_holdout(list(benign_intents) + list(attack_intents), 0.5, seed)
    test_b = np.array([i in held for i in benign_intents], bool)
    test_a = np.array([i in held for i in attack_intents], bool)
    fit_b = ~test_b
    assert set(np.asarray(benign_intents)[fit_b]).isdisjoint(
        set(np.asarray(benign_intents)[test_b]) | set(np.asarray(attack_intents)[test_a]))
    return fit_b, test_b, test_a


def score_decisions(benign, attack, fit_b):
    threshold = float(np.quantile(benign[fit_b], 1 - BUDGET))
    return benign > threshold, attack > threshold


def joint_decisions(benign, attack, fit_b):
    calibration = [r for r, keep in zip(benign, fit_b) if keep]
    rows = [CalibrationRow(r.base_cos, r.words, r.excess_span,
                           answer_loss=r.adl_best) for r in calibration]
    fence = fit_rule_fence(rows, "either", budget=BUDGET,
                           embedder="intfloat/e5-small-v2", policy=POLICY, echo_min=1)
    columns = DumpedColumns()
    fence, _ = joint_fence(fence, [(None, r) for r in calibration], columns, BUDGET)
    return tuple(np.asarray(blocks_under(fence, [(None, r) for r in arm], columns), bool)
                 for arm in (benign, attack))


def rates(blocked_b, blocked_a, test_b, test_a, poisoned):
    success = test_a & poisoned
    assert test_b.any() and test_a.any() and success.any()
    return {"fpr": float(blocked_b[test_b].mean()),
            "br_all": float(blocked_a[test_a].mean()),
            "br_success": float(blocked_a[success].mean())}


def aggregate(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "sd_across_splits": float(values.std()),
            "min": float(values.min()), "max": float(values.max())}


def run(root: Path, splits=200):
    sources = {}

    def read(path):
        payload = path.read_bytes()
        namespace, relative = ("data", path.relative_to(root)) if path.is_relative_to(root) else (
            "code", path.relative_to(REPO_ROOT))
        sources[f"{namespace}:{relative.as_posix()}"] = {
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        return json.loads(payload)

    paper = root / "runs/paper"
    published = read(REPO_ROOT / "experiments/paper/results/answer_check_20260909/main_table.json")
    old_baselines = read(paper / "perrow/ci_table1.json")
    result = {"schema_version": 1, "protocol": {
        "budget": BUDGET, "holdout": 0.5, "seeds": [0, splits - 1], "n_splits": splits,
        "split_unit": "intent_id", "split_function": "sha256(seed:intent_id), first 8 bytes < 0.5",
        "fit_population": "benign calibration intents only",
        "evaluation_population": "benign and attack rows from held-out intents only",
        "aggregation": "arithmetic mean of per-split rates; overlapping splits are not independent trials",
        "success_labels": "frozen poisoned flags used by the original main table; no answer re-judging",
        "retrieval_conditioning": "none; candidate-pair detection, not end-to-end ASR",
        "policy": POLICY, "answer_rule": "either", "echo_min": 1, "storage_dtype": "float16",
        "model_calls": 0,
    }, "sets": {}, "sources": sources}
    for tag, short, stem in SET_NAMES:
        dump_path = paper / f"perrow/answer_check_rows/e5-small-v2__{stem}.jsonl"
        payload = dump_path.read_bytes()
        sources[f"data:{dump_path.relative_to(root)}"] = {
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        benign, attack = load_dumped_rows(dump_path)
        bi = [r.intent_id for r in benign]
        ai = [r.intent_id for r in attack]
        assert all(r.poisoned is not None for r in attack)
        poisoned = np.array([r.poisoned is True for r in attack])
        old = published["cells"][f"e5-small-v2|{stem}"]
        assert (len(benign), len(attack)) == (old["n_genuine"], old["n_attack"])
        assert int(poisoned.sum()) == old_baselines[tag]["n_poisoned"]
        sweep = read(paper / f"granularity32/scores/{short}.json")["rows"]
        sb = [r for r in sweep if r["arm"] == "genuine"]
        sa = [r for r in sweep if r["arm"] == "attack"]
        assert [r["intent_id"] for r in sb] == bi and [r["intent_id"] for r in sa] == ai
        methods = {"Cosine": ([-r["cos"] for r in sb], [-r["cos"] for r in sa])}
        ppl = read(paper / f"operating/{stem}_baseline_scores.json")["cond. perplexity"]
        assert ppl["benign_intents"] == bi and len(ppl["attack"]) == len(ai)
        methods["Perplexity"] = (ppl["benign"], ppl["attack"])
        for name, filename, score in (
            ("Multi-model", f"multiview_{short}_e5.json",
             lambda r: -min(r["cos_primary"], r["cos_aux0"], r["cos_aux1"])),
            ("Key salting", f"salting_{short}_e5.json", lambda r: -r["prefix:0"]),
            ("LLM judge", f"judge_vote_{short}.json", lambda r: r["score"]),
            ("Erase-and-check", f"eac_{short}.json", lambda r: r["p_max"]),
        ):
            rows = read(paper / "perrow" / filename)["rows"]
            arms = [[r for r in rows if r["arm"] == arm] for arm in ("genuine", "attack")]
            for actual, expected in zip(arms, (benign, attack)):
                assert [(r["intent_id"], r["record_id"]) for r in actual] == [
                    (r.intent_id, r.record_id) for r in expected], (tag, name, "alignment")
            methods[name] = tuple([score(r) for r in arm] for arm in arms)
        methods = {name: tuple(np.asarray(arm, float) for arm in pair)
                   for name, pair in methods.items()}
        assert all(np.isfinite(arm).all() for pair in methods.values() for arm in pair)
        full_b, full_a = np.ones(len(bi), bool), np.ones(len(ai), bool)
        parity = {}
        for name, pair in methods.items():
            actual = rates(*score_decisions(*pair, full_b), full_b, full_a, poisoned)
            expected = old_baselines[tag]["methods"][BASELINES[name]]
            assert np.isclose(actual["br_all"], expected["att"][0], rtol=0, atol=1e-12), (tag, name)
            assert np.isclose(actual["br_success"], expected["succ"][0], rtol=0, atol=1e-12), (tag, name)
            parity[name] = actual
        parity["Ours"] = rates(*joint_decisions(benign, attack, full_b), full_b, full_a, poisoned)
        for metric, field in (("br_all", "tpr_all_planted_joint_eta"),
                              ("br_success", "tpr_poisoned_joint_eta"),
                              ("fpr", "benign_block_rate_in_sample_joint_eta")):
            assert np.isclose(parity["Ours"][metric], old["answer_rules"]["either"][field],
                              rtol=0, atol=1e-12), (tag, metric)
        counts = {k: [] for k in ("calibration_benign", "test_benign", "test_attack", "test_success")}
        measured = {name: [] for name in (*methods, "Ours")}
        for seed in range(splits):
            fit_b, test_b, test_a = split_masks(bi, ai, seed)
            for key, value in zip(counts, (fit_b.sum(), test_b.sum(), test_a.sum(), (test_a & poisoned).sum())):
                counts[key].append(int(value))
            assert min(fit_b.sum(), test_b.sum(), test_a.sum()) >= 20
            for name, pair in methods.items():
                measured[name].append(rates(*score_decisions(*pair, fit_b), test_b, test_a, poisoned))
            measured["Ours"].append(rates(*joint_decisions(benign, attack, fit_b), test_b, test_a, poisoned))
        result["sets"][tag] = {"benign_corpus": "NQ" if tag == "KCA" else "ComQA",
            "n_benign": len(bi), "n_attack": len(ai), "n_success": int(poisoned.sum()),
            "n_benign_intents": len(set(bi)), "n_attack_intents": len(set(ai)),
            "split_counts": {key: aggregate(value) for key, value in counts.items()},
            "full_cohort_parity": parity,
            "methods": {name: {key: aggregate([r[key] for r in values])
                               for key in ("fpr", "br_all", "br_success")}
                        for name, values in measured.items()}}
        result["sets"][tag]["methods"]["LaCache"] = {
            "status": "unverified", "reason": "Available summaries lack per-row answer-prefix scores needed to refit and evaluate common intent holdouts.",
            "fpr": None, "br_all": None, "br_success": None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=200)
    args = parser.parse_args()
    if args.splits < 1:
        parser.error("--splits must be positive")
    result = run(data_root(args.data_root), args.splits)
    result["source_code"] = {"path": str(Path(__file__).relative_to(REPO_ROOT)),
                             "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for tag, cell in result["sets"].items():
        for name, metrics in cell["methods"].items():
            if metrics.get("status") == "unverified":
                print(tag, name, "unverified")
            else:
                print(tag, name, *(f"{k}={metrics[k]['mean']:.6f}" for k in ("fpr", "br_all", "br_success")))


if __name__ == "__main__":
    main()
