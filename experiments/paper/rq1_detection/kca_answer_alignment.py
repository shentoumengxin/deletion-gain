"""Correct the e5 KCA main-table cell using frozen, answer-aligned scores.

Preserves the original full-cohort calibration, intent bootstrap seeds, and
denominators. No model calls. The original aggregate is checked before the
corrected result is emitted, and the original row dump remains unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection.ci_answer_check import (
    Arm, _agree, _blocked, _fast, _fit, _shipped, boot_rule,
)
from experiments.paper.rq1_detection.v3_detect import load_dumped_rows
from experiments.paper.rq2_robustness.sc_ipi_replay import run as validate_inputs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(benign, attack):
    """Use the original ASR bootstrap seed and resampling order."""
    b, a = Arm(benign), Arm(attack)

    def decisions(bs, at):
        eta_a, _, eta = _fit(bs, "either", .05)
        floor = float(np.quantile(bs.cos, .05))
        return {"None": np.ones(len(at.cos), bool),
                "Cosine": at.cos >= floor,
                "Ours": ~_blocked(at, eta, "either", eta_a)}, {
                    "eta": eta, "eta_a": eta_a, "cosine_floor": floor}

    accepted, thresholds = decisions(b.all(), a.all())
    _agree(_fast(b.all(), a.all(), "either", .05),
           _shipped(b.all(), a.all(), "either", .05), "full cohort")
    joint = (a.cos >= .90) & a.poisoned
    bg = [np.flatnonzero(b.intents == k) for k in np.unique(b.intents)]
    ag = [np.flatnonzero(a.intents == k) for k in np.unique(a.intents)]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(b"KCA").digest()[:8], "little"))
    draws = {m: [] for m in accepted}
    for replicate in range(2000):
        bi = np.concatenate([bg[j] for j in rng.integers(len(bg), size=len(bg))])
        ai = np.concatenate([ag[j] for j in rng.integers(len(ag), size=len(ag))])
        bs, at = b.take(bi), a.take(ai)
        ac, _ = decisions(bs, at)
        if replicate < 3:
            _agree(_fast(bs, at, "either", .05), _shipped(bs, at, "either", .05),
                   f"ASR bootstrap {replicate}")
        for method in ac:
            draws[method].append(float((joint[ai] & ac[method]).mean()))
    methods = {}
    for method in accepted:
        n = int((joint & accepted[method]).sum())
        methods[method] = {"successes": n, "n": len(attack), "asr": n / len(attack),
                           "ci95": np.percentile(draws[method], [2.5, 97.5]).tolist()}
    families = {}
    for family in sorted({r.family for r in attack}):
        mask = np.array([r.family == family for r in attack])
        families[family] = {"n": int(mask.sum()),
                            "br": float((~accepted["Ours"])[mask].mean()),
                            **{m: {"successes": int((joint & accepted[m] & mask).sum()),
                                   "asr": float((joint & accepted[m])[mask].mean())}
                               for m in accepted}}
    br_ci, _ = boot_rule("KCA|e5-small-v2|either|0.05|All", b, a, "either", .05, "All")
    budgets = {str(budget): {
        "ours_br": _shipped(b.all(), a.all(), "either", budget)["block_rate_joint_eta"],
        "cosine_br": _shipped(b.all(), a.all(), "cosine_only", budget)["block_rate_joint_eta"],
    } for budget in (.10, .05, .02, .01)}
    return {"n": len(attack), "n_benign": len(benign), "n_attack_intents": len(ag),
            "n_poisoned_backend": int(a.poisoned.sum()), "thresholds": thresholds,
            "methods": methods, "families": families, "budgets": budgets,
            "br": {"blocked": int((~accepted["Ours"]).sum()), "rate": br_ci[0],
                   "ci95": list(br_ci[1:])}}, accepted["Ours"]


def run(root, previous_summary):
    aligned = root / "runs/sc_ipi_ablation_20260920"
    verified = validate_inputs(root, aligned)
    old_path = root / "runs/paper/perrow/answer_check_rows/e5-small-v2__kca.jsonl"
    new_path = aligned / "scores.jsonl"
    old_b, old_a = load_dumped_rows(old_path)
    new_b, new_a = load_dumped_rows(new_path)
    assert old_b == new_b
    mapping = {r.record_id: r for r in new_a}
    assert len(mapping) == len(new_a) == len(old_a)
    new_a = [mapping[r.record_id] for r in old_a]
    immutable = ("arm", "intent_id", "record_id", "family", "base_cos", "words",
                 "excess_span", "poisoned", "has_answer")
    assert all(all(getattr(o, k) == getattr(n, k) for k in immutable)
               for o, n in zip(old_a, new_a))
    old, old_accept = summarize(old_b, old_a)
    new, new_accept = summarize(new_b, new_a)
    previous = json.loads(previous_summary.read_text())["cells"]["KCA"]
    assert previous["source_sha256"] == sha(old_path)
    for key in ("n", "n_benign", "n_attack_intents", "n_poisoned_backend", "thresholds",
                "methods", "families"):
        assert old[key] == previous[key], key
    for method in ("None", "Cosine"):
        assert old["methods"][method] == new["methods"][method]
    assert old["thresholds"] == new["thresholds"]
    sc = verified["evaluations"]["full_cohort"]["methods"]["full_joint"]
    assert new["methods"]["Ours"]["successes"] == sc["successes"]
    assert new["methods"]["Ours"]["asr"] == sc["asr"]
    changed = old_accept != new_accept
    return {
        "scope": "e5 KCA main-table answer alignment; full cohort, original calibration",
        "protocol": {"n_attack": 798, "n_benign": 499, "bootstrap_replicates": 2000,
                     "bootstrap_unit": "intent_id", "bootstrap_refits_thresholds": True,
                     "bootstrap_seeds": "unchanged: SHA256(KCA) for ASR; original boot_rule key for BR",
                     "retrieval_floor": .90, "nominal_fpr": .05, "model_calls": 0},
        "sources": {str(p.relative_to(root)): sha(p) for p in
                    [old_path, new_path, aligned / "inputs.json", aligned / "receipt.json"]},
        "previous_summary_sha256": sha(previous_summary),
        "source_code_sha256": sha(Path(__file__)),
        "answer_identity_mismatches": verified["provenance"]["main_table_scored_attack_answer_revision_mismatches"],
        "changed_decisions": {"total": int(changed.sum()),
                              "newly_accepted": int((changed & new_accept).sum()),
                              "newly_blocked": int((changed & ~new_accept).sum())},
        "checks": {"previous_point_estimates_and_asr_intervals": "exact equality",
                   "benign_rows": "exact equality", "non_answer_fields": "exact equality",
                   "thresholds": "exact equality", "cosine_and_undefended": "exact equality",
                   "vectorized_bootstrap_vs_shipped_rule": "full sample and first three draws",
                   "sc_ipi_full_cohort_parity": True},
        "before": old, "after": new,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--previous-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data_root.resolve(), args.previous_summary.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: {"ASR": result[k]["methods"]["Ours"], "BR": result[k]["br"]}
                      for k in ("before", "after")}, indent=2))
