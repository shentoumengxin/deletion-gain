"""Assemble the twelve detection cells into one small JSON for the code branch."""
import json, glob, os

from sentry.artifacts import data_path
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, help="reviewable output; do not overwrite the published summary before comparing provenance")
args = parser.parse_args()
D = str(data_path("runs", "answer_check_20260909"))
RULES = ("dg_only", "adl", "echo", "either", "cosine_only")
cells, policies = {}, set()
for f in sorted(glob.glob(f"{D}/cells/*.json")):
    d = json.load(open(f))
    enc, st = os.path.basename(f)[:-5].split("__")
    policies.add(d["policy"])
    keep = {k: d.get(k) for k in ("policy", "embedder", "storage_dtype", "budget", "holdout",
                                  "seed", "n_genuine", "n_attack", "excess_auroc",
                                  "excess_auroc_cos_matched", "matched_support",
                                  "cosine_auroc", "cosine_block_rate", "cosine_threshold")}
    keep["answers"] = d.get("answers")
    keep["answer_rules"] = {r: {k: v for k, v in (d.get("answer_rules") or {}).get(r, {}).items()
                                if not isinstance(v, (list,))} for r in RULES}
    keep["answer_rules_pairing"] = d.get("answer_rules_pairing")
    cells[f"{enc}|{st}"] = keep
assert len(policies) == 1, f"cells were scored under more than one cut: {policies}"
out = {
    "what": "Detection under the answer-checked rule: three attack sets x four encoders.",
    "policy": policies.pop(),
    "budget": 0.05,
    "echo_min": 1,
    "read_this_first": [
        "The `either` column the paper quotes is the *_joint_eta family: eta_a is fitted on the "
        "benign arm, then the DG height is refitted given it so the conjunction spends the whole "
        "benign budget. Reading the conjunction at DG's own height understates it.",
        "Answer coverage is 1.00 on both arms of every cell; a cell with partial coverage would "
        "degrade toward dg_only and answer_coverage says so.",
        "Benign rates are single-seed holdout draws; the 200-split mean on these arms is near 0.055.",
        "The three attack classes move in different directions and must not be pooled.",
    ],
    "source": f"{D}/cells/*.json",
    "report": "experiments/paper/results/answer_check_20260909/REPORT.md",
    "cells": cells,
}
p = args.out
json.dump(out, open(p, "w"), indent=1)
print("wrote", p, os.path.getsize(p), "bytes;", len(cells), "cells")
