"""Figure 3 left panel: strict success of search candidates without a defense and after Ours.

Inputs: the `--dump-viable` JSONL of each rule-informed attacker draw, one line per
retrievable candidate (cos >= 0.90) with `round`, `evades_joint` and `poisoned_strict`.
Every dumped line counts once, duplicate texts included, as in
supp_search_summary.py.

For each group (all rounds, round 1, round 2), the output gives strict successes and
candidates among
- `retrieved`: every retrievable candidate, i.e. native retrieval with no defense;
- `accepted`: the candidates the deployed rule serves (`evades_joint`);
- `blocked`: the rest.
`round` is 0 for the attacker's first round and 1 for the round rewritten after the
DG feedback on the first. The pooled `accepted` counts must equal
search_4p2.json -> rule_informed_draws.figure3_left_pooled.deployed_joint_rule.overall.

Only counting, no model call. Run where the dumps live (cpu-server):
  python -m experiments.paper.rq2_robustness.supp_search_figure3 \
      --dump draw1/viable.jsonl --dump draw2/viable.jsonl --dump draw3/viable.jsonl \
      --out search_figure3_rounds.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GROUPS = {"overall": lambda r: True,
          "round1": lambda r: r["round"] == 0,
          "round2": lambda r: r["round"] == 1}
SETS = {"retrieved": lambda r: True,
        "accepted": lambda r: r["evades_joint"],
        "blocked": lambda r: not r["evades_joint"]}


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def counts(rows: list[dict]) -> dict:
    out = {}
    for group, in_group in GROUPS.items():
        out[group] = {}
        for name, keep in SETS.items():
            sel = [r for r in rows if in_group(r) and keep(r)]
            out[group][name] = {"n_candidates": len(sel),
                                "strict": int(sum(r["poisoned_strict"] for r in sel))}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", action="append", required=True,
                        help="one draw's --dump-viable JSONL; repeat per draw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    dumps = [load(p) for p in args.dump]
    assert all({r["round"] for r in d} <= {0, 1} for d in dumps)
    pooled = counts([r for d in dumps for r in d])
    for group in GROUPS:
        cells = pooled[group]
        assert cells["retrieved"]["n_candidates"] == (cells["accepted"]["n_candidates"]
                                                      + cells["blocked"]["n_candidates"])
    out = {"definition": ("strict successes / candidates among retrievable search "
                          "candidates (retrieved), those the deployed rule serves "
                          "(accepted) and the rest (blocked); round1 = first attacker "
                          "round, round2 = round rewritten after DG feedback; pooled over "
                          "the three rule-informed draws"),
           "sources": [{"dump": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                        "n_rows": len(d)} for p, d in zip(args.dump, dumps)],
           "pooled": pooled,
           "per_draw": [counts(d) for d in dumps]}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for group, cells in pooled.items():
        print(group, {k: f"{v['strict']}/{v['n_candidates']}" for k, v in cells.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
