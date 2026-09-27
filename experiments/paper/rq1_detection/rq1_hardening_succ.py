"""The Succ. column for the hardening row, so the table has no blanks.

Every other row in the detection table reports the block rate twice: over all attempted
plants, and over the subset the judge confirmed actually poisoned the victim. The hardening
row was published with three dashes there, which reads as a property of the method. It is
not -- the per-row salted scores exist, they simply were not joined to the poisoning flags.
This does the join.

The threshold is the one the Att. column already uses: the benign quantile at the same
budget, fitted on the same benign arm. Only the attack arm narrows, exactly as it does for
the other rows, so the two columns stay comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_flags(path):
    flags = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            f = json.loads(line)
            flags[f["record_id"]] = f
    return flags


def poisoned_of(record_id, flags):
    """Direct join, then the KCA f2 gcg-f2-NNNN -> kca-f2-NNNN remap, as v3_detect does."""
    if record_id in flags:
        return flags[record_id]
    if record_id and record_id.startswith("gcg-f2-"):
        alt = "kca-f2-" + record_id[len("gcg-f2-"):]
        if alt in flags:
            return flags[alt]
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", action="append", required=True, help="SET=path")
    p.add_argument("--flags", required=True)
    p.add_argument("--salt", default="salting, compressive k=128")
    p.add_argument("--budgets", default="0.05,0.01")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    flags = load_flags(args.flags)
    budgets = [float(b) for b in args.budgets.split(",")]
    report = {"salt": args.salt, "flags": args.flags, "sets": {}}

    print(f"  {'set':6s}{'budget':>8}{'Att.':>9}{'Succ.':>9}{'n_pois':>9}{'n_att':>8}")
    for spec in args.scores:
        tag, path = spec.split("=", 1)
        blob = json.loads(Path(path).read_text(encoding="utf-8"))[args.salt]
        attack = np.asarray(blob["attack"], float)
        benign = np.asarray(blob["benign"], float)
        ids = blob["attack_record_ids"]
        assert len(ids) == len(attack), "record ids and scores are out of step"

        keep, missing = [], 0
        for i, rid in enumerate(ids):
            f = poisoned_of(rid, flags)
            if f is None:
                missing += 1
            elif f.get("poisoned"):
                keep.append(i)
        cell = {"n_attack": len(attack), "n_poisoned": len(keep),
                "n_flag_missing": missing, "at": {}}
        for b in budgets:
            thr = float(np.quantile(benign, 1.0 - b))
            cell["at"][f"{b:g}"] = {
                "block_all": float((attack > thr).mean()),
                "block_poisoned": float((attack[keep] > thr).mean()) if keep else None}
            c = cell["at"][f"{b:g}"]
            print(f"  {tag:6s}{b:>8g}{c['block_all']:>9.3f}"
                  f"{(c['block_poisoned'] if c['block_poisoned'] is not None else float('nan')):>9.3f}"
                  f"{len(keep):>9}{len(attack):>8}")
        report["sets"][tag] = cell

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
