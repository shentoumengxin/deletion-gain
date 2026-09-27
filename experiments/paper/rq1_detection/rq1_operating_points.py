"""What the "@5% FPR" label actually costs, and what the same tests do at 1%.

Every block rate in the paper is read against a threshold placed at the benign 95th
percentile of the *scoring* arm. That is a nominal budget: fit on a benign sample, and the
rate you achieve on benign traffic you have not seen is a different number. Reporting the
nominal one alone invites the reader to assume they coincide, and on \\comqa{} they do not
-- the fitted boundary lands at 8.8% out of sample on a single split.

This script separates the two, for every statistic for which per-row scores exist, and
adds the budget a cache actually wants:

``nominal``   threshold at the benign ``1-budget`` quantile of the whole benign arm, block
              rate read against it. This is what the tables report; its in-sample FPR is
              the budget by construction.
``realized``  fit the threshold on half the benign *intents*, then read the false-block
              rate on the other half and the block rate on all attacks. Repeated over many
              intent-grouped splits, so the answer is a distribution rather than one draw.

Both are reported at 5% and at 1%. A cache earns its keep through hit rate, so 1% is the
budget a deployment would actually pick, and a defense that only works at 5% should have to
say so.

No leakage: splits are by ``intent_id``, never by row (two entries for one intent share an
anchor and most of their words).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DEPLOYED = "multi[count:4+width:2:cap16]/runs"


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC with ties at half credit."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    values = np.concatenate([pos, neg])
    # average ranks within tie groups
    sorted_vals = values[order]
    start = 0
    for i in range(1, len(sorted_vals) + 1):
        if i == len(sorted_vals) or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def intent_holdout(intents, holdout: float, seed: int) -> set:
    """Deterministic hash split over intents, matching sentry.cache.defense.calibrate."""
    held = set()
    for intent in sorted(set(intents)):
        digest = hashlib.sha256(f"{seed}:{intent}".encode("utf-8")).digest()
        if int.from_bytes(digest[:8], "big") / 2 ** 64 < holdout:
            held.add(intent)
    return held


def operating_point(att, ben, ben_intents, budget, n_splits=200, holdout=0.5, seed0=0):
    """Nominal (in-sample) and realized (held-out) behaviour at one budget."""
    att, ben = np.asarray(att, float), np.asarray(ben, float)
    ben_intents = np.asarray(ben_intents, dtype=object)

    thr = float(np.quantile(ben, 1.0 - budget))
    cell = {"nominal_threshold": thr,
            "nominal_br": float((att > thr).mean()),
            "nominal_fpr_in_sample": float((ben > thr).mean())}

    fprs, brs = [], []
    for s in range(n_splits):
        held = intent_holdout(ben_intents, holdout, seed0 + s)
        mask = np.array([i in held for i in ben_intents])
        if mask.sum() < 20 or (~mask).sum() < 20:
            continue
        t = float(np.quantile(ben[~mask], 1.0 - budget))
        fprs.append(float((ben[mask] > t).mean()))
        brs.append(float((att > t).mean()))
    cell.update(realized_fpr_mean=float(np.mean(fprs)), realized_fpr_sd=float(np.std(fprs)),
                realized_fpr_p05=float(np.quantile(fprs, 0.05)),
                realized_fpr_p95=float(np.quantile(fprs, 0.95)),
                realized_br_mean=float(np.mean(brs)), realized_br_sd=float(np.std(brs)),
                n_splits=len(fprs))
    return cell


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", action="append", required=True,
                   help="SET=path to a rq3_granularity --dump-scores json")
    p.add_argument("--extra", action="append", default=[],
                   help="SET=path to a json of {method: {attack: [...], benign: [...], "
                        "benign_intents: [...]}} for statistics not in the sweep dump. "
                        "Repeatable per set: pass one per source file rather than "
                        "pre-merging them, so no derived merge artifact has to be kept.")
    p.add_argument("--policy", default=DEPLOYED)
    p.add_argument("--budgets", default="0.05,0.01")
    p.add_argument("--splits", type=int, default=200)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    budgets = [float(b) for b in args.budgets.split(",")]
    report = {"policy": args.policy, "budgets": budgets, "n_splits": args.splits, "sets": {}}

    extra_paths: dict[str, list[str]] = {}
    for spec in args.extra:
        tag, path = spec.split("=", 1)
        extra_paths.setdefault(tag, []).append(path)

    for spec in args.scores:
        tag, path = spec.split("=", 1)
        dump = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = dump["specs"].index(args.policy)
        rows = dump["rows"]
        att_rows = [r for r in rows if r["arm"] == "attack"]
        ben_rows = [r for r in rows if r["arm"] == "genuine"]
        ben_intents = [r["intent_id"] for r in ben_rows]

        methods = {
            "deletion gain": ([r["exc"][idx] for r in att_rows],
                              [r["exc"][idx] for r in ben_rows]),
            # the cache's own score, negated so "higher = more suspicious" throughout
            "cosine": ([-r["cos"] for r in att_rows], [-r["cos"] for r in ben_rows]),
        }
        for extra_path in extra_paths.get(tag, []):
            for name, blob in json.loads(
                    Path(extra_path).read_text(encoding="utf-8")).items():
                methods[name] = (blob["attack"], blob["benign"])
                if blob.get("benign_intents"):
                    methods[name] = (blob["attack"], blob["benign"], blob["benign_intents"])

        cells = {}
        for name, payload in methods.items():
            att, ben = payload[0], payload[1]
            ints = payload[2] if len(payload) > 2 else ben_intents
            cells[name] = {"n_attack": len(att), "n_benign": len(ben),
                           "auroc": auroc(np.asarray(att, float), np.asarray(ben, float)),
                           "at": {f"{b:g}": operating_point(att, ben, ints, b, args.splits)
                                  for b in budgets}}
        report["sets"][tag] = {"source": path, "n_rows": len(rows), "methods": cells}

        print(f"\n=== {tag} ===")
        head = f"  {'method':<16}{'AUC':>7}" + "".join(
            f"{'BR@'+f'{b:g}':>9}{'realFPR':>9}{'realBR':>9}" for b in budgets)
        print(head)
        for name, cell in cells.items():
            line = f"  {name:<16}{cell['auroc']:>7.3f}"
            for b in budgets:
                c = cell["at"][f"{b:g}"]
                line += (f"{c['nominal_br']:>9.3f}"
                         f"{c['realized_fpr_mean']:>7.3f}±{c['realized_fpr_sd']:<.2f}"
                         f"{c['realized_br_mean']:>9.3f}")
            print(line)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
