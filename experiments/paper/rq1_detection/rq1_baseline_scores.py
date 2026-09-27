"""Per-row scores for the tunable baselines, so their operating point can be moved.

`rq2_baselines.py` reports each signal at one budget and keeps only the summary. A
reviewer who asks "what does conditional perplexity block at 1% FPR?" cannot be answered
from a summary, and 5% is not the budget a cache would pick -- a cache earns its keep
through hit rate, so every point of false blocking is paid for directly.

This re-scores the same rows the detection table used and writes the raw arrays, in the
shape `rq1_operating_points.py --extra` consumes. Two signals only: the ones with a
continuous score and therefore a movable threshold. The LLM judge returns a verdict, not a
score, so it has exactly one operating point and appears at its own achieved benign rate;
LaCache needs a decode per row and is not re-run here.

Rows are loaded through `v3_detect.load_rows`, not reimplemented, so the arms, the anchor
choice and the drop rule are the ones the table was built on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", required=True)
    p.add_argument("--attack-role", action="append", required=True)
    p.add_argument("--benign-generator", action="append", required=True)
    p.add_argument("--signals", default="binli,perplexity")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from experiments.paper.rq1_detection.v3_detect import load_rows

    rows, comp = load_rows(args.eval, args.attack_role, args.benign_generator)
    n_att = sum(1 for r in rows if r["arm"] == "attack")
    n_ben = len(rows) - n_att
    print(f"{n_att} attack / {n_ben} benign rows", flush=True)

    signals = {}
    wanted = [s for s in args.signals.split(",") if s]
    if "binli" in wanted:
        from experiments.paper.baselines.defense.binli import BiNLI
        signals["bidirectional NLI"] = BiNLI()
    if "perplexity" in wanted:
        from experiments.paper.baselines.defense.perplexity import PerplexityAsymmetry
        signals["cond. perplexity"] = PerplexityAsymmetry()

    out = {}
    for name, signal in signals.items():
        start = time.perf_counter()
        scores = []
        for i, row in enumerate(rows):
            scores.append(float(signal.score(row["anchor"], row["text"], 0.0, None)))
            if i and i % 200 == 0:
                rate = (time.perf_counter() - start) / i
                print(f"  {name}: {i}/{len(rows)}  {rate*1000:.0f} ms/row  "
                      f"eta {rate*(len(rows)-i)/60:.1f} min", flush=True)
        wall = time.perf_counter() - start
        out[name] = {
            "attack": [s for s, r in zip(scores, rows) if r["arm"] == "attack"],
            "benign": [s for s, r in zip(scores, rows) if r["arm"] == "genuine"],
            "benign_intents": [r["intent_id"] for r in rows if r["arm"] == "genuine"],
            "ms_per_row": wall / len(rows) * 1000,
        }
        a = np.array(out[name]["attack"]); b = np.array(out[name]["benign"])
        for budget in (0.05, 0.01):
            t = float(np.quantile(b, 1 - budget))
            print(f"  {name}: BR@{budget:g} = {(a > t).mean():.3f}", flush=True)
        print(f"  {name}: {wall/len(rows)*1000:.1f} ms/row", flush=True)

    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
