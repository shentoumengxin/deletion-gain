"""RQ2: the deletion test against one representative of each defense paradigm.

``STORY.md`` §4 sorts existing defenses into three families and diagnoses each. This puts
one representative of each on the same rows, in the entry side, and reports the number
a cache operator actually lives with.

| paradigm (`STORY.md` §4) | representative | what it looks at |
|---|---|---|
| similarity / representation | cosine | how close the two texts are |
| point-wise prompt defense | conditional perplexity asymmetry | how surprising each text is given the other |
| post-hit semantic verification | bidirectional NLI entailment | whether the two texts entail each other |
| ours | ``excess`` | whether deleting part of the entry improves its match |

**The headline metric is block@5%, not AUROC.** A prior evaluation of these same signals
(recorded in the project's notes, on a different corpus) found AUROC 0.83–0.91 across
them while TPR at FPR ≤ 1% was **0.000 for every one of them, and for their ensemble**.
AUROC measures ranking; a cache has to draw a line. Both are reported here so the gap
between them is visible rather than argued.

Latency per call is timed in the same pass, because the cost axis is half of RQ2's
claim: ``excess`` is 17 dot products against a cross-encoder forward pass.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from sentry.cache.defense.calibrate import (
    auroc, filter_benign_generators, load_pair_rows, parse_policy,
)
from sentry.cache.defense.deletion import build_profile, excess


def block_at_budget(attack_scores, benign_scores, budget=0.05):
    """Share of attacks caught when the line spends ``budget`` on benign traffic.

    The deployment number. A defense is bought at a false-block price and this is what
    that price buys.
    """
    if len(attack_scores) < 2 or len(benign_scores) < 2:
        return float("nan")
    line = float(np.quantile(benign_scores, 1.0 - budget))
    return float((np.asarray(attack_scores) > line).mean())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--embedder", default="intfloat/e5-small-v2")
    parser.add_argument("--policy", default="count:6")
    parser.add_argument("--benign-generator", action="append", default=["human_comqa"])
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--limit-per-arm", type=int, default=0,
                        help="cap rows per family for a quick pass; 0 = all")
    parser.add_argument("--skip", action="append", default=[],
                        help="signal to skip, e.g. --skip perplexity")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    from sentry.embeddings import TransformerCLSEmbedder

    rows, _ = load_pair_rows(args.records)
    rows = filter_benign_generators(rows, args.benign_generator)
    if args.limit_per_arm:
        per_family = defaultdict(list)
        for r in rows:
            per_family[r.family].append(r)
        rows = [r for fam in per_family for r in per_family[fam][:args.limit_per_arm]]

    policy = parse_policy(args.policy)
    embedder = TransformerCLSEmbedder(args.embedder)
    print(f"{len(rows)} entry-side rows | {args.embedder} | policy {policy.fingerprint()}",
          flush=True)

    signals = {}
    if "binli" not in args.skip:
        from experiments.paper.baselines.defense.binli import BiNLI as _BiNLI
        signals["binli"] = _BiNLI()
    if "perplexity" not in args.skip:
        from experiments.paper.baselines.defense.perplexity import PerplexityAsymmetry as _PPL
        signals["perplexity"] = _PPL()

    # --- ours + cosine, one batched encode over the pool -------------------
    pool: dict[str, int] = {}

    def intern(t):
        if t not in pool:
            pool[t] = len(pool)
        return pool[t]

    keep = []
    for row in rows:
        profile = build_profile(row.text, embedder, policy)
        if not profile.judgeable:
            continue
        keep.append((row, profile, intern(row.anchor)))
    texts = [None] * len(pool)
    for t, i in pool.items():
        texts[i] = t
    chunks = [embedder.encode(texts[i:i + 256]) for i in range(0, len(texts), 256)]
    matrix = np.concatenate(chunks, axis=0)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-12)

    start = time.perf_counter()
    scored = []
    for row, profile, ai in keep:
        reading = excess(profile, matrix[ai])
        scored.append({"row": row, "cosine": -reading.base_cos,   # lower cos = suspicious
                       "excess": reading.excess_span})
    ours_ms = (time.perf_counter() - start) * 1000 / max(len(keep), 1)
    print(f"  excess: {ours_ms:.4f} ms/row (dot products only; the encode above is "
          f"insertion-time work, not serving)", flush=True)

    # --- baselines, timed --------------------------------------------------
    timings = {"excess": ours_ms}
    for name, signal in signals.items():
        start = time.perf_counter()
        for record in scored:
            row = record["row"]
            record[name] = float(signal.score(row.anchor, row.text, 0.0, None))
        timings[name] = (time.perf_counter() - start) * 1000 / max(len(scored), 1)
        print(f"  {name}: {timings[name]:.2f} ms/row", flush=True)

    methods = ["cosine", "excess"] + list(signals)
    families = sorted({r["row"].family for r in scored if r["row"].arm == "attack"})
    benign = [r for r in scored if r["row"].arm == "genuine"]

    report = {"embedder": args.embedder, "policy": policy.fingerprint(),
              "budget": args.budget, "latency_ms_per_row": timings,
              "n_benign": len(benign), "families": {}}

    print(f"\n  AUROC / block@{args.budget:.0%}  (higher is better; block is the one "
          f"that decides deployment)")
    header = "".join(f"{m:>22}" for m in methods)
    print(f"  {'family':<30}{'n':>6}{header}")
    for family in families + ["POOLED"]:
        attacks = [r for r in scored if r["row"].arm == "attack"
                   and (family == "POOLED" or r["row"].family == family)]
        if len(attacks) < 2:
            continue
        cells, line = {}, ""
        for m in methods:
            a = np.array([r[m] for r in attacks])
            b = np.array([r[m] for r in benign])
            au, bl = auroc(a, b), block_at_budget(a, b, args.budget)
            cells[m] = {"auroc": au, "block": bl}
            line += f"{au:>11.3f}{bl:>11.3f}"
        report["families"][family] = {"n_attack": len(attacks), **cells}
        print(f"  {family:<30}{len(attacks):>6}{line}")

    print("\n  Each pair is (AUROC, block@budget). A signal can rank well and still")
    print("  catch almost nothing once the line is drawn -- that gap is the point.")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
