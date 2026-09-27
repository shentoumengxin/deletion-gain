"""RQ3: is `excess` reliable for the reason we say it is?

The method rests on two ingredients, and the claim is that both are load-bearing:

1. **It is relational.** Shortened versions are scored against the arriving query, not
   against each other. Take the anchor away and there should be nothing left.
2. **The reference is the text itself.** What is read is how much a *deletion* improves
   the match — not how well each part scores on its own. Score the parts directly and
   the signal should go.

`DELETION_TEST.md` §5 measured both under the previous analysis code and found each
ablation destroys the signal. This reproduces them through the deployed code, so the
claim rests on the statistic that actually ships rather than on a predecessor.

Two more RQ3 questions ride along in the same pass because the vectors are already in
memory: what the winning deletion removes (§3.4's mechanism claim), and whether the span
policy matters once its population effect is controlled.

Every variant is reported as AUROC and block@5% against the same benign arm, so the
columns are comparable down the page.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sentry.cache.defense.calibrate import (
    auroc, filter_benign_generators, load_pair_rows, parse_policy,
)
from sentry.cache.defense.deletion import build_profile, excess, unit
from sentry.cache.defense.spans import SpanPolicy, build_spans


def block_at_budget(attack, benign, budget=0.05):
    if len(attack) < 2 or len(benign) < 2:
        return float("nan")
    line = float(np.quantile(benign, 1.0 - budget))
    return float((np.asarray(attack) > line).mean())


def spread_stats(segment_scores: np.ndarray, floor: float) -> dict:
    """Statistics over how each segment scores *on its own* against the anchor.

    This is ablation 2. `DELETION_TEST.md` §5 rescales past the anisotropy floor first —
    two unrelated texts already score ~0.87 on e5, so raw differences are compressed into
    the top thirteen percent of the range and any spread statistic reads mostly noise.
    ``g`` maps the floor to zero and 1.0 to one.
    """
    g = np.clip((segment_scores - floor) / max(1.0 - floor, 1e-9), 0.0, None)
    level = float(g.mean())
    ordered = np.sort(g)
    m = len(ordered)
    denom = m * ordered.sum()
    gini = (float((2.0 * np.arange(1, m + 1) - m - 1).dot(ordered) / denom)
            if denom > 1e-12 else 0.0)
    return {
        "span_cv": float(g.std() / max(level, 1e-9)),
        "span_range": float(g.max() - g.min()),
        "span_gini": gini,
        "span_dead_frac": float((g <= 0.0).mean()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--embedder", default="intfloat/e5-small-v2")
    parser.add_argument("--benign-generator", action="append", default=["human_comqa"])
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    from sentry.embeddings import TransformerCLSEmbedder

    rows, _ = load_pair_rows(args.records)
    rows = filter_benign_generators(rows, args.benign_generator)
    embedder = TransformerCLSEmbedder(args.embedder)
    count6, width3 = parse_policy("count:6"), parse_policy("width:3")
    print(f"{len(rows)} entry-side rows | {args.embedder}", flush=True)

    pool: dict[str, int] = {}

    def intern(t):
        if t not in pool:
            pool[t] = len(pool)
        return pool[t]

    # Everything the four questions need, interned once.
    plan = []
    for row in rows:
        parts6 = count6.segments(row.text)
        if len(parts6) < count6.min_segments:
            continue
        spans6 = build_spans(parts6)
        parts3 = width3.segments(row.text)
        judgeable3 = len(parts3) >= width3.min_segments
        plan.append({
            "row": row,
            "anchor": intern(row.anchor),
            "whole": intern(row.text),
            "segments": [intern(p) for p in parts6],
            "spans": [intern(t) for t in spans6.span_texts],
            "span_names": spans6.span_names,
            "n_seg": len(parts6),
            "w3": ([intern(t) for t in build_spans(parts3).span_texts]
                   if judgeable3 else None),
            "w3_whole": intern(row.text) if judgeable3 else None,
        })

    texts = [None] * len(pool)
    for t, i in pool.items():
        texts[i] = t
    print(f"{len(plan)} judgeable rows, {len(texts)} distinct texts to encode",
          flush=True)
    chunks = [embedder.encode(texts[i:i + 256]) for i in range(0, len(texts), 256)]
    matrix = np.concatenate(chunks, axis=0)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-12)

    # The anisotropy floor: what two unrelated texts in this space already score.
    sample = matrix[[p["whole"] for p in plan[:400]]]
    gram = sample @ sample.T
    floor = float(np.median(gram[np.triu_indices(len(sample), k=1)]))
    print(f"anisotropy floor = {floor:.4f} (unrelated pairs already score this)",
          flush=True)

    scored = []
    for p in plan:
        a = matrix[p["anchor"]]
        base = float(matrix[p["whole"]] @ a)
        span_scores = matrix[p["spans"]] @ a
        rec = {"row": p["row"], "n_seg": p["n_seg"],
               "excess": float(span_scores.max()) - base,
               "cos": base}

        # --- ablation 1: no anchor. How far apart do the two sides of a cut sit?
        # Reads the text against itself instead of against the query.
        seps = []
        names = list(p["span_names"])
        for cut in range(1, p["n_seg"]):
            try:
                i = names.index(f"pre{cut}")
                j = names.index(f"suf{cut}")
            except ValueError:
                continue
            seps.append(float(matrix[p["spans"][i]] @ matrix[p["spans"][j]]))
        # lower self-similarity = more suspicious, so negate to keep "higher = worse"
        rec["no_anchor_split"] = -min(seps) if seps else float("nan")

        # --- ablation 2: score each span on its own instead of scoring deletions
        rec.update(spread_stats(matrix[p["segments"]] @ a, floor))

        # --- mechanism: which end won, and how much it kept
        best = int(np.argmax(span_scores))
        name = p["span_names"][best]
        cut = int(name[3:])
        rec["best_end"] = "prefix" if name.startswith("pre") else "suffix"
        rec["best_kept"] = ((cut if name.startswith("pre") else p["n_seg"] - cut)
                            / p["n_seg"])

        # --- span policy, on rows judgeable under BOTH cuts
        if p["w3"] is not None:
            w3_scores = matrix[p["w3"]] @ a
            rec["excess_w3"] = float(w3_scores.max()) - float(matrix[p["w3_whole"]] @ a)
        scored.append(rec)

    benign = [r for r in scored if r["row"].arm == "genuine"]
    attacks = [r for r in scored if r["row"].arm == "attack"]
    report = {"embedder": args.embedder, "anisotropy_floor": floor,
              "n_benign": len(benign), "n_attack": len(attacks), "variants": {}}

    variants = ["excess", "no_anchor_split", "span_cv", "span_range", "span_gini",
                "span_dead_frac"]
    print(f"\n  RQ3.3 -- both ingredients removed, one at a time")
    print(f"  {'variant':<20}{'what it reads':<44}{'AUROC':>8}{'block@5%':>10}")
    labels = {
        "excess": "ours: deletion, against the anchor",
        "no_anchor_split": "ablation 1: text against ITSELF, no anchor",
        "span_cv": "ablation 2: each span alone, spread (CV)",
        "span_range": "ablation 2: each span alone, range",
        "span_gini": "ablation 2: each span alone, Gini",
        "span_dead_frac": "ablation 2: share of spans at the floor",
    }
    for name in variants:
        a = np.array([r[name] for r in attacks], dtype=float)
        b = np.array([r[name] for r in benign], dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        au, bl = auroc(a, b), block_at_budget(a, b, args.budget)
        report["variants"][name] = {"auroc": au, "block": bl}
        print(f"  {name:<20}{labels[name]:<44}{au:>8.3f}{bl:>10.3f}")

    print(f"\n  RQ3.1 -- what the winning deletion removes")
    print(f"  {'family':<32}{'n':>6}{'prefix share':>14}{'median kept':>13}")
    for family in sorted({r["row"].family for r in attacks}) + ["canonical"]:
        group = [r for r in scored if r["row"].family == family]
        if not group:
            continue
        share = float(np.mean([r["best_end"] == "prefix" for r in group]))
        kept = float(np.median([r["best_kept"] for r in group]))
        report.setdefault("mechanism", {})[family] = {
            "n": len(group), "prefix_share": share, "median_kept": kept}
        print(f"  {family:<32}{len(group):>6}{share:>14.3f}{kept:>13.2f}")

    both = [r for r in scored if "excess_w3" in r]
    if both:
        ba = [r for r in both if r["row"].arm == "attack"]
        bb = [r for r in both if r["row"].arm == "genuine"]
        print(f"\n  RQ3.4 -- span policy, on the {len(both)} rows judgeable under BOTH")
        print(f"  {'policy':<20}{'AUROC':>8}{'block@5%':>10}")
        for key, label in (("excess", "count:6"), ("excess_w3", "width:3")):
            a = np.array([r[key] for r in ba])
            b = np.array([r[key] for r in bb])
            au, bl = auroc(a, b), block_at_budget(a, b, args.budget)
            report.setdefault("span_policy", {})[label] = {
                "auroc": au, "block": bl, "n_rows": len(both)}
            print(f"  {label:<20}{au:>8.3f}{bl:>10.3f}")
        print("  (comparing on the intersection removes the population effect: three-word")
        print("   segments drop every text under ~7 words, which is not a policy effect)")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
