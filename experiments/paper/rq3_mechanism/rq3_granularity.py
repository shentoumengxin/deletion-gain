"""RQ3 granularity sweep: is multi[count:6+width:2:cap16]/runs the right cut?

The deployed policy asserts a mechanism: a fixed count makes every deletion drop the
same FRACTION of the text, so an injected clause shorter than one coarse segment never
gets isolated; the width cut fixes that; the union dominates both. The only prior
measurement (experiments/paper/results/rq4_fix/policy.txt) compared three policies
on the old 2,175-row set -- the granularity space itself was never swept. This script
sweeps it on the v3 eval sets (LMP-800 / SCP-800, ComQA benign arm, e5, float16,
budget 0.05) and produces the table the claim needs: TPR@5% stratified by
INJECTED-SPAN LENGTH -- the number of words the attack text adds over its intent's
canonical question (NOT the `payload` field, which is only the wrong-answer literal).

Pattern: intern every distinct variant text across ALL policies into one pool, encode
the pool ONCE, then score each policy from the matrix (ablations_v3.py's pattern).
Scoring parity with v3_detect.py / calibrate.evaluate_policy: float16-stored unit span
vectors dotted against the float64 unit anchor; flat fence at the (1-budget) benign
quantile; block rates read against the fence fitted on the WHOLE benign arm
(seed-independent headline); the seed-0 realized rate read on the held-out intent half.

Judgeability: the deployed floor is min_segments=3; a cell that cannot judge a text
routes it to the serving hook's unjudgeable branch (a miss). Cells therefore report
both the judgeable-only rates AND the effective rates that charge unjudgeable rows as
misses (blocked attacks / lost benign). count:2 is degenerate under the deployed
floor (every text has 2 segments); it is reported under an explicit floor of 2.

Modes:
  --mode sweep   heavy: encode + score + write --out JSON
  --mode bench   light: single-thread numpy serving ms per cell; run it under
                 OMP_NUM_THREADS=1. Methodology identical to bench_excess_ms.py
                 (which produced <server-workdir>/out/v3/excess_serving_ms.json):
                 float16 span block .astype(float32) @ float32 anchor, then max;
                 >=1000 repeats, median + p95 reported. Updates --out in place.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from sentry.cache.defense.calibrate import (
    auroc, filter_benign_generators, load_pair_rows, matched_auroc, parse_policy,
    _intent_holdout,
)
from sentry.cache.defense.fence import CalibrationRow, ExcessFence, achieved_block_rate

DIM = 384
DEPLOYED = "multi[count:6+width:2:cap16]/runs"
KEY_POLICIES = ("count:6/runs", "width:2:cap16/runs", "width:1:cap16/runs", DEPLOYED)
#: injected-span-length buckets in words (lo, hi inclusive; None = open).
BUCKETS = ((0, 0), (1, 2), (3, 4), (5, 8), (9, 16), (17, None))

#: parity targets from the v3 detection matrix (results/v3/detection/e5-small-v2/*.json),
#: same eval files, same deployed policy, float16, budget 0.05, seed 0. Reported as a
#: delta, not asserted: the pool is batch-encoded, the matrix cells encoded per row, and
#: CPU transformer inference is batch-composition-sensitive at the ~1e-5 level.
PARITY = {
    "LMP": {"excess_auroc": 0.9623822645290581, "block": 0.87375,
            "threshold_scoring": 0.010257290824995042,
            "in_sample": 0.050100200400801605, "seed0": 0.08368200836820083},
    "SCP": {"excess_auroc": 0.9868760076544065, "block": 0.9624060150375939,
            "threshold_scoring": 0.010257290824995042,
            "in_sample": 0.050100200400801605, "seed0": 0.08368200836820083},
    # KCA rides the NQ benign arm, so its fence height and realized rate differ from the
    # two ComQA sets; values from results/v3/detection/e5-small-v2/kca.json.
    "KCA": {"excess_auroc": 0.9895279280365242, "block": 0.9411027568922306,
            "threshold_scoring": 0.00437273782533198,
            "in_sample": 0.050100200400801605, "seed0": 0.029197080291970802},
}
#: The LMP set was renamed CAP in the v3 write-up; accept either --set-name so the
#: parity check still fires.
PARITY["CAP"] = PARITY["LMP"]


def build_grid() -> tuple[list[str], dict[str, list[str]]]:
    """The swept cells, deduplicated, each tagged with the sweep(s) it belongs to."""
    order: list[str] = []
    groups: dict[str, list[str]] = {}

    def add(spec: str, group: str) -> None:
        if spec not in groups:
            groups[spec] = []
            order.append(spec)
        groups[spec].append(group)

    for n in (2, 3, 4, 6, 8, 12):
        add(f"count:{n}", "count-sweep/span")
        add(f"count:{n}/runs", "count-sweep/runs")
    for w in (1, 2, 3, 4):
        add(f"width:{w}:cap16", "width-sweep/span")
        add(f"width:{w}:cap16/runs", "width-sweep/runs")
    for w in (1, 2, 3):
        add(f"multi[count:6+width:{w}:cap16]/runs", "multi-width-sweep")
    for n in (3, 4, 6, 8):
        add(f"multi[count:{n}+width:2:cap16]/runs", "multi-count-sweep")
    # Fill the interior of the coarse x fine matrix. The two sweeps above are lines
    # through the deployed cell (count fixed at 6, then width fixed at 2); a trade-off
    # plot needs the plane, and a multi cell's variant set is the UNION of its two
    # components' sets, so every text these add is already interned by the count and
    # width sweeps -- they cost scoring time, not encoding time.
    for n in (3, 4, 8):
        for w in (1, 3):
            add(f"multi[count:{n}+width:{w}:cap16]/runs", "multi-matrix")
    add(DEPLOYED, "deployed-reference")
    return order, groups


def bucket_label(delta: int) -> str:
    for lo, hi in BUCKETS:
        if delta >= lo and (hi is None or delta <= hi):
            if hi is None:
                return f"{lo}+"
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
    return "?"


def dist(values) -> dict:
    a = np.asarray(sorted(values), dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "min": float(a[0]), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "mean": float(a.mean()),
            "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
            "max": float(a[-1])}


def canonical_words(eval_path: str) -> dict[str, int]:
    out = {}
    for line in Path(eval_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("query_role") == "canonical":
            out[r["intent_id"]] = len(r["text"].split())
    return out


# --------------------------------------------------------------------------- sweep
def sweep(args) -> int:
    from sentry.cache.defense.spans import shortened

    print(f"EFFECTIVE ARGS: mode=sweep eval={args.eval} set={args.set_name} "
          f"embedder={args.embedder} attack-role={args.attack_role} "
          f"benign-generator={args.benign_generator} budget={args.budget} "
          f"storage=float16 holdout={args.holdout} seed={args.seed} "
          f"limit={args.limit} out={args.out}", flush=True)

    import torch
    torch.set_num_threads(args.threads)
    from sentry.embeddings import TransformerCLSEmbedder
    embedder = TransformerCLSEmbedder(args.embedder, batch_size=args.encode_batch)

    order, groups = build_grid()
    policies = {spec: parse_policy(spec) for spec in order}

    rows, comp = load_pair_rows(args.eval, (args.attack_role,))
    rows = filter_benign_generators(rows, args.benign_generator)
    print(f"composition: {comp.kept} dropped_no_anchor={comp.dropped_no_anchor}",
          flush=True)
    if args.limit:
        benign = [r for r in rows if r.arm == "genuine"][: args.limit]
        attack = [r for r in rows if r.arm == "attack"][: args.limit]
        rows = benign + attack
        print(f"LIMIT: {len(rows)} rows", flush=True)

    canon = canonical_words(args.eval)

    pool: dict[str, int] = {}

    def intern(t: str) -> int:
        if t not in pool:
            pool[t] = len(pool)
        return pool[t]

    plans = []
    for row in rows:
        delta, delta_raw = None, None
        if row.arm == "attack":
            cw = canon.get(row.intent_id)
            delta_raw = (len(row.text.split()) - cw) if cw is not None else None
            delta = None if delta_raw is None else max(0, delta_raw)
        entry = {"row": row, "whole": intern(row.text), "anchor": intern(row.anchor),
                 "words": len(row.text.split()),
                 "delta": delta, "delta_raw": delta_raw,
                 "cells": {}}
        for spec in order:
            sp = shortened(policies[spec], row.text)
            entry["cells"][spec] = {
                "idx": np.array([intern(t) for t in sp.span_texts], dtype=np.int64),
                "seg": sp.segment_count,
                "n_del": len(sp.deletion_texts),
                "j3": (sp.segment_count >= 3 and len(sp.span_texts) > 0
                       and len(sp.deletion_texts) > 0),
                "j2": (sp.segment_count >= 2 and len(sp.span_texts) > 0
                       and len(sp.deletion_texts) > 0),
                "names": sp.span_names if spec == DEPLOYED else None,
            }
        plans.append(entry)

    texts = [None] * len(pool)
    for t, i in pool.items():
        texts[i] = t
    print(f"{len(plans)} rows, {len(texts)} distinct texts to encode once", flush=True)
    t0 = time.perf_counter()
    chunks = []
    for start in range(0, len(texts), 4096):
        chunks.append(embedder.encode(texts[start:start + 4096]))
        if (start // 4096) % 5 == 0:
            print(f"  encoded {min(start + 4096, len(texts))}/{len(texts)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    M = np.concatenate(chunks, axis=0).astype(np.float64)
    M /= np.linalg.norm(M, axis=1, keepdims=True).clip(1e-12)
    M16 = M.astype(np.float16)
    print(f"encoded in {time.perf_counter() - t0:.0f}s; matrix {M.shape}", flush=True)

    # ---- score every (row, cell) from the matrix: parity with deletion.excess ----
    for pl in plans:
        a = M[pl["anchor"]]
        pl["base"] = float(M16[pl["whole"]] @ a)
        pl["exc"] = {}
        pl["dep_best"] = None
        for spec in order:
            cell = pl["cells"][spec]
            if len(cell["idx"]) == 0:
                pl["exc"][spec] = None
                continue
            scores = M16[cell["idx"]] @ a
            best = int(np.argmax(scores))
            pl["exc"][spec] = float(scores[best]) - pl["base"]
            if spec == DEPLOYED and cell["names"] is not None:
                pl["dep_best"] = cell["names"][best]

    n_benign_total = sum(1 for pl in plans if pl["row"].arm == "genuine")
    n_attack_total = len(plans) - n_benign_total

    if args.dump_scores:
        # Per-row, per-cell scores, so a policy can be CHOSEN on one half of the intents
        # and REPORTED on the other (rq3_selection_split.py). Without that split, every
        # "cell X beats the deployed cell" reading of this sweep is an argmax over 32
        # cells taken on the same rows it is reported on.
        dump = {"set": args.set_name, "eval": args.eval, "embedder": args.embedder,
                "budget": args.budget, "specs": order,
                "rows": [{"intent_id": pl["row"].intent_id, "arm": pl["row"].arm,
                          "family": pl["row"].family, "cos": pl["base"],
                          "words": pl["words"],
                          "exc": [pl["exc"][sp] for sp in order],
                          "j3": [bool(pl["cells"][sp]["j3"]) for sp in order],
                          "j2": [bool(pl["cells"][sp]["j2"]) for sp in order]}
                         for pl in plans]}
        Path(args.dump_scores).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_scores).write_text(json.dumps(dump), encoding="utf-8")
        print(f"wrote scores dump {args.dump_scores} "
              f"({len(plans)} rows x {len(order)} cells)", flush=True)

    # ---- injected-span length distribution --------------------------------------
    att_plans = [pl for pl in plans if pl["row"].arm == "attack"]
    inj = {"definition": "max(0, len(words(attack_text)) - len(words(canonical_text "
                         "of same intent))). NOT the `payload` field (the wrong-answer "
                         "literal): the injected clause is longer.",
           "overall": dist([pl["delta"] for pl in att_plans if pl["delta"] is not None]),
           "n_delta_missing": sum(1 for pl in att_plans if pl["delta"] is None),
           "n_raw_negative_clamped": sum(1 for pl in att_plans
                                         if pl["delta_raw"] is not None
                                         and pl["delta_raw"] < 0),
           "raw_unclamped": dist([pl["delta_raw"] for pl in att_plans
                                  if pl["delta_raw"] is not None]),
           "bucket_counts": dict(Counter(bucket_label(pl["delta"]) for pl in att_plans
                                         if pl["delta"] is not None)),
           "per_family": {}}
    for fam in sorted({pl["row"].family for pl in att_plans}):
        vals = [pl["delta"] for pl in att_plans
                if pl["row"].family == fam and pl["delta"] is not None]
        inj["per_family"][fam] = dist(vals)
        inj["per_family"][fam]["bucket_counts"] = dict(
            Counter(bucket_label(v) for v in vals))

    # ---- per-cell evaluation -----------------------------------------------------
    grid = {}
    fences = {}
    for spec in order:
        pol = policies[spec]
        floor, degenerate = 3, None
        jkey = "j3"
        jud = [pl for pl in plans if pl["cells"][spec]["j3"]]
        nb = sum(1 for pl in jud if pl["row"].arm == "genuine")
        na = len(jud) - nb
        if nb < 4 or na < 2:
            floor, jkey = 2, "j2"
            jud = [pl for pl in plans if pl["cells"][spec]["j2"]]
            nb = sum(1 for pl in jud if pl["row"].arm == "genuine")
            na = len(jud) - nb
            degenerate = ("every text is unjudgeable under the deployed floor "
                          "min_segments=3 (the cut yields <3 segments); reported "
                          "under an explicit floor of 2 instead")
        ben = [pl for pl in jud if pl["row"].arm == "genuine"]
        att = [pl for pl in jud if pl["row"].arm == "attack"]
        if len(ben) < 4 or len(att) < 2:
            grid[spec] = {"policy": pol.fingerprint(), "groups": groups[spec],
                          "degenerate": "too few judgeable rows even at floor 2",
                          "n_benign_judgeable": len(ben), "n_attack_judgeable": len(att)}
            continue

        b_exc = np.array([pl["exc"][spec] for pl in ben])
        a_exc = np.array([pl["exc"][spec] for pl in att])
        b_cos = np.array([pl["base"] for pl in ben])
        a_cos = np.array([pl["base"] for pl in att])

        def cal(group):
            return [CalibrationRow(pl["base"], pl["words"], pl["exc"][spec])
                    for pl in group]

        scoring_fence = ExcessFence.fit_flat(cal(ben), budget=args.budget,
                                             embedder=embedder.model_name,
                                             policy=pol.fingerprint())
        fit_intents, eval_intents = _intent_holdout(
            [pl["row"] for pl in ben], holdout=args.holdout, seed=args.seed)
        fit_rows = [pl for pl in ben if pl["row"].intent_id in fit_intents]
        eval_rows = [pl for pl in ben if pl["row"].intent_id in eval_intents]
        holdout_fence = (ExcessFence.fit_flat(cal(fit_rows), budget=args.budget,
                                              embedder=embedder.model_name,
                                              policy=pol.fingerprint())
                         if len(fit_rows) >= 4 else None)

        def blocked(group):
            return [bool(scoring_fence.blocks(pl["base"], pl["words"], pl["exc"][spec]))
                    for pl in group]

        att_blocked = blocked(att)
        ben_blocked = blocked(ben)
        n_att_unj = n_attack_total - len(att)
        n_ben_unj = n_benign_total - len(ben)
        matched, support = matched_auroc(a_exc, b_exc, a_cos, b_cos, width=0.01)
        cosine_fence = float(np.quantile(b_cos, args.budget))

        families = {}
        for fam in sorted({pl["row"].family for pl in att}):
            fa = [pl for pl in att if pl["row"].family == fam]
            fa_exc = np.array([pl["exc"][spec] for pl in fa])
            fa_cos = np.array([pl["base"] for pl in fa])
            fm, fsup = matched_auroc(fa_exc, b_exc, fa_cos, b_cos, width=0.01)
            families[fam] = {"n": len(fa), "excess_auroc": auroc(fa_exc, b_exc),
                             "matched": fm, "support": fsup,
                             "block_at_budget": float(np.mean(blocked(fa)))}

        # cost: dot products per hit = stored span vectors read at serving. The whole
        # vector is retrieval's own dot and not charged (same convention as
        # excess_serving_ms.json / the v3 cost table). Storage charges whole + spans
        # at float16; deletions (diagnostic, also persisted by entry_store) reported
        # separately.
        dots = {"pooled": [len(pl["cells"][spec]["idx"]) for pl in jud],
                "attack": [len(pl["cells"][spec]["idx"]) for pl in att],
                "benign": [len(pl["cells"][spec]["idx"]) for pl in ben]}
        n_del = [pl["cells"][spec]["n_del"] for pl in jud]
        storage = [(d + 1) * DIM * 2 for d in dots["pooled"]]
        storage_del = [(d + 1 + nd) * DIM * 2 for d, nd in zip(dots["pooled"], n_del)]

        def strat_over(plans):
            out = {}
            for lo, hi in BUCKETS:
                label = f"{lo}+" if hi is None else (f"{lo}" if lo == hi
                                                     else f"{lo}-{hi}")
                in_b = [pl for pl in plans if pl["delta"] is not None
                        and pl["delta"] >= lo and (hi is None or pl["delta"] <= hi)]
                in_j = [pl for pl in in_b if pl["cells"][spec][jkey]]
                blk = sum(1 for pl in in_j
                          if scoring_fence.blocks(pl["base"], pl["words"],
                                                  pl["exc"][spec]))
                out[label] = {
                    "n": len(in_b), "n_judgeable": len(in_j), "n_blocked": blk,
                    "tpr_judgeable": (blk / len(in_j)) if in_j else None,
                    "tpr_effective": ((blk + len(in_b) - len(in_j)) / len(in_b))
                                     if in_b else None}
            return out

        strat = strat_over(att_plans)
        # Injection length and template are almost collinear on SCP -- scp_z injects
        # 2-6 words, scp_p 7-14, scp_i 94-216 -- so a bucket-vs-bucket comparison across
        # the whole set cannot tell "the cut resolves short payloads" apart from "the cut
        # happens to suit zero-shot templates". Stratifying WITHIN a family is the
        # comparison that separates them: scp_z spans the 1-2/3-4 buckets and scp_p the
        # 5-8/9-16 buckets on its own. Any length claim has to survive here.
        strat_by_family = {
            fam: strat_over([pl for pl in att_plans if pl["row"].family == fam])
            for fam in sorted({pl["row"].family for pl in att_plans})
        }

        grid[spec] = {
            "policy": pol.fingerprint(), "groups": groups[spec], "floor": floor,
            "degenerate": degenerate,
            "n_benign_judgeable": len(ben), "n_attack_judgeable": len(att),
            "n_benign_unjudgeable": n_ben_unj, "n_attack_unjudgeable": n_att_unj,
            "excess_auroc": auroc(a_exc, b_exc),
            "excess_auroc_cos_matched": matched, "matched_support": support,
            "block_at_budget": float(np.mean(att_blocked)),
            "block_effective": (sum(att_blocked) + n_att_unj) / n_attack_total,
            "threshold_scoring": float(scoring_fence.coefficients[0]),
            "threshold_holdout": (float(holdout_fence.coefficients[0])
                                  if holdout_fence else None),
            "benign_block_in_sample": float(np.mean(ben_blocked)),
            "benign_effective_in_sample":
                (sum(ben_blocked) + n_ben_unj) / n_benign_total,
            "realized_benign_seed0": (achieved_block_rate(holdout_fence, cal(eval_rows))
                                      if holdout_fence and eval_rows else None),
            "n_fit_intents": len(fit_intents), "n_eval_intents": len(eval_intents),
            "cosine_auroc": auroc(-a_cos, -b_cos),
            "cosine_block_rate": float((a_cos < cosine_fence).mean()),
            "families": families,
            "dots_per_hit": {k: {"mean": float(np.mean(v)),
                                 "median": float(np.median(v)),
                                 "p95": float(np.percentile(v, 95))}
                             for k, v in dots.items()},
            "variants_stored_mean": float(np.mean([d + nd for d, nd
                                                   in zip(dots["pooled"], n_del)])),
            "storage_bytes_f16": {"mean": float(np.mean(storage)),
                                  "median": float(np.median(storage)),
                                  "p95": float(np.percentile(storage, 95))},
            "storage_bytes_f16_with_deletions_mean": float(np.mean(storage_del)),
            "stratified_tpr": strat,
            "stratified_tpr_by_family": strat_by_family,
        }
        fences[spec] = scoring_fence
        g = grid[spec]
        print(f"{spec:<38} AUROC {g['excess_auroc']:.4f}  matched "
              f"{g['excess_auroc_cos_matched']:.4f}"
              f"({g['matched_support']:>4})  block {g['block_at_budget']:.4f}  "
              f"thr {g['threshold_scoring']:.6f}  dots {g['dots_per_hit']['pooled']['mean']:.1f}  "
              f"nb/na {len(ben)}/{len(att)}"
              + (f"  [floor={floor}]" if floor != 3 else ""), flush=True)

    # count:2 span == count:2 runs by construction (two halves either way); note it.
    if "count:2" in grid and "count:2/runs" in grid and "degenerate" in grid["count:2"]:
        for spec in ("count:2", "count:2/runs"):
            if grid[spec].get("degenerate"):
                grid[spec]["degenerate"] += ("; note count:2's span and runs forms "
                                             "produce the identical two variants")

    # ---- marginal value of the deployed union ------------------------------------
    c6r, w2r = "count:6/runs", "width:2:cap16/runs"
    marginal = {"cells": {}}
    for spec in (c6r, w2r, DEPLOYED):
        g = grid.get(spec, {})
        marginal["cells"][spec] = {k: g.get(k) for k in
                                   ("block_at_budget", "excess_auroc",
                                    "threshold_scoring")}
        if "dots_per_hit" in g:
            marginal["cells"][spec]["dots_mean"] = g["dots_per_hit"]["pooled"]["mean"]
    common = [pl for pl in att_plans
              if all(pl["cells"][s]["j3"] for s in (c6r, w2r, DEPLOYED))]
    if all(s in fences for s in (c6r, w2r, DEPLOYED)):
        tab = Counter()
        for pl in common:
            key = (bool(fences[c6r].blocks(pl["base"], pl["words"], pl["exc"][c6r])),
                   bool(fences[w2r].blocks(pl["base"], pl["words"], pl["exc"][w2r])),
                   bool(fences[DEPLOYED].blocks(pl["base"], pl["words"],
                                                pl["exc"][DEPLOYED])))
            tab[key] += 1
        marginal["overlap_n_common_attacks"] = len(common)
        marginal["overlap"] = {f"count={k[0]},width={k[1]},multi={k[2]}": v
                               for k, v in sorted(tab.items())}
        marginal["multi_only_over_count"] = sum(
            v for k, v in tab.items() if k[2] and not k[0])
        marginal["multi_only_over_width"] = sum(
            v for k, v in tab.items() if k[2] and not k[1])
        marginal["multi_missed_caught_by_either"] = sum(
            v for k, v in tab.items() if not k[2] and (k[0] or k[1]))
    # which component supplied the winning variant under the deployed union.
    # shortened() dedups union variants crediting the FIRST component (count:6), so the
    # width share below is a LOWER bound on what the fine cut contributes.
    attribution = {"note": "argmax variant's component under the deployed union; "
                           "variants shared by both cuts are credited to count:6 "
                           "(dedup order), so the width share is a lower bound"}
    for label, group in (
            ("blocked_attacks",
             [pl for pl in att_plans if pl["cells"][DEPLOYED]["j3"]
              and DEPLOYED in fences
              and fences[DEPLOYED].blocks(pl["base"], pl["words"], pl["exc"][DEPLOYED])]),
            ("all_attacks", [pl for pl in att_plans if pl["cells"][DEPLOYED]["j3"]]),
            ("benign", [pl for pl in plans if pl["row"].arm == "genuine"
                        and pl["cells"][DEPLOYED]["j3"]])):
        comps = Counter(pl["dep_best"].split("#")[0] for pl in group
                        if pl["dep_best"] is not None)
        attribution[label] = {"n": len(group), "by_component": dict(comps)}
    marginal["winning_component"] = attribution

    report = {
        "set": args.set_name, "eval": args.eval, "embedder": args.embedder,
        "attack_role": args.attack_role, "benign_generator": args.benign_generator,
        "budget": args.budget, "storage_dtype": "float16", "fence_form": "flat",
        "holdout": args.holdout, "seed": args.seed, "dim": DIM,
        "limit": args.limit or None,
        "n_benign_total": n_benign_total, "n_attack_total": n_attack_total,
        "n_distinct_texts_encoded": len(texts),
        "composition": {"kept": comp.kept, "dropped_no_anchor": comp.dropped_no_anchor},
        "injected_span_length": inj,
        "key_policies": list(KEY_POLICIES),
        "grid": grid,
        "marginal": marginal,
        "command": " ".join(sys.argv),
    }
    if args.set_name in PARITY and DEPLOYED in grid and not args.limit:
        exp = PARITY[args.set_name]
        got = grid[DEPLOYED]
        report["deployed_parity_check"] = {
            "against": f"results/v3/detection/e5-small-v2/{args.set_name.lower()}.json",
            "expected": exp,
            "got": {"excess_auroc": got["excess_auroc"],
                    "block": got["block_at_budget"],
                    "threshold_scoring": got["threshold_scoring"],
                    "in_sample": got["benign_block_in_sample"],
                    "seed0": got["realized_benign_seed0"]},
            "note": "pool is batch-encoded once, the detection cell encoded per row; "
                    "CPU transformer inference is batch-composition-sensitive, so "
                    "agreement is expected to ~1e-4, not bit-exact",
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


# --------------------------------------------------------------------------- bench
def time_hit(n_spans: int, dtype, repeats: int) -> list[float]:
    """One hit: n_spans stored vectors dotted with the anchor, then the max.

    Verbatim methodology of bench_excess_ms.py (excess_serving_ms.json)."""
    rng = np.random.default_rng(0)
    block = rng.standard_normal((n_spans, DIM)).astype(dtype)
    block /= np.linalg.norm(block.astype(np.float32), axis=1,
                            keepdims=True).astype(dtype)
    anchor = rng.standard_normal(DIM).astype(np.float32)
    anchor /= np.linalg.norm(anchor)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sims = block.astype(np.float32) @ anchor
        _ = float(sims.max())
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def bench(args) -> int:
    print(f"EFFECTIVE ARGS: mode=bench out={args.out} repeats={args.repeats} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    data = json.loads(Path(args.out).read_text(encoding="utf-8"))
    cache: dict[int, dict] = {}

    def measure(n: int) -> dict:
        if n not in cache:
            s = time_hit(n, np.float16, args.repeats)
            cache[n] = {"n_spans": n,
                        "ms_median": round(statistics.median(s), 5),
                        "ms_p95": round(sorted(s)[int(0.95 * len(s))], 5)}
        return cache[n]

    for spec, cell in data["grid"].items():
        d = cell.get("dots_per_hit", {}).get("pooled")
        if not d:
            continue
        cell["serving_ms"] = {
            "mean_spans": measure(max(1, int(round(d["mean"])))),
            "median": measure(max(1, int(d["median"]))),
            "p95": measure(max(1, int(d["p95"]))),
        }
        print(f"{spec:<38} spans(mean)={cell['serving_ms']['mean_spans']['n_spans']:>4} "
              f"-> {cell['serving_ms']['mean_spans']['ms_median']:.4f} ms", flush=True)
    data["bench_env"] = {
        "numpy": np.__version__, "dim": DIM, "repeats": args.repeats,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "note": "single-thread numpy, float16 storage, methodology of "
                "bench_excess_ms.py; serving path only (stored span vectors dotted "
                "with the query embedding the cache already computed, then a max)",
        "bench_command": " ".join(sys.argv),
    }
    Path(args.out).write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"updated {args.out}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("sweep", "bench"), default="sweep")
    p.add_argument("--eval", help="lmp_eval.jsonl / scp_eval.jsonl")
    p.add_argument("--embedder", default="intfloat/e5-small-v2")
    p.add_argument("--attack-role", default="ndss",
                   help="'ndss' for LMP, 'scp' for SCP")
    p.add_argument("--benign-generator", action="append", default=None)
    p.add_argument("--set-name", default="")
    p.add_argument("--budget", type=float, default=0.05)
    p.add_argument("--holdout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--encode-batch", type=int, default=256)
    p.add_argument("--limit", type=int, default=0,
                   help="smoke: keep only the first N rows per arm")
    p.add_argument("--repeats", type=int, default=2000, help="bench repeats")
    p.add_argument("--dump-scores", default="",
                   help="also write per-row per-cell excess/cosine here, for the "
                        "held-out policy-selection check (rq3_selection_split.py)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.benign_generator is None:
        args.benign_generator = ["human_comqa"]
    if args.mode == "bench":
        return bench(args)
    if not args.eval:
        p.error("--eval is required in sweep mode")
    return sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
