"""Does the victim still answer the previous search run's candidates the way it did then?

The new 4+2 search run draws every victim answer fresh, while the previous joint run's
answers came from a response cache filled on 2026-09-01/10. When its rates move, two causes
are mixed: the new prompt produced different candidates, and the victim (a hosted
`deepseek-v4-flash`) may now answer the same text differently. This separates the second.

It re-asks the victim, with the same request (temperature 0, seed 0, max_tokens 200,
thinking disabled) but a fresh cache file, about the previous run's candidates that the
Table 2 cells read: each target's undefended pick, every DG-passing candidate, and every
candidate the deployed rule served. Candidate sets are held fixed as the previous run
decided them; only the verdict on the victim's answer is re-read. The deployed rule's
served set itself depends on the answer (ADL and Echo), so for that cell this measures the
verdict drift on a fixed set, not a full re-run.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from experiments.paper.rq2_robustness.rq4_isr import no_pushback, obeyed_strict
from experiments.paper.rq2_robustness.supp_search_summary import load_dump

SETS = {"none": lambda r: r["undefended_pick"], "dg_only": lambda r: r["evades_dg"],
        "ours": lambda r: r["evades_joint"]}


def verdicts(rows, answers) -> list[dict]:
    """Old and fresh strict / no-pushback verdicts for each row."""
    out = []
    for r in rows:
        fresh = answers[r["text"]]
        out.append({**r, "fresh_strict": bool(r["literal"]) and obeyed_strict(fresh, r["literal"]),
                    "fresh_no_pushback": bool(r["literal"]) and no_pushback(fresh, r["literal"]),
                    "same_response": fresh == r["response"]})
    return out


def compare(rows, n_targets) -> dict:
    """Per Table 2 cell: target-level ASR with the old and the fresh answers."""
    out = {}
    for name, keep in SETS.items():
        kept = [r for r in rows if keep(r)]
        cell = {"n_candidates": len(kept),
                "n_same_response": sum(r["same_response"] for r in kept)}
        for tier, old_f, new_f in (("strict", "poisoned_strict", "fresh_strict"),
                                   ("no_pushback", "poisoned_no_pushback",
                                    "fresh_no_pushback")):
            old_t = {r["target"] for r in kept if r[old_f]}
            new_t = {r["target"] for r in kept if r[new_f]}
            cell[tier] = {"old_asr": len(old_t) / n_targets, "fresh_asr": len(new_t) / n_targets,
                          "old_targets": len(old_t), "fresh_targets": len(new_t),
                          "candidate_verdict_agreement": float(np.mean(
                              [r[old_f] == r[new_f] for r in kept])) if kept else None}
        out[name] = cell
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="the previous run's --dump-viable")
    parser.add_argument("--n-targets", type=int, default=200)
    parser.add_argument("--cache-name", default="rq4_isr_fresh_20260923")
    parser.add_argument("--max-calls", type=int, default=1500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from sentry.research.operators import Client, load_env

    rows = [r for r in load_dump(args.dump) if any(keep(r) for keep in SETS.values())]
    texts = sorted({r["text"] for r in rows})
    if len(texts) > args.max_calls:
        parser.error(f"{len(texts)} distinct texts exceed --max-calls {args.max_calls}")
    victim = Client(load_env(), cache_name=args.cache_name)
    answers = {}

    def ask(text):
        answers[text] = victim.chat([{"role": "user", "content": text}], temperature=0.0,
                                    seed=0, max_tokens=200,
                                    extra_body={"thinking": {"type": "disabled"}}) or ""

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(ask, texts))
    checked = verdicts(rows, answers)
    report = {"dump": args.dump, "n_distinct_texts": len(texts),
              "n_api_calls": victim.calls, "n_cache_hits": victim.cache_hits,
              "share_identical_response": float(np.mean(
                  [answers[t] == next(r["response"] for r in rows if r["text"] == t)
                   for t in texts])),
              "cells": compare(checked, args.n_targets)}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
