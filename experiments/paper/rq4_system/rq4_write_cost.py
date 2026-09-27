"""The other half of the cost: what the defense charges on the write path.

\\S\\ref{sec:rq4} prices the serving path -- the dot products a hit pays for -- and stops
there. A cache operator also pays at insertion, once per entry, and pays for the vectors
forever. Reporting only the read side understates the bill, and quoting storage as "a few
kilobytes" hides which vectors are actually being counted.

Three quantities, measured rather than derived:

``served``     the shortened versions the serving statistic reads. This is the number the
               serving-cost figures already quote, and it is what a dot product is paid for.
``encoded``    the encoder forward passes one insertion costs. Strictly larger than
               ``served``: :func:`build_profile` also embeds the one-segment *deletions*,
               which the deployed statistic never reads.
``persisted``  the vectors the store actually writes -- whole, spans and deletions -- which
               is what the disk bill is computed from.

The gap between the second and the first is the honest finding here: the deletion vectors
are diagnostic, the served statistic is a max over spans alone, and a deployment that never
inspects them can drop them. Both totals are reported so the saving is visible and the
number in the paper says which one it is.

Everything is measured against the real span policy through the real ``build_profile``, so
a change to the cut moves these numbers automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BYTES = {"float16": 2, "float32": 4, "float64": 8}


def describe(values, scale=1.0):
    a = np.asarray(values, dtype=float) * scale
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, 0.95)), "max": float(a.max()),
            "min": float(a.min()), "n": int(a.size)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", required=True)
    p.add_argument("--attack-role", action="append", required=True)
    p.add_argument("--benign-generator", action="append", required=True)
    p.add_argument("--embedder", default="intfloat/e5-small-v2")
    p.add_argument("--policy", default="multi[count:4+width:2:cap16]/runs")
    p.add_argument("--storage-dtype", default="float16")
    p.add_argument("--arm", default="attack", choices=["attack", "genuine", "all"],
                   help="which arm the per-entry averages are taken over; the serving-cost "
                        "figures use the attack arm, so this defaults to the same")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent))
    from experiments.paper.rq1_detection.v3_detect import load_rows
    from sentry.cache.defense.calibrate import parse_policy
    from sentry.cache.defense.deletion import build_profile
    from sentry.cache.defense.spans import shortened
    from sentry.embeddings import TransformerCLSEmbedder

    policy = parse_policy(args.policy)
    rows, _ = load_rows(args.eval, args.attack_role, args.benign_generator)
    if args.arm != "all":
        rows = [r for r in rows if r["arm"] == args.arm]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} entries on the {args.arm} arm, policy {args.policy}", flush=True)

    embedder = TransformerCLSEmbedder(args.embedder)

    served, encoded, persisted, words, wall = [], [], [], [], []
    dimension = None
    for i, row in enumerate(rows):
        text = row["text"]
        variants = shortened(policy, text)
        start = time.perf_counter()
        profile = build_profile(text, embedder, policy, storage_dtype=args.storage_dtype)
        wall.append(time.perf_counter() - start)
        if not profile.judgeable:
            continue
        dimension = profile.dimension
        served.append(int(profile.spans.shape[0]))
        # one forward pass per distinct shortened text, plus the entry itself
        encoded.append(len(variants.all_texts) + 1)
        persisted.append(1 + int(profile.spans.shape[0]) + int(profile.deletions.shape[0]))
        words.append(profile.words)
        if i and i % 200 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    unit_bytes = BYTES[args.storage_dtype] * dimension
    served_only = [1 + s for s in served]

    report = {
        "eval": args.eval, "policy": args.policy, "embedder": args.embedder,
        "arm": args.arm, "n_entries": len(served), "dimension": dimension,
        "storage_dtype": args.storage_dtype, "bytes_per_vector": unit_bytes,
        "words_per_entry": describe(words),
        "serving_dot_products_per_hit": describe(served),
        "insertion_encoder_forward_passes": describe(encoded),
        "insertion_wall_ms": describe(wall, 1000.0),
        "vectors_persisted_as_shipped": describe(persisted),
        "vectors_persisted_spans_only": describe(served_only),
        "kb_per_entry_as_shipped": describe(persisted, unit_bytes / 1024.0),
        "kb_per_entry_spans_only": describe(served_only, unit_bytes / 1024.0),
        "storage_inflation_as_shipped": describe(persisted),
        "note": ("inflation is relative to an undefended cache, which stores one vector per "
                 "entry, so the vector counts are themselves the inflation factors. "
                 "'as shipped' counts the one-segment deletion vectors the store writes "
                 "but the served statistic never reads; 'spans only' is what a deployment "
                 "that drops them would pay."),
    }
    for key in ("mean", "median", "p95", "max"):
        report.setdefault("mb_per_10k_entries", {})[key] = {
            "as_shipped": report["kb_per_entry_as_shipped"][key] * 10000 / 1024.0,
            "spans_only": report["kb_per_entry_spans_only"][key] * 10000 / 1024.0,
        }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    d = report
    print(f"\n  words/entry            mean {d['words_per_entry']['mean']:.1f}")
    print(f"  dot products at serving mean {d['serving_dot_products_per_hit']['mean']:.1f}"
          f"  median {d['serving_dot_products_per_hit']['median']:.0f}"
          f"  max {d['serving_dot_products_per_hit']['max']:.0f}")
    print(f"  encoder passes at insert mean {d['insertion_encoder_forward_passes']['mean']:.1f}"
          f"  median {d['insertion_encoder_forward_passes']['median']:.0f}"
          f"  max {d['insertion_encoder_forward_passes']['max']:.0f}")
    print(f"  insertion wall           mean {d['insertion_wall_ms']['mean']:.1f} ms"
          f"  p95 {d['insertion_wall_ms']['p95']:.1f} ms")
    print(f"  vectors kept (as shipped) mean {d['vectors_persisted_as_shipped']['mean']:.1f}"
          f"   spans only {d['vectors_persisted_spans_only']['mean']:.1f}")
    print(f"  KB/entry (as shipped)    mean {d['kb_per_entry_as_shipped']['mean']:.1f}"
          f"   spans only {d['kb_per_entry_spans_only']['mean']:.1f}")
    print(f"  MB per 10k entries       as shipped "
          f"{d['mb_per_10k_entries']['mean']['as_shipped']:.0f}"
          f"   spans only {d['mb_per_10k_entries']['mean']['spans_only']:.0f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
