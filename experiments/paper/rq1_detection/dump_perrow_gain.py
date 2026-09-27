"""Per-row (base_cos, excess_span) dump for one eval set under one cut.

Feeds figures/src/fig3_coexistence.py. Uses the authoritative serving path --
``deletion.build_profile`` + ``deletion.excess``, per row, float16 storage -- so the rows
are on exactly the footing Table 1's cells are computed on, rather than a batched
re-encode (which agrees only to the ~1e-4 batch-composition band).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sentry.cache.defense.calibrate import (
    filter_benign_generators, load_pair_rows, parse_policy,
)
from sentry.cache.defense.deletion import build_profile, excess as excess_of


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--embedder", default="intfloat/e5-small-v2")
    p.add_argument("--attack-role", required=True)
    p.add_argument("--benign-generator", action="append", required=True)
    p.add_argument("--storage-dtype", default="float16")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    print(f"EFFECTIVE ARGS: eval={args.eval} policy={args.policy} "
          f"embedder={args.embedder} storage={args.storage_dtype} out={args.out}",
          flush=True)

    import torch
    torch.set_num_threads(args.threads)
    from sentry.embeddings import TransformerCLSEmbedder
    embedder = TransformerCLSEmbedder(args.embedder, batch_size=64)

    policy = parse_policy(args.policy)
    rows, comp = load_pair_rows(args.eval, (args.attack_role,))
    rows = filter_benign_generators(rows, args.benign_generator)
    print(f"composition: {comp.kept} dropped_no_anchor={comp.dropped_no_anchor}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n, skipped = 0, 0
    with out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            anchor = embedder.encode([r.anchor])[0]
            prof = build_profile(r.text, embedder, policy,
                                 storage_dtype=args.storage_dtype)
            if not prof.judgeable:
                skipped += 1
                continue
            rd = excess_of(prof, anchor)
            fh.write(json.dumps({
                "record_id": getattr(r, "record_id", None),
                "arm": r.arm, "family": r.family, "intent_id": r.intent_id,
                "words": len(r.text.split()),
                "base_cos": rd.base_cos, "excess_span": rd.excess_span,
            }, ensure_ascii=False) + "\n")
            n += 1
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    print(f"wrote {out}: {n} rows ({skipped} unjudgeable, omitted)", flush=True)
    print("command: " + " ".join(sys.argv), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
